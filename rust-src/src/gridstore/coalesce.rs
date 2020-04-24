use std::borrow::Borrow;
use std::cmp::{Ordering, Reverse};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::sync::Arc;

use failure::Error;
use indexmap::map::{Entry as IndexMapEntry, IndexMap};
use itertools::Itertools;
use min_max_heap::MinMaxHeap;
use ordered_float::OrderedFloat;
use rayon::prelude::*;

use crate::gridstore::common::*;
use crate::gridstore::stackable::{stackable, StackableNode, StackableTree};
use crate::gridstore::store::GridStore;

/// Takes a vector of phrasematch subqueries (stack) and match options, gets matching grids, sorts the grids,
/// and returns a result of a sorted vector of contexts (lists of grids with added metadata)
pub fn coalesce<T: Borrow<GridStore> + Clone + Debug>(
    stack: Vec<PhrasematchSubquery<T>>,
    match_opts: &MatchOpts,
) -> Result<Vec<CoalesceContext>, Error> {
    let contexts = if stack.len() <= 1 {
        coalesce_single(&stack[0], match_opts)?
    } else {
        coalesce_multi(stack, match_opts)?
    };

    let mut out = Vec::with_capacity(MAX_CONTEXTS);
    if !contexts.is_empty() {
        let max_relevance = contexts[0].relev;
        let mut sets: HashSet<u64> = HashSet::new();
        for context in contexts {
            if out.len() >= MAX_CONTEXTS {
                break;
            }
            // 0.25 is the smallest allowed relevance
            if max_relevance - context.relev >= 0.25 {
                break;
            }
            let inserted = sets.insert(context.entries[0].tmp_id.into());
            if inserted {
                out.push(context);
            }
        }
    }
    Ok(out)
}

fn grid_to_coalesce_entry<T: Borrow<GridStore> + Clone>(
    grid: &MatchEntry,
    subquery: &PhrasematchSubquery<T>,
    match_opts: &MatchOpts,
    phrasematch_id: u32,
) -> CoalesceEntry {
    // Zoom has been adjusted in coalesce_multi, or correct zoom has been passed in for coalesce_single
    debug_assert!(match_opts.zoom == subquery.store.borrow().zoom);
    let relevance = grid.grid_entry.relev * subquery.weight;

    CoalesceEntry {
        grid_entry: GridEntry { relev: relevance, ..grid.grid_entry },
        matches_language: grid.matches_language,
        idx: subquery.idx,
        tmp_id: ((subquery.idx as u32) << 25) + grid.grid_entry.id,
        mask: subquery.mask,
        distance: grid.distance,
        scoredist: grid.scoredist,
        phrasematch_id,
    }
}

fn coalesce_single<T: Borrow<GridStore> + Clone>(
    subquery: &PhrasematchSubquery<T>,
    match_opts: &MatchOpts,
) -> Result<Vec<CoalesceContext>, Error> {
    let bigger_max = 2 * MAX_CONTEXTS;

    let grids = subquery.store.borrow().streaming_get_matching(
        &subquery.match_keys[0].key,
        match_opts,
        bigger_max,
    )?;
    let mut max_relevance: f64 = 0.;
    let mut previous_id: u32 = 0;
    let mut previous_relevance: f64 = 0.;
    let mut previous_scoredist: f64 = 0.;
    let mut min_scoredist = std::f64::MAX;
    let mut feature_count: usize = 0;

    let mut coalesced: HashMap<u32, CoalesceEntry> = HashMap::new();

    for grid in grids {
        let coalesce_entry = grid_to_coalesce_entry(&grid, subquery, match_opts, 0);

        // If it's the same feature as the last one, but a lower scoredist don't add it
        if previous_id == coalesce_entry.grid_entry.id
            && coalesce_entry.scoredist <= previous_scoredist
        {
            continue;
        }

        if feature_count > bigger_max {
            if coalesce_entry.scoredist < min_scoredist {
                continue;
            } else if coalesce_entry.grid_entry.relev < previous_relevance {
                // Grids should be sorted by relevance coming out of get_matching,
                // so if it's lower than the last relevance, stop
                break;
            }
        }

        if max_relevance - coalesce_entry.grid_entry.relev >= 0.25 {
            break;
        }
        if coalesce_entry.grid_entry.relev > max_relevance {
            max_relevance = coalesce_entry.grid_entry.relev;
        }

        // Save current values before mocing into coalesced
        let current_id = coalesce_entry.grid_entry.id;
        let current_relev = coalesce_entry.grid_entry.relev;
        let current_scoredist = coalesce_entry.scoredist;

        // If it's the same feature as one that's been added before, but a higher scoredist, update the entry
        match coalesced.entry(current_id) {
            Entry::Occupied(mut already_coalesced) => {
                if current_scoredist > already_coalesced.get().scoredist
                    && current_relev >= already_coalesced.get().grid_entry.relev
                {
                    already_coalesced.insert(coalesce_entry);
                }
            }
            Entry::Vacant(entry) => {
                entry.insert(coalesce_entry);
            }
        }

        if previous_id != current_id {
            feature_count += 1;
        }
        if match_opts.proximity.is_none() && feature_count > bigger_max {
            break;
        }
        if current_scoredist < min_scoredist {
            min_scoredist = current_scoredist;
        }
        previous_id = current_id;
        previous_relevance = current_relev;
        previous_scoredist = current_scoredist;
    }

    let mut contexts: Vec<CoalesceContext> = coalesced
        .iter()
        .map(|(_, entry)| CoalesceContext {
            entries: vec![entry.clone()],
            mask: entry.mask,
            relev: entry.grid_entry.relev,
        })
        .collect();

    contexts.sort_by_key(|context| {
        Reverse((
            OrderedFloat(context.relev),
            OrderedFloat(context.entries[0].scoredist),
            context.entries[0].grid_entry.x,
            context.entries[0].grid_entry.y,
            context.entries[0].grid_entry.id,
        ))
    });

    contexts.truncate(MAX_CONTEXTS);
    Ok(contexts)
}

fn coalesce_multi<T: Borrow<GridStore> + Clone>(
    mut stack: Vec<PhrasematchSubquery<T>>,
    match_opts: &MatchOpts,
) -> Result<Vec<CoalesceContext>, Error> {
    stack.sort_by_key(|subquery| (subquery.store.borrow().zoom, subquery.idx));

    let mut coalesced: HashMap<(u16, u16, u16), Vec<CoalesceContext>> = HashMap::new();
    let mut contexts: Vec<CoalesceContext> = Vec::new();

    let mut max_relevance: f64 = 0.;

    let mut zoom_adjusted_match_options = match_opts.clone();

    for (i, subquery) in stack.iter().enumerate() {
        let mut to_add_to_coalesced: HashMap<(u16, u16, u16), Vec<CoalesceContext>> =
            HashMap::new();
        let compatible_zooms: Vec<u16> = stack
            .iter()
            .filter_map(|subquery_b| {
                if subquery.idx == subquery_b.idx
                    || subquery.store.borrow().zoom < subquery_b.store.borrow().zoom
                {
                    None
                } else {
                    Some(subquery_b.store.borrow().zoom)
                }
            })
            .dedup()
            .collect();

        if zoom_adjusted_match_options.zoom != subquery.store.borrow().zoom {
            zoom_adjusted_match_options = match_opts.adjust_to_zoom(subquery.store.borrow().zoom);
        }

        let grids = subquery.store.borrow().streaming_get_matching(
            &subquery.match_keys[0].key,
            &zoom_adjusted_match_options,
            MAX_GRIDS_PER_PHRASE,
        )?;

        for grid in grids.take(MAX_GRIDS_PER_PHRASE) {
            let coalesce_entry =
                grid_to_coalesce_entry(&grid, subquery, &zoom_adjusted_match_options, 0);

            let zxy = (subquery.store.borrow().zoom, grid.grid_entry.x, grid.grid_entry.y);

            let mut context_mask = coalesce_entry.mask;
            let mut context_relevance = coalesce_entry.grid_entry.relev;
            let mut entries: Vec<CoalesceEntry> = vec![coalesce_entry];

            // See which other zooms are compatible.
            // These should all be lower zooms, so "zoom out" by dividing by 2^(difference in zooms)
            for other_zoom in compatible_zooms.iter() {
                let scale_factor: u16 = 1 << (subquery.store.borrow().zoom - *other_zoom);
                let other_zxy = (
                    *other_zoom,
                    entries[0].grid_entry.x / scale_factor,
                    entries[0].grid_entry.y / scale_factor,
                );

                if let Some(already_coalesced) = coalesced.get(&other_zxy) {
                    let mut prev_mask = 0;
                    let mut prev_relev: f64 = 0.;
                    for parent_context in already_coalesced {
                        for parent_entry in &parent_context.entries {
                            // this cover is functionally identical with previous and
                            // is more relevant, replace the previous.
                            if parent_entry.mask == prev_mask
                                && parent_entry.grid_entry.relev > prev_relev
                            {
                                entries.pop();
                                entries.push(parent_entry.clone());
                                // Update the context-level aggregate relev
                                context_relevance -= prev_relev;
                                context_relevance += parent_entry.grid_entry.relev;

                                prev_mask = parent_entry.mask;
                                prev_relev = parent_entry.grid_entry.relev;
                            } else if (context_mask & parent_entry.mask) == 0 {
                                entries.push(parent_entry.clone());

                                context_relevance += parent_entry.grid_entry.relev;
                                context_mask = context_mask | parent_entry.mask;

                                prev_mask = parent_entry.mask;
                                prev_relev = parent_entry.grid_entry.relev;
                            }
                        }
                    }
                }
            }
            if context_relevance > max_relevance {
                max_relevance = context_relevance;
            }

            if i == (stack.len() - 1) {
                if entries.len() == 1 {
                    // Slightly penalize contexts that have no stacking
                    context_relevance -= 0.01;
                } else if entries[0].mask > entries[1].mask {
                    // Slightly penalize contexts in ascending order
                    context_relevance -= 0.01
                }

                if max_relevance - context_relevance < 0.25 {
                    contexts.push(CoalesceContext {
                        entries,
                        mask: context_mask,
                        relev: context_relevance,
                    });
                }
            } else if i == 0 || entries.len() > 1 {
                if let Some(already_coalesced) = to_add_to_coalesced.get_mut(&zxy) {
                    already_coalesced.push(CoalesceContext {
                        entries,
                        mask: context_mask,
                        relev: context_relevance,
                    });
                } else {
                    to_add_to_coalesced.insert(
                        zxy,
                        vec![CoalesceContext {
                            entries,
                            mask: context_mask,
                            relev: context_relevance,
                        }],
                    );
                }
            }
        }
        for (to_add_zxy, to_add_context) in to_add_to_coalesced {
            if let Some(existing_vector) = coalesced.get_mut(&to_add_zxy) {
                existing_vector.extend(to_add_context);
            } else {
                coalesced.insert(to_add_zxy, to_add_context);
            }
        }
    }

    for (_, matched) in coalesced {
        for context in matched {
            if max_relevance - context.relev < 0.25 {
                contexts.push(context);
            }
        }
    }

    contexts.sort_by_key(|context| {
        (
            Reverse(OrderedFloat(context.relev)),
            Reverse(OrderedFloat(context.entries[0].scoredist)),
            context.entries[0].idx,
            Reverse(context.entries[0].grid_entry.x),
            Reverse(context.entries[0].grid_entry.y),
            Reverse(context.entries[0].grid_entry.id),
        )
    });

    Ok(contexts)
}

type TreeCoalesceState = HashMap<(u16, u16), Vec<CoalesceContext>>;
struct CoalesceStep<'a, T: Borrow<GridStore> + Clone + Debug> {
    node: &'a StackableNode<'a, T>,
    prev_state: Option<Arc<TreeCoalesceState>>,
    prev_zoom: u16,
    match_opts: MatchOpts,
}

impl<T: Borrow<GridStore> + Clone + Debug> CoalesceStep<'_, T> {
    fn new<'a>(
        node: &'a StackableNode<'a, T>,
        prev_state: Option<Arc<TreeCoalesceState>>,
        prev_zoom: u16,
        match_opts: &MatchOpts,
    ) -> CoalesceStep<'a, T> {
        let subquery = node.phrasematch.expect("phrasematch required");
        let match_opts = if match_opts.zoom == subquery.store.borrow().zoom {
            match_opts.clone()
        } else {
            match_opts.adjust_to_zoom(subquery.store.borrow().zoom)
        };
        CoalesceStep { node, prev_state, prev_zoom, match_opts }
    }
}

impl<T: Borrow<GridStore> + Clone + Debug> Ord for CoalesceStep<'_, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.node.max_relev).cmp(&OrderedFloat(other.node.max_relev))
    }
}
impl<T: Borrow<GridStore> + Clone + Debug> PartialOrd for CoalesceStep<'_, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: Borrow<GridStore> + Clone + Debug> PartialEq for CoalesceStep<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        OrderedFloat(self.node.max_relev) == OrderedFloat(other.node.max_relev)
    }
}
impl<T: Borrow<GridStore> + Clone + Debug> Eq for CoalesceStep<'_, T> {}

struct KeyFetchStep<T: Borrow<GridStore> + Clone + Debug> {
    key_id: u32,
    subquery: PhrasematchSubquery<T>,
    key: MatchKey,
    match_opts: MatchOpts,
    is_single: bool,
}

// this is the thing that comes out of the first phase of two-phase coalesce
// for single coalesce, we just do everything in phase 1, whereas for multi-coalesce,
// we only do the first part, depending what kind of node we're on, we'll return different things
enum KeyFetchResult {
    Single(ConstrainedPriorityQueue<CoalesceContext>),
    Multi((u32, Vec<MatchEntry>)),
}

fn penalize_multi_context(context: &mut CoalesceContext) {
    // penalize single-entry stacks and ascending stacks for... some reason?
    if context.entries.len() == 1 || context.entries[0].mask > context.entries[1].mask {
        context.relev -= 0.01
    }
}

pub const COALESCE_CHUNK_SIZE: usize = 8;

pub fn tree_coalesce<T: Borrow<GridStore> + Clone + Debug + Send + Sync>(
    stack_tree: &StackableTree<T>,
    match_opts: &MatchOpts,
) -> Result<Vec<CoalesceContext>, Error> {
    debug_assert!(stack_tree.root.phrasematch.is_none(), "no phrasematch on root node");

    let mut contexts: ConstrainedPriorityQueue<CoalesceContext> =
        ConstrainedPriorityQueue::new(MAX_CONTEXTS * 20);
    let mut steps: MinMaxHeap<CoalesceStep<T>> = MinMaxHeap::new();
    let mut data_cache: HashMap<u32, Vec<MatchEntry>> = HashMap::new();

    for child_idx in &stack_tree.root.children {
        if let Some(node) = stack_tree.arena.get(*child_idx) {
            // push the first set of nodes into the queue
            steps.push(CoalesceStep::new(&node, None, 0, match_opts));
        }
    }

    while steps.len() > 0 {
        // as long as there's still work to do, we'll execute it a chunk at a time, peeling off
        // the next few best nodes, and executing on them in parallel
        let mut step_chunk = Vec::with_capacity(COALESCE_CHUNK_SIZE);
        let mut keys = Vec::new();
        for _i in 0..COALESCE_CHUNK_SIZE {
            if let Some(step) = steps.pop_max() {
                // if we've already gotten as many items as we're going to return, only keep processing
                // if anything we have left has the possibility of beating our worst current result
                if contexts.len() >= contexts.max_size {
                    if step.node.max_relev
                        <= contexts.peek_min().expect("contexts can't be empty").relev
                    {
                        break;
                    }
                }

                // if this is a single item with no parents or children, we can do a more-efficient
                // coalesce operation in the first phase, rather, than a two-phase coalesce
                let is_single = step.prev_state.is_none() && step.node.children.len() == 0;

                let subquery = step
                    .node
                    .phrasematch
                    .as_ref()
                    .expect("phrasematch must be set on non-root tree nodes");

                for key_group in subquery.match_keys.iter() {
                    if is_single || !data_cache.contains_key(&key_group.id) {
                        keys.push(KeyFetchStep {
                            key_id: key_group.id,
                            subquery: (*subquery).clone(),
                            key: key_group.key.clone(),
                            match_opts: step.match_opts.clone(),
                            is_single,
                        });
                    }
                }

                if !is_single {
                    step_chunk.push(step);
                }
            }
        }

        // phase 1: we get any data we don't already have in cache (and for single coalesce, we
        // just do the whole operation)
        let key_data: Vec<Result<_, Error>> = keys
            .into_par_iter()
            .map(|key_step| {
                if key_step.is_single {
                    // this is a first-level node with no children, so short-circuit to a single-coalesce
                    // stategy
                    //
                    // we're not stacking this on top of anything, and we're not stacking anything else
                    // on top of this, so we can grab a minimal set of elements here
                    let bigger_max = 2 * MAX_CONTEXTS;

                    // call tree_coalesce_single on each key group
                    let mut step_contexts: ConstrainedPriorityQueue<CoalesceContext> =
                        ConstrainedPriorityQueue::new(MAX_CONTEXTS);

                    let grids = key_step.subquery.store.borrow().streaming_get_matching(
                        &key_step.key,
                        &key_step.match_opts,
                        // double to give us some sorting wiggle room
                        bigger_max,
                    )?;

                    let coalesced = tree_coalesce_single(
                        &key_step.subquery,
                        &key_step.match_opts,
                        grids,
                        key_step.key_id,
                    )?;

                    for entry in coalesced {
                        step_contexts.push(entry);
                    }

                    Ok(KeyFetchResult::Single(step_contexts))
                } else {
                    let data: Vec<_> = key_step
                        .subquery
                        .store
                        .borrow()
                        .streaming_get_matching(
                            &key_step.key,
                            &key_step.match_opts,
                            MAX_GRIDS_PER_PHRASE,
                        )?
                        .take(MAX_GRIDS_PER_PHRASE)
                        .collect();
                    Ok(KeyFetchResult::Multi((key_step.key_id, data)))
                }
            })
            .collect();

        for result in key_data {
            match result? {
                KeyFetchResult::Single(phrasematch_contexts) => {
                    // for coalesce single we got back full-on contexts
                    for context in phrasematch_contexts {
                        contexts.push(context);
                    }
                }
                KeyFetchResult::Multi((key_id, data)) => {
                    // for coalesce multi we got back cached data to be used in the next step
                    data_cache.insert(key_id, data);
                }
            }
        }

        // phase 2: for complex coalesce, we do the coalescing in a second phase now that the data has been
        // fetched
        let chunk_results: Vec<Result<(Vec<CoalesceContext>, Vec<CoalesceStep<'_, T>>), Error>> =
            step_chunk
                .into_par_iter()
                .map(|step| {
                    let subquery = step
                        .node
                        .phrasematch
                        .as_ref()
                        .expect("phrasematch must be set on non-root tree nodes");

                    let mut phrasematch_contexts: Vec<CoalesceContext> = Vec::new();

                    let scale_factor: u16 = 1 << (subquery.store.borrow().zoom - step.prev_zoom);

                    let mut state: TreeCoalesceState = TreeCoalesceState::new();

                    for key_group in subquery.match_keys.iter() {
                        let grids = data_cache
                            .get(&key_group.id)
                            .expect("data must have been pre-collected");

                        let mut step_contexts: ConstrainedPriorityQueue<CoalesceContext> =
                            ConstrainedPriorityQueue::new(MAX_CONTEXTS);

                        if let Some(prev_state) = &step.prev_state {
                            // we're stacking on top of something that was already there
                            for grid in grids.iter() {
                                let prev_zoom_xy = (
                                    grid.grid_entry.x / scale_factor,
                                    grid.grid_entry.y / scale_factor,
                                );

                                if let Some(already_coalesced) = prev_state.get(&prev_zoom_xy) {
                                    let entry = grid_to_coalesce_entry(
                                        &grid,
                                        &subquery,
                                        &step.match_opts,
                                        key_group.id,
                                    );
                                    for parent_context in already_coalesced {
                                        let mut new_context = parent_context.clone();
                                        new_context.entries.insert(0, entry.clone());

                                        new_context.mask = new_context.mask | subquery.mask;
                                        new_context.relev += entry.grid_entry.relev;

                                        let mut out_context = new_context.clone();
                                        penalize_multi_context(&mut out_context);
                                        step_contexts.push(out_context);

                                        if step.node.children.len() > 0 {
                                            // only bother with getting ready to recurse if we have any children to
                                            // operate on
                                            let state_vec = state
                                                .entry((grid.grid_entry.x, grid.grid_entry.y))
                                                .or_insert_with(|| vec![]);
                                            state_vec.push(new_context);
                                        }
                                    }
                                }
                            }
                        } else {
                            // there's nothing to stack on already there, but we'll be stacking on this in
                            // the future
                            for grid in grids.iter() {
                                let entry = grid_to_coalesce_entry(
                                    &grid,
                                    &subquery,
                                    &step.match_opts,
                                    key_group.id,
                                );
                                let context = CoalesceContext {
                                    mask: subquery.mask,
                                    relev: entry.grid_entry.relev,
                                    entries: vec![entry],
                                };

                                let mut out_context = context.clone();
                                penalize_multi_context(&mut out_context);
                                step_contexts.push(out_context);

                                let state_vec = state
                                    .entry((grid.grid_entry.x, grid.grid_entry.y))
                                    .or_insert_with(|| vec![]);
                                state_vec.push(context);
                            }
                        }
                        phrasematch_contexts.extend(step_contexts.into_iter());
                    }

                    let mut next_steps = Vec::with_capacity(step.node.children.len());
                    if state.len() > 0 {
                        let state = Arc::new(state);
                        for child_idx in step.node.children.iter() {
                            if let Some(child) = stack_tree.arena.get(*child_idx) {
                                next_steps.push(CoalesceStep::new(
                                    &child,
                                    Some(state.clone()),
                                    subquery.store.borrow().zoom,
                                    match_opts,
                                ));
                            }
                        }
                    }

                    Ok((phrasematch_contexts, next_steps))
                })
                .collect();

        for result in chunk_results {
            let (phrasematch_contexts, next_steps) = result?;
            for context in phrasematch_contexts {
                contexts.push(context);
            }
            for step in next_steps {
                steps.push(step);
            }
        }
    }

    // other stuff that ought to happen here:
    // - deduplication? if we have the same mask, same stack, better relevance, we should prefer it
    // - the thing where we don't allow jumps down in relevance that are bigger than 0.25
    // - way smarter stopping earlier, sorting, cutting off, etc.
    // - there's a relevance penalty for ascending vs. descending stuff for some reason... maybe
    //   we just shouldn't do that anymore though?

    let mut contexts = contexts.into_vec_desc();
    // if the same ID occurs more than once, just keep the first (highest-relevance) one
    let mut unique_ids: HashSet<u64> = HashSet::with_capacity(contexts.len());
    contexts.retain(|context| unique_ids.insert(context.entries[0].tmp_id.into()));
    Ok(contexts)
}

fn tree_coalesce_single<T: Borrow<GridStore> + Clone, U: Iterator<Item = MatchEntry>>(
    subquery: &PhrasematchSubquery<T>,
    match_opts: &MatchOpts,
    grids: U,
    phrasematch_id: u32,
) -> Result<impl Iterator<Item = CoalesceContext>, Error> {
    let bigger_max = 2 * MAX_CONTEXTS;

    let mut max_relevance: f64 = 0.;
    let mut previous_id: u32 = 0;
    let mut previous_relevance: f64 = 0.;
    let mut previous_scoredist: f64 = 0.;
    let mut min_scoredist = std::f64::MAX;
    let mut feature_count: usize = 0;

    let mut coalesced: HashMap<u32, CoalesceEntry> = HashMap::new();

    for grid in grids {
        let coalesce_entry = grid_to_coalesce_entry(&grid, &subquery, match_opts, phrasematch_id);

        // If it's the same feature as the last one, but a lower scoredist don't add it
        if previous_id == coalesce_entry.grid_entry.id
            && coalesce_entry.scoredist <= previous_scoredist
        {
            continue;
        }

        if feature_count > bigger_max {
            if coalesce_entry.scoredist < min_scoredist {
                continue;
            } else if coalesce_entry.grid_entry.relev < previous_relevance {
                // Grids should be sorted by relevance coming out of get_matching,
                // so if it's lower than the last relevance, stop
                break;
            }
        }

        if max_relevance - coalesce_entry.grid_entry.relev >= 0.25 {
            break;
        }
        if coalesce_entry.grid_entry.relev > max_relevance {
            max_relevance = coalesce_entry.grid_entry.relev;
        }

        // Save current values before mocing into coalesced
        let current_id = coalesce_entry.grid_entry.id;
        let current_relev = coalesce_entry.grid_entry.relev;
        let current_scoredist = coalesce_entry.scoredist;

        // If it's the same feature as one that's been added before, but a higher scoredist, update the entry
        match coalesced.entry(current_id) {
            Entry::Occupied(mut already_coalesced) => {
                if current_scoredist > already_coalesced.get().scoredist
                    && current_relev >= already_coalesced.get().grid_entry.relev
                {
                    already_coalesced.insert(coalesce_entry);
                }
            }
            Entry::Vacant(entry) => {
                entry.insert(coalesce_entry);
            }
        }

        if previous_id != current_id {
            feature_count += 1;
        }
        if match_opts.proximity.is_none() && feature_count > bigger_max {
            break;
        }
        if current_scoredist < min_scoredist {
            min_scoredist = current_scoredist;
        }
        previous_id = current_id;
        previous_relevance = current_relev;
        previous_scoredist = current_scoredist;
    }

    let mut ids: Vec<_> = coalesced.keys().cloned().collect();
    ids.sort();
    let contexts = ids.into_iter().map(move |id| {
        let entry = coalesced.remove(&id).expect("hashmap must contain key");
        CoalesceContext { mask: entry.mask, relev: entry.grid_entry.relev, entries: vec![entry] }
    });

    Ok(contexts)
}

pub fn collapse_phrasematches<T: Borrow<GridStore> + Clone + Debug>(
    phrasematches: Vec<PhrasematchSubquery<T>>,
) -> Vec<PhrasematchSubquery<T>> {
    let mut phrasematch_results: Vec<PhrasematchSubquery<T>> = Vec::new();
    let mut phrasematch_map = IndexMap::new();
    let mut group_hash;
    for phrasematch in phrasematches.into_iter() {
        group_hash = (OrderedFloat(phrasematch.weight), phrasematch.idx, phrasematch.mask);

        match phrasematch_map.entry(group_hash) {
            IndexMapEntry::Vacant(entry) => {
                let pm = PhrasematchSubquery {
                    store: phrasematch.store,
                    idx: phrasematch.idx,
                    non_overlapping_indexes: phrasematch.non_overlapping_indexes,
                    weight: phrasematch.weight,
                    mask: phrasematch.mask,
                    match_keys: phrasematch.match_keys,
                };
                entry.insert(pm);
            }
            IndexMapEntry::Occupied(mut grouped_phrasematch) => {
                grouped_phrasematch.get_mut().match_keys.push(phrasematch.match_keys[0].clone());
            }
        }
    }
    for (_key, val) in phrasematch_map {
        phrasematch_results.push(val);
    }
    phrasematch_results
}

pub fn stack_and_coalesce<T: Borrow<GridStore> + Clone + Debug + Send + Sync>(
    phrasematches: &Vec<PhrasematchSubquery<T>>,
    match_opts: &MatchOpts,
) -> Result<Vec<CoalesceContext>, Error> {
    // currently stackable requires double-wrapping the phrasematches vector, which requires an
    // extra clone; ideally we wouldn't do that
    let collapsed_phrasematches = collapse_phrasematches(phrasematches.to_vec());
    let tree = stackable(&collapsed_phrasematches);
    tree_coalesce(&tree, &match_opts)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::gridstore::builder::*;
    use crate::gridstore::common::MatchPhrase::Range;
    use fixedbitset::FixedBitSet;

    #[test]
    fn collapse_phrasematches_test() {
        let directory: tempfile::TempDir = tempfile::tempdir().unwrap();
        let mut builder = GridStoreBuilder::new(directory.path()).unwrap();

        let key = GridKey { phrase_id: 1, lang_set: 1 };

        let entries = vec![
            GridEntry { id: 2, x: 2, y: 2, relev: 0.8, score: 3, source_phrase_hash: 0 },
            GridEntry { id: 3, x: 3, y: 3, relev: 1., score: 1, source_phrase_hash: 1 },
            GridEntry { id: 1, x: 1, y: 1, relev: 1., score: 7, source_phrase_hash: 2 },
        ];
        builder.insert(&key, entries).expect("Unable to insert record");
        builder.finish().unwrap();
        let store1 = GridStore::new_with_options(directory.path(), 14, 1, 200.).unwrap();

        let a1 = PhrasematchSubquery {
            store: &store1,
            idx: 2,
            non_overlapping_indexes: FixedBitSet::with_capacity(128),
            weight: 0.5,
            mask: 1,
            match_keys: vec![MatchKeyWithId {
                key: MatchKey { match_phrase: Range { start: 0, end: 1 }, lang_set: 0 },
                id: 1,
            }],
        };

        let a2 = PhrasematchSubquery {
            store: &store1,
            idx: 2,
            non_overlapping_indexes: FixedBitSet::with_capacity(128),
            weight: 0.5,
            mask: 1,
            match_keys: vec![MatchKeyWithId {
                key: MatchKey { match_phrase: Range { start: 0, end: 1 }, lang_set: 0 },
                id: 2,
            }],
        };
        let phrasematch_results = vec![a1, a2];
        let collapsed_phrasematch = collapse_phrasematches(phrasematch_results.to_vec());
        assert_eq!(
            collapsed_phrasematch[0].match_keys.len(),
            2,
            "phrasematch match_keys with the same idx, weight and mask are grouped together"
        );
        assert_eq!(collapsed_phrasematch[0].match_keys[0].id, 1);
        assert_eq!(collapsed_phrasematch[0].match_keys[1].id, 2);
    }
}
