use std::borrow::Borrow;
use std::cmp::Reverse;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

use failure::Error;
use itertools::Itertools;
use ordered_float::OrderedFloat;

use crate::gridstore::common::*;
use crate::gridstore::stackable::{stackable, StackableNode};
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

pub fn tree_coalesce<T: Borrow<GridStore> + Clone + Debug>(
    stack_tree: &StackableNode<T>,
    match_opts: &MatchOpts,
) -> Result<Vec<CoalesceContext>, Error> {
    // the "tree" is just a node with no phrasematch; assure that this is the case
    debug_assert!(stack_tree.phrasematch.is_none(), "no phrasematch on root node");

    let mut contexts: Vec<CoalesceContext> = Vec::new();

    for node in &stack_tree.children {
        let subquery =
            node.phrasematch.as_ref().expect("phrasematch must be set on non-root tree nodes");

        let mut zoom_adjusted_match_options = match_opts.clone();
        if zoom_adjusted_match_options.zoom != subquery.store.borrow().zoom {
            zoom_adjusted_match_options = match_opts.adjust_to_zoom(subquery.store.borrow().zoom);
        }

        if node.children.len() == 0 {
            // we're not stacking this on top of anything, and we're not stacking anything else
            // on top of this, so we can grab a minimal set of elements here
            let bigger_max = 2 * MAX_CONTEXTS;

            // call tree_coalesce_single on each key group
            for key_group in subquery.match_keys.iter() {
                let grids = subquery.store.borrow().streaming_get_matching(
                    &key_group.key,
                    &zoom_adjusted_match_options,
                    // double to give us some sorting wiggle room
                    bigger_max,
                )?;

                let coalesced = tree_coalesce_single(
                    &subquery,
                    &zoom_adjusted_match_options,
                    grids,
                    key_group.id,
                )?;

                contexts.extend(coalesced);
            }
        } else {
            // we need lots of grids because we don't know where the things we're stacking on top
            // will be
            let mut prev_state: TreeCoalesceState = TreeCoalesceState::new();

            for key_group in subquery.match_keys.iter() {
                let grids = subquery.store.borrow().streaming_get_matching(
                    &key_group.key,
                    &zoom_adjusted_match_options,
                    MAX_GRIDS_PER_PHRASE,
                )?;

                for grid in grids.take(MAX_GRIDS_PER_PHRASE) {
                    let entry = grid_to_coalesce_entry(
                        &grid,
                        &subquery,
                        &zoom_adjusted_match_options,
                        key_group.id,
                    );
                    let context = CoalesceContext {
                        mask: subquery.mask,
                        relev: entry.grid_entry.relev,
                        entries: vec![entry],
                    };

                    contexts.push(context.clone());

                    let state_vec = prev_state
                        .entry((grid.grid_entry.x, grid.grid_entry.y))
                        .or_insert_with(|| vec![]);
                    state_vec.push(context);
                }
            }

            let mut multi_contexts = Vec::new();
            tree_recurse(
                &node,
                match_opts,
                &prev_state,
                subquery.store.borrow().zoom,
                &mut multi_contexts,
            )?;

            // penalize singnle-entry stacks and ascending stacks for... some reason?
            for mut context in multi_contexts {
                if context.entries.len() == 1 || context.entries[0].mask > context.entries[1].mask {
                    context.relev -= 0.01
                }
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

    // other stuff that ought to happen here:
    // - deduplication? if we have the same mask, same stack, better relevance, we should prefer it
    // - the thing where we don't allow jumps down in relevance that are bigger than 0.25
    // - way smarter stopping earlier, sorting, cutting off, etc.
    // - there's a relevance penalty for ascending vs. descending stuff for some reason... maybe
    //   we just shouldn't do that anymore though?
    contexts.truncate(MAX_CONTEXTS * 40);

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

    let contexts = coalesced.into_iter().map(|(_, entry)| CoalesceContext {
        entries: vec![entry.clone()],
        mask: entry.mask,
        relev: entry.grid_entry.relev,
    });

    Ok(contexts)
}

fn tree_recurse<T: Borrow<GridStore> + Clone + Debug>(
    node: &StackableNode<T>,
    match_opts: &MatchOpts,
    prev_state: &TreeCoalesceState,
    prev_zoom: u16,
    mut contexts: &mut Vec<CoalesceContext>,
) -> Result<(), Error> {
    for child in &node.children {
        // we need lots of grids because we don't know where the things we're stacking on top
        // will be
        let subquery =
            child.phrasematch.as_ref().expect("phrasematch must be set on non-root tree nodes");

        let mut zoom_adjusted_match_options = match_opts.clone();
        if zoom_adjusted_match_options.zoom != subquery.store.borrow().zoom {
            zoom_adjusted_match_options = match_opts.adjust_to_zoom(subquery.store.borrow().zoom);
        }

        let scale_factor: u16 = 1 << (subquery.store.borrow().zoom - prev_zoom);

        let mut state: TreeCoalesceState = TreeCoalesceState::new();

        for key_group in subquery.match_keys.iter() {
            let grids = subquery.store.borrow().streaming_get_matching(
                &key_group.key,
                &zoom_adjusted_match_options,
                MAX_GRIDS_PER_PHRASE,
            )?;

            for grid in grids.take(MAX_GRIDS_PER_PHRASE) {
                let prev_zoom_xy =
                    (grid.grid_entry.x / scale_factor, grid.grid_entry.y / scale_factor);

                if let Some(already_coalesced) = prev_state.get(&prev_zoom_xy) {
                    let entry = grid_to_coalesce_entry(
                        &grid,
                        &subquery,
                        &zoom_adjusted_match_options,
                        key_group.id,
                    );
                    for parent_context in already_coalesced {
                        let mut new_context = parent_context.clone();
                        new_context.entries.insert(0, entry.clone());

                        new_context.mask = new_context.mask | subquery.mask;
                        new_context.relev += entry.grid_entry.relev;

                        contexts.push(new_context.clone());

                        if child.children.len() > 0 {
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
        }

        if state.len() > 0 {
            tree_recurse(&child, match_opts, &state, subquery.store.borrow().zoom, &mut contexts)?;
        }
    }
    Ok(())
}

pub fn collapse_phrasematches<T: Borrow<GridStore> + Clone + Debug>(
    phrasematches: Vec<PhrasematchSubquery<T>>,
) -> Vec<PhrasematchSubquery<T>> {
    let mut phrasematch_results: Vec<PhrasematchSubquery<T>> = Vec::new();
    let mut phrasematch_map = HashMap::new();
    let mut group_hash;
    for phrasematch in phrasematches.into_iter() {
        group_hash = (OrderedFloat(phrasematch.weight), phrasematch.idx, phrasematch.mask);

        match phrasematch_map.entry(group_hash) {
            Entry::Vacant(entry) => {
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
            Entry::Occupied(mut grouped_phrasematch) => {
                grouped_phrasematch.get_mut().match_keys.push(phrasematch.match_keys[0].clone());
            }
        }
    }
    for (_key, val) in phrasematch_map {
        phrasematch_results.push(val);
    }
    phrasematch_results
}

pub fn stack_and_coalesce<T: Borrow<GridStore> + Clone + Debug>(
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
            non_overlapping_indexes: HashSet::new(),
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
            non_overlapping_indexes: HashSet::new(),
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
