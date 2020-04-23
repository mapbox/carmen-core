#![allow(dead_code)]
use std::borrow::Borrow;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;

use crate::gridstore::common::*;
use crate::gridstore::store::*;

use fixedbitset::FixedBitSet;
use generational_arena::{Arena, Index as ArenaIndex};
use ordered_float::OrderedFloat;

#[derive(Debug, Clone)]
pub struct StackableNode<'a, T: Borrow<GridStore> + Clone + Debug> {
    pub phrasematch: Option<&'a PhrasematchSubquery<T>>,
    pub children: Vec<ArenaIndex>,
    pub bmask: FixedBitSet,
    pub mask: u32,
    pub idx: u16,
    pub max_relev: f64,
    pub zoom: u16,
}

impl<'a, T: Borrow<GridStore> + Clone + Debug> StackableNode<'a, T> {
    fn is_leaf(&self) -> bool {
        self.children.len() == 0
    }
}

//tree traversal used only for tests
pub fn bfs<T: Borrow<GridStore> + Clone + Debug>(tree: StackableTree<T>) -> Vec<StackableNode<T>> {
    let mut node_vec: Vec<StackableNode<T>> = vec![];
    let mut stack: Vec<_> = vec![];

    stack.push(tree.root.clone());

    while stack.len() > 0 {
        let node = stack.pop().unwrap();
        node_vec.push(node.clone());
        for maybe_child in node.children {
            if let Some(child) = tree.arena.get(maybe_child) {
                stack.push(child.clone());
            }
        }
    }
    return node_vec;
}

pub const LEAF_SOFT_MAX: usize = 1024;

#[derive(Debug, Clone)]
pub struct ArenaManager<'a, T: Borrow<GridStore> + Clone + Debug> {
    arena: Arena<StackableNode<'a, T>>,
    // this is a map from max_relev to the count of leaves with that max_relev, and a list of all
    // the arena indexes (leaf or otherwise) with that max_index
    relev_map: HashMap<OrderedFloat<f64>, (usize, Vec<ArenaIndex>)>,
    min_relev: OrderedFloat<f64>,
    total_leaves: usize,
    soft_max: usize,
}

impl<'a, T: Borrow<GridStore> + Clone + Debug> ArenaManager<'a, T> {
    fn new() -> Self {
        ArenaManager {
            arena: Arena::new(),
            relev_map: HashMap::new(),
            min_relev: OrderedFloat(std::f64::MAX),
            total_leaves: 0,
            soft_max: LEAF_SOFT_MAX,
        }
    }

    #[inline(always)]
    fn is_full(&self) -> bool {
        self.total_leaves >= self.soft_max
    }

    fn add(&mut self, node: StackableNode<'a, T>) -> Option<ArenaIndex> {
        let max_relev = OrderedFloat(node.max_relev);
        let is_leaf = node.children.len() == 0;
        let old_total_leaves = self.total_leaves;

        if old_total_leaves >= self.soft_max && max_relev < self.min_relev {
            // we're constrained, and this is worse than our worst, so don't keep it
            None
        } else {
            // we're definitely going to add this
            let arena_index = self.arena.insert(node);

            let mut relev_entry = self.relev_map.entry(max_relev).or_insert((0, Vec::new()));

            if is_leaf {
                relev_entry.0 += 1;
                self.total_leaves += 1;
            }
            relev_entry.1.push(arena_index);

            if old_total_leaves < self.soft_max {
                // this was an unconstrained add, so just update min if necessary and move on
                if max_relev < self.min_relev {
                    self.min_relev = max_relev;
                }
            } else {
                // this was a constrained add. if we inserted into the min bucket, we're done,
                // but if we added to some better bin than that, we may be able to cull the min
                // bucket and choose a new min
                if max_relev > self.min_relev {
                    // if we're inserting into a bin other than the worst one, we might be able
                    // to cull the worst one
                    let min_count =
                        self.relev_map.get(&self.min_relev).expect("must contain min_relev").0;
                    let total_without_min = self.total_leaves - min_count;
                    if total_without_min >= self.soft_max {
                        // we can make due without the minimum bin
                        self.cull_min();
                    }
                }
            }
            Some(arena_index)
        }
    }

    fn cull_min(&mut self) {
        if let Some((min_leaf_count, min_nodes)) = self.relev_map.remove(&self.min_relev) {
            self.total_leaves -= min_leaf_count;
            for node_index in min_nodes {
                self.arena.remove(node_index);
            }
        }
        // pick a new min
        self.min_relev =
            self.relev_map.keys().min().map(|min| *min).unwrap_or(OrderedFloat(std::f64::MAX));
    }

    #[inline(always)]
    pub fn get(&self, index: ArenaIndex) -> Option<&StackableNode<'a, T>> {
        self.arena.get(index)
    }
}

#[derive(Debug, Clone)]
pub struct StackableTree<'a, T: Borrow<GridStore> + Clone + Debug> {
    pub root: StackableNode<'a, T>,
    pub arena: ArenaManager<'a, T>,
}

struct PhrasematchBin<'a, T: Borrow<GridStore> + Clone + Debug> {
    phrasematches: Vec<&'a PhrasematchSubquery<T>>,
    max_relev: OrderedFloat<f64>,
    max_relev_after_this: OrderedFloat<f64>,
}

pub fn stackable<'a, T: Borrow<GridStore> + Clone + Debug>(
    phrasematches: &'a Vec<PhrasematchSubquery<T>>,
) -> StackableTree<'a, T> {
    let mut arena: ArenaManager<'a, T> = ArenaManager::new();

    let mut binned_phrasematches: BTreeMap<u16, PhrasematchBin<'a, T>> = BTreeMap::new();
    for phrasematch in phrasematches {
        let mut bin = binned_phrasematches.entry(phrasematch.store.borrow().type_id).or_insert(
            PhrasematchBin {
                phrasematches: Vec::new(),
                max_relev: OrderedFloat(0.0),
                max_relev_after_this: OrderedFloat(0.0),
            },
        );
        if phrasematch.weight > *bin.max_relev {
            bin.max_relev = OrderedFloat(phrasematch.weight);
        }
        bin.phrasematches.push(phrasematch);
    }

    let mut binned_phrasematches: Vec<_> =
        binned_phrasematches.into_iter().map(|(_k, v)| v).collect();
    // calculate the max_relev_after_this sums
    let mut sum_so_far = 0.0;
    for bin in binned_phrasematches.iter_mut().rev() {
        bin.max_relev_after_this = OrderedFloat(sum_so_far);
        sum_so_far = sum_so_far + *bin.max_relev;
    }

    let root = binned_stackable(
        &binned_phrasematches,
        None,
        FixedBitSet::with_capacity(128),
        0,
        129,
        0.0,
        0,
        0,
        &mut arena,
    );
    StackableTree { root, arena }
}

fn binned_stackable<'b, 'a: 'b, T: Borrow<GridStore> + Clone + Debug>(
    binned_phrasematches: &'b Vec<PhrasematchBin<'a, T>>,
    current_phrasematch: Option<&'a PhrasematchSubquery<T>>,
    bmask: FixedBitSet,
    mask: u32,
    idx: u16,
    relev_so_far: f64,
    zoom: u16,
    start_type_idx: usize,
    arena: &mut ArenaManager<'a, T>,
) -> StackableNode<'a, T> {
    let mut node = StackableNode {
        phrasematch: current_phrasematch,
        children: vec![],
        mask: mask,
        bmask: bmask,
        idx: idx,
        max_relev: relev_so_far,
        zoom: zoom,
    };

    for (type_idx, phrasematch_group) in
        binned_phrasematches.iter().enumerate().skip(start_type_idx)
    {
        for phrasematch in phrasematch_group.phrasematches.iter() {
            if (node.mask & phrasematch.mask) == 0
                && phrasematch.non_overlapping_indexes.contains(node.idx as usize) == false
            {
                let target_relev = relev_so_far + phrasematch.weight;
                let max_possible_relev = target_relev + *phrasematch_group.max_relev_after_this;
                if arena.is_full() && max_possible_relev < *arena.min_relev {
                    continue;
                }

                let target_mask = &phrasematch.mask | node.mask;
                let mut target_bmask: FixedBitSet = node.bmask.clone();
                target_bmask.union_with(&phrasematch.non_overlapping_indexes);

                let child_node = binned_stackable(
                    &binned_phrasematches,
                    Some(&phrasematch),
                    target_bmask,
                    target_mask,
                    phrasematch.idx,
                    target_relev,
                    phrasematch.store.borrow().zoom,
                    type_idx + 1,
                    arena,
                );

                let max_relev = child_node.max_relev;

                if let Some(arena_index) = arena.add(child_node) {
                    node.children.push(arena_index);

                    if max_relev > node.max_relev {
                        node.max_relev = max_relev;
                    }
                }
            }
        }
    }
    node
}
/*

*/

#[cfg(test)]
mod test {
    use super::*;
    use crate::gridstore::builder::*;
    use crate::gridstore::common::MatchPhrase::Range;

    #[test]
    fn simple_stackable_test() {
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
        let store2 = GridStore::new_with_options(directory.path(), 14, 2, 200.).unwrap();

        let a1 = PhrasematchSubquery {
            store: &store1,
            idx: 1,
            non_overlapping_indexes: FixedBitSet::with_capacity(128),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                key: MatchKey { match_phrase: Range { start: 0, end: 1 }, lang_set: 0 },
                id: 0,
            }],
            mask: 2,
        };

        let b1 = PhrasematchSubquery {
            store: &store2,
            idx: 2,
            non_overlapping_indexes: FixedBitSet::with_capacity(128),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                key: MatchKey { match_phrase: Range { start: 0, end: 1 }, lang_set: 0 },
                id: 1,
            }],
            mask: 1,
        };

        let b2 = PhrasematchSubquery {
            store: &store2,
            idx: 2,
            non_overlapping_indexes: FixedBitSet::with_capacity(128),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                key: MatchKey { match_phrase: Range { start: 0, end: 1 }, lang_set: 0 },
                id: 2,
            }],
            mask: 1,
        };

        let phrasematch_results = vec![a1, b1, b2];

        let tree = stackable(&phrasematch_results);
        let a1_children_ids: Vec<u32> = tree
            .arena
            .get(tree.root.children[0])
            .unwrap()
            .children
            .iter()
            .map(|node_idx| {
                tree.arena
                    .get(*node_idx)
                    .unwrap()
                    .phrasematch
                    .as_ref()
                    .map(|p| p.match_keys[0].id)
                    .unwrap()
            })
            .collect();
        assert_eq!(vec![1, 2], a1_children_ids, "a1 can stack with b1 and b2");
        let b1_children_ids: Vec<u32> = tree
            .arena
            .get(tree.root.children[1])
            .unwrap()
            .children
            .iter()
            .map(|node_idx| {
                tree.arena
                    .get(*node_idx)
                    .unwrap()
                    .phrasematch
                    .as_ref()
                    .map(|p| p.match_keys[0].id)
                    .unwrap()
            })
            .collect();
        assert_eq!(0, b1_children_ids.len(), "b1 cannot stack with b2, same nmask");
        let b2_children_ids: Vec<u32> = tree
            .arena
            .get(tree.root.children[2])
            .unwrap()
            .children
            .iter()
            .map(|node_idx| {
                tree.arena
                    .get(*node_idx)
                    .unwrap()
                    .phrasematch
                    .as_ref()
                    .map(|p| p.match_keys[0].id)
                    .unwrap()
            })
            .collect();
        assert_eq!(0, b2_children_ids.len(), "b2 cannot stack with b1, same nmask");
    }

    #[test]
    fn bmask_stackable_test() {
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
        let store = GridStore::new_with_options(directory.path(), 14, 1, 200.).unwrap();
        let mut a1_bmask: FixedBitSet = FixedBitSet::with_capacity(128);
        a1_bmask.insert(0);
        a1_bmask.insert(1);
        let mut b1_bmask: FixedBitSet = FixedBitSet::with_capacity(128);
        b1_bmask.insert(1);
        b1_bmask.insert(0);

        let a1 = PhrasematchSubquery {
            store: &store,
            idx: 1,
            non_overlapping_indexes: FixedBitSet::with_capacity(128),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                key: MatchKey { match_phrase: Range { start: 0, end: 1 }, lang_set: 0 },
                id: 0,
            }],
            mask: 1,
        };

        let b1 = PhrasematchSubquery {
            store: &store,
            idx: 1,
            non_overlapping_indexes: FixedBitSet::with_capacity(128),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                key: MatchKey { match_phrase: Range { start: 0, end: 1 }, lang_set: 0 },
                id: 1,
            }],
            mask: 1,
        };
        let phrasematch_results = vec![a1, b1];
        let tree = stackable(&phrasematch_results);

        let bmask_stacks: Vec<bool> = bfs(tree).iter().map(|node| node.is_leaf()).collect();
        assert_eq!(bmask_stacks[1], true, "a1 cannot stack with b1 since a1's bmask contains the idx of b1 - so they don't have any children");
        assert_eq!(bmask_stacks[2], true, "b1 cannot stack with a1 since b1's bmask contains the idx of a1 - so they don't have any children");
    }

    #[test]
    fn mask_stackable_test() {
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
        let store = GridStore::new_with_options(directory.path(), 14, 1, 200.).unwrap();

        let a1 = PhrasematchSubquery {
            store: &store,
            idx: 1,
            non_overlapping_indexes: FixedBitSet::with_capacity(128),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                key: MatchKey { match_phrase: Range { start: 0, end: 1 }, lang_set: 0 },
                id: 0,
            }],
            mask: 1,
        };

        let b1 = PhrasematchSubquery {
            store: &store,
            idx: 1,
            non_overlapping_indexes: FixedBitSet::with_capacity(128),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                key: MatchKey { match_phrase: Range { start: 0, end: 1 }, lang_set: 0 },
                id: 1,
            }],
            mask: 1,
        };
        let phrasematch_results = vec![a1, b1];
        let tree = stackable(&phrasematch_results);
        let mask_stacks: Vec<bool> = bfs(tree).iter().map(|node| node.is_leaf()).collect();
        assert_eq!(mask_stacks[1], true, "a1 and b1 cannot stack since they have the same mask - so they don't have any children");
        assert_eq!(mask_stacks[2], true, "a1 and b1 cannot stack since they have the same mask - so they don't have any children");
    }

    #[test]
    fn binned_stackable_test() {
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
        let store = GridStore::new_with_options(directory.path(), 14, 1, 200.).unwrap();

        let a1 = PhrasematchSubquery {
            store: &store,
            idx: 1,
            non_overlapping_indexes: FixedBitSet::with_capacity(128),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                key: MatchKey { match_phrase: Range { start: 0, end: 1 }, lang_set: 0 },
                id: 0,
            }],
            mask: 1,
        };

        let b1 = PhrasematchSubquery {
            store: &store,
            idx: 1,
            non_overlapping_indexes: FixedBitSet::with_capacity(128),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                key: MatchKey { match_phrase: Range { start: 0, end: 1 }, lang_set: 0 },
                id: 1,
            }],
            mask: 1,
        };
        let phrasematch_results = vec![a1, b1];
        let tree = stackable(&phrasematch_results);
        println!("{:?}", tree);
    }
}
