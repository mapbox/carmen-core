#![allow(dead_code)]
use std::borrow::Borrow;
use std::collections::{BTreeMap};
use std::fmt::Debug;

use crate::gridstore::common::*;
use crate::gridstore::store::*;
use fixedbitset::FixedBitSet;

#[derive(Debug, Clone)]
pub struct StackableNode<'a, T: Borrow<GridStore> + Clone + Debug> {
    pub phrasematch: Option<&'a PhrasematchSubquery<T>>,
    pub children: Vec<StackableNode<'a, T>>,
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
pub fn bfs<T: Borrow<GridStore> + Clone + Debug>(root: StackableNode<T>) -> Vec<StackableNode<T>> {
    let mut node_vec: Vec<StackableNode<T>> = vec![];
    let mut stack: Vec<_> = vec![];

    stack.push(root);

    while stack.len() > 0 {
        let node = stack.pop().unwrap();
        node_vec.push(node.clone());
        for child in node.children {
            stack.push(child);
        }
    }
    return node_vec;
}

pub fn stackable<'a, T: Borrow<GridStore> + Clone + Debug>(
    phrasematches: &'a Vec<PhrasematchSubquery<T>>,
) -> StackableNode<'a, T> {
    let mut binned_phrasematches: BTreeMap<u16, Vec<&'a PhrasematchSubquery<T>>> = BTreeMap::new();
    for phrasematch in phrasematches {
        binned_phrasematches
            .entry(phrasematch.store.borrow().type_id)
            .or_insert(Vec::new())
            .push(phrasematch);
    }
    let binned_phrasematches: Vec<_> = binned_phrasematches.into_iter().map(|(_k, v)| v).collect();
    binned_stackable(&binned_phrasematches, None, FixedBitSet::with_capacity(128), 0, 129, 0.0, 0, 0)
}

fn binned_stackable<'b, 'a: 'b, T: Borrow<GridStore> + Clone + Debug>(
    binned_phrasematch: &'b Vec<Vec<&'a PhrasematchSubquery<T>>>,
    phrasematch_result: Option<&'a PhrasematchSubquery<T>>,
    bmask: FixedBitSet,
    mask: u32,
    idx: u16,
    max_relev: f64,
    zoom: u16,
    start_type_idx: usize,
) -> StackableNode<'a, T> {
    let mut node = StackableNode {
        phrasematch: phrasematch_result,
        children: vec![],
        mask: mask,
        bmask: bmask,
        idx: idx,
        max_relev: max_relev,
        zoom: zoom,
    };

    for (type_idx, phrasematch_group) in binned_phrasematch.iter().enumerate().skip(start_type_idx)
    {
        for phrasematches in phrasematch_group.iter() {
            if (node.mask & phrasematches.mask) == 0
                && phrasematches.non_overlapping_indexes.contains(node.idx as usize) == false
            {
                let target_mask = &phrasematches.mask | node.mask;
                let target_bmask: FixedBitSet = node.bmask.clone();
                let phrasematch_bmask: FixedBitSet =
                    phrasematches.non_overlapping_indexes.clone();
                target_bmask.intersection(&phrasematch_bmask);
                let target_relev = 0.0 + phrasematches.weight;

                node.children.push(binned_stackable(
                    &binned_phrasematch,
                    Some(&phrasematches),
                    target_bmask,
                    target_mask,
                    phrasematches.idx,
                    target_relev,
                    phrasematches.store.borrow().zoom,
                    type_idx + 1,
                ));
            }
        }
    }
    node
}

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
        let a1_children_ids: Vec<u32> = tree.clone().children[0]
            .clone()
            .children
            .iter()
            .map(|node| node.phrasematch.as_ref().map(|p| p.match_keys[0].id).unwrap())
            .collect();
        assert_eq!(vec![1, 2], a1_children_ids, "a1 can stack with b1 and b2");
        let b1_children_ids: Vec<u32> = tree.clone().children[1]
            .clone()
            .children
            .iter()
            .map(|node| node.phrasematch.as_ref().map(|p| p.match_keys[0].id).unwrap())
            .collect();
        assert_eq!(0, b1_children_ids.len(), "b1 cannot stack with b2, same nmask");
        let b2_children_ids: Vec<u32> = tree.clone().children[2]
            .clone()
            .children
            .iter()
            .map(|node| node.phrasematch.as_ref().map(|p| p.match_keys[0].id).unwrap())
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
        let mut a1_bmask: FixedBitSet =FixedBitSet::with_capacity(128);
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
    }
}
