#![allow(dead_code)]
use ordered_float::OrderedFloat;
use std::borrow::Borrow;
use std::cmp::Reverse;
use std::collections::HashSet;
use std::fmt::Debug;

use crate::gridstore::common::*;
use crate::gridstore::store::*;

#[derive(Debug, Clone)]
pub struct StackableNode<'a, T: Borrow<GridStore> + Clone + Debug> {
    pub phrasematch: Option<&'a PhrasematchSubquery<T>>,
    pub children: Vec<StackableNode<'a, T>>,
    pub nmask: u32,
    pub bmask: HashSet<u16>,
    pub mask: u32,
    pub idx: u16,
    pub max_relev: f64,
    pub zoom: u16,
}

#[derive(Debug, Clone)]
pub struct StackableNodeCopy<'a, T: Borrow<GridStore> + Clone + Debug> {
    pub phrasematch: Option<&'a PhrasematchSubquery<T>>,
    pub children: Vec<StackableNodeCopy<'a, T>>,
    pub bmask: HashSet<u16>,
    pub mask: u32,
    pub idx: u16,
    pub max_relev: f64,
    pub zoom: u16,
}

impl<'a, T: Borrow<GridStore> + Clone + Debug> StackableNodeCopy<'a, T> {
    fn is_leaf(&self) -> bool {
        self.children.len() == 0
    }
}

//tree traversal used only for tests
pub fn bfs<T: Borrow<GridStore> + Clone + Debug>(
    root: StackableNodeCopy<T>,
) -> Vec<StackableNodeCopy<T>> {
    let mut node_vec: Vec<StackableNodeCopy<T>> = vec![];
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
    phrasematch_results: &'a Vec<PhrasematchSubquery<T>>,
    phrasematch_result: Option<&'a PhrasematchSubquery<T>>,
    nmask: u32,
    bmask: HashSet<u16>,
    mask: u32,
    idx: u16,
    max_relev: f64,
    zoom: u16,
) -> StackableNode<'a, T> {
    let mut node = StackableNode {
        phrasematch: phrasematch_result,
        children: vec![],
        mask: mask,
        bmask: bmask,
        nmask: nmask,
        idx: idx,
        max_relev: max_relev,
        zoom: zoom,
    };

    for phrasematches in phrasematch_results.iter() {
        if node.phrasematch.is_some() {
            if node.zoom > phrasematches.store.borrow().zoom {
                continue;
            } else if node.zoom == phrasematches.store.borrow().zoom {
                if node.idx > phrasematches.idx {
                    continue;
                }
            }
        }

        if (node.nmask & (1u32 << phrasematches.store.borrow().type_id as u32)) == 0
            && (node.mask & phrasematches.mask) == 0
            && phrasematches.non_overlapping_indexes.contains(&node.idx) == false
        {
            let target_nmask = &(1u32 << phrasematches.store.borrow().type_id as u32) | node.nmask;
            let target_mask = &phrasematches.mask | node.mask;
            let mut target_bmask: HashSet<u16> = node.bmask.iter().cloned().collect();
            let phrasematch_bmask: HashSet<u16> =
                phrasematches.non_overlapping_indexes.iter().cloned().collect();
            target_bmask.extend(&phrasematch_bmask);
            let target_relev = 0.0 + phrasematches.weight;

            node.children.push(stackable(
                &phrasematch_results,
                Some(&phrasematches),
                target_nmask,
                target_bmask,
                target_mask,
                phrasematches.idx,
                target_relev,
                phrasematches.store.borrow().zoom,
            ));
        }
    }

    node.children.sort_by_key(|node| Reverse(OrderedFloat(node.max_relev)));

    if !node.children.is_empty() {
        node.max_relev = node.max_relev + node.children[0].max_relev;
    }

    node
}

pub fn binned_stackable<'a, T: Borrow<GridStore> + Clone + Debug>(
    binned_phrasematch: &'a Vec<Vec<PhrasematchSubquery<T>>>,
    phrasematch_result: Option<&'a PhrasematchSubquery<T>>,
    bmask: HashSet<u16>,
    mask: u32,
    idx: u16,
    max_relev: f64,
    zoom: u16,
    start_type_idx: usize,
) -> StackableNodeCopy<'a, T> {
    let mut node = StackableNodeCopy {
        phrasematch: phrasematch_result,
        children: vec![],
        mask: mask,
        bmask: bmask,
        idx: idx,
        max_relev: max_relev,
        zoom: zoom,
    };

    for (type_idx, phrasematch_group) in binned_phrasematch.iter().enumerate().skip(start_type_idx) {
        for phrasematches in phrasematch_group.iter() {
            if (node.mask & phrasematches.mask) == 0
                && phrasematches.non_overlapping_indexes.contains(&node.idx) == false
            {
                let target_mask = &phrasematches.mask | node.mask;
                let mut target_bmask: HashSet<u16> = node.bmask.iter().cloned().collect();
                let phrasematch_bmask: HashSet<u16> =
                    phrasematches.non_overlapping_indexes.iter().cloned().collect();
                target_bmask.extend(&phrasematch_bmask);
                let target_relev = 0.0 + phrasematches.weight;

                node.children.push(binned_stackable(
                    &binned_phrasematch,
                    Some(&phrasematches),
                    target_bmask,
                    target_mask,
                    phrasematches.idx,
                    target_relev,
                    phrasematches.store.borrow().zoom,
                    type_idx + 1
                ));
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

    use std::collections::BTreeMap;

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
            non_overlapping_indexes: HashSet::new(),
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
            non_overlapping_indexes: HashSet::new(),
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
            non_overlapping_indexes: HashSet::new(),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                key: MatchKey { match_phrase: Range { start: 0, end: 1 }, lang_set: 0 },
                id: 2,
            }],
            mask: 1,
        };

        let phrasematch_results = vec![a1, b1, b2];
        let mut binned_phrasematch: BTreeMap<u16, Vec<PhrasematchSubquery<&GridStore>>> =
            BTreeMap::new();

        for phrasematch in phrasematch_results {
            binned_phrasematch
                .entry(phrasematch.store.borrow().type_id)
                .or_insert(Vec::new())
                .push(phrasematch);
        }
        let binned_phrasematch: Vec<_> = binned_phrasematch.into_iter().map(|(_k, v)| v).collect();

        let tree = binned_stackable(&binned_phrasematch, None, HashSet::new(), 0, 129, 0.0, 0, 0);
        println!("{:?}", tree);
        // let a1_children_ids: Vec<u32> = tree.clone().children[0]
        //     .clone()
        //     .children
        //     .iter()
        //     .map(|node| node.phrasematch.as_ref().map(|p| p.match_keys[0].id).unwrap())
        //     .collect();
        // assert_eq!(vec![1, 2], a1_children_ids, "a1 can stack with b1 and b2");
        // let b1_children_ids: Vec<u32> = tree.clone().children[1]
        //     .clone()
        //     .children
        //     .iter()
        //     .map(|node| node.phrasematch.as_ref().map(|p| p.match_keys[0].id).unwrap())
        //     .collect();
        // assert_eq!(0, b1_children_ids.len(), "b1 cannot stack with b2, same nmask");
        // let b2_children_ids: Vec<u32> = tree.clone().children[2]
        //     .clone()
        //     .children
        //     .iter()
        //     .map(|node| node.phrasematch.as_ref().map(|p| p.match_keys[0].id).unwrap())
        //     .collect();
        // assert_eq!(0, b2_children_ids.len(), "b2 cannot stack with b1, same nmask");
    }

    // #[test]
    // fn bmask_stackable_test() {
    //     let directory: tempfile::TempDir = tempfile::tempdir().unwrap();
    //     let mut builder = GridStoreBuilder::new(directory.path()).unwrap();
    //
    //     let key = GridKey { phrase_id: 1, lang_set: 1 };
    //
    //     let entries = vec![
    //         GridEntry { id: 2, x: 2, y: 2, relev: 0.8, score: 3, source_phrase_hash: 0 },
    //         GridEntry { id: 3, x: 3, y: 3, relev: 1., score: 1, source_phrase_hash: 1 },
    //         GridEntry { id: 1, x: 1, y: 1, relev: 1., score: 7, source_phrase_hash: 2 },
    //     ];
    //     builder.insert(&key, entries).expect("Unable to insert record");
    //     builder.finish().unwrap();
    //     let store = GridStore::new_with_options(directory.path(), 14, 1, 200.).unwrap();
    //     let mut a1_bmask: HashSet<u16> = HashSet::new();
    //     a1_bmask.insert(0);
    //     a1_bmask.insert(1);
    //     let mut b1_bmask: HashSet<u16> = HashSet::new();
    //     b1_bmask.insert(1);
    //     b1_bmask.insert(0);
    //
    //     let a1 = PhrasematchSubquery {
    //         store: &store,
    //         idx: 1,
    //         non_overlapping_indexes: HashSet::new(),
    //         weight: 0.5,
    //         match_keys: vec![MatchKeyWithId {
    //             key: MatchKey { match_phrase: Range { start: 0, end: 1 }, lang_set: 0 },
    //             id: 0,
    //         }],
    //         mask: 1,
    //     };
    //
    //     let b1 = PhrasematchSubquery {
    //         store: &store,
    //         idx: 1,
    //         non_overlapping_indexes: HashSet::new(),
    //         weight: 0.5,
    //         match_keys: vec![MatchKeyWithId {
    //             key: MatchKey { match_phrase: Range { start: 0, end: 1 }, lang_set: 0 },
    //             id: 1,
    //         }],
    //         mask: 1,
    //     };
    //     let phrasematch_results = vec![a1, b1];
    //     let tree = binned_stackable(&phrasematch_results, None, HashSet::new(), 0, 129, 0.0, 0);
    //     let bmask_stacks: Vec<bool> = bfs(tree).iter().map(|node| node.is_leaf()).collect();
    //     assert_eq!(bmask_stacks[1], true, "a1 cannot stack with b1 since a1's bmask contains the idx of b1 - so they don't have any children");
    //     assert_eq!(bmask_stacks[2], true, "b1 cannot stack with a1 since b1's bmask contains the idx of a1 - so they don't have any children");
    // }

    // #[test]
    // fn mask_stackable_test() {
    //     let directory: tempfile::TempDir = tempfile::tempdir().unwrap();
    //     let mut builder = GridStoreBuilder::new(directory.path()).unwrap();
    //
    //     let key = GridKey { phrase_id: 1, lang_set: 1 };
    //
    //     let entries = vec![
    //         GridEntry { id: 2, x: 2, y: 2, relev: 0.8, score: 3, source_phrase_hash: 0 },
    //         GridEntry { id: 3, x: 3, y: 3, relev: 1., score: 1, source_phrase_hash: 1 },
    //         GridEntry { id: 1, x: 1, y: 1, relev: 1., score: 7, source_phrase_hash: 2 },
    //     ];
    //     builder.insert(&key, entries).expect("Unable to insert record");
    //     builder.finish().unwrap();
    //     let store = GridStore::new_with_options(directory.path(), 14, 1, 200.).unwrap();
    //
    //     let a1 = PhrasematchSubquery {
    //         store: &store,
    //         idx: 1,
    //         non_overlapping_indexes: HashSet::new(),
    //         weight: 0.5,
    //         match_keys: vec![MatchKeyWithId {
    //             key: MatchKey { match_phrase: Range { start: 0, end: 1 }, lang_set: 0 },
    //             id: 0,
    //         }],
    //         mask: 1,
    //     };
    //
    //     let b1 = PhrasematchSubquery {
    //         store: &store,
    //         idx: 1,
    //         non_overlapping_indexes: HashSet::new(),
    //         weight: 0.5,
    //         match_keys: vec![MatchKeyWithId {
    //             key: MatchKey { match_phrase: Range { start: 0, end: 1 }, lang_set: 0 },
    //             id: 1,
    //         }],
    //         mask: 1,
    //     };
    //     let phrasematch_results = vec![a1, b1];
    //     let tree = binned_stackable(&phrasematch_results, None, HashSet::new(), 0, 129, 0.0, 0);
    //     let mask_stacks: Vec<bool> = bfs(tree).iter().map(|node| node.is_leaf()).collect();
    //     assert_eq!(mask_stacks[1], true, "a1 and b1 cannot stack since they have the same mask - so they don't have any children");
    //     assert_eq!(mask_stacks[2], true, "a1 and b1 cannot stack since they have the same mask - so they don't have any children");
    // }
    //
    // #[test]
    // fn binned_stackable_test() {
    //     let directory: tempfile::TempDir = tempfile::tempdir().unwrap();
    //     let mut builder = GridStoreBuilder::new(directory.path()).unwrap();
    //
    //     let key = GridKey { phrase_id: 1, lang_set: 1 };
    //
    //     let entries = vec![
    //         GridEntry { id: 2, x: 2, y: 2, relev: 0.8, score: 3, source_phrase_hash: 0 },
    //         GridEntry { id: 3, x: 3, y: 3, relev: 1., score: 1, source_phrase_hash: 1 },
    //         GridEntry { id: 1, x: 1, y: 1, relev: 1., score: 7, source_phrase_hash: 2 },
    //     ];
    //     builder.insert(&key, entries).expect("Unable to insert record");
    //     builder.finish().unwrap();
    //     let store = GridStore::new_with_options(directory.path(), 14, 1, 200.).unwrap();
    //
    //     let a1 = PhrasematchSubquery {
    //         store: &store,
    //         idx: 1,
    //         non_overlapping_indexes: HashSet::new(),
    //         weight: 0.5,
    //         match_keys: vec![MatchKeyWithId {
    //             key: MatchKey { match_phrase: Range { start: 0, end: 1 }, lang_set: 0 },
    //             id: 0,
    //         }],
    //         mask: 1,
    //     };
    //
    //     let b1 = PhrasematchSubquery {
    //         store: &store,
    //         idx: 1,
    //         non_overlapping_indexes: HashSet::new(),
    //         weight: 0.5,
    //         match_keys: vec![MatchKeyWithId {
    //             key: MatchKey { match_phrase: Range { start: 0, end: 1 }, lang_set: 0 },
    //             id: 1,
    //         }],
    //         mask: 1,
    //     };
    //     let phrasematch_results = vec![a1, b1];
    //     let tree = binned_stackable(&phrasematch_results, None, HashSet::new(), 0, 129, 0.0, 0);
    //     println!("{:?}", tree);
    // }
}
