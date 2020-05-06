use carmen_core::gridstore::*;
use test_utils::*;

use fixedbitset::FixedBitSet;

const ALL_LANGUAGES: u128 = u128::max_value();

#[test]
fn coalesce_single_test_proximity_quadrants() {
    let directory: tempfile::TempDir = tempfile::tempdir().unwrap();
    let mut builder = GridStoreBuilder::new(directory.path()).unwrap();

    let key = GridKey { phrase_id: 1, lang_set: 1 };

    let entries = vec![
        GridEntry { id: 1, x: 200, y: 200, relev: 1., score: 1, source_phrase_hash: 0 }, // ne
        GridEntry { id: 2, x: 200, y: 0, relev: 1., score: 1, source_phrase_hash: 0 },   // se
        GridEntry { id: 3, x: 0, y: 0, relev: 1., score: 1, source_phrase_hash: 0 },     // sw
        GridEntry { id: 4, x: 0, y: 200, relev: 1., score: 1, source_phrase_hash: 0 },   // nw
    ];
    builder.insert(&key, entries).expect("Unable to insert record");

    builder.finish().unwrap();

    let store = GridStore::new_with_options(directory.path(), 14, 1, 200.).unwrap();
    let subquery = PhrasematchSubquery {
        store: &store,
        idx: 1,
        non_overlapping_indexes: FixedBitSet::with_capacity(128),
        weight: 1.,
        match_keys: vec![MatchKeyWithId {
            nearby_only: false,
            id: 0,
            key: MatchKey { match_phrase: MatchPhrase::Range { start: 1, end: 3 }, lang_set: 1 },
        }],
        mask: 1 << 0,
    };
    let stack = vec![subquery];

    println!("Coalesce single - NE proximity");
    let match_opts = MatchOpts {
        zoom: 14,
        proximity: Some([110, 115]), // NE proximity point
        ..MatchOpts::default()
    };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    assert_eq!(result, tree_result);
    let result_ids: Vec<u32> =
        result.iter().map(|context| context.entries[0].grid_entry.id).collect();
    let result_distances: Vec<f64> =
        result.iter().map(|context| round(context.entries[0].distance, 0)).collect();
    assert_eq!(result_ids, [1, 4, 2, 3], "Results are in the order ne, nw, se, sw");
    assert_eq!(result_distances, [124.0, 139.0, 146.0, 159.0], "Result distances are correct");

    println!("Coalesce single - SE proximity");
    let match_opts = MatchOpts {
        zoom: 14,
        proximity: Some([110, 85]), // SE proximity point
        ..MatchOpts::default()
    };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    assert_eq!(result, tree_result);
    let result_ids: Vec<u32> =
        result.iter().map(|context| context.entries[0].grid_entry.id).collect();
    let result_distances: Vec<f64> =
        result.iter().map(|context| round(context.entries[0].distance, 0)).collect();
    assert_eq!(result_ids, [2, 3, 1, 4], "Results are in the order se, sw, ne, nw");
    assert_eq!(result_distances, [124.0, 139.0, 146.0, 159.0], "Result distances are correct");

    println!("Coalesce single - SW proximity");
    let match_opts = MatchOpts {
        zoom: 14,
        proximity: Some([90, 85]), // SW proximity point
        ..MatchOpts::default()
    };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    assert_eq!(result, tree_result);
    let result_ids: Vec<u32> =
        result.iter().map(|context| context.entries[0].grid_entry.id).collect();
    let result_distances: Vec<f64> =
        result.iter().map(|context| round(context.entries[0].distance, 0)).collect();
    assert_eq!(result_ids, [3, 2, 4, 1], "Results are in the order sw, se, nw, ne");
    assert_eq!(result_distances, [124.0, 139.0, 146.0, 159.0], "Result distances are correct");

    println!("Coalesce single - NW proximity");
    let match_opts = MatchOpts {
        zoom: 14,
        proximity: Some([90, 115]), // NW proximity point
        ..MatchOpts::default()
    };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    assert_eq!(result, tree_result);
    let result_ids: Vec<u32> =
        result.iter().map(|context| context.entries[0].grid_entry.id).collect();
    let result_distances: Vec<f64> =
        result.iter().map(|context| round(context.entries[0].distance, 0)).collect();
    assert_eq!(result_ids, [4, 1, 3, 2], "Results are in the order nw, ne, sw, se");
    assert_eq!(result_distances, [124.0, 139.0, 146.0, 159.0], "Result distances are correct");
}

#[test]
fn coalesce_single_test_proximity_basic() {
    let directory: tempfile::TempDir = tempfile::tempdir().unwrap();
    let mut builder = GridStoreBuilder::new(directory.path()).unwrap();

    let key = GridKey { phrase_id: 1, lang_set: 1 };

    let entries = vec![
        GridEntry { id: 1, x: 2, y: 2, relev: 1., score: 1, source_phrase_hash: 0 },
        GridEntry { id: 2, x: 2, y: 0, relev: 1., score: 1, source_phrase_hash: 0 },
        GridEntry { id: 3, x: 0, y: 0, relev: 1., score: 1, source_phrase_hash: 0 },
        GridEntry { id: 4, x: 0, y: 2, relev: 1., score: 1, source_phrase_hash: 0 },
    ];
    builder.insert(&key, entries).expect("Unable to insert record");

    builder.finish().unwrap();

    let store = GridStore::new_with_options(directory.path(), 14, 1, 200.).unwrap();
    let subquery = PhrasematchSubquery {
        store: &store,
        idx: 1,
        non_overlapping_indexes: FixedBitSet::with_capacity(128),
        weight: 1.,
        match_keys: vec![MatchKeyWithId {
            nearby_only: false,
            id: 0,
            key: MatchKey { match_phrase: MatchPhrase::Range { start: 1, end: 3 }, lang_set: 1 },
        }],
        mask: 1 << 0,
    };
    let stack = vec![subquery];
    let match_opts = MatchOpts { zoom: 14, proximity: Some([2, 2]), ..MatchOpts::default() };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    assert_eq!(result, tree_result);
    let result_ids: Vec<u32> =
        result.iter().map(|context| context.entries[0].grid_entry.id).collect();
    assert_eq!(
        result_ids,
        [1, 2, 4, 3],
        "Results with the same relev and score should be ordered by distance"
    );

    let result_distances: Vec<f64> =
        result.iter().map(|context| round(context.entries[0].distance, 2)).collect();
    assert_eq!(
        result_distances,
        [0.00, 2.00, 2.00, 2.83],
        "Results with the same relev and score should be ordered by distance"
    );
}

#[test]
fn coalesce_single_test_language_penalty() {
    let directory: tempfile::TempDir = tempfile::tempdir().unwrap();
    let mut builder = GridStoreBuilder::new(directory.path()).unwrap();

    let key = GridKey { phrase_id: 1, lang_set: 1 };

    let entries = vec![
        GridEntry { id: 1, x: 2, y: 2, relev: 1., score: 1, source_phrase_hash: 0 },
        GridEntry { id: 2, x: 2, y: 0, relev: 1., score: 1, source_phrase_hash: 0 },
        GridEntry { id: 3, x: 0, y: 0, relev: 1., score: 1, source_phrase_hash: 0 },
        GridEntry { id: 4, x: 0, y: 2, relev: 1., score: 1, source_phrase_hash: 0 },
    ];
    builder.insert(&key, entries).expect("Unable to insert record");
    builder.finish().unwrap();

    let store = GridStore::new_with_options(directory.path(), 14, 1, 1.).unwrap();
    let subquery = PhrasematchSubquery {
        store: &store,
        idx: 1,
        non_overlapping_indexes: FixedBitSet::with_capacity(128),
        weight: 1.,
        match_keys: vec![MatchKeyWithId {
            nearby_only: false,
            id: 0,
            key: MatchKey { match_phrase: MatchPhrase::Range { start: 1, end: 3 }, lang_set: 2 },
        }],
        mask: 1 << 0,
    };
    let stack = vec![subquery.clone()];
    let match_opts = MatchOpts { zoom: 14, proximity: Some([2, 2]), ..MatchOpts::default() };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    assert_eq!(result, tree_result);
    #[cfg_attr(rustfmt, rustfmt::skip)]
    {
        assert_eq!(result[0].relev, 1., "Contexts inside the proximity radius don't get a cross langauge penalty");
        assert_eq!(result[0].entries[0].grid_entry.relev, 1., "Grids inside the proximity radius don't get a cross language penalty");
        assert_eq!(result[0].entries[0].matches_language, false, "Matches language property is correctly set on CoalesceEntry");
        assert_eq!(result[1].relev, 0.96, "Contexts outside the proximity radius get a cross langauge penalty");
        assert_eq!(result[1].entries[0].grid_entry.relev, 0.96, "Grids outside the proximity radius get a cross language penalty");
        assert_eq!(result[1].entries[0].matches_language, false, "Matches language property is correctly set on CoalesceEntry");
    }
    let match_opts = MatchOpts { zoom: 14, ..MatchOpts::default() };
    let stack = vec![subquery.clone()];
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    assert_eq!(result, tree_result);
    #[cfg_attr(rustfmt, rustfmt::skip)]
    {
        assert_eq!(result[0].relev, 0.96, "With no proximity, cross language contexts get a penalty");
        assert_eq!(result[0].entries[0].grid_entry.relev, 0.96, "With no proximity, cross language grids get a penalty");
        assert_eq!(result[0].entries[0].matches_language, false, "Matches language property is correctly set on CoalesceEntry");
    }
}

#[test]
fn coalesce_multi_test_language_penalty() {
    // Add more specific layer into a store
    let store1 = create_store(
        vec![StoreEntryBuildingBlock {
            grid_key: GridKey { phrase_id: 1, lang_set: 1 },
            entries: vec![
                GridEntry { id: 1, x: 2, y: 2, relev: 1., score: 1, source_phrase_hash: 0 },
                GridEntry { id: 2, x: 12800, y: 12800, relev: 1., score: 1, source_phrase_hash: 0 },
            ],
        }],
        1,
        14,
        1,
        FixedBitSet::with_capacity(128),
        200.,
    );

    // Add less specific layer into a store
    let store2 = create_store(
        vec![StoreEntryBuildingBlock {
            grid_key: GridKey { phrase_id: 2, lang_set: 1 },
            entries: vec![
                GridEntry { id: 3, x: 0, y: 0, relev: 1., score: 1, source_phrase_hash: 0 },
                GridEntry { id: 4, x: 50, y: 50, relev: 1., score: 1, source_phrase_hash: 0 },
            ],
        }],
        2,
        6,
        0,
        FixedBitSet::with_capacity(128),
        200.,
    );

    // Subqueries with a different language set
    println!("Coalesce multi - Subqueries with different language set from grids, with proximity");
    let stack = vec![
        PhrasematchSubquery {
            store: &store1.store,
            idx: store1.idx,
            non_overlapping_indexes: store1.non_overlapping_indexes.clone(),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                nearby_only: false,
                id: 0,
                key: MatchKey {
                    match_phrase: MatchPhrase::Range { start: 1, end: 3 },
                    lang_set: 2,
                },
            }],
            mask: 1 << 0,
        },
        PhrasematchSubquery {
            store: &store2.store,
            idx: store2.idx,
            non_overlapping_indexes: store2.non_overlapping_indexes.clone(),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                nearby_only: false,
                id: 1,
                key: MatchKey {
                    match_phrase: MatchPhrase::Range { start: 1, end: 3 },
                    lang_set: 2,
                },
            }],
            mask: 1 << 1,
        },
    ];

    let match_opts = MatchOpts { zoom: 14, proximity: Some([2, 2]), ..MatchOpts::default() };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    assert_eq!(result, tree_result);
    #[cfg_attr(rustfmt, rustfmt::skip)]
    {
        assert_eq!(result[0].relev, 1., "Contexts inside the proximity radius don't get a cross langauge penalty");
        assert_eq!(result[0].entries[0].grid_entry.relev, 0.5, "Grids inside the proximity radius don't get a cross language penalty");
        assert_eq!(result[0].entries[0].matches_language, false, "matches_language property is correctly set on CoalesceEntry");
        assert_eq!(result[1].relev, 0.96, "Contexts outside the proximity radius get a cross langauge penalty");
        assert_eq!(result[1].entries[0].grid_entry.relev, 0.48, "1st grid outside the proximity radius get a cross language penalty");
        assert_eq!(result[1].entries[1].grid_entry.relev, 0.48, "2nd grid outside the proximity radius gets a cross language penalty");
        assert_eq!(result[1].entries[0].matches_language, false, "matches_language property is correctly set on 1st CoalesceEntry in context");
        assert_eq!(result[1].entries[1].matches_language, false, "matches_language property is correctly set on 2nd CoalesceEntry in context");
    }
    println!("Coalesce multi - Subqueires with different lang set from grids, no proximity");
    let match_opts = MatchOpts { zoom: 14, ..MatchOpts::default() };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    assert_eq!(result, tree_result);
    #[cfg_attr(rustfmt, rustfmt::skip)]
    {
        assert_eq!(result[0].relev, 0.96, "Cross language contexts get a penalty");
        assert_eq!(result[0].entries[0].grid_entry.relev, 0.48, "Cross language grids get a penalty");
        assert_eq!(result[0].entries[0].matches_language, false, "matches_language property is correctly set on CoalesceEntry");
    }
}

#[test]
fn coalesce_single_test() {
    let store = create_store(
        vec![StoreEntryBuildingBlock {
            grid_key: GridKey { phrase_id: 1, lang_set: 1 },
            entries: vec![
                GridEntry { id: 1, x: 1, y: 1, relev: 1., score: 3, source_phrase_hash: 0 },
                GridEntry { id: 2, x: 2, y: 2, relev: 0.8, score: 3, source_phrase_hash: 0 },
                GridEntry { id: 3, x: 3, y: 3, relev: 1., score: 1, source_phrase_hash: 0 },
            ],
        }],
        1,
        6,
        0,
        FixedBitSet::with_capacity(128),
        40.,
    );
    let subquery = PhrasematchSubquery {
        store: &store.store,
        idx: store.idx,
        non_overlapping_indexes: store.non_overlapping_indexes.clone(),
        weight: 1.,
        match_keys: vec![MatchKeyWithId {
            nearby_only: false,
            id: 0,
            key: MatchKey { match_phrase: MatchPhrase::Range { start: 1, end: 3 }, lang_set: 1 },
        }],
        mask: 1 << 0,
    };
    let stack = vec![subquery];

    // Test default opts - no proximity or bbox
    println!("Coalsece single - no proximity, no bbox");
    let match_opts = MatchOpts { zoom: 6, ..MatchOpts::default() };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    assert_eq!(result, tree_result);

    #[cfg_attr(rustfmt, rustfmt::skip)]
    {
        assert_eq!(result[0].relev, 1., "1st result has relevance 1");
        assert_eq!(result[0].entries.len(), 1, "1st result has one coalesce entry");
        assert_eq!(result[0].entries[0].matches_language, true, "1st result is a language match");
        assert_eq!(result[0].entries[0].distance, 0., "1st result has distance 0");
        assert_eq!(result[0].entries[0].idx, 1, "1st result has idx of subquery");
        assert_eq!(result[0].entries[0].mask, 1 << 0, "1st result has original mask");
        assert_eq!(result[0].entries[0].scoredist, 3., "1st result scoredist is the grid score");
        assert_eq!(result[0].entries[0].grid_entry, GridEntry {
                id: 1,
                x: 1,
                y: 1,
                relev: 1.,
                score: 3,
                source_phrase_hash: 0,
            }, "1st result grid entry is the highest relevance and score");
        assert_eq!(result[1].relev, 1., "2nd result has relevance 1");
        assert_eq!(result[1].entries.len(), 1, "2nd result has one coalesce entry");
        assert_eq!(result[1].entries[0].matches_language, true, "2nd result is a language match");
        assert_eq!(result[1].entries[0].distance, 0., "2nd result has distance 0");
        assert_eq!(result[1].entries[0].idx, 1, "2nd result has idx of subquery");
        assert_eq!(result[1].entries[0].mask, 1 << 0, "2nd result has original mask");
        assert_eq!(result[1].entries[0].scoredist, 1., "2nd result scoredist is the grid score");
        assert_eq!(result[1].entries[0].grid_entry, GridEntry {
                id: 3,
                x: 3,
                y: 3,
                relev: 1.,
                score: 1,
                source_phrase_hash: 0,
            }, "2nd result grid entry is the highest relevance, lower score");
        assert_eq!(result[2].relev, 0.8, "3rd result has relevance 0.8");
        assert_eq!(result[2].entries.len(), 1, "3rd result has one coalesce entry");
        assert_eq!(result[2].entries[0].matches_language, true, "3rd result is a language match");
        assert_eq!(result[2].entries[0].distance, 0., "3rd result has distance 0");
        assert_eq!(result[2].entries[0].idx, 1, "3rd result has idx of subquery");
        assert_eq!(result[2].entries[0].mask, 1 << 0, "3rd result has original mask");
        assert_eq!(result[2].entries[0].scoredist, 3., "3rd result scoredist is the grid score");
        assert_eq!(result[2].entries[0].grid_entry, GridEntry {
                id: 2,
                x: 2,
                y: 2,
                relev: 0.8,
                score: 3,
                source_phrase_hash: 0,
            }, "3rd result grid entry is the lowest relevance, even though score is higher than 2nd");
    }
    // Test opts with proximity
    println!("Coalsece single - with proximity");
    let match_opts = MatchOpts { zoom: 6, proximity: Some([3, 3]), ..MatchOpts::default() };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    assert_eq!(result, tree_result);
    #[cfg_attr(rustfmt, rustfmt::skip)]
    {
        assert_eq!(result[0].entries[0].grid_entry.id, 3, "1st result is the closest, even if its a slightly lower score");
        assert_eq!(result[1].entries[0].grid_entry.id, 1, "2nd result is farther away than 3rd but has a higher relevance");
        assert_eq!(result[2].entries[0].grid_entry.id, 2, "3rd is closer but has a lower relevance");
    }
    assert_eq!(
        result[0],
        CoalesceContext {
            mask: 1 << 0,
            relev: 1.,
            entries: vec![CoalesceEntry {
                phrasematch_id: 0,
                matches_language: true,
                idx: 1,
                tmp_id: 33554435,
                mask: 1 << 0,
                distance: 0.,
                scoredist: 1.5839497841387566,
                grid_entry: GridEntry {
                    id: 3,
                    x: 3,
                    y: 3,
                    relev: 1.,
                    score: 1,
                    source_phrase_hash: 0,
                }
            }],
        },
        "1st result has expected properties"
    );
    assert_eq!(
        result[1],
        CoalesceContext {
            mask: 1 << 0,
            relev: 1.,
            entries: vec![CoalesceEntry {
                phrasematch_id: 0,
                matches_language: true,
                idx: 1,
                tmp_id: 33554433,
                mask: 1 << 0,
                distance: 2.8284271247461903,
                scoredist: 1.109893833332405,
                grid_entry: GridEntry {
                    id: 1,
                    x: 1,
                    y: 1,
                    relev: 1.,
                    score: 3,
                    source_phrase_hash: 0,
                }
            }],
        },
        "2nd result has expected properties"
    );
    assert_eq!(
        result[2],
        CoalesceContext {
            mask: 1 << 0,
            relev: 0.8,
            entries: vec![CoalesceEntry {
                phrasematch_id: 0,
                matches_language: true,
                idx: 1,
                tmp_id: 33554434,
                mask: 1 << 0,
                distance: 1.4142135623730951,
                // Has the same scoredist as 2nd result because they're both beyond proximity radius
                scoredist: 1.109893833332405,
                grid_entry: GridEntry {
                    id: 2,
                    x: 2,
                    y: 2,
                    relev: 0.8,
                    score: 3,
                    source_phrase_hash: 0,
                }
            }],
        },
        "2nd result has expected properties"
    );

    // Test with bbox
    println!("Coalsece single - with bbox");
    let match_opts = MatchOpts { zoom: 6, bbox: Some([1, 1, 1, 1]), ..MatchOpts::default() };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    assert_eq!(result, tree_result);
    assert_eq!(result[0].entries.len(), 1, "Only one result is within the bbox");
    assert_eq!(result[0].entries[0].grid_entry.id, 1, "Result is the one that's within the bbox");
    assert_eq!(
        result[0],
        CoalesceContext {
            mask: 1 << 0,
            relev: 1.,
            entries: vec![CoalesceEntry {
                phrasematch_id: 0,
                matches_language: true,
                idx: 1,
                tmp_id: 33554433,
                mask: 1 << 0,
                distance: 0.,
                scoredist: 3.,
                grid_entry: GridEntry {
                    id: 1,
                    x: 1,
                    y: 1,
                    relev: 1.,
                    score: 3,
                    source_phrase_hash: 0,
                }
            }],
        },
        "Result has expected properties"
    );

    // Test with bbox and proximity
    println!("Coalesce single - with bbox and proximity");
    let match_opts = MatchOpts { zoom: 6, bbox: Some([1, 1, 1, 1]), proximity: Some([1, 1]) };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    assert_eq!(result, tree_result);
    assert_eq!(result[0].entries.len(), 1, "Only one result is within the bbox");
    assert_eq!(
        result[0],
        CoalesceContext {
            mask: 1 << 0,
            relev: 1.,
            entries: vec![CoalesceEntry {
                phrasematch_id: 0,
                matches_language: true,
                idx: 1,
                tmp_id: 33554433,
                mask: 1 << 0,
                distance: 0.,
                scoredist: 1.7322531402718835,
                grid_entry: GridEntry {
                    id: 1,
                    x: 1,
                    y: 1,
                    relev: 1.,
                    score: 3,
                    source_phrase_hash: 0,
                }
            }],
        },
        "Result has expected properties, including scoredist"
    );
    // TODO: test with more than one result within bbox, to make sure results are still ordered by proximity?
}

#[test]
fn coalesce_single_languages_test() {
    let directory: tempfile::TempDir = tempfile::tempdir().unwrap();
    let mut builder = GridStoreBuilder::new(directory.path()).unwrap();
    let lang_sets: [Vec<u32>; 4] = [vec![0], vec![1], vec![0, 1], vec![2]];
    // Load each grid_entry with a grid key for each language
    for (i, langs) in lang_sets.iter().enumerate() {
        let lang_set = langarray_to_langfield(&langs[..]);
        let key = GridKey { phrase_id: 1, lang_set };
        let grid_entry =
            GridEntry { id: i as u32, x: 1, y: 1, relev: 1., score: 0, source_phrase_hash: 0 };
        builder.insert(&key, vec![grid_entry]).expect("Unable to insert record");
    }
    builder.finish().unwrap();

    let store = GridStore::new_with_options(directory.path(), 6, 1, 200.).unwrap();
    // Test query with all languages
    println!("Coalesce single - all languages");
    let subquery = PhrasematchSubquery {
        store: &store,
        idx: 1,
        non_overlapping_indexes: FixedBitSet::with_capacity(128),
        weight: 1.,
        match_keys: vec![MatchKeyWithId {
            nearby_only: false,
            id: 0,
            key: MatchKey {
                match_phrase: MatchPhrase::Range { start: 1, end: 3 },
                lang_set: ALL_LANGUAGES,
            },
        }],
        mask: 1 << 0,
    };
    let stack = vec![subquery];
    let match_opts = MatchOpts { zoom: 6, ..MatchOpts::default() };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    assert_eq!(result, tree_result);

    #[cfg_attr(rustfmt, rustfmt::skip)]
    {
        assert_eq!(result.len(), 4, "Returns 4 results");
        assert_eq!(result[0].relev, 1., "1st result has relevance of 1");
        assert_eq!(result[0].entries[0].grid_entry.id, 3, "1st result has highest grid id, which is the tiebreaker for sorting");
        assert_eq!(result[0].entries[0].grid_entry.relev, 1., "1st result grid has original relevance");
        assert_eq!(result[0].entries[0].matches_language, true, "1st result matches language");
        assert_eq!(result[1].relev, 1., "2nd result has original relevance");
        assert_eq!(result[1].entries[0].grid_entry.id, 2, "2nd result is the 2nd highest grid id");
        assert_eq!(result[1].entries[0].grid_entry.relev, 1., "2nd result grid has original relevance");
        assert_eq!(result[1].entries[0].matches_language, true, "2nd result matches language");
        assert_eq!(result[2].relev, 1., "3rd result has original relevance");
        assert_eq!(result[2].entries[0].grid_entry.id, 1, "3rd result is the 3rd highest grid id");
        assert_eq!(result[2].entries[0].grid_entry.relev, 1., "3rd result grid has original relevance");
        assert_eq!(result[2].entries[0].matches_language, true, "3rd result matches language");
        assert_eq!(result[3].relev, 1., "4th result has original relevance");
        assert_eq!(result[3].entries[0].grid_entry.id, 0, "4th result is the 4th highest grid id");
        assert_eq!(result[3].entries[0].grid_entry.relev, 1., "4th result grid has original relevance");
        assert_eq!(result[3].entries[0].matches_language, true, "4th result matches language");
    }

    // Test lanuage 0
    println!("Coalesce single - language 0, language matching 2 grids");
    let subquery = PhrasematchSubquery {
        store: &store,
        idx: 1,
        non_overlapping_indexes: FixedBitSet::with_capacity(128),
        weight: 1.,
        match_keys: vec![MatchKeyWithId {
            nearby_only: false,
            id: 0,
            key: MatchKey {
                match_phrase: MatchPhrase::Range { start: 1, end: 3 },
                lang_set: langarray_to_langfield(&[0]),
            },
        }],
        mask: 1 << 0,
    };
    let stack = vec![subquery];
    let match_opts = MatchOpts { zoom: 6, ..MatchOpts::default() };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    assert_eq!(result, tree_result);

    #[cfg_attr(rustfmt, rustfmt::skip)]
    {
        assert_eq!(result.len(), 4, "Returns 4 results");
        assert_eq!(result[0].relev, 1., "1st result has relevance of 1");
        assert_eq!(result[0].entries[0].grid_entry.id, 2, "1st result is a grid with 0 in the lang set, and highest grid id");
        assert_eq!(result[0].entries[0].grid_entry.relev, 1., "1st result grid has original relevance");
        assert_eq!(result[0].entries[0].matches_language, true, "1st result matches language");
        assert_eq!(result[1].relev, 1., "2nd result has original relevance");
        assert_eq!(result[1].entries[0].grid_entry.id, 0, "2nd result is a grid with 0 in the lang set");
        assert_eq!(result[1].entries[0].grid_entry.relev, 1., "2nd result grid has original relevance");
        assert_eq!(result[1].entries[0].matches_language, true, "2nd result matches language");
        assert_eq!(result[2].relev, 0.96, "3rd result has reduced relevance");
        assert_eq!(result[2].entries[0].grid_entry.id, 3, "3rd result is a grid that doesnt include lang 0");
        assert_eq!(result[2].entries[0].grid_entry.relev, 0.96, "3rd result grid has reduced relevance");
        assert_eq!(result[2].entries[0].matches_language, false, "3rd result does not match language");
        assert_eq!(result[3].relev, 0.96, "4th result has reduced relevance");
        assert_eq!(result[3].entries[0].grid_entry.id, 1, "4th result is the 4th highest grid id");
        assert_eq!(result[3].entries[0].grid_entry.relev, 0.96, "4th result grid has reduced relevance");
        assert_eq!(result[3].entries[0].matches_language, false, "4th result does not match language");
    }

    println!("Coalesce single - language 3, language matching no grids");

    let subquery = PhrasematchSubquery {
        store: &store,
        idx: 1,
        non_overlapping_indexes: FixedBitSet::with_capacity(128),
        weight: 1.,
        match_keys: vec![MatchKeyWithId {
            nearby_only: false,
            id: 0,
            key: MatchKey {
                match_phrase: MatchPhrase::Range { start: 1, end: 3 },
                lang_set: langarray_to_langfield(&[3]),
            },
        }],
        mask: 1 << 0,
    };
    let stack = vec![subquery];
    let match_opts = MatchOpts { zoom: 6, ..MatchOpts::default() };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    assert_eq!(result, tree_result);

    #[cfg_attr(rustfmt, rustfmt::skip)]
    {
        assert_eq!(result.len(), 4, "Returns 4 results");
        assert_eq!(result[0].relev, 0.96, "1st result has reduced relevance");
        assert_eq!(result[0].entries[0].grid_entry.id, 3, "1st result has highest grid id, which is the tiebreaker for sorting");
        assert_eq!(result[0].entries[0].grid_entry.relev, 0.96, "1st result grid has reduced relevance");
        assert_eq!(result[0].entries[0].matches_language, false, "1st result does not match language");
        assert_eq!(result[1].relev, 0.96, "2nd result has reduced relevance");
        assert_eq!(result[1].entries[0].grid_entry.id, 2, "2nd result is the 2nd highest grid id");
        assert_eq!(result[1].entries[0].grid_entry.relev, 0.96, "2nd result grid has reduced relevance");
        assert_eq!(result[1].entries[0].matches_language, false, "2nd result does not match language");
        assert_eq!(result[2].relev, 0.96, "3rd result has reduced relevance");
        assert_eq!(result[2].entries[0].grid_entry.id, 1, "3rd result is the 3rd highest grid id");
        assert_eq!(result[2].entries[0].grid_entry.relev, 0.96, "3rd result grid has reduced relevance");
        assert_eq!(result[2].entries[0].matches_language, false, "3rd result does not match language");
        assert_eq!(result[3].relev, 0.96, "4th result has reduced relevance");
        assert_eq!(result[3].entries[0].grid_entry.id, 0, "4th result is the 4th highest grid id");
        assert_eq!(result[3].entries[0].grid_entry.relev, 0.96, "4th result grid has reduced relevance");
        assert_eq!(result[3].entries[0].matches_language, false, "4th result does not match language");
    }
}

#[test]
fn coalesce_single_nearby_only() {
    let directory: tempfile::TempDir = tempfile::tempdir().unwrap();
    let mut builder = GridStoreBuilder::new(directory.path()).unwrap();

    let key = GridKey { phrase_id: 1, lang_set: 1 };

    let entries = vec![
        GridEntry { id: 1, x: 100, y: 100, relev: 1., score: 1, source_phrase_hash: 0 },
        GridEntry { id: 2, x: 50, y: 50, relev: 1., score: 1, source_phrase_hash: 0 },
        GridEntry { id: 3, x: 90, y: 90, relev: 1., score: 1, source_phrase_hash: 0 },
        GridEntry { id: 4, x: 200, y: 200, relev: 1., score: 1, source_phrase_hash: 0 },
    ];
    builder.insert(&key, entries).expect("Unable to insert record");

    builder.finish().unwrap();

    let store = GridStore::new_with_options(directory.path(), 14, 1, 200.).unwrap();
    let subquery = PhrasematchSubquery {
        store: &store,
        idx: 1,
        non_overlapping_indexes: FixedBitSet::with_capacity(128),
        weight: 1.,
        match_keys: vec![MatchKeyWithId {
            nearby_only: true,
            id: 0,
            key: MatchKey { match_phrase: MatchPhrase::Range { start: 1, end: 3 }, lang_set: 1 },
        }],
        mask: 1 << 0,
    };
    let stack = vec![subquery];
    let match_opts = MatchOpts { zoom: 14, proximity: Some([100, 100]), ..MatchOpts::default() };
    let tree = stackable(&stack);
    let tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    let result_ids: Vec<u32> =
        tree_result.iter().map(|context| context.entries[0].grid_entry.id).collect();
    assert_eq!(
        result_ids,
        [1, 3],
        "Results with the same relev and score should be ordered by distance"
    );
}

#[test]
fn coalesce_multi_test() {
    // Add more specific layer into a store
    let store1 = create_store(
        vec![StoreEntryBuildingBlock {
            grid_key: GridKey { phrase_id: 1, lang_set: 1 },
            entries: vec![
                GridEntry { id: 1, x: 1, y: 1, relev: 1., score: 1, source_phrase_hash: 0 },
                // TODO: this isn't a real tile at zoom 1. Maybe pick more realistic test case?
                GridEntry { id: 2, x: 2, y: 2, relev: 1., score: 1, source_phrase_hash: 0 },
            ],
        }],
        0,
        1,
        0,
        FixedBitSet::with_capacity(128),
        40.,
    );

    let store2 = create_store(
        vec![StoreEntryBuildingBlock {
            grid_key: GridKey { phrase_id: 2, lang_set: 1 },
            entries: vec![
                GridEntry { id: 1, x: 1, y: 1, relev: 1., score: 3, source_phrase_hash: 0 },
                GridEntry { id: 2, x: 2, y: 2, relev: 1., score: 3, source_phrase_hash: 0 },
                GridEntry { id: 3, x: 3, y: 3, relev: 1., score: 1, source_phrase_hash: 0 },
            ],
        }],
        1,
        2,
        1,
        FixedBitSet::with_capacity(128),
        40.,
    );

    let stack = vec![
        PhrasematchSubquery {
            store: &store1.store,
            idx: store1.idx,
            non_overlapping_indexes: store1.non_overlapping_indexes.clone(),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                nearby_only: false,
                id: 0,
                key: MatchKey {
                    match_phrase: MatchPhrase::Range { start: 1, end: 3 },
                    lang_set: 1,
                },
            }],
            mask: 1 << 1,
        },
        PhrasematchSubquery {
            store: &store2.store,
            idx: store2.idx,
            non_overlapping_indexes: store2.non_overlapping_indexes.clone(),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                nearby_only: false,
                id: 1,
                key: MatchKey {
                    match_phrase: MatchPhrase::Range { start: 1, end: 3 },
                    lang_set: 1,
                },
            }],
            mask: 1 << 0,
        },
    ];

    // Test coalesce multi with no proximity or bbox
    println!("Coalsece multi - no proximity no bbox");
    let match_opts = MatchOpts { zoom: 6, ..MatchOpts::default() };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    assert_eq!(result, tree_result);
    assert_eq!(result[0].relev, 1., "1st result has relevance 1");
    assert_eq!(result[0].mask, 3, "1st result context has correct mask");
    assert_eq!(result[0].entries.len(), 2, "1st result has 2 coalesce entries");
    assert_eq!(
        result[0].entries[0],
        CoalesceEntry {
            phrasematch_id: 0,
            matches_language: true,
            idx: 1,
            tmp_id: 33554434,
            mask: 1 << 0,
            distance: 0.,
            scoredist: 3.,
            grid_entry: GridEntry {
                id: 2,
                x: 2,
                y: 2,
                relev: 0.5,
                score: 3,
                source_phrase_hash: 0,
            }
        },
        "1st result 1st entry is the highest score from the higher zoom index"
    );
    assert_eq!(
        result[0].entries[1],
        CoalesceEntry {
            phrasematch_id: 0,
            matches_language: true,
            idx: 0,
            tmp_id: 1,
            mask: 1 << 1,
            distance: 0.,
            scoredist: 1.,
            grid_entry: GridEntry {
                id: 1,
                x: 1,
                y: 1,
                relev: 0.5,
                score: 1,
                source_phrase_hash: 0,
            }
        },
        "1st result 2nd entry is the overelpping grid from the lower zoom index"
    );
    assert_eq!(result[1].relev, 1., "2nd result has relevance 1");
    assert_eq!(result[1].mask, 3, "2nd result context has correct mask");
    assert_eq!(result[1].entries.len(), 2, "2nd result has 2 coalesce entries");
    assert_eq!(
        result[1].entries[0],
        CoalesceEntry {
            phrasematch_id: 0,
            matches_language: true,
            idx: 1,
            tmp_id: 33554435,
            mask: 1 << 0,
            distance: 0.,
            scoredist: 1.,
            grid_entry: GridEntry {
                id: 3,
                x: 3,
                y: 3,
                relev: 0.5,
                score: 1,
                source_phrase_hash: 0,
            }
        },
        "2nd result 1st entry is the lower score grid that overlaps with a grid "
    );
    assert_eq!(
        result[0].entries[1],
        CoalesceEntry {
            phrasematch_id: 0,
            matches_language: true,
            idx: 0,
            tmp_id: 1,
            mask: 1 << 1,
            distance: 0.,
            scoredist: 1.,
            grid_entry: GridEntry {
                id: 1,
                x: 1,
                y: 1,
                relev: 0.5,
                score: 1,
                source_phrase_hash: 0,
            }
        },
        "2nd result 2nd entry is the overlapping grid from the lower zoom index"
    );

    // Test coalesce multi with proximity
    println!("Coalesce multi - with proximity");
    let match_opts = MatchOpts { zoom: 2, proximity: Some([3, 3]), ..MatchOpts::default() };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    assert_eq!(result, tree_result);
    assert_eq!(result[0].relev, 1., "1st result context has relevance 1");
    assert_eq!(result[0].mask, 3, "1st result context has correct mask");
    assert_eq!(result[0].entries.len(), 2, "1st result has 2 coalesce entries");
    assert_eq!(
        result[0].entries[0],
        CoalesceEntry {
            phrasematch_id: 0,
            matches_language: true,
            idx: 1,
            tmp_id: 33554435,
            mask: 1 << 0,
            distance: 0.,
            scoredist: 1.5839497841387566,
            grid_entry: GridEntry {
                id: 3,
                x: 3,
                y: 3,
                relev: 0.5,
                score: 1,
                source_phrase_hash: 0,
            }
        },
        "1st result 1st entry is closest entry in the higher zoom index"
    );
    assert_eq!(
        result[0].entries[1],
        CoalesceEntry {
            phrasematch_id: 0,
            matches_language: true,
            idx: 0,
            tmp_id: 1,
            mask: 1 << 1,
            distance: 0.,
            scoredist: 1.5839497841387566,
            grid_entry: GridEntry {
                id: 1,
                x: 1,
                y: 1,
                relev: 0.5,
                score: 1,
                source_phrase_hash: 0,
            }
        },
        "1st result 2nd entry is the overlapping entry, the distance for the outer entry is 0"
    );
    assert_eq!(result[1].entries.len(), 2, "2nd result has 2 coalesce entries");
    assert_eq!(
        result[1].entries[0],
        CoalesceEntry {
            phrasematch_id: 0,
            matches_language: true,
            idx: 1,
            tmp_id: 33554434,
            mask: 1 << 0,
            distance: 1.4142135623730951,
            scoredist: 1.109893833332405,
            grid_entry: GridEntry {
                id: 2,
                x: 2,
                y: 2,
                relev: 0.5,
                score: 3,
                source_phrase_hash: 0,
            }
        },
        "2nd result 1st entry is the farther away entry from the higher zoom index"
    );
    assert_eq!(
        result[1].entries[1],
        CoalesceEntry {
            phrasematch_id: 0,
            matches_language: true,
            idx: 0,
            tmp_id: 1,
            mask: 1 << 1,
            distance: 0.,
            scoredist: 1.5839497841387566,
            grid_entry: GridEntry {
                id: 1,
                x: 1,
                y: 1,
                relev: 0.5,
                score: 1,
                source_phrase_hash: 0,
            }
        },
        "2nd result 2nd entry is the overlapping entry, the distance for the outer entry is 0"
    );
}

#[test]
fn coalesce_multi_languages_test() {
    // Store 1 with grids in all languages
    let store1 = create_store(
        vec![StoreEntryBuildingBlock {
            grid_key: GridKey { phrase_id: 1, lang_set: ALL_LANGUAGES },
            entries: vec![GridEntry {
                id: 1,
                x: 1,
                y: 1,
                relev: 1.,
                score: 1,
                source_phrase_hash: 0,
            }],
        }],
        0,
        1,
        0,
        FixedBitSet::with_capacity(128),
        200.,
    );

    // Store 2 with grids in multiple language sets
    let store2 = create_store(
        vec![
            // Insert grid with lang_set 1
            StoreEntryBuildingBlock {
                grid_key: GridKey { phrase_id: 2, lang_set: langarray_to_langfield(&[1]) },
                entries: vec![GridEntry {
                    id: 2,
                    x: 1,
                    y: 1,
                    relev: 1.,
                    score: 1,
                    source_phrase_hash: 0,
                }],
            },
            // Insert grid with lang_set 0
            StoreEntryBuildingBlock {
                grid_key: GridKey { phrase_id: 2, lang_set: langarray_to_langfield(&[0]) },
                entries: vec![GridEntry {
                    id: 3,
                    x: 1,
                    y: 1,
                    relev: 1.,
                    score: 1,
                    source_phrase_hash: 0,
                }],
            },
        ],
        1,
        1,
        1,
        FixedBitSet::with_capacity(128),
        200.,
    );

    // Test ALL LANGUAGES
    println!("Coalesce multi - all languages");
    let stack = vec![
        PhrasematchSubquery {
            store: &store1.store,
            idx: store1.idx,
            non_overlapping_indexes: store1.non_overlapping_indexes.clone(),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                nearby_only: false,
                id: 0,
                key: MatchKey {
                    match_phrase: MatchPhrase::Range { start: 1, end: 3 },
                    lang_set: ALL_LANGUAGES,
                },
            }],
            mask: 1 << 1,
        },
        PhrasematchSubquery {
            store: &store2.store,
            idx: store2.idx,
            non_overlapping_indexes: store2.non_overlapping_indexes.clone(),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                nearby_only: false,
                id: 1,
                key: MatchKey {
                    match_phrase: MatchPhrase::Range { start: 1, end: 3 },
                    lang_set: ALL_LANGUAGES,
                },
            }],
            mask: 1 << 0,
        },
    ];
    let match_opts = MatchOpts { zoom: 6, ..MatchOpts::default() };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    assert_eq!(result, tree_result);
    #[cfg_attr(rustfmt, rustfmt::skip)]
    {
        assert_eq!(result.len(), 2, "Two results are returned");
        assert_eq!(result[0].entries.len(), 2, "1st context has two entries");
        assert_eq!(result[0].relev, 1., "1st context has relevance of 1");
        assert_eq!(result[0].entries[0].grid_entry.id, 3, "1st entry in 1st result has highest grid id, which is the tiebreaker for sorting");
        assert_eq!(result[0].entries[0].grid_entry.relev, 0.5, "1st entry in 1st result has original relevance" );
        assert_eq!(result[0].entries[0].matches_language, true, "1st entry in 1st result matches language" );
        assert_eq!(result[0].entries[1].grid_entry.id, 1, "2nd entry in 1st result is the overapping grid" );
        assert_eq!(result[0].entries[1].grid_entry.relev, 0.5, "2nd entry in 1st result has original relevance" );
        assert_eq!(result[0].entries[1].matches_language, true, "2nd entry in 1st result matches language" );
        assert_eq!(result[1].entries.len(), 2, "2nd context has two entries");
        assert_eq!(result[1].relev, 1., "2nd context has relevance of 1");
        assert_eq!(result[1].entries[0].grid_entry.id, 2, "1st entry in 2nd result is the lower grid id" );
        assert_eq!(result[1].entries[0].grid_entry.relev, 0.5, "1st entry in 2nd result has original relevance" );
        assert_eq!(result[1].entries[0].matches_language, true, "1st entry in 2nd result matches language" );
        assert_eq!(result[1].entries[1].grid_entry.id, 1, "2nd entry in 2nd result is the overlapping grid" );
        assert_eq!(result[1].entries[1].grid_entry.relev, 0.5, "2nd entry in 2nd result has original relevance" );
        assert_eq!(result[1].entries[1].matches_language, true, "2nd entry in 2nd result matches language" );
    }

    // Test language 0
    println!("Coalesce multi - language 0");
    let stack = vec![
        PhrasematchSubquery {
            store: &store1.store,
            idx: store1.idx,
            non_overlapping_indexes: store1.non_overlapping_indexes.clone(),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                nearby_only: false,
                id: 0,
                key: MatchKey {
                    match_phrase: MatchPhrase::Range { start: 1, end: 3 },
                    lang_set: ALL_LANGUAGES,
                },
            }],
            mask: 1 << 1,
        },
        PhrasematchSubquery {
            store: &store2.store,
            idx: store2.idx,
            non_overlapping_indexes: store2.non_overlapping_indexes.clone(),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                nearby_only: false,
                id: 1,
                key: MatchKey {
                    match_phrase: MatchPhrase::Range { start: 1, end: 3 },
                    lang_set: langarray_to_langfield(&[0]),
                },
            }],
            mask: 1 << 0,
        },
    ];
    let match_opts = MatchOpts { zoom: 6, ..MatchOpts::default() };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    assert_eq!(result, tree_result);
    #[cfg_attr(rustfmt, rustfmt::skip)]
    {
        assert_eq!(result.len(), 2, "Two results are returned");
        assert_eq!(result[0].entries.len(), 2, "1st context has two entries");
        assert_eq!(result[0].relev, 1., "1st context has relevance of 1");
        assert_eq!(result[0].entries[0].grid_entry.id, 3, "1st entry in 1st result is the id of the better language match");
        assert_eq!(result[0].entries[0].grid_entry.relev, 0.5, "1st entry in 1st result has original relevance");
        assert_eq!(result[0].entries[0].matches_language, true, "1st entry in 1st result matches language");
        assert_eq!(result[0].entries[1].grid_entry.id, 1, "2nd entry in 1st result is the overapping grid");
        assert_eq!(result[0].entries[1].grid_entry.relev, 0.5, "2nd entry in 1st result has original relevance because the grid has all languages");
        assert_eq!(result[0].entries[1].matches_language, true, "2nd entry in 1st result matches language");
        assert_eq!(result[1].entries.len(), 2, "2nd context has two entries");
        assert_eq!(result[1].relev, 0.98, "2nd context has lower overall relevance due to language penalty");
        assert_eq!(result[1].entries[0].grid_entry.id, 2, "1st entry in 2nd result has the id of the worse language match");
        assert_eq!(result[1].entries[0].grid_entry.relev, 0.48, "1st entry in 2nd result has lower relevance due to language penalty");
        assert_eq!(result[1].entries[0].matches_language, false, "1st entry in 2nd result does not match language");
        assert_eq!(result[1].entries[1].grid_entry.id, 1, "2nd entry in 2nd result is the overlapping grid");
        assert_eq!(result[1].entries[1].grid_entry.relev, 0.5, "2nd entry in 2nd result has original relevance because the grid has all languages");
        assert_eq!(result[1].entries[1].matches_language, true, "2nd entry in 2nd result matches language");
    }

    // Test language 3
    println!("Coalsece multi - language 3");
    let stack = vec![
        PhrasematchSubquery {
            store: &store1.store,
            idx: store1.idx,
            non_overlapping_indexes: store1.non_overlapping_indexes.clone(),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                nearby_only: false,
                id: 0,
                key: MatchKey {
                    match_phrase: MatchPhrase::Range { start: 1, end: 3 },
                    lang_set: ALL_LANGUAGES,
                },
            }],
            mask: 1 << 1,
        },
        PhrasematchSubquery {
            store: &store2.store,
            idx: store2.idx,
            non_overlapping_indexes: store2.non_overlapping_indexes.clone(),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                nearby_only: false,
                id: 1,
                key: MatchKey {
                    match_phrase: MatchPhrase::Range { start: 1, end: 3 },
                    lang_set: langarray_to_langfield(&[3]),
                },
            }],
            mask: 1 << 0,
        },
    ];
    let match_opts = MatchOpts { zoom: 6, ..MatchOpts::default() };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    assert_eq!(result, tree_result);
    #[cfg_attr(rustfmt, rustfmt::skip)]
    {
        assert_eq!(result.len(), 2, "Two results are returned");
        assert_eq!(result[0].entries.len(), 2, "1st context has two entries");
        assert_eq!(result[0].relev, 0.98, "1st context has lower overall relevance due to language penalty");
        assert_eq!(result[0].entries[0].grid_entry.id, 3, "1st entry in 1st result has highest grid id, which is the tiebreaker for sorting");
        assert_eq!(result[0].entries[0].grid_entry.relev, 0.48, "1st entry in 1st result has lower relevance due to language penalty");
        assert_eq!(result[0].entries[0].matches_language, false, "1st entry in 1st result does not match language");
        assert_eq!(result[0].entries[1].grid_entry.id, 1, "2nd entry in 1st result is the overapping grid");
        assert_eq!(result[0].entries[1].grid_entry.relev, 0.5, "2nd entry in 1st result has original relevance because the grid has all languages");
        assert_eq!(result[0].entries[1].matches_language, true, "2nd entry in 1st result matches language");
        assert_eq!(result[1].entries.len(), 2, "2nd context has two entries");
        assert_eq!(result[1].relev, 0.98, "2nd context has lower overall relevance due to language penalty");
        assert_eq!(result[1].entries[0].grid_entry.id, 2, "1st entry in 2nd result has the id of the other grid");
        assert_eq!(result[1].entries[0].grid_entry.relev, 0.48, "1st entry in 2nd result has lower relevance due to language penalty");
        assert_eq!(result[1].entries[0].matches_language, false, "1st entry in 2nd result does not match language");
        assert_eq!(result[1].entries[1].grid_entry.id, 1, "2nd entry in 2nd result is the overlapping grid");
        assert_eq!(result[1].entries[1].grid_entry.relev, 0.5, "2nd entry in 2nd result has original relevance because the grid has all languages");
        assert_eq!(result[1].entries[1].matches_language, true, "2nd entry in 2nd result matches language");
    }
}

#[test]
fn coalesce_multi_scoredist() {
    // Add more specific layer into a store
    let store1 = create_store(
        vec![StoreEntryBuildingBlock {
            grid_key: GridKey { phrase_id: 1, lang_set: 0 },
            entries: vec![GridEntry {
                id: 1,
                x: 0,
                y: 0,
                relev: 1.,
                score: 1,
                source_phrase_hash: 0,
            }],
        }],
        0,
        0,
        0,
        FixedBitSet::with_capacity(128),
        200.,
    );

    // Add less specific layer into a store
    let store2 = create_store(
        vec![StoreEntryBuildingBlock {
            grid_key: GridKey { phrase_id: 2, lang_set: 0 },
            entries: vec![
                GridEntry { id: 2, x: 4800, y: 6200, relev: 1., score: 7, source_phrase_hash: 0 },
                GridEntry { id: 3, x: 4600, y: 6200, relev: 1., score: 1, source_phrase_hash: 0 },
            ],
        }],
        1,
        14,
        1,
        FixedBitSet::with_capacity(128),
        200.,
    );

    let stack = vec![
        PhrasematchSubquery {
            store: &store1.store,
            idx: store1.idx,
            non_overlapping_indexes: store1.non_overlapping_indexes.clone(),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                nearby_only: false,
                id: 0,
                key: MatchKey {
                    match_phrase: MatchPhrase::Range { start: 1, end: 3 },
                    lang_set: 0,
                },
            }],
            mask: 1 << 1,
        },
        PhrasematchSubquery {
            store: &store2.store,
            idx: store2.idx,
            non_overlapping_indexes: store2.non_overlapping_indexes.clone(),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                nearby_only: false,
                id: 1,
                key: MatchKey {
                    match_phrase: MatchPhrase::Range { start: 1, end: 3 },
                    lang_set: 0,
                },
            }],
            mask: 1 << 0,
        },
    ];
    // Closer proximity to one grid
    println!("Coalesce multi - proximity very close to one grid");
    let match_opts = MatchOpts { zoom: 14, proximity: Some([4601, 6200]), ..MatchOpts::default() };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    assert_eq!(result, tree_result);
    assert_eq!(result[0].entries[0].grid_entry.id, 3, "Closer feature is 1st");
    assert_eq!(result[1].entries[0].grid_entry.id, 2, "Farther feature is 2nd");
    assert_eq!(
        result[0].entries[0].distance < result[1].entries[0].distance,
        true,
        "1st grid in 1st context is closer than 1st grid in 2nd context"
    );

    // Proximity is still close to same grid, but less close
    println!("Coalesce multi - proximity less close to one grid");
    let match_opts = MatchOpts { zoom: 14, proximity: Some([4610, 6200]), ..MatchOpts::default() };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    assert_eq!(result, tree_result);
    assert_eq!(result[0].entries[0].grid_entry.id, 3, "Farther feature with higher score is 1st");
    assert_eq!(result[1].entries[0].grid_entry.id, 2, "Closer feature with lower score is 2nd");
    assert_eq!(
        result[0].entries[0].distance > result[1].entries[0].distance,
        false,
        "1st grid in 1st context is not closer than 1st grid in 2nd context"
    );
}

// TODO: language tests
#[test]
fn coalesce_multi_test_bbox() {
    let store1 = create_store(
        vec![StoreEntryBuildingBlock {
            grid_key: GridKey { phrase_id: 1, lang_set: ALL_LANGUAGES },
            entries: vec![
                GridEntry { id: 1, x: 0, y: 0, relev: 0.8, score: 1, source_phrase_hash: 0 },
                GridEntry { id: 2, x: 1, y: 1, relev: 1., score: 1, source_phrase_hash: 0 },
            ],
        }],
        0,
        1,
        0,
        FixedBitSet::with_capacity(128),
        200.,
    );
    let store2 = create_store(
        vec![StoreEntryBuildingBlock {
            grid_key: GridKey { phrase_id: 2, lang_set: ALL_LANGUAGES },
            entries: vec![
                GridEntry { id: 3, x: 3, y: 0, relev: 1., score: 1, source_phrase_hash: 0 },
                GridEntry { id: 4, x: 0, y: 3, relev: 1., score: 1, source_phrase_hash: 0 },
            ],
        }],
        1,
        2,
        1,
        FixedBitSet::with_capacity(128),
        200.,
    );

    let store3 = create_store(
        vec![StoreEntryBuildingBlock {
            grid_key: GridKey { phrase_id: 3, lang_set: ALL_LANGUAGES },
            entries: vec![
                GridEntry { id: 5, x: 21, y: 7, relev: 1., score: 1, source_phrase_hash: 0 },
                GridEntry { id: 6, x: 21, y: 18, relev: 1., score: 1, source_phrase_hash: 0 },
            ],
        }],
        2,
        5,
        2,
        FixedBitSet::with_capacity(128),
        200.,
    );

    let stack = vec![
        PhrasematchSubquery {
            store: &store1.store,
            idx: store1.idx,
            non_overlapping_indexes: store1.non_overlapping_indexes.clone(),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                nearby_only: false,
                id: 0,
                key: MatchKey {
                    match_phrase: MatchPhrase::Range { start: 1, end: 3 },
                    lang_set: ALL_LANGUAGES,
                },
            }],
            mask: 1 << 1,
        },
        PhrasematchSubquery {
            store: &store2.store,
            idx: store2.idx,
            non_overlapping_indexes: store2.non_overlapping_indexes.clone(),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                nearby_only: false,
                id: 1,
                key: MatchKey {
                    match_phrase: MatchPhrase::Range { start: 1, end: 3 },
                    lang_set: ALL_LANGUAGES,
                },
            }],
            mask: 1 << 0,
        },
    ];
    // Test bbox at zoom 1 that should contain 2 grids
    println!("Coalesce multi - bbox at lower zoom of subquery");
    let match_opts = MatchOpts { zoom: 1, bbox: Some([0, 0, 1, 0]), ..MatchOpts::default() };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let _tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    // assert_eq!(result, tree_result);
    assert_eq!(result.len(), 2, "Bbox [1,0,0,1,0] - 2 results are within the bbox");
    assert_eq!(
        (result[0].entries[0].grid_entry.x, result[0].entries[0].grid_entry.y),
        (3, 0),
        "Bbox [1,0,0,1,0] - 1st result is zxy 2/3/0, and the higher relevance grid within the bbox"
    );
    assert_eq!(
        (result[1].entries[0].grid_entry.x, result[1].entries[0].grid_entry.y),
        (0, 0),
        "Bbox [1,0,0,1,0] - 2nd result is zxy 1/0/0"
    );
    // Test bbox at zoom 2 that should contain 2 grids
    println!("Coalesce multi - bbox at higher zoom of subquery");
    let match_opts = MatchOpts { zoom: 2, bbox: Some([0, 0, 1, 3]), ..MatchOpts::default() };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let _tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    // assert_eq!(result, tree_result);
    assert_eq!(result.len(), 2, "Bbox [2,0,0,1,3] - 2 results are within the bbox");
    assert_eq!(
        (result[0].entries[0].grid_entry.x, result[0].entries[0].grid_entry.y),
        (0, 3),
        "Bbox [2,0,0,1,3] - 1st result is zxy 2/0/3"
    );
    assert_eq!(
        (result[1].entries[0].grid_entry.x, result[1].entries[0].grid_entry.y),
        (0, 0),
        "Bbox [2,0,0,1,3] - 2nd result is zxy 1/0/0"
    );

    // Test bbox at zoom 6 that should contain 2 grids
    println!("Coalesce multi - bbox at zoom 6");
    let match_opts = MatchOpts { zoom: 6, bbox: Some([14, 30, 15, 64]), ..MatchOpts::default() };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let _tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    // assert_eq!(result, tree_result);
    assert_eq!(result.len(), 2, "Bbox [6,14,30,15,64] - 2 results are within the bbox");
    assert_eq!(
        (result[0].entries[0].grid_entry.x, result[0].entries[0].grid_entry.y),
        (0, 3),
        "Bbox [6,14,30,15,64] - 1st result is zxy 2/0/3"
    );
    assert_eq!(
        (result[1].entries[0].grid_entry.x, result[1].entries[0].grid_entry.y),
        (0, 0),
        "Bbox [6,14,30,15,64] - 2nd result is zxy 1/0/0"
    );

    // Test bbox at lower zoom than either of the expected results
    println!("Coalesce multi - bbox at lower zoom than either of the expected results");
    let stack = vec![
        PhrasematchSubquery {
            store: &store2.store,
            idx: store2.idx,
            non_overlapping_indexes: store2.non_overlapping_indexes.clone(),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                nearby_only: false,
                id: 0,
                key: MatchKey {
                    match_phrase: MatchPhrase::Range { start: 1, end: 4 },
                    lang_set: ALL_LANGUAGES,
                },
            }],
            mask: 1 << 1,
        },
        PhrasematchSubquery {
            store: &store3.store,
            idx: store3.idx,
            non_overlapping_indexes: store3.non_overlapping_indexes.clone(),
            weight: 0.5,
            match_keys: vec![MatchKeyWithId {
                nearby_only: false,
                id: 1,
                key: MatchKey {
                    match_phrase: MatchPhrase::Range { start: 1, end: 4 },
                    lang_set: ALL_LANGUAGES,
                },
            }],
            mask: 1 << 0,
        },
    ];
    let match_opts = MatchOpts { zoom: 1, bbox: Some([0, 0, 1, 0]), ..MatchOpts::default() };
    let result = coalesce(stack.iter().map(|s| s.clone().into()).collect(), &match_opts).unwrap();
    let tree = stackable(&stack);
    let _tree_result = truncate_coalesce_results(tree_coalesce(&tree, &match_opts).unwrap());
    // assert_eq!(result, tree_result);
    assert_eq!(result.len(), 2, "Bbox [1,0,0,1,0] - 2 results are within the bbox");
    assert_eq!(
        (result[0].entries[0].grid_entry.x, result[0].entries[0].grid_entry.y),
        (3, 0),
        "Bbox [1,0,0,1,0] - 1st result is xzy 2/3/0"
    );
    assert_eq!(
        (result[1].entries[0].grid_entry.x, result[1].entries[0].grid_entry.y),
        (21, 7),
        "Bbox [1,0,0,1,0] - 2nd result is xzy 5/20/7"
    );
}

#[cfg(test)]
fn truncate_coalesce_results(results: Vec<CoalesceContext>) -> Vec<CoalesceContext> {
    let mut new_results = Vec::new();
    let max_relevance = if results.len() == 0 { 1.0 } else { results[0].relev };
    for r in results {
        if max_relevance - r.relev < 0.25 {
            let context = r.clone();
            new_results.push(context);
        }
    }
    new_results
}

// TODO: add proximity test with max score
// TODO: add sort tests?
