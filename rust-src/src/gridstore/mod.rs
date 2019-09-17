mod builder;
mod coalesce;
mod common;
mod gridstore_format;
mod spatial;
mod store;

pub use builder::*;
pub use coalesce::coalesce;
pub use common::*;
pub use store::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn combined_test() {
        let directory: tempfile::TempDir = tempfile::tempdir().unwrap();
        let mut builder = GridStoreBuilder::new(directory.path()).unwrap();

        let key = GridKey { phrase_id: 1, lang_set: 1 };

        let mut entries = vec![
            GridEntry { id: 2, x: 2, y: 2, relev: 0.8, score: 3, source_phrase_hash: 0 },
            GridEntry { id: 3, x: 3, y: 3, relev: 1., score: 1, source_phrase_hash: 1 },
            GridEntry { id: 1, x: 1, y: 1, relev: 1., score: 7, source_phrase_hash: 2 },
        ];
        builder.insert(&key, entries.clone()).expect("Unable to insert record");

        builder.finish().unwrap();

        let reader = GridStore::new(directory.path()).unwrap();
        let record: Vec<_> = reader.get(&key).unwrap().unwrap().collect();

        entries.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(
            record, entries,
            "identical entries come out as went in, in reverse-sorted order"
        );

        {
            let key = GridKey { phrase_id: 2, lang_set: 1 };
            let record = reader.get(&key).expect("Failed to get key");
            assert!(record.is_none(), "Retrieved no results");
        }
    }

    #[test]
    fn renumber_test() {
        let directory: tempfile::TempDir = tempfile::tempdir().unwrap();
        let mut builder = GridStoreBuilder::new(directory.path()).unwrap();

        // phrase IDs are descending, grid IDs are ascending
        let items = vec![
            (
                GridKey { phrase_id: 2, lang_set: 1 },
                GridEntry { id: 0, x: 1, y: 1, relev: 1., score: 7, source_phrase_hash: 2 },
            ),
            (
                GridKey { phrase_id: 1, lang_set: 1 },
                GridEntry { id: 1, x: 1, y: 1, relev: 1., score: 7, source_phrase_hash: 2 },
            ),
            (
                GridKey { phrase_id: 0, lang_set: 1 },
                GridEntry { id: 2, x: 1, y: 1, relev: 1., score: 7, source_phrase_hash: 2 },
            ),
        ];

        for (key, val) in items {
            builder.insert(&key, vec![val]).expect("Unable to insert record");
        }
        builder.renumber(&vec![2, 1, 0]).unwrap();
        // after renumbering, the IDs should match
        builder.finish().unwrap();

        let reader = GridStore::new(directory.path()).unwrap();

        for id in 0..=2 {
            let entries: Vec<_> =
                reader.get(&GridKey { phrase_id: id, lang_set: 1 }).unwrap().unwrap().collect();
            assert_eq!(id, entries[0].id);
        }
    }

    #[test]
    fn phrase_hash_test() {
        let directory: tempfile::TempDir = tempfile::tempdir().unwrap();
        let mut builder = GridStoreBuilder::new(directory.path()).unwrap();

        let key = GridKey { phrase_id: 1, lang_set: 1 };

        let mut entries = vec![
            GridEntry { id: 1, x: 1, y: 1, relev: 1.0, score: 1, source_phrase_hash: 0 },
            GridEntry { id: 1, x: 1, y: 1, relev: 0.6, score: 1, source_phrase_hash: 2 },
            GridEntry { id: 1, x: 1, y: 1, relev: 0.4, score: 1, source_phrase_hash: 3 },
        ];
        builder.insert(&key, entries.clone()).expect("Unable to insert record");

        builder.finish().unwrap();

        let reader = GridStore::new(directory.path()).unwrap();
        let record: Vec<_> = reader.get(&key).unwrap().unwrap().collect();

        entries.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(
            record, entries,
            "identical entries come out as went in, in reverse-sorted order"
        );
    }

    #[test]
    fn cover_test() {
        let directory: tempfile::TempDir = tempfile::tempdir().unwrap();
        let mut builder = GridStoreBuilder::new(directory.path()).unwrap();

        let key = GridKey { phrase_id: 1, lang_set: 1 };

        let entries = vec![
            GridEntry { id: 1, x: 1, y: 1, relev: 1., score: 1, source_phrase_hash: 0 },
            GridEntry { id: 1, x: 1, y: 2, relev: 1., score: 1, source_phrase_hash: 0 },
            GridEntry { id: 1, x: 2, y: 1, relev: 1., score: 1, source_phrase_hash: 0 },
        ];
        builder.insert(&key, entries.clone()).expect("Unable to insert record");

        builder.finish().unwrap();

        let reader = GridStore::new(directory.path()).unwrap();
        let record: Vec<_> = reader.get(&key).unwrap().unwrap().collect();

        // Results come back morton order. Maybe we should implement a custom partial_cmp
        assert_eq!(record[0], entries[1], "expected first result");
        assert_eq!(record[1], entries[2], "expected second result");
        assert_eq!(record[2], entries[0], "expected second result");
    }

    #[test]
    fn score_test() {
        let directory: tempfile::TempDir = tempfile::tempdir().unwrap();
        let mut builder = GridStoreBuilder::new(directory.path()).unwrap();

        let key = GridKey { phrase_id: 1, lang_set: 1 };

        let mut entries = vec![
            GridEntry { id: 1, x: 1, y: 1, relev: 1., score: 1, source_phrase_hash: 0 },
            GridEntry { id: 1, x: 1, y: 1, relev: 1., score: 7, source_phrase_hash: 0 },
        ];
        builder.insert(&key, entries.clone()).expect("Unable to insert record");

        builder.finish().unwrap();

        let reader = GridStore::new(directory.path()).unwrap();
        let record: Vec<_> = reader.get(&key).unwrap().unwrap().collect();

        entries.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(
            record, entries,
            "identical entries come out as went in, in reverse-sorted order"
        );
    }

    #[test]
    fn matching_test() {
        let directory: tempfile::TempDir = tempfile::tempdir().unwrap();
        let mut builder = GridStoreBuilder::new(directory.path()).unwrap();

        let keys = vec![
            GridKey { phrase_id: 1, lang_set: 1 },
            GridKey { phrase_id: 1, lang_set: 2 },
            GridKey { phrase_id: 2, lang_set: 1 },
            GridKey { phrase_id: 1, lang_set: 1 },
        ];

        let mut i = 0;
        for key in keys.iter() {
            for _j in 0..2 {
                #[cfg_attr(rustfmt, rustfmt::skip)]
                let entries = vec![
                    GridEntry { id: i, x: (2 * i) as u16, y: 1, relev: 1., score: 1, source_phrase_hash: 0 },
                    GridEntry { id: i + 1, x: ((2 * i) + 1) as u16, y: 1, relev: 1., score: 7, source_phrase_hash: 0 },
                    GridEntry { id: i + 2, x: ((2 * i) + 2) as u16, y: 1, relev: 1., score: 7, source_phrase_hash: 0 },
                    GridEntry { id: i + 3, x: ((2 * i) + 1) as u16, y: 1, relev: 1., score: 7, source_phrase_hash: 0 },
                ];
                i += 4;

                builder.insert(key, entries).expect("Unable to insert record");
            }
        }

        builder.finish().unwrap();

        let reader = GridStore::new(directory.path()).unwrap();

        let search_key =
            MatchKey { match_phrase: MatchPhrase::Range { start: 1, end: 2 }, lang_set: 1 };
        let records: Vec<_> =
            reader.get_matching(&search_key, &MatchOpts::default()).unwrap().collect();
        #[cfg_attr(rustfmt, rustfmt::skip)]
        assert_eq!(
            records,
            [
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 58, y: 1, id: 30, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 57, y: 1, id: 31, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 57, y: 1, id: 29, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 1, x: 56, y: 1, id: 28, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 1.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 7, x: 26, y: 1, id: 14, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 7, x: 25, y: 1, id: 15, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 7, x: 25, y: 1, id: 13, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 1, x: 24, y: 1, id: 12, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 1.0 }
            ]
        );

        let search_key =
            MatchKey { match_phrase: MatchPhrase::Range { start: 1, end: 3 }, lang_set: 1 };
        let records: Vec<_> =
            reader.get_matching(&search_key, &MatchOpts::default()).unwrap().collect();
        #[cfg_attr(rustfmt, rustfmt::skip)]
        assert_eq!(
            records,
            [
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 58, y: 1, id: 30, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 57, y: 1, id: 31, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 57, y: 1, id: 29, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 42, y: 1, id: 22, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 41, y: 1, id: 23, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 41, y: 1, id: 21, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 1, x: 56, y: 1, id: 28, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 1.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 1, x: 40, y: 1, id: 20, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 1.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 7, x: 26, y: 1, id: 14, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 7, x: 25, y: 1, id: 15, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 7, x: 25, y: 1, id: 13, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 1, x: 24, y: 1, id: 12, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 1.0 }
            ]
        );

        let search_key =
            MatchKey { match_phrase: MatchPhrase::Range { start: 1, end: 3 }, lang_set: 0 };
        let records: Vec<_> =
            reader.get_matching(&search_key, &MatchOpts::default()).unwrap().collect();
        #[cfg_attr(rustfmt, rustfmt::skip)]
        assert_eq!(
            records,
            [
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 7, x: 58, y: 1, id: 30, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 7, x: 57, y: 1, id: 31, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 7, x: 57, y: 1, id: 29, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 7, x: 42, y: 1, id: 22, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 7, x: 41, y: 1, id: 23, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 7, x: 41, y: 1, id: 21, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 7, x: 26, y: 1, id: 14, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 7, x: 25, y: 1, id: 15, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 7, x: 25, y: 1, id: 13, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 1, x: 24, y: 1, id: 12, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 1.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 1, x: 56, y: 1, id: 28, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 1.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 1, x: 40, y: 1, id: 20, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 1.0 }
            ]
        );

        let search_key =
            MatchKey { match_phrase: MatchPhrase::Range { start: 1, end: 3 }, lang_set: 2 };
        let records: Vec<_> =
            reader.get_matching(&search_key, &MatchOpts::default()).unwrap().collect();
        #[cfg_attr(rustfmt, rustfmt::skip)]
        assert_eq!(
            records,
            [
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 26, y: 1, id: 14, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 25, y: 1, id: 15, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 25, y: 1, id: 13, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 1, x: 24, y: 1, id: 12, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 1.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 7, x: 42, y: 1, id: 22, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 7, x: 41, y: 1, id: 23, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 7, x: 41, y: 1, id: 21, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 7, x: 58, y: 1, id: 30, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 7, x: 57, y: 1, id: 31, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 7, x: 57, y: 1, id: 29, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 1, x: 56, y: 1, id: 28, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 1.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 1, x: 40, y: 1, id: 20, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 1.0 }
            ]
        );

        let search_key =
            MatchKey { match_phrase: MatchPhrase::Range { start: 1, end: 3 }, lang_set: 3 };
        let records: Vec<_> =
            reader.get_matching(&search_key, &MatchOpts::default()).unwrap().collect();
        #[cfg_attr(rustfmt, rustfmt::skip)]
        assert_eq!(
            records,
            [
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 58, y: 1, id: 30, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 57, y: 1, id: 31, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 57, y: 1, id: 29, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 42, y: 1, id: 22, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 41, y: 1, id: 23, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 41, y: 1, id: 21, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 26, y: 1, id: 14, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 25, y: 1, id: 15, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 25, y: 1, id: 13, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 1, x: 24, y: 1, id: 12, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 1.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 1, x: 56, y: 1, id: 28, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 1.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 1, x: 40, y: 1, id: 20, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 1.0 }
            ]
        );

        let search_key =
            MatchKey { match_phrase: MatchPhrase::Range { start: 1, end: 1 }, lang_set: 1 };
        let records: Vec<_> =
            reader.get_matching(&search_key, &MatchOpts::default()).unwrap().collect();
        assert_eq!(records, []);

        let search_key =
            MatchKey { match_phrase: MatchPhrase::Range { start: 3, end: 4 }, lang_set: 1 };
        let records: Vec<_> =
            reader.get_matching(&search_key, &MatchOpts::default()).unwrap().collect();
        assert_eq!(records, []);

        let search_key =
            MatchKey { match_phrase: MatchPhrase::Range { start: 1, end: 3 }, lang_set: 1 };
        let records: Vec<_> = reader
            .get_matching(
                &search_key,
                &MatchOpts { bbox: Some([26, 0, 41, 2]), ..MatchOpts::default() },
            )
            .unwrap()
            .collect();
        #[cfg_attr(rustfmt, rustfmt::skip)]
        assert_eq!(
            records,
            [
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 41, y: 1, id: 23, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 41, y: 1, id: 21, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 7.0 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 1, x: 40, y: 1, id: 20, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 1.0 },
                MatchEntry { grid_entry: GridEntry { relev: 0.96, score: 7, x: 26, y: 1, id: 14, source_phrase_hash: 0 }, matches_language: false, distance: 0.0, scoredist: 7.0 }
            ]
        );

        // Search just below existing records where z-order curve overlaps with bbox, but we do not
        // want records.
        let search_key =
            MatchKey { match_phrase: MatchPhrase::Range { start: 1, end: 3 }, lang_set: 1 };
        let records: Vec<_> = reader
            .get_matching(
                &search_key,
                &MatchOpts { bbox: Some([0, 2, 100, 2]), proximity: None, ..MatchOpts::default() },
            )
            .unwrap()
            .collect();
        assert_eq!(records.len(), 0, "no matching recods in bbox");

        // Search where neigther z-order curve or actual x,y overlap with bbox.
        let search_key =
            MatchKey { match_phrase: MatchPhrase::Range { start: 1, end: 3 }, lang_set: 1 };
        let records: Vec<_> = reader
            .get_matching(
                &search_key,
                &MatchOpts {
                    bbox: Some([100, 100, 100, 100]),
                    proximity: None,
                    ..MatchOpts::default()
                },
            )
            .unwrap()
            .collect();
        assert_eq!(records.len(), 0, "no matching recods in bbox");

        let search_key =
            MatchKey { match_phrase: MatchPhrase::Range { start: 1, end: 3 }, lang_set: 2 };
        let records: Vec<_> = reader
            .get_matching(
                &search_key,
                &MatchOpts {
                    bbox: None,
                    proximity: Some(Proximity { point: [26, 1], radius: 1000. }),
                    ..MatchOpts::default()
                },
            )
            .unwrap()
            .collect();
        #[cfg_attr(rustfmt, rustfmt::skip)]
        assert_eq!(
            records,
            [
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 26, y: 1, id: 14, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 15750.000000000002 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 25, y: 1, id: 15, source_phrase_hash: 0 }, matches_language: true, distance: 1.0, scoredist: 12600.000000000002 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 25, y: 1, id: 13, source_phrase_hash: 0 }, matches_language: true, distance: 1.0, scoredist: 12600.000000000002 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 1, x: 24, y: 1, id: 12, source_phrase_hash: 0 }, matches_language: true, distance: 2.0, scoredist: 913.3852617539986 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 41, y: 1, id: 23, source_phrase_hash: 0 }, matches_language: false, distance: 15.0, scoredist: 840.0000000000002 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 41, y: 1, id: 21, source_phrase_hash: 0 }, matches_language: false, distance: 15.0, scoredist: 840.0000000000002 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 42, y: 1, id: 22, source_phrase_hash: 0 }, matches_language: false, distance: 16.0, scoredist: 787.5000000000001 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 57, y: 1, id: 31, source_phrase_hash: 0 }, matches_language: false, distance: 31.0, scoredist: 406.4516129032259 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 57, y: 1, id: 29, source_phrase_hash: 0 }, matches_language: false, distance: 31.0, scoredist: 406.4516129032259 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 58, y: 1, id: 30, source_phrase_hash: 0 }, matches_language: false, distance: 32.0, scoredist: 393.75000000000006 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 1, x: 40, y: 1, id: 20, source_phrase_hash: 0 }, matches_language: false, distance: 14.0, scoredist: 130.48360882199978 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 1, x: 56, y: 1, id: 28, source_phrase_hash: 0 }, matches_language: false, distance: 30.0, scoredist: 60.89235078359991 }
            ]
        );

        let search_key =
            MatchKey { match_phrase: MatchPhrase::Range { start: 1, end: 3 }, lang_set: 2 };
        let records: Vec<_> = reader
            .get_matching(
                &search_key,
                &MatchOpts {
                    bbox: Some([10, 0, 41, 2]),
                    proximity: Some(Proximity { point: [26, 1], radius: 1000. }),
                    ..MatchOpts::default()
                },
            )
            .unwrap()
            .collect();
        #[cfg_attr(rustfmt, rustfmt::skip)]
        assert_eq!(
            records,
            [
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 26, y: 1, id: 14, source_phrase_hash: 0 }, matches_language: true, distance: 0.0, scoredist: 15750.000000000002 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 25, y: 1, id: 15, source_phrase_hash: 0 }, matches_language: true, distance: 1.0, scoredist: 12600.000000000002 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 25, y: 1, id: 13, source_phrase_hash: 0 }, matches_language: true, distance: 1.0, scoredist: 12600.000000000002 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 1, x: 24, y: 1, id: 12, source_phrase_hash: 0 }, matches_language: true, distance: 2.0, scoredist: 913.3852617539986 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 41, y: 1, id: 23, source_phrase_hash: 0 }, matches_language: false, distance: 15.0, scoredist: 840.0000000000002 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 7, x: 41, y: 1, id: 21, source_phrase_hash: 0 }, matches_language: false, distance: 15.0, scoredist: 840.0000000000002 },
                MatchEntry { grid_entry: GridEntry { relev: 1.0, score: 1, x: 40, y: 1, id: 20, source_phrase_hash: 0 }, matches_language: false, distance: 14.0, scoredist: 130.48360882199978 }
            ]
        );

        let listed_keys: Result<Vec<_>, _> = reader.keys().collect();
        let mut orig_keys = keys.clone();
        orig_keys.sort();
        orig_keys.dedup();
        assert_eq!(listed_keys.unwrap(), orig_keys);
    }

    #[test]
    fn prefix_test() {
        let directory: tempfile::TempDir = tempfile::tempdir().unwrap();
        let mut builder = GridStoreBuilder::new(directory.path()).unwrap();

        let alphabet = "abcdefghijklmnopqrstuvwxyz";
        let phrases: Vec<String> = alphabet
            .bytes()
            .flat_map(move |l1| {
                alphabet.bytes().flat_map(move |l2| {
                    alphabet.bytes().map(move |l3| String::from_utf8(vec![l1, l2, l3]).unwrap())
                })
            })
            .take(5000)
            .collect();

        // insert phrases
        for i in 0..=(phrases.len() as u32) {
            let key = GridKey { phrase_id: i, lang_set: 1 };
            let entries = vec![GridEntry {
                id: i,
                x: i as u16,
                y: 1,
                relev: 1.,
                score: 1,
                source_phrase_hash: 0,
            }];
            builder.insert(&key, entries).expect("Unable to insert record");
        }

        // calculate bins
        let mut bins: BTreeMap<u8, u32> = BTreeMap::new();
        for (i, phrase) in phrases.iter().enumerate() {
            // insert the first occurrence of every prefix
            bins.entry(phrase.bytes().next().unwrap()).or_insert(i as u32);
        }
        let mut boundaries: Vec<_> = bins.values().cloned().collect();
        boundaries.push(phrases.len() as u32);

        builder.load_bin_boundaries(boundaries);

        builder.finish().unwrap();

        let reader = GridStore::new(directory.path()).unwrap();

        let find_range = |prefix: &str| {
            let start = phrases
                .iter()
                .enumerate()
                .find(|(_, phrase)| phrase.starts_with(prefix))
                .unwrap()
                .0;
            let end = phrases
                .iter()
                .enumerate()
                .rev()
                .find(|(_, phrase)| phrase.starts_with(prefix))
                .unwrap()
                .0
                + 1;
            (start as u32, end as u32)
        };

        let starts_with_b = find_range("b");
        let starts_with_bc = find_range("bc");

        // query that we expect to use the pre-cached ranges
        let search_key = MatchKey {
            match_phrase: MatchPhrase::Range { start: starts_with_b.0, end: starts_with_b.1 },
            lang_set: 1,
        };
        let mut records: Vec<_> =
            reader.get_matching(&search_key, &MatchOpts::default()).unwrap().collect();
        records.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut expected = Vec::new();
        for i in starts_with_b.0..starts_with_b.1 {
            expected.push(MatchEntry {
                grid_entry: GridEntry {
                    relev: 1.0,
                    score: 1,
                    x: i as u16,
                    y: 1,
                    id: i,
                    source_phrase_hash: 0,
                },
                matches_language: true,
                distance: 0.0,
                scoredist: 1.0,
            })
        }
        assert_eq!(records, expected);

        // query that we expect not to use the precached ranges
        let search_key = MatchKey {
            match_phrase: MatchPhrase::Range { start: starts_with_bc.0, end: starts_with_bc.1 },
            lang_set: 1,
        };
        let mut records: Vec<_> =
            reader.get_matching(&search_key, &MatchOpts::default()).unwrap().collect();
        records.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut expected = Vec::new();
        for i in starts_with_bc.0..starts_with_bc.1 {
            expected.push(MatchEntry {
                grid_entry: GridEntry {
                    relev: 1.0,
                    score: 1,
                    x: i as u16,
                    y: 1,
                    id: i,
                    source_phrase_hash: 0,
                },
                matches_language: true,
                distance: 0.0,
                scoredist: 1.0,
            })
        }
        assert_eq!(records, expected);
    }
}
