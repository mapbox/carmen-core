mod common;
mod builder;
mod store;
mod gridstore_generated;

pub use common::*;
pub use builder::*;
pub use store::*;

#[test]
fn combined_test() {
    let directory: tempfile::TempDir = tempfile::tempdir().unwrap();
    let mut builder = GridStoreBuilder::new(directory.path()).unwrap();

    let key = GridKey { phrase_id: 1, lang_set: 1 };

    let mut entries = vec![
        GridEntry { id: 2, x: 2, y: 2, relev: 0.8, score: 3, source_phrase_hash: 0 },
        GridEntry { id: 3, x: 3, y: 3, relev: 1., score: 1, source_phrase_hash: 1 },
        GridEntry { id: 1, x: 1, y: 1, relev: 1., score: 7, source_phrase_hash: 2 }
    ];
    builder.insert(&key, &entries).expect("Unable to insert record");

    builder.finish().unwrap();

    let reader = GridStore::new(directory.path()).unwrap();
    let record: Vec<_> = reader.get(&key).unwrap().unwrap().collect();

    entries.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert_eq!(record, entries, "identical entries come out as went in, in reverse-sorted order");

    {
        let key = GridKey { phrase_id: 2, lang_set: 1 };
        let record = reader.get(&key).expect("Failed to get key");
        assert!(record.is_none(), "Retrieved no results");
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
        GridEntry { id: 1, x: 1, y: 1, relev: 0.4, score: 1, source_phrase_hash: 3 }
    ];
    builder.insert(&key, &entries).expect("Unable to insert record");

    builder.finish().unwrap();

    let reader = GridStore::new(directory.path()).unwrap();
    let record: Vec<_> = reader.get(&key).unwrap().unwrap().collect();

    entries.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert_eq!(record, entries, "identical entries come out as went in, in reverse-sorted order");
}

#[test]
fn cover_test() {
    let directory: tempfile::TempDir = tempfile::tempdir().unwrap();
    let mut builder = GridStoreBuilder::new(directory.path()).unwrap();

    let key = GridKey { phrase_id: 1, lang_set: 1 };

    let entries = vec![
        GridEntry { id: 1, x: 1, y: 1, relev: 1., score: 1, source_phrase_hash: 0 },
        GridEntry { id: 1, x: 1, y: 2, relev: 1., score: 1, source_phrase_hash: 0 },
        GridEntry { id: 1, x: 2, y: 1, relev: 1., score: 1, source_phrase_hash: 0 }
    ];
    builder.insert(&key, &entries).expect("Unable to insert record");

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
    builder.insert(&key, &entries).expect("Unable to insert record");

    builder.finish().unwrap();

    let reader = GridStore::new(directory.path()).unwrap();
    let record: Vec<_> = reader.get(&key).unwrap().unwrap().collect();

    entries.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert_eq!(record, entries, "identical entries come out as went in, in reverse-sorted order");
}
