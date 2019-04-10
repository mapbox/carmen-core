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
        GridEntry {
            id: 2,
            x: 2,
            y: 2,
            relev: 0.8,
            score: 3,
            source_phrase_hash: 0
        },
        GridEntry {
            id: 3,
            x: 3,
            y: 3,
            relev: 1.,
            score: 1,
            source_phrase_hash: 1
        },
        GridEntry {
            id: 1,
            x: 1,
            y: 1,
            relev: 1.,
            score: 7,
            source_phrase_hash: 2
        }
    ];
    builder.insert(&key, &entries).expect("Unable to insert record");

    builder.finish().unwrap();

    let reader = GridStore::new(directory.path()).unwrap();
    let record: Vec<_> = reader.get(&key).unwrap().unwrap().collect();

    entries.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert_eq!(record, entries, "identical entries come out as went in, in reverse-sorted order");
}