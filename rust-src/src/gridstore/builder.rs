use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};

use itertools::Itertools;
use morton::interleave_morton;
use rocksdb::DB;

use crate::gridstore::common::*;
use crate::gridstore::gridstore_generated::*;

type BuilderEntry = BTreeMap<u8, BTreeMap<u32, Vec<u32>>>;

pub struct GridStoreBuilder {
    path: PathBuf,
    data: BTreeMap<GridKey, BuilderEntry>,
}

/// Extends a BuildEntry with the given values.
fn extend_entries(builder_entry: &mut BuilderEntry, values: &[GridEntry]) -> () {
    for (rs, values) in
        &values.into_iter().group_by(|value| (relev_float_to_int(value.relev) << 4) | value.score)
    {
        let rs_entry = builder_entry.entry(rs).or_insert_with(|| BTreeMap::new());
        for (zcoord, values) in
            &values.into_iter().group_by(|value| interleave_morton(value.x, value.y))
        {
            let zcoord_entry = rs_entry.entry(zcoord).or_insert_with(|| Vec::new());
            for value in values {
                let id_phrase: u32 = (value.id << 8) | (value.source_phrase_hash as u32);
                zcoord_entry.push(id_phrase);
            }
        }
    }
}

impl GridStoreBuilder {
    /// Makes a new GridStoreBuilder with a particular filename.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        Ok(GridStoreBuilder { path: path.as_ref().to_owned(), data: BTreeMap::new() })
    }

    /// Inserts a new GridStore entry with the given values.
    pub fn insert(&mut self, key: &GridKey, values: &[GridEntry]) -> Result<(), Box<dyn Error>> {
        let mut to_insert = BuilderEntry::new();
        extend_entries(&mut to_insert, values);
        self.data.insert(key.to_owned(), to_insert);
        Ok(())
    }

    ///  Appends a values to and existing GridStore entry.
    pub fn append(&mut self, key: &GridKey, values: &[GridEntry]) -> Result<(), Box<dyn Error>> {
        let mut to_append = self.data.entry(key.to_owned()).or_insert_with(|| BuilderEntry::new());
        extend_entries(&mut to_append, values);
        Ok(())
    }

    /// [wip] Writes data to disk.
    pub fn finish(mut self) -> Result<(), Box<Error>> {
        let db = DB::open_default(&self.path)?;
        let mut db_key: Vec<u8> = Vec::with_capacity(MAX_KEY_LENGTH);
        for (grid_key, value) in self.data.iter_mut() {
            // figure out the key
            db_key.clear();
            // type marker is 0 -- regular entry
            grid_key.write_to(0, &mut db_key)?;

            // figure out the value
            let mut fb_builder = flatbuffers::FlatBufferBuilder::new();
            let mut rses: Vec<_> = Vec::new();
            for (rs, coord_group) in value.iter_mut().rev() {
                let mut coords: Vec<_> = Vec::new();
                for (coord, ids) in coord_group.iter_mut().rev() {
                    // reverse sort
                    ids.sort_by(|a, b| b.cmp(a));
                    ids.dedup();

                    let fb_ids = fb_builder.create_vector(&ids);
                    let fb_coord = Coord::create(
                        &mut fb_builder,
                        &CoordArgs { coord: *coord, ids: Some(fb_ids) },
                    );
                    coords.push(fb_coord);
                }
                let fb_coords = fb_builder.create_vector(&coords);
                let fb_rs = RelevScore::create(
                    &mut fb_builder,
                    &RelevScoreArgs { relev_score: *rs, coords: Some(fb_coords) },
                );
                rses.push(fb_rs);
            }
            let fb_rses = fb_builder.create_vector(&rses);
            let record = PhraseRecord::create(
                &mut fb_builder,
                &PhraseRecordArgs { relev_scores: Some(fb_rses) },
            );
            fb_builder.finish(record, None);

            let db_data = fb_builder.finished_data();

            db.put(&db_key, &db_data)?;
        }
        drop(db);
        Ok(())
    }
}

#[cfg(test)]
use tempfile;

#[test]
fn extend_entry_test() {
    let mut entry = BuilderEntry::new();

    extend_entries(
        &mut entry,
        &vec![GridEntry { id: 1, x: 1, y: 1, relev: 1., score: 7, source_phrase_hash: 2 }],
    );

    // relev 3 (0011) with score 7 (0111) -> 55
    let grids = entry.get(&55);
    assert_ne!(grids, None, "Retrieve grids based on relev and score");

    // x:1, y:1 -> z-order 3
    let vals = grids.unwrap().get(&3);
    assert_ne!(vals, None, "Retrieve entries based on z-order");
    // id 1 (1 << 8 == 256) with phrase 2 => 258
    assert_eq!(vals.unwrap()[0], 258, "TODO");
}

#[test]
fn insert_test() {
    let directory: tempfile::TempDir = tempfile::tempdir().unwrap();
    let mut builder = GridStoreBuilder::new(directory.path()).unwrap();

    let key = GridKey { phrase_id: 1, lang_set: 1 };

    builder
        .insert(
            &key,
            &vec![
                GridEntry { id: 2, x: 2, y: 2, relev: 0.8, score: 3, source_phrase_hash: 0 },
                GridEntry { id: 3, x: 3, y: 3, relev: 1., score: 1, source_phrase_hash: 1 },
                GridEntry { id: 1, x: 1, y: 1, relev: 1., score: 7, source_phrase_hash: 2 },
            ],
        )
        .expect("Unable to insert record");

    assert_ne!(builder.path.to_str(), None);
    assert_eq!(builder.data.len(), 1, "Gridstore has one entry");

    let entry = builder.data.get(&key);
    assert_ne!(entry, None);
    assert_eq!(entry.unwrap().len(), 3, "Entry contains three grids");

    builder.finish().unwrap();
}

#[test]
fn append_test() {
    let directory: tempfile::TempDir = tempfile::tempdir().unwrap();
    let mut builder = GridStoreBuilder::new(directory.path()).unwrap();

    let key = GridKey { phrase_id: 1, lang_set: 1 };

    builder
        .insert(
            &key,
            &vec![GridEntry { id: 2, x: 2, y: 2, relev: 0.8, score: 3, source_phrase_hash: 0 }],
        )
        .expect("Unable to insert record");

    builder
        .append(
            &key,
            &vec![
                GridEntry { id: 3, x: 3, y: 3, relev: 1., score: 1, source_phrase_hash: 1 },
                GridEntry { id: 1, x: 1, y: 1, relev: 1., score: 7, source_phrase_hash: 2 },
            ],
        )
        .expect("Unable to append grids");

    assert_ne!(builder.path.to_str(), None);
    assert_eq!(builder.data.len(), 1, "Gridstore has one entry");

    let entry = builder.data.get(&key);
    assert_ne!(entry, None);
    assert_eq!(entry.unwrap().len(), 3, "Entry contains three grids");

    builder.finish().unwrap();
}
