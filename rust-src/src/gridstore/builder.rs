use std::collections::{btree_map::Entry, BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use failure::{Error, Fail};
use itertools::Itertools;
use morton::interleave_morton;
use rocksdb::{DBCompressionType, Options, DB};

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

fn copy_entries(source_entry: &BuilderEntry, destination_entry: &mut BuilderEntry) -> () {
    for (rs, values) in source_entry.iter() {
        let rs_entry = destination_entry.entry(*rs).or_insert_with(|| BTreeMap::new());
        for (zcoord, values) in values.iter() {
            let zcoord_entry = rs_entry.entry(*zcoord).or_insert_with(|| Vec::new());
            zcoord_entry.extend(values.iter());
        }
    }
}

fn get_fb_value(value: &mut BuilderEntry) -> Result<Vec<u8>, Error> {
    let mut fb_builder = flatbuffers::FlatBufferBuilder::new();
    let mut rses: Vec<_> = Vec::new();

    let mut id_lists: HashMap<_, u32> = HashMap::new();
    let mut id_idx = 0;

    for (rs, coord_group) in value.iter_mut().rev() {
        let mut coords: Vec<_> = Vec::new();
        for (coord, ids) in coord_group.iter_mut().rev() {
            // reverse sort
            ids.sort_by(|a, b| b.cmp(a));
            ids.dedup();

            let idx = id_lists.entry(ids).or_insert_with(|| {
                let new_id = id_idx;
                id_idx += 1;
                new_id
            });

            //let fb_ids = fb_builder.create_vector(&ids);
            let fb_coord = Coord::new(*coord, *idx);
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

    let mut id_lists: Vec<_> = id_lists.into_iter().collect();
    id_lists.sort_by_key(|e| e.1);
    let fb_id_lists: Vec<_> = id_lists.into_iter().map(|(list, _)| {
        let list = fb_builder.create_vector(list);
        IdList::create(&mut fb_builder, &IdListArgs { ids: Some(list) })
    }).collect();
    let fb_id_lists = fb_builder.create_vector(&fb_id_lists);

    let record =
        PhraseRecord::create(&mut fb_builder, &PhraseRecordArgs { relev_scores: Some(fb_rses), id_lists: Some(fb_id_lists) });
    fb_builder.finish(record, None);
    Ok(fb_builder.finished_data().to_vec())
}

impl GridStoreBuilder {
    /// Makes a new GridStoreBuilder with a particular filename.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        Ok(GridStoreBuilder { path: path.as_ref().to_owned(), data: BTreeMap::new() })
    }

    /// Inserts a new GridStore entry with the given values.
    pub fn insert(&mut self, key: &GridKey, values: &[GridEntry]) -> Result<(), Error> {
        let mut to_insert = BuilderEntry::new();
        extend_entries(&mut to_insert, values);
        self.data.insert(key.to_owned(), to_insert);
        Ok(())
    }

    ///  Appends a values to and existing GridStore entry.
    pub fn append(&mut self, key: &GridKey, values: &[GridEntry]) -> Result<(), Error> {
        let mut to_append = self.data.entry(key.to_owned()).or_insert_with(|| BuilderEntry::new());
        extend_entries(&mut to_append, values);
        Ok(())
    }

    /// In situations under which data has been inserted using temporary phrase IDs, renumber
    /// the data in the index to use final phrase IDs, given a temporary-to-final-ID mapping
    pub fn renumber(&mut self, tmp_phrase_ids_to_ids: &[u32]) -> Result<(), Error> {
        let mut old_data: BTreeMap<GridKey, BuilderEntry> = BTreeMap::new();
        std::mem::swap(&mut old_data, &mut self.data);

        for (key, value) in old_data.into_iter() {
            let new_phrase_id = tmp_phrase_ids_to_ids
                .get(key.phrase_id as usize)
                .ok_or_else(|| BuildError::OutOfBoundsRenumberEntry { tmp_id: key.phrase_id })?;
            let new_key = GridKey { phrase_id: *new_phrase_id, ..key };
            match self.data.entry(new_key) {
                Entry::Vacant(v) => {
                    v.insert(value);
                }
                Entry::Occupied(_) => {
                    return Err(Error::from(BuildError::DuplicateRenumberEntry {
                        target_id: *new_phrase_id,
                    }))
                }
            };
        }
        Ok(())
    }

    /// Writes data to disk.
    pub fn finish(mut self) -> Result<(), Error> {
        let mut opts = Options::default();
        opts.set_disable_auto_compactions(true);
        opts.set_compression_type(DBCompressionType::Lz4hc);
        opts.create_if_missing(true);

        let db = DB::open(&opts, &self.path)?;
        let mut db_key: Vec<u8> = Vec::with_capacity(MAX_KEY_LENGTH);

        let grouped = self.data.iter_mut().group_by(|(key, _value)| (key.phrase_id >> 10) << 10);

        for (group_id, group_value) in grouped.into_iter() {
            let mut lang_set_map: HashMap<u128, BuilderEntry> = HashMap::new();

            for (grid_key, value) in group_value {
                // figure out the key
                db_key.clear();
                // type marker is 0 -- regular entry
                grid_key.write_to(0, &mut db_key)?;

                let mut grouped_entry =
                    lang_set_map.entry(grid_key.lang_set).or_insert_with(|| BuilderEntry::new());
                copy_entries(&value, &mut grouped_entry);
                // figure out the value
                let db_data = get_fb_value(value)?;
                db.put(&db_key, &db_data)?;
            }
            for (lang_set, mut builder_entry) in lang_set_map.into_iter() {
                db_key.clear();
                let group_key = GridKey { phrase_id: group_id, lang_set };
                group_key.write_to(1, &mut db_key)?;
                let grouped_db_data = get_fb_value(&mut builder_entry)?;
                db.put(&db_key, &grouped_db_data)?;
            }
        }

        db.compact_range(None::<&[u8]>, None::<&[u8]>);
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

#[derive(Debug, Fail)]
enum BuildError {
    #[fail(display = "duplicate rename entry: {}", target_id)]
    DuplicateRenumberEntry { target_id: u32 },
    #[fail(display = "out of bounds: {}", tmp_id)]
    OutOfBoundsRenumberEntry { tmp_id: u32 },
}
