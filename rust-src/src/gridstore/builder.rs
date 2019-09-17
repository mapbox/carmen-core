use std::collections::hash_map::Entry as HmEntry;
use std::collections::{btree_map::Entry, BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use failure::{Error, Fail};
use itertools::Itertools;
use morton::interleave_morton;
use rocksdb::{DBCompressionType, Options, DB};
use smallvec::{smallvec, SmallVec};

use crate::gridstore::common::*;
use crate::gridstore::gridstore_format;

type BuilderEntry = HashMap<u8, HashMap<u32, SmallVec<[u32; 4]>>>;

pub struct GridStoreBuilder {
    path: PathBuf,
    data: BTreeMap<GridKey, BuilderEntry>,
    bin_boundaries: Vec<u32>,
}

/// Extends a BuildEntry with the given values.
fn extend_entries(builder_entry: &mut BuilderEntry, values: Vec<GridEntry>) -> () {
    for (rs, rs_values) in somewhat_eager_groupby(values.into_iter(), |value| {
        (relev_float_to_int(value.relev) << 4) | value.score
    }) {
        let rs_entry =
            builder_entry.entry(rs).or_insert_with(|| HashMap::with_capacity(rs_values.len()));
        for (zcoord, zc_values) in
            &rs_values.into_iter().group_by(|value| interleave_morton(value.x, value.y))
        {
            let id_phrases =
                zc_values.map(|value| (value.id << 8) | (value.source_phrase_hash as u32));
            match rs_entry.entry(zcoord) {
                HmEntry::Vacant(e) => {
                    e.insert(id_phrases.collect());
                }
                HmEntry::Occupied(mut e) => {
                    e.get_mut().extend(id_phrases);
                }
            }
        }
    }
}

fn copy_entries(source_entry: &BuilderEntry, destination_entry: &mut BuilderEntry) -> () {
    for (rs, values) in source_entry.iter() {
        let rs_entry = destination_entry.entry(*rs).or_insert_with(|| HashMap::new());
        for (zcoord, values) in values.iter() {
            let zcoord_entry = rs_entry.entry(*zcoord).or_insert_with(|| SmallVec::new());
            zcoord_entry.extend(values.iter().cloned());
        }
    }
}

fn get_encoded_value(value: BuilderEntry) -> Result<Vec<u8>, Error> {
    let mut builder = gridstore_format::Writer::new();

    let mut items: Vec<(_, _)> = value.into_iter().collect();
    items.sort_by(|a, b| b.0.cmp(&a.0));

    let mut rses: Vec<_> = Vec::with_capacity(items.len());

    let mut id_lists: HashMap<_, gridstore_format::FixedVecOffset<u32>> = HashMap::new();

    for (rs, coord_group) in items.into_iter() {
        let mut inner_items: Vec<(_, _)> = coord_group.into_iter().collect();
        inner_items.sort_by(|a, b| b.0.cmp(&a.0));

        let mut coords: Vec<_> = Vec::with_capacity(inner_items.len());

        for (coord, mut ids) in inner_items.into_iter() {
            // reverse sort
            ids.sort_by(|a, b| b.cmp(a));
            ids.dedup();

            let encoded_ids =
                id_lists.entry(ids.clone()).or_insert_with(|| builder.write_fixed_vec(&ids));

            let encoded_coord = gridstore_format::Coord { coord, ids: encoded_ids.clone() };
            coords.push(encoded_coord);
        }
        let encoded_coords = builder.write_uniform_vec(&coords);
        let encoded_rs = gridstore_format::RelevScore { relev_score: rs, coords: encoded_coords };
        rses.push(encoded_rs);
    }

    let encoded_rses = builder.write_var_vec(&rses);

    let record = gridstore_format::PhraseRecord { relev_scores: encoded_rses };
    builder.write_fixed_scalar(record);

    Ok(builder.finish())
}

impl GridStoreBuilder {
    /// Makes a new GridStoreBuilder with a particular filename.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        Ok(GridStoreBuilder {
            path: path.as_ref().to_owned(),
            data: BTreeMap::new(),
            bin_boundaries: Vec::new(),
        })
    }

    /// Inserts a new GridStore entry with the given values.
    pub fn insert(&mut self, key: &GridKey, values: Vec<GridEntry>) -> Result<(), Error> {
        let mut to_insert = BuilderEntry::new();
        extend_entries(&mut to_insert, values);
        self.data.insert(key.to_owned(), to_insert);
        Ok(())
    }

    ///  Appends a values to and existing GridStore entry.
    pub fn append(&mut self, key: &GridKey, values: Vec<GridEntry>) -> Result<(), Error> {
        let mut to_append = self.data.entry(key.to_owned()).or_insert_with(|| BuilderEntry::new());
        extend_entries(&mut to_append, values);
        Ok(())
    }

    pub fn compact_append(
        &mut self,
        key: &GridKey,
        relev: f64,
        score: u8,
        id: u32,
        source_phrase_hash: u8,
        coords: &[(u16, u16)],
    ) {
        let to_append =
            self.data.entry(key.to_owned()).or_insert_with(|| BuilderEntry::with_capacity(1));

        let relev_score = (relev_float_to_int(relev) << 4) | score;
        let id_hash = smallvec![(id << 8) | (source_phrase_hash as u32)];
        let rs_entry =
            to_append.entry(relev_score).or_insert_with(|| HashMap::with_capacity(coords.len()));
        for pair in coords {
            let zcoord = interleave_morton(pair.0, pair.1);
            match rs_entry.entry(zcoord) {
                HmEntry::Vacant(e) => {
                    e.insert(id_hash.clone());
                }
                HmEntry::Occupied(mut e) => {
                    e.get_mut().extend_from_slice(&id_hash);
                }
            }
        }
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

    pub fn load_bin_boundaries(&mut self, bin_boundaries: Vec<u32>) -> Result<(), Error> {
        self.bin_boundaries = bin_boundaries;
        Ok(())
    }

    /// Writes data to disk.
    pub fn finish(self) -> Result<(), Error> {
        let mut opts = Options::default();
        opts.set_disable_auto_compactions(true);
        opts.set_compression_type(DBCompressionType::Lz4);
        opts.create_if_missing(true);

        let db = DB::open(&opts, &self.path)?;
        let mut db_key: Vec<u8> = Vec::with_capacity(MAX_KEY_LENGTH);

        let mut bin_seq = self.bin_boundaries.iter().cloned().peekable();
        let mut current_bin = None;
        let mut next_boundary = 0u32;
        let grouped = somewhat_eager_groupby(self.data.into_iter(), |(key, _value)| {
            while key.phrase_id >= next_boundary {
                current_bin = bin_seq.next();
                next_boundary = *(bin_seq.peek().unwrap_or(&std::u32::MAX));
            }

            current_bin
        });

        for (group_id, group_value) in grouped {
            let mut lang_set_map: HashMap<u128, BuilderEntry> = HashMap::new();

            for (grid_key, value) in group_value.into_iter() {
                // figure out the key
                db_key.clear();
                // type marker is 0 -- regular entry
                grid_key.write_to(0, &mut db_key)?;

                let mut grouped_entry =
                    lang_set_map.entry(grid_key.lang_set).or_insert_with(|| BuilderEntry::new());
                copy_entries(&value, &mut grouped_entry);
                // figure out the value
                let db_data = get_encoded_value(value)?;
                db.put(&db_key, &db_data)?;
            }
            if let Some(group_id) = group_id {
                for (lang_set, builder_entry) in lang_set_map.into_iter() {
                    db_key.clear();
                    let group_key = GridKey { phrase_id: group_id, lang_set };
                    group_key.write_to(1, &mut db_key)?;
                    let grouped_db_data = get_encoded_value(builder_entry)?;
                    db.put(&db_key, &grouped_db_data)?;
                }
            }
        }

        // bake the prefix boundaries
        let mut encoded_boundaries: Vec<u8> = Vec::with_capacity(self.bin_boundaries.len() * 4);
        for boundary in self.bin_boundaries {
            encoded_boundaries.extend_from_slice(&boundary.to_le_bytes());
        }
        db.put("~BOUNDS", &encoded_boundaries)?;

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
        vec![GridEntry { id: 1, x: 1, y: 1, relev: 1., score: 7, source_phrase_hash: 2 }],
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
            vec![
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
            vec![GridEntry { id: 2, x: 2, y: 2, relev: 0.8, score: 3, source_phrase_hash: 0 }],
        )
        .expect("Unable to insert record");

    builder
        .append(
            &key,
            vec![
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
