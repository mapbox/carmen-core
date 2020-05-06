use std::cmp::Ordering;
use std::collections::HashSet;
use std::convert::TryInto;
use std::path::{Path, PathBuf};

use byteorder::{BigEndian, ReadBytesExt};
use failure::Error;
use itertools::Itertools;
use min_max_heap::MinMaxHeap;
use morton::deinterleave_morton;
use ordered_float::OrderedFloat;
use rocksdb::{Direction, IteratorMode, Options, DB};
use serde::Serialize;

use crate::gridstore::common::*;
use crate::gridstore::gridstore_format;
use crate::gridstore::spatial;

#[derive(Debug, Serialize)]
pub struct GridStore {
    #[serde(skip_serializing)]
    db: DB,
    #[serde(skip_serializing)]
    pub bin_boundaries: HashSet<u32>,
    pub path: PathBuf,
    // options:
    pub zoom: u16,
    pub type_id: u16,
    pub coalesce_radius: f64,
    #[serde(default)]
    pub might_be_slow: bool,
    pub bbox: [u16; 4],
}

#[inline]
fn decode_value<T: AsRef<[u8]>>(value: T) -> impl Iterator<Item = GridEntry> {
    let record_ref = {
        let value_ref: &[u8] = value.as_ref();
        // this is pretty sketch: we're opting out of compiler lifetime protection
        // for this reference. This usage should be safe though, because we'll move the
        // reference and the underlying owned object around together as a unit (the
        // tuple below) so that when we pull the reference into the inner closures,
        // we'll drag the owned object along, and won't drop it until the whole
        // nest of closures is deleted
        let static_ref: &'static [u8] = unsafe { std::mem::transmute(value_ref) };
        (value, static_ref)
    };
    let reader = gridstore_format::Reader::new(record_ref.1);
    let record = { gridstore_format::read_phrase_record_from(&reader) };

    let iter = gridstore_format::read_var_vec_raw(record_ref.1, record.relev_scores)
        .into_iter()
        .flat_map(move |rs_obj| {
            // grab a reference to the outer object to make sure it doesn't get freed
            let _ref = &record_ref;

            let relev_score = rs_obj.relev_score;
            let relev = relev_int_to_float(relev_score >> 4);
            // mask for the least significant four bits
            let score = relev_score & 15;

            let nested_ref = record_ref.1;
            gridstore_format::read_uniform_vec_raw(record_ref.1, rs_obj.coords)
                .into_iter()
                .flat_map(move |coords_obj| {
                    let (x, y) = deinterleave_morton(coords_obj.coord);

                    gridstore_format::read_fixed_vec_raw(nested_ref, coords_obj.ids)
                        .into_iter()
                        .map(move |id_comp| {
                            let id = id_comp >> 8;
                            let source_phrase_hash = (id_comp & 255) as u8;
                            GridEntry { relev, score, x, y, id, source_phrase_hash }
                        })
                })
        });
    iter
}

#[inline]
fn decode_matching_value<T: AsRef<[u8]>>(
    value: T,
    match_opts: &MatchOpts,
    matches_language: bool,
    coalesce_radius: f64,
) -> impl Iterator<Item = MatchEntry> {
    let match_opts = match_opts.clone();

    let record_ref = {
        let value_ref: &[u8] = value.as_ref();
        // this is pretty sketch: we're opting out of compiler lifetime protection
        // for this reference. This usage should be safe though, because we'll move the
        // reference and the underlying owned object around together as a unit (the
        // tuple below) so that when we pull the reference into the inner closures,
        // we'll drag the owned object along, and won't drop it until the whole
        // nest of closures is deleted
        let static_ref: &'static [u8] = unsafe { std::mem::transmute(value_ref) };
        (value, static_ref)
    };
    let reader = gridstore_format::Reader::new(record_ref.1);
    let record = { gridstore_format::read_phrase_record_from(&reader) };

    let relevs = gridstore_format::read_var_vec_raw(record_ref.1, record.relev_scores)
        .into_iter()
        .map(|rs_obj| {
            let relev_score = rs_obj.relev_score;
            let relev = relev_int_to_float(relev_score >> 4);
            // mask for the least significant four bits
            let score = relev_score & 15;
            (relev, score, rs_obj)
        });

    let iter = somewhat_eager_groupby(relevs.into_iter(), |(relev, _, _)| *relev)
        .into_iter()
        .flat_map(move |(relev, score_groups)| {
            // grab a reference to the outer object to make sure it doesn't get freed
            let _ref = &record_ref;

            let match_opts = match_opts.clone();
            let nested_ref = _ref.1;
            let coords_per_score = score_groups.into_iter().map(move |(_, score, rs_obj)| {
                let coords_vec = gridstore_format::read_uniform_vec_raw(nested_ref, rs_obj.coords);
                let coords =
                    match &match_opts {
                        MatchOpts { bbox: None, proximity: None, .. } => {
                            Some(Box::new(coords_vec.into_iter())
                                as Box<dyn Iterator<Item = gridstore_format::Coord>>)
                        }
                        MatchOpts { bbox: Some(bbox), proximity: None, .. } => {
                            match spatial::bbox_filter(coords_vec, *bbox) {
                                Some(v) => Some(Box::new(v)
                                    as Box<dyn Iterator<Item = gridstore_format::Coord>>),
                                None => None,
                            }
                        }
                        MatchOpts { bbox: None, proximity: Some(prox_pt), .. } => {
                            match spatial::proximity(coords_vec, *prox_pt) {
                                Some(v) => Some(Box::new(v)
                                    as Box<dyn Iterator<Item = gridstore_format::Coord>>),
                                None => None,
                            }
                        }
                        MatchOpts { bbox: Some(bbox), proximity: Some(prox_pt), .. } => {
                            match spatial::bbox_proximity_filter(coords_vec, *bbox, *prox_pt) {
                                Some(v) => Some(Box::new(v)
                                    as Box<dyn Iterator<Item = gridstore_format::Coord>>),
                                None => None,
                            }
                        }
                    };

                let coords = coords.unwrap_or_else(|| {
                    Box::new((Option::<gridstore_format::Coord>::None).into_iter())
                        as Box<dyn Iterator<Item = gridstore_format::Coord>>
                });
                let match_opts = match_opts.clone();
                coords.map(move |coords_obj| {
                    let (x, y) = deinterleave_morton(coords_obj.coord);

                    let (distance, within_radius, scoredist) = match &match_opts {
                        MatchOpts { proximity: Some(prox_pt), zoom, .. } => {
                            let distance = spatial::tile_dist(prox_pt[0], prox_pt[1], x, y);
                            (
                                distance,
                                // The proximity radius calculation is also done in scoredist
                                // There could be an opportunity to optimize by doing it once
                                distance <= spatial::proximity_radius(*zoom, coalesce_radius),
                                spatial::scoredist(*zoom, distance, score, coalesce_radius),
                            )
                        }
                        _ => (0f64, false, score as f64),
                    };
                    (distance, within_radius, score, scoredist, x, y, coords_obj)
                })
            });

            let all_coords = coords_per_score.kmerge_by(
            |
                (_distance1, _within_radius1, _score1, scoredist1, _x1, _y1, _coords_obj1),
                (_distance2, _within_radius2, _score2, scoredist2, _x2, _y2, _coords_obj2)
            | {
                scoredist1.partial_cmp(scoredist2).unwrap() == Ordering::Greater
            });

            let nested_ref = record_ref.1;
            all_coords.flat_map(
                move |(distance, within_radius, score, scoredist, x, y, coords_obj)| {
                    let ids = gridstore_format::read_fixed_vec_raw(nested_ref, coords_obj.ids);

                    ids.into_iter().map(move |id_comp| {
                        let id = id_comp >> 8;
                        let source_phrase_hash = (id_comp & 255) as u8;
                        MatchEntry {
                            grid_entry: GridEntry {
                                relev: relev
                                    * (if matches_language || within_radius {
                                        1f64
                                    } else {
                                        0.96f64
                                    }),
                                score,
                                x,
                                y,
                                id,
                                source_phrase_hash,
                            },
                            matches_language,
                            distance,
                            scoredist,
                        }
                    })
                },
            )
        });
    iter
}

struct QueueElement<T: Iterator<Item = MatchEntry>> {
    next_entry: MatchEntry,
    entry_iter: T,
}

impl<T: Iterator<Item = MatchEntry>> QueueElement<T> {
    fn sort_key(&self) -> (OrderedFloat<f64>, OrderedFloat<f64>, bool, u16, u16, u32) {
        (
            OrderedFloat(self.next_entry.grid_entry.relev),
            OrderedFloat(self.next_entry.scoredist),
            self.next_entry.matches_language,
            self.next_entry.grid_entry.x,
            self.next_entry.grid_entry.y,
            self.next_entry.grid_entry.id,
        )
    }
}

impl<T: Iterator<Item = MatchEntry>> Ord for QueueElement<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sort_key().cmp(&other.sort_key())
    }
}

impl<T: Iterator<Item = MatchEntry>> PartialOrd for QueueElement<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Iterator<Item = MatchEntry>> PartialEq for QueueElement<T> {
    fn eq(&self, other: &Self) -> bool {
        self.sort_key() == other.sort_key()
    }
}

impl<T: Iterator<Item = MatchEntry>> Eq for QueueElement<T> {}

impl GridStore {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        GridStore::new_with_options(path, 6, 0, 0.0, [0, 0, 63, 63])
    }

    pub fn mark_slow_indexes(zoom: u16) -> bool {
        return if zoom == 14 { true } else { false };
    }

    pub fn new_with_options<P: AsRef<Path>>(
        path: P,
        zoom: u16,
        type_id: u16,
        coalesce_radius: f64,
        bbox: [u16; 4],
    ) -> Result<Self, Error> {
        let path = path.as_ref().to_owned();
        let mut opts = Options::default();
        opts.set_read_only(true);
        opts.set_allow_mmap_reads(true);
        let db = DB::open(&opts, &path)?;

        let bin_boundaries: HashSet<u32> = match db.get("~BOUNDS")? {
            Some(entry) => {
                let encoded_boundaries: &[u8] = entry.as_ref();
                encoded_boundaries
                    .chunks(4)
                    .filter_map(|chunk| {
                        if chunk.len() == 4 {
                            Some(u32::from_le_bytes(chunk.try_into().unwrap()))
                        } else {
                            None
                        }
                    })
                    .collect()
            }
            None => HashSet::new(),
        };

        let might_be_slow = GridStore::mark_slow_indexes(zoom);
        Ok(GridStore { db, path, bin_boundaries, zoom, type_id, coalesce_radius, bbox, might_be_slow })
    }

    #[inline(never)]
    pub fn get(&self, key: &GridKey) -> Result<Option<impl Iterator<Item = GridEntry>>, Error> {
        let mut db_key: Vec<u8> = Vec::new();
        key.write_to(TypeMarker::SinglePhrase, &mut db_key)?;

        Ok(match self.db.get(&db_key)? {
            Some(value) => Some(decode_value(value)),
            None => None,
        })
    }

    pub fn streaming_get_matching(
        &self,
        match_key: &MatchKey,
        match_opts: &MatchOpts,
        max_values: usize,
    ) -> Result<impl Iterator<Item = MatchEntry>, Error> {
        let (fetch_start, fetch_end, fetch_type_marker) = match match_key.match_phrase {
            MatchPhrase::Exact(id) => (id, id + 1, TypeMarker::SinglePhrase),
            MatchPhrase::Range { start, end } => {
                if self.bin_boundaries.contains(&start) && self.bin_boundaries.contains(&end) {
                    (start, end, TypeMarker::PrefixBin)
                } else {
                    (start, end, TypeMarker::SinglePhrase)
                }
            }
        };

        let match_opts = match_opts.clone();

        let mut range_key = match_key.clone();
        range_key.match_phrase = MatchPhrase::Range { start: fetch_start, end: fetch_end };
        let mut db_key: Vec<u8> = Vec::new();
        range_key.write_start_to(fetch_type_marker, &mut db_key)?;

        let db_iter = self
            .db
            .iterator(IteratorMode::From(&db_key, Direction::Forward))
            .take_while(|(k, _)| range_key.matches_key(fetch_type_marker, k).unwrap());

        let mut pri_queue = MinMaxHeap::<QueueElement<_>>::new();

        for (key, value) in db_iter {
            let matches_language = match_key.matches_language(&key).unwrap();
            let mut entry_iter =
                decode_matching_value(value, &match_opts, matches_language, self.coalesce_radius);
            if let Some(next_entry) = entry_iter.next() {
                let queue_element = QueueElement { next_entry, entry_iter };
                if pri_queue.len() >= max_values {
                    let worst_entry = pri_queue.peek_min().unwrap();
                    if worst_entry >= &queue_element {
                        continue;
                    } else {
                        pri_queue.replace_min(queue_element);
                    }
                } else {
                    pri_queue.push(queue_element);
                }
            }
        }

        let iter = std::iter::from_fn(move || {
            if let Some(mut best_entry) = pri_queue.peek_max_mut() {
                if let Some(mut next_entry) = best_entry.entry_iter.next() {
                    std::mem::swap(&mut next_entry, &mut (best_entry.next_entry));
                    Some(next_entry)
                } else {
                    let best_entry = best_entry.pop();
                    Some(best_entry.next_entry)
                }
            } else {
                None
            }
        });
        Ok(iter)
    }

    pub fn keys<'i>(&'i self) -> impl Iterator<Item = Result<GridKey, Error>> + 'i {
        let db_iter = self.db.iterator(IteratorMode::Start);
        db_iter.take_while(|(key, _)| key[0] == 0).map(|(key, _)| {
            let phrase_id = (&key[1..]).read_u32::<BigEndian>()?;

            let key_lang_partial = &key[5..];
            let lang_set: u128 = if key_lang_partial.len() == 0 {
                // 0-length language array is the shorthand for "matches everything"
                std::u128::MAX
            } else {
                let mut key_lang_full = [0u8; 16];
                key_lang_full[(16 - key_lang_partial.len())..].copy_from_slice(key_lang_partial);

                (&key_lang_full[..]).read_u128::<BigEndian>()?
            };

            Ok(GridKey { phrase_id, lang_set })
        })
    }

    pub fn iter<'i>(
        &'i self,
    ) -> impl Iterator<Item = Result<(GridKey, Vec<GridEntry>), Error>> + 'i {
        let db_iter = self.db.iterator(IteratorMode::Start);
        db_iter.take_while(|(key, _)| key[0] == 0).map(|(key, value)| {
            let phrase_id = (&key[1..]).read_u32::<BigEndian>()?;

            let key_lang_partial = &key[5..];
            let lang_set: u128 = if key_lang_partial.len() == 0 {
                // 0-length language array is the shorthand for "matches everything"
                std::u128::MAX
            } else {
                let mut key_lang_full = [0u8; 16];
                key_lang_full[(16 - key_lang_partial.len())..].copy_from_slice(key_lang_partial);

                (&key_lang_full[..]).read_u128::<BigEndian>()?
            };

            let entries: Vec<_> = decode_value(value).collect();

            Ok((GridKey { phrase_id, lang_set }, entries))
        })
    }
}
