use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};
use flatbuffers;
use itertools::Itertools;
use morton::deinterleave_morton;
use ordered_float::OrderedFloat;
use rocksdb::{Direction, IteratorMode, DB};

use crate::gridstore::common::*;
use crate::gridstore::gridstore_generated::*;
use crate::gridstore::spatial;

pub struct GridStore {
    db: DB,
}

// this is a bit of a hack -- it constructs a flatbuffers vector bounded by the lifetime
// of the underlying buffer, rather than by the lifetime of its parent vector, in the event
// that vectors are nested
fn get_vector<'a, T: 'a>(
    buf: &'a [u8],
    table: &flatbuffers::Table,
    field: flatbuffers::VOffsetT,
) -> Option<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<T>>> {
    let o = table.vtable().get(field) as usize;
    if o == 0 {
        return None;
    }

    let addr = table.loc + o;
    let offset = (&buf[addr..(addr + 4)]).read_u32::<LittleEndian>().unwrap() as usize;
    Some(flatbuffers::Vector::new(buf, addr + offset))
}

// the groupby in itertools doesn't take ownership of the thing it's grouping, instead returning
// groups that reference an unowned buffer -- this is tricky for lifetime purposes if you want to
// return an iterator based on a groupby. This version makes a slightly different tradeoff -- it
// takes ownership, and eagerly collects each group into a vector as it goes. So it's still
// lazy-ish (in the sense that it doesn't advance beyond the current group), but more eager than
// the itertools version
fn somewhat_eager_groupby<T: Iterator, F, K>(
    mut it: T,
    key: F,
) -> impl Iterator<Item = (K, Vec<T::Item>)>
where
    K: Sized + Copy + PartialEq,
    F: Fn(&T::Item) -> K,
{
    let mut curr_key: Option<K> = None;
    let mut running_group: Vec<T::Item> = Vec::new();
    let mut done = false;

    std::iter::from_fn(move || {
        if done {
            return None;
        }

        loop {
            let item = it.next();
            if let Some(val) = item {
                let k = key(&val);
                match &curr_key {
                    None => {
                        curr_key = Some(k);
                        running_group.push(val);
                    }
                    Some(o) => {
                        if *o != k {
                            let mut out_vec = Vec::new();
                            std::mem::swap(&mut out_vec, &mut running_group);
                            let to_return = Some((*o, out_vec));

                            running_group.push(val);
                            curr_key = Some(k);

                            return to_return;
                        } else {
                            running_group.push(val);
                        }
                    }
                };
            } else {
                match &curr_key {
                    None => return None,
                    Some(o) => {
                        let mut out_vec = Vec::new();
                        std::mem::swap(&mut out_vec, &mut running_group);
                        let to_return = Some((*o, out_vec));

                        done = true;
                        return to_return;
                    }
                }
            }
        }
    })
}

#[test]
fn eager_test() {
    let a = vec![1, 1, 1, 2, 3, 4, 4, 4, 7, 7, 8];
    let b: Vec<_> = somewhat_eager_groupby(a.into_iter(), |x| *x).collect();
    assert_eq!(
        b,
        vec![
            (1, vec![1, 1, 1]),
            (2, vec![2]),
            (3, vec![3]),
            (4, vec![4, 4, 4]),
            (7, vec![7, 7]),
            (8, vec![8])
        ]
    );

    let a = vec![(1, 'a'), (1, 'b'), (2, 'b'), (3, 'z'), (4, 'a'), (4, 'a')];
    let b: Vec<_> = somewhat_eager_groupby(a.into_iter(), |x| (*x).0).collect();
    assert_eq!(
        b,
        vec![
            (1, vec![(1, 'a'), (1, 'b')]),
            (2, vec![(2, 'b')]),
            (3, vec![(3, 'z')]),
            (4, vec![(4, 'a'), (4, 'a')])
        ]
    );
}

impl GridStore {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let path = path.as_ref().to_owned();
        let db = DB::open_default(&path)?;
        Ok(GridStore { db })
    }

    pub fn get(
        &self,
        key: &GridKey,
    ) -> Result<Option<impl Iterator<Item = GridEntry>>, Box<Error>> {
        let mut db_key: Vec<u8> = Vec::new();
        key.write_to(0, &mut db_key)?;

        Ok(match self.db.get(&db_key)? {
            Some(value) => {
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
                let record = get_root_as_phrase_record(record_ref.1);
                let rs_vec = get_vector::<RelevScore>(
                    record_ref.1,
                    &record._tab,
                    PhraseRecord::VT_RELEV_SCORES,
                )
                .unwrap();

                let iter = rs_vec.iter().flat_map(move |rs_obj| {
                    let relev_score = rs_obj.relev_score();
                    let relev = relev_int_to_float(relev_score >> 4);
                    // mask for the least significant four bits
                    let score = relev_score & 15;

                    let coords =
                        get_vector::<Coord>(record_ref.1, &rs_obj._tab, RelevScore::VT_COORDS)
                            .unwrap();

                    coords.into_iter().flat_map(move |coords_obj| {
                        let (x, y) = deinterleave_morton(coords_obj.coord());

                        coords_obj.ids().unwrap().iter().map(move |id_comp| {
                            let id = id_comp >> 8;
                            let source_phrase_hash = (id_comp & 255) as u8;
                            GridEntry { relev, score, x, y, id, source_phrase_hash }
                        })
                    })
                });
                Some(iter)
            }
            None => None,
        })
    }

    // this is only called this because of inertia -- I'm open to a rename
    pub fn get_matching(
        &self,
        match_key: &MatchKey,
        match_opts: &MatchOpts,
    ) -> Result<Box<dyn Iterator<Item = MatchEntry>>, Box<Error>> {
        match match_opts {
            MatchOpts { bbox: None, .. } => Ok(Box::new(self.global_get_matching(match_key, &match_opts)?)),
            MatchOpts { bbox: Some(bbox), .. } => {
                let bbox: [u16; 4] = bbox.clone();
                let out = self.global_get_matching(match_key, &match_opts)?.filter(move |entry| {
                    entry.grid_entry.x >= bbox[0]
                        && entry.grid_entry.x <= bbox[2]
                        && entry.grid_entry.y >= bbox[1]
                        && entry.grid_entry.y <= bbox[3]
                });
                Ok(Box::new(out))
            }
        }
    }

    fn global_get_matching(
        &self,
        match_key: &MatchKey,
        match_opts: &MatchOpts,
    ) -> Result<impl Iterator<Item = MatchEntry>, Box<Error>> {
        let mut db_key: Vec<u8> = Vec::new();
        match_key.write_start_to(0, &mut db_key)?;

        let match_opts = match_opts.clone();

        let db_iter = self
            .db
            .iterator(IteratorMode::From(&db_key, Direction::Forward))
            .take_while(|(k, _)| match_key.matches_key(k).unwrap());
        let mut lang_match_refs: Vec<(Box<[u8]>, &'static [u8])> = Vec::new();
        let mut lang_mismatch_refs: Vec<(Box<[u8]>, &'static [u8])> = Vec::new();
        for (key, value) in db_iter {
            let record_ref = {
                let value_ref: &[u8] = value.as_ref();
                // same approach as in get above -- maybe sketchy
                let static_ref: &'static [u8] = unsafe { std::mem::transmute(value_ref) };
                (value, static_ref)
            };
            if match_key.matches_language(&key).unwrap() {
                lang_match_refs.push(record_ref);
            } else {
                lang_mismatch_refs.push(record_ref);
            }
        }

        let all_refs = vec![lang_match_refs, lang_mismatch_refs];
        let out = all_refs.into_iter().enumerate().flat_map(move |(i, ref_set)| {
            let matches_language = i == 0;

            // eagerly bucket all the relev/score chunks from all the groups
            // DB entries into a single set
            let mut coords_for_rs = BTreeMap::new();
            for record_ref in &ref_set {
                let record = get_root_as_phrase_record(record_ref.1);
                let rs_vec = get_vector::<RelevScore>(
                    record_ref.1,
                    &record._tab,
                    PhraseRecord::VT_RELEV_SCORES,
                )
                .unwrap();

                for rs_obj in rs_vec {
                    let relev_score = rs_obj.relev_score();
                    let relev = relev_int_to_float(relev_score >> 4);
                    // mask for the least significant four bits
                    let score = relev_score & 15;

                    let coords_vec = get_vector::<Coord>(record_ref.1, &rs_obj._tab, RelevScore::VT_COORDS).unwrap();
                    let coords = match match_opts {
                        MatchOpts { bbox: None, .. } => Box::new(coords_vec.into_iter()) as Box<Iterator<Item=Coord>>,
                        MatchOpts { bbox: Some(bbox), .. } => {
                            let bbox: [u16; 4] = bbox.clone();
                            Box::new(spatial::bbox_filter(coords_vec, bbox)) as Box<Iterator<Item=Coord>>
                        },
                    };

                    let slot =
                        coords_for_rs.entry((OrderedFloat(relev), score)).or_insert_with(|| vec![]);
                    slot.push(coords);
                }
            }

            coords_for_rs.into_iter().rev().flat_map(move |((relev, score), coord_sets)| {
                // this is necessitated by the unsafe hackery above: we need to grab a reference
                // to ref_set so that it gets moved into the closure, so that its memory doesn't
                // get freed before we're done with it
                let _ref_set = &ref_set;
                let relev = relev.into_inner();
                // for each relev/score, lazily k-way-merge the child entities by z-order curve value
                let merged = coord_sets
                    .into_iter()
                    .kmerge_by(|a, b| a.coord().cmp(&b.coord()) == Ordering::Greater)
                    .map(|coords_obj| (coords_obj.coord(), coords_obj));

                // group together entries from different keys that have the same z-order coordinate
                somewhat_eager_groupby(merged, |a| (*a).0).flat_map(
                    move |(coord, coords_obj_group)| {
                        let (x, y) = deinterleave_morton(coord);

                        // get all the feature IDs from all the entries with the same XY, and eagerly
                        // combine them and sort descending if necessary (if there's only one entry,
                        // it's already sorted)
                        let all_ids: Vec<u32> = match coords_obj_group.len() {
                            0 => Vec::new(),
                            1 => coords_obj_group[0].1.ids().unwrap().iter().collect(),
                            _ => {
                                let mut ids = Vec::new();
                                for (_, coords_obj) in coords_obj_group {
                                    ids.extend(coords_obj.ids().unwrap().iter());
                                }
                                ids.sort_by(|a, b| b.cmp(a));
                                ids.dedup();
                                ids
                            }
                        };

                        all_ids.into_iter().map(move |id_comp| {
                            let id = id_comp >> 8;
                            let source_phrase_hash = (id_comp & 255) as u8;
                            MatchEntry {
                                grid_entry: GridEntry {
                                    relev,
                                    score,
                                    x,
                                    y,
                                    id,
                                    source_phrase_hash,
                                },
                                matches_language: matches_language,
                            }
                        })
                    },
                )
            })
        });
        Ok(out)
    }
}
