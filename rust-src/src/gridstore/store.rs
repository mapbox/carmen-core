use std::error::Error;
use std::path::Path;

use morton::deinterleave_morton;
use rocksdb::{DB, IteratorMode, Direction};
use flatbuffers;
use byteorder::{LittleEndian, ReadBytesExt};

use crate::gridstore::common::*;
use crate::gridstore::gridstore_generated::*;

pub struct GridStore {
    db: DB
}

// this is a bit of a hack -- it constructs a flatbuffers vector bounded by the lifetime
// of the underlying buffer, rather than by the lifetime of its parent vector, in the event
// that vectors are nested
fn get_vector<'a, T: 'a>(buf: &'a [u8], table: &flatbuffers::Table, field: flatbuffers::VOffsetT)
    -> Option<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<T>>> {
    let o = table.vtable().get(field) as usize;
    if o == 0 {
        return None;
    }

    let addr = table.loc + o;
    let offset = (&buf[addr..(addr + 4)]).read_u32::<LittleEndian>().unwrap() as usize;
    Some(flatbuffers::Vector::new(buf, addr + offset))
}

impl GridStore {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let path = path.as_ref().to_owned();
        let db = DB::open_default(&path)?;
        Ok(GridStore { db })
    }

    pub fn get(&self, key: &GridKey) -> Result<Option<impl Iterator<Item=GridEntry>>, Box<Error>> {
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
                let rs_vec = get_vector::<RelevScore>(record_ref.1, &record._tab, PhraseRecord::VT_RELEV_SCORES).unwrap();

                let iter = rs_vec.iter().flat_map(move |rs_obj| {
                    let relev_score = rs_obj.relev_score();
                    let relev = relev_int_to_float(relev_score >> 4);
                    // mask for the least significant four bits
                    let score = relev_score & 15;

                    let coords = get_vector::<Coord>(record_ref.1, &rs_obj._tab, RelevScore::VT_COORDS).unwrap();

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
            },
            None => None,
        })
    }

    // this is only called this because of inertia -- I'm open to a rename
    pub fn get_matching(&self, match_key: &MatchKey) -> Result<impl Iterator<Item=MatchEntry>, Box<Error>> {
        let mut db_key: Vec<u8> = Vec::new();
        match_key.write_start_to(0, &mut db_key)?;

        let db_iter = self.db
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

            ref_set.into_iter().flat_map(move |record_ref| {
                let record = get_root_as_phrase_record(record_ref.1);
                let rs_vec = get_vector::<RelevScore>(record_ref.1, &record._tab, PhraseRecord::VT_RELEV_SCORES).unwrap();

                rs_vec.iter().flat_map(move |rs_obj| {
                    let relev_score = rs_obj.relev_score();
                    let relev = relev_int_to_float(relev_score >> 4);
                    // mask for the least significant four bits
                    let score = relev_score & 15;

                    let coords = get_vector::<Coord>(record_ref.1, &rs_obj._tab, RelevScore::VT_COORDS).unwrap();

                    coords.into_iter().flat_map(move |coords_obj| {
                        let (x, y) = deinterleave_morton(coords_obj.coord());

                        coords_obj.ids().unwrap().iter().map(move |id_comp| {
                            let id = id_comp >> 8;
                            let source_phrase_hash = (id_comp & 255) as u8;
                            MatchEntry {
                                grid_entry: GridEntry { relev, score, x, y, id, source_phrase_hash },
                                matches_language: matches_language
                            }
                        })
                    })
                })
            })
        });
        Ok(out)
    }
}
