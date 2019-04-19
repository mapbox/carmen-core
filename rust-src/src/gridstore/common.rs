use std::error::Error;

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, PartialOrd, Ord, PartialEq, Eq, Clone)]
pub struct GridKey {
    pub phrase_id: u32,
    pub lang_set: u128,
}

impl GridKey {
    pub fn write_to(&self, type_marker: u8, db_key: &mut Vec<u8>) -> Result<(), Box<Error>> {
        db_key.push(type_marker);
        // next goes the ID
        db_key.write_u32::<BigEndian>(self.phrase_id)?;
        // now the language ID
        match self.lang_set {
            std::u128::MAX => { /* do nothing -- this is the all-languages marker */ }
            0 => {
                db_key.push(0);
            }
            _ => {
                let lang_set = self.lang_set.to_be_bytes();
                let iter = lang_set.iter().skip_while(|byte| **byte == 0u8);
                db_key.extend(iter);
            }
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Debug, PartialOrd, Ord, PartialEq, Eq, Clone)]
pub enum MatchPhrase {
    Exact(u32),
    Range { start: u32, end: u32 },
}

#[derive(Serialize, Deserialize, Debug, PartialOrd, Ord, PartialEq, Eq, Clone)]
pub struct MatchKey {
    pub match_phrase: MatchPhrase,
    pub lang_set: u128,
}

impl MatchKey {
    pub fn write_start_to(&self, type_marker: u8, db_key: &mut Vec<u8>) -> Result<(), Box<Error>> {
        db_key.push(type_marker);
        // next goes the ID
        let start = match self.match_phrase {
            MatchPhrase::Exact(phrase_id) => phrase_id,
            MatchPhrase::Range { start, .. } => start,
        };
        db_key.write_u32::<BigEndian>(start)?;
        Ok(())
    }

    pub fn matches_key(&self, db_key: &[u8]) -> Result<bool, Box<Error>> {
        let key_phrase = (&db_key[1..]).read_u32::<BigEndian>()?;
        Ok(match self.match_phrase {
            MatchPhrase::Exact(phrase_id) => phrase_id == key_phrase,
            MatchPhrase::Range { start, end } => start <= key_phrase && key_phrase < end,
        })
    }

    pub fn matches_language(&self, db_key: &[u8]) -> Result<bool, Box<Error>> {
        let key_lang_partial = &db_key[5..];
        if key_lang_partial.len() == 0 {
            // 0-length language array is the shorthand for "matches everything"
            return Ok(true);
        }

        let mut key_lang_full = [0u8; 16];
        key_lang_full[(16 - key_lang_partial.len())..].copy_from_slice(key_lang_partial);

        let key_lang_set: u128 = (&key_lang_full[..]).read_u128::<BigEndian>()?;

        Ok(self.lang_set & key_lang_set != 0)
    }
}

// keys consist of a marker byte indicating type (regular entry, prefix cache, etc.) followed by
// a 32-bit phrase ID followed by a variable-length set of bytes for language -- everything after
// the phrase ID is assumed to be language, and it might be up to 128 bits long, but we'll strip
// leading (in a big-endian sense/most-significant sense) zero bytes for compactness
pub const MAX_KEY_LENGTH: usize = 1 + (32 / 8) + (128 / 8);

#[derive(Serialize, Deserialize, Debug, PartialOrd, PartialEq)]
pub struct GridEntry {
    // these will be truncated to 4 bits apiece
    pub relev: f32,
    pub score: u8,
    pub x: u16,
    pub y: u16,
    // this will be truncated to 24 bits
    pub id: u32,
    pub source_phrase_hash: u8,
}

#[derive(Serialize, Deserialize, Debug, PartialOrd, PartialEq)]
pub struct MatchEntry {
    pub grid_entry: GridEntry,
    pub matches_language: bool,
}

#[inline]
pub fn relev_float_to_int(relev: f32) -> u8 {
    if relev == 0.4 {
        0
    } else if relev == 0.6 {
        1
    } else if relev == 0.8 {
        2
    } else {
        3
    }
}

#[inline]
pub fn relev_int_to_float(relev: u8) -> f32 {
    match relev {
        0 => 0.4,
        1 => 0.6,
        2 => 0.8,
        _ => 1.,
    }
}
