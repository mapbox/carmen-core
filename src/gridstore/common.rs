use std::error::Error;

use serde::{Serialize, Deserialize};
use byteorder::{BigEndian, WriteBytesExt};

#[derive(Serialize, Deserialize, Debug, PartialOrd, Ord, PartialEq, Eq, Clone)]
pub struct GridKey {
    pub phrase_id: u32,
    pub lang_set: u128
}

impl GridKey {
    pub fn write_to(&self, type_marker: u8, db_key: &mut Vec<u8>) -> Result<(), Box<Error>> {
        db_key.push(type_marker);
        // next goes the ID
        db_key.write_u32::<BigEndian>(self.phrase_id)?;
        // now the language ID
        match self.lang_set {
            std::u128::MAX => { /* do nothing -- this is the all-languages marker */ },
            0 => { db_key.push(0); },
            _ => {
                let lang_set = self.lang_set.to_be_bytes();
                let iter = lang_set.iter().skip_while(|byte| **byte == 0u8);
                db_key.extend(iter);
            }
        }
        Ok(())
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
    pub source_phrase_hash: u8
}

#[inline]
pub fn relev_float_to_int(relev: f32) -> u8 {
    if relev == 0.4 { 0 }
    else if relev == 0.6 { 1 }
    else if relev == 0.8 { 2 }
    else { 3 }
}

#[inline]
pub fn relev_int_to_float(relev: u8) -> f32 {
    match relev {
        0 => 0.4,
        1 => 0.6,
        2 => 0.8,
        _ => 1.
    }
}