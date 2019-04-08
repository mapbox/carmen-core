use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, PartialOrd, Ord, PartialEq, Eq, Clone)]
pub struct GridKey {
    pub phrase_id: u32,
    pub lang_set: u128
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