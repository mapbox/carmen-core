use core::cmp::{Ordering, Reverse};
use std::borrow::Borrow;

use crate::gridstore::spatial::adjust_bbox_zoom;
use crate::gridstore::store::GridStore;

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use failure::Error;
use fixedbitset::FixedBitSet;
use min_max_heap::MinMaxHeap;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize, Serializer};

#[derive(Copy, Clone, Debug)]
pub enum TypeMarker {
    SinglePhrase = 0,
    PrefixBin = 1,
}

#[derive(Serialize, Deserialize, Debug, PartialOrd, Ord, PartialEq, Eq, Clone)]
pub struct GridKey {
    pub phrase_id: u32,
    pub lang_set: u128,
}

impl GridKey {
    pub fn write_to(&self, type_marker: TypeMarker, db_key: &mut Vec<u8>) -> Result<(), Error> {
        db_key.push(type_marker as u8);
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
    pub fn write_start_to(
        &self,
        type_marker: TypeMarker,
        db_key: &mut Vec<u8>,
    ) -> Result<(), Error> {
        db_key.push(type_marker as u8);
        // next goes the ID
        let start = match self.match_phrase {
            MatchPhrase::Exact(phrase_id) => phrase_id,
            MatchPhrase::Range { start, .. } => start,
        };
        db_key.write_u32::<BigEndian>(start)?;
        Ok(())
    }

    pub fn matches_key(&self, type_marker: TypeMarker, db_key: &[u8]) -> Result<bool, Error> {
        let key_phrase = (&db_key[1..]).read_u32::<BigEndian>()?;
        if db_key[0] != (type_marker as u8) {
            return Ok(false);
        }
        Ok(match self.match_phrase {
            MatchPhrase::Exact(phrase_id) => phrase_id == key_phrase,
            MatchPhrase::Range { start, end } => start <= key_phrase && key_phrase < end,
        })
    }

    pub fn matches_language(&self, db_key: &[u8]) -> Result<bool, Error> {
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

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct MatchOpts {
    pub bbox: Option<[u16; 4]>,
    pub proximity: Option<[u16; 2]>,
    pub zoom: u16,
}

impl Default for MatchOpts {
    fn default() -> Self {
        MatchOpts { bbox: None, proximity: None, zoom: 16 }
    }
}

pub const EARTH_CIRC_IN_MILES: f64 = 24901.0;
pub const NEARBY_RADIUS: f64 = 25.0;

impl MatchOpts {
    pub fn adjust_to_zoom(&self, target_z: u16) -> MatchOpts {
        if self.zoom == target_z {
            self.clone()
        } else {
            let adjusted_proximity = match &self.proximity {
                Some([x, y]) => {
                    if target_z < self.zoom {
                        // If this is a zoom out, divide by 2 for every level of zooming out.
                        let zoom_levels = self.zoom - target_z;
                        // Shifting to the right by a number is the same as dividing by 2 that number of times.
                        Some([x >> zoom_levels, y >> zoom_levels])
                    } else {
                        // If this is a zoom in, choose the closest to the middle of the possible tiles at the higher zoom level.
                        // The scale of the coordinates for zooming in is 2^(difference in zs).
                        let scale_multiplier = 1 << (target_z - self.zoom);
                        // Pick a coordinate halfway between the possible higher zoom tiles,
                        // subtracting one to pick the one on the top left of the four middle tiles for consistency.
                        let mid_coord_adjuster = scale_multiplier / 2 - 1;
                        let adjusted_x = x * scale_multiplier + mid_coord_adjuster;
                        let adjusted_y = y * scale_multiplier + mid_coord_adjuster;

                        Some([adjusted_x, adjusted_y])
                    }
                }
                None => None,
            };

            let adjusted_bbox = self.bbox.map(|bbox| adjust_bbox_zoom(bbox, self.zoom, target_z));

            MatchOpts { zoom: target_z, proximity: adjusted_proximity, bbox: adjusted_bbox }
        }
    }

    pub fn with_nearby_only(&self) -> MatchOpts {
        let mut constrained = self.clone();
        let prox = if let Some(prox) = constrained.proximity {
            prox.clone()
        } else {
            return constrained;
        };

        let miles_per_tile = EARTH_CIRC_IN_MILES / ((1 << constrained.zoom) as f64);
        let padding = (NEARBY_RADIUS / miles_per_tile).ceil() as u16;

        let mut new_box: [u16; 4] = [
            if prox[0] < padding { 0 } else { prox[0] - padding }, // prevent overflows because this is unsigned
            if prox[1] < padding { 0 } else { prox[1] - padding }, // ditto
            prox[0] + padding,
            prox[1] + padding,
        ];

        if let Some(old_box) = constrained.bbox {
            new_box = [
                std::cmp::max(old_box[0], new_box[0]),
                std::cmp::max(old_box[1], new_box[1]),
                std::cmp::min(old_box[2], new_box[2]),
                std::cmp::min(old_box[3], new_box[3]),
            ]
        }

        constrained.bbox = Some(new_box);
        constrained
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use once_cell::sync::Lazy;

    fn matchopts_proximity_generator(point: [u16; 2], zoom: u16) -> MatchOpts {
        MatchOpts { proximity: Some(point), zoom: zoom, ..MatchOpts::default() }
    }

    #[test]
    fn adjust_to_zoom_test_proximity() {
        static MATCH_OPTS_PROXIMITY: Lazy<(MatchOpts, MatchOpts, MatchOpts)> = Lazy::new(|| {
            let match_opts1 = matchopts_proximity_generator([2, 28], 14);
            let match_opts2 = matchopts_proximity_generator([11, 25], 6);
            let match_opts3 = matchopts_proximity_generator([6, 6], 4);
            (match_opts1, match_opts2, match_opts3)
        });

        let adjusted_match_opts1 = MATCH_OPTS_PROXIMITY.0.adjust_to_zoom(6);
        assert_eq!(
            adjusted_match_opts1.zoom, 6,
            "Adjusted MatchOpts should have target zoom as zoom"
        );
        assert_eq!(adjusted_match_opts1.proximity.unwrap(), [0, 0], "should be 0,0");

        let adjusted_match_opts2 = MATCH_OPTS_PROXIMITY.1.adjust_to_zoom(8);
        assert_eq!(
            adjusted_match_opts2.zoom, 8,
            "Adjusted MatchOpts should have target zoom as zoom"
        );
        assert_eq!(adjusted_match_opts2.proximity.unwrap(), [45, 101], "Should be 45, 101");

        let same_zoom = MATCH_OPTS_PROXIMITY.2.adjust_to_zoom(4);
        assert_eq!(same_zoom, MATCH_OPTS_PROXIMITY.2, "If the zoom is the same as the original, adjusted MatchOpts should be a clone of the original");
        let zoomed_out_1z = MATCH_OPTS_PROXIMITY.2.adjust_to_zoom(3);
        let proximity_out_1z = zoomed_out_1z.proximity.unwrap();
        assert_eq!(proximity_out_1z, [3, 3], "4/6/6 zoomed out to zoom 3 should be 3/3/3");
        assert_eq!(zoomed_out_1z.zoom, 3, "The adjusted zoom should be the target zoom");

        let zoomed_out_2z = MATCH_OPTS_PROXIMITY.2.adjust_to_zoom(2);
        let proximity_out_2z = zoomed_out_2z.proximity.unwrap();
        assert_eq!(proximity_out_2z, [1, 1], "4/6/6 zoomed out to zoom 2 should be 2/1/1");

        let zoomed_in_1z = MATCH_OPTS_PROXIMITY.2.adjust_to_zoom(5);
        let proximity_in_1z = zoomed_in_1z.proximity.unwrap();
        assert_eq!(proximity_in_1z, [12, 12], "4/6/6 zoomed in to zoom 5 should be 5/12/12");
        assert_eq!(zoomed_in_1z.zoom, 5, "The adjusted zoom should be the target zoom");

        let zoomed_in_2z = MATCH_OPTS_PROXIMITY.2.adjust_to_zoom(6);
        let proximity_in_2z = zoomed_in_2z.proximity.unwrap();
        assert_eq!(proximity_in_2z, [25, 25], "4/6/6 zoomed in to zoom 6 should be 6/25/25");

        let zoomed_in_3z = MATCH_OPTS_PROXIMITY.2.adjust_to_zoom(7);
        let proximity_in_3z = zoomed_in_3z.proximity.unwrap();
        assert_eq!(proximity_in_3z, [51, 51], "4/6/6 zoomed in to zoom 7 should be 7/51/51");
    }

    fn matchopts_bbox_generator(bbox: [u16; 4], zoom: u16) -> MatchOpts {
        MatchOpts { bbox: Some(bbox), zoom: zoom, ..MatchOpts::default() }
    }

    #[test]
    fn adjust_to_zoom_text_bbox() {
        static MATCH_OPTS_BBOX: Lazy<(
            MatchOpts,
            MatchOpts,
            MatchOpts,
            MatchOpts,
            MatchOpts,
            MatchOpts,
        )> = Lazy::new(|| {
            let match_opts1 = matchopts_bbox_generator([32760, 32758, 32767, 32714], 15);
            let match_opts2 = matchopts_bbox_generator([6, 4, 7, 5], 4);
            let match_opts3 = matchopts_bbox_generator([6, 5, 7, 6], 4);
            let match_opts4 = matchopts_bbox_generator([3, 3, 3, 3], 3);
            let match_opts5 = matchopts_bbox_generator([5, 3, 7, 4], 3);
            let match_opts6 = matchopts_bbox_generator([6, 3, 8, 4], 5);
            (match_opts1, match_opts2, match_opts3, match_opts4, match_opts5, match_opts6)
        });
        // Test bottom right most tile at highest zoom
        let zoomed_in_16 = MATCH_OPTS_BBOX.0.adjust_to_zoom(16);
        assert_eq!(
            zoomed_in_16.bbox.unwrap(),
            [65520, 65516, 65535, 65429],
            "does not error while zooming into the right most tile on the highest zoom level"
        );

        // Test case where single parent tile contains entire bbox
        let zoomed_out_1z = MATCH_OPTS_BBOX.1.adjust_to_zoom(3);
        assert_eq!(zoomed_out_1z.bbox.unwrap(), [3,2,3,2], "Bbox covering 4 tiles zoomed out 1z can be 1 parent tile if it contains all 4 original tiles");
        assert_eq!(zoomed_out_1z.zoom, 3, "The adjusted zoom should be the target zoom");
        let zoomed_back_in_1z = zoomed_out_1z.adjust_to_zoom(4);
        assert_eq!(
            zoomed_back_in_1z, MATCH_OPTS_BBOX.1,
            "The zoomed in bbox from 1 parent tile should include the 4 tiles it contains"
        );

        // Test case where higher zoom level bbox spans multiple parent tiles
        let zoomed_out_1z_2 = MATCH_OPTS_BBOX.2.adjust_to_zoom(3);
        assert_eq!(
            zoomed_out_1z_2.bbox.unwrap(),
            [3, 2, 3, 3],
            "Bboxes that span two parent tiles should return a bbox that includes both parent tiles"
        );
        let zoomed_back_in_1z_2 = zoomed_out_1z_2.adjust_to_zoom(4);
        assert_eq!(
            zoomed_back_in_1z_2.bbox.unwrap(),
            [6, 4, 7, 7],
            "The zoomed in bbox from 2 parent tiles should include all 8 tiles they contain"
        );

        // Gut check simple case
        assert_eq!(
            MATCH_OPTS_BBOX.3.adjust_to_zoom(4).bbox.unwrap(),
            [6, 6, 7, 7],
            "[3,3,3,3] is correctly scaled to zoom 4"
        );
        assert_eq!(
            MATCH_OPTS_BBOX.3.adjust_to_zoom(5).bbox.unwrap(),
            [12, 12, 15, 15],
            "[3,3,3,3] is correctly scaled to zoom 5"
        );

        // Multi-tile parent bbox zoom in
        assert_eq!(
            MATCH_OPTS_BBOX.4.adjust_to_zoom(4).bbox.unwrap(),
            [10, 6, 15, 9],
            "Multi-tile parent zoomed in one zoom level includes all the higher-zoom tiles"
        );
        assert_eq!(
            MATCH_OPTS_BBOX.4.adjust_to_zoom(5).bbox.unwrap(),
            [20, 12, 31, 19],
            "Multi-tile parent zoomed in two zoom levels includes all the higher-zoom tiles"
        );

        // Multi-parent, multi-tile bbox zoomed out
        assert_eq!(
            MATCH_OPTS_BBOX.5.adjust_to_zoom(4).bbox.unwrap(),
            [3, 1, 4, 2],
            "Multi-tile parent zoomed in one zoom level includes all the higher-zoom tiles"
        );
    }

    #[test]
    fn nearby_only() {
        let opts = matchopts_proximity_generator([100, 100], 14);
        assert_eq!(
            opts.with_nearby_only(),
            MatchOpts { bbox: Some([83, 83, 117, 117]), proximity: Some([100, 100]), zoom: 14 }
        );

        let opts = matchopts_proximity_generator([100, 100], 6);
        assert_eq!(
            opts.with_nearby_only(),
            MatchOpts { bbox: Some([99, 99, 101, 101]), proximity: Some([100, 100]), zoom: 6 }
        );

        // truncate at the antemeridian
        let opts = matchopts_proximity_generator([5, 5], 14);
        assert_eq!(
            opts.with_nearby_only(),
            MatchOpts { bbox: Some([0, 0, 22, 22]), proximity: Some([5, 5]), zoom: 14 }
        );

        // test interaction between existing bbox and limiter
        let mut opts = matchopts_proximity_generator([100, 100], 14);
        opts.bbox = Some([90, 70, 115, 180]);
        assert_eq!(
            opts.with_nearby_only(),
            MatchOpts { bbox: Some([90, 83, 115, 117]), proximity: Some([100, 100]), zoom: 14 }
        );
    }
}

// keys consist of a marker byte indicating type (regular entry, prefix cache, etc.) followed by
// a 32-bit phrase ID followed by a variable-length set of bytes for language -- everything after
// the phrase ID is assumed to be language, and it might be up to 128 bits long, but we'll strip
// leading (in a big-endian sense/most-significant sense) zero bytes for compactness
pub const MAX_KEY_LENGTH: usize = 1 + (32 / 8) + (128 / 8);

// The max number of contexts to return from Coalesce
pub const MAX_CONTEXTS: usize = 40;

// limit to 100,000 records -- we may want to experiment with this number; it was 500k in
// carmen-cache, but hopefully we're sorting more intelligently on the way in here so
// shouldn't need as many records. Still, we should limit it somehow.
pub const MAX_GRIDS_PER_PHRASE: usize = 100_000;

#[derive(Serialize, Deserialize, Debug, PartialOrd, PartialEq, Clone)]
pub struct GridEntry {
    // these will be truncated to 4 bits apiece
    pub relev: f64,
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
    pub distance: f64,
    pub scoredist: f64,
}

#[derive(Serialize, Deserialize, Debug, PartialOrd, PartialEq, Clone)]
pub struct CoalesceEntry {
    pub grid_entry: GridEntry,
    pub matches_language: bool,
    pub idx: u16,
    pub tmp_id: u32,
    pub mask: u32,
    pub distance: f64,
    pub scoredist: f64,
    pub phrasematch_id: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CoalesceContext {
    pub mask: u32,
    pub relev: f64,
    pub entries: Vec<CoalesceEntry>,
}

impl CoalesceContext {
    #[inline(always)]
    fn sort_key(&self) -> (OrderedFloat<f64>, OrderedFloat<f64>, Reverse<u16>, u16, u16, u32) {
        (
            OrderedFloat(self.relev),
            OrderedFloat(self.entries[0].scoredist),
            Reverse(self.entries[0].idx),
            self.entries[0].grid_entry.x,
            self.entries[0].grid_entry.y,
            self.entries[0].grid_entry.id,
        )
    }
}

impl Ord for CoalesceContext {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sort_key().cmp(&other.sort_key())
    }
}
impl PartialOrd for CoalesceContext {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for CoalesceContext {
    fn eq(&self, other: &Self) -> bool {
        self.sort_key() == other.sort_key()
    }
}
impl Eq for CoalesceContext {}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MatchKeyWithId {
    pub key: MatchKey,
    #[serde(default)]
    pub nearby_only: bool,
    pub id: u32,
}

#[derive(Serialize, Debug, Clone)]
pub struct PhrasematchSubquery<T: Borrow<GridStore> + Clone> {
    pub store: T,
    pub idx: u16,
    #[serde(serialize_with = "serialize_fixedbitset")]
    pub non_overlapping_indexes: FixedBitSet, // the field formerly known as bmask
    pub weight: f64,
    pub mask: u32,
    pub match_keys: Vec<MatchKeyWithId>,
}

fn serialize_fixedbitset<S>(bits: &FixedBitSet, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.collect_seq(bits.ones())
}

pub struct ConstrainedPriorityQueue<T: Ord> {
    pub max_size: usize,
    heap: MinMaxHeap<T>,
}

impl<T: Ord> ConstrainedPriorityQueue<T> {
    pub fn new(max_size: usize) -> Self {
        ConstrainedPriorityQueue { max_size, heap: MinMaxHeap::new() }
    }

    pub fn push(&mut self, element: T) -> bool {
        if self.heap.len() >= self.max_size {
            if let Some(min) = self.heap.peek_min() {
                if element > *min {
                    self.heap.replace_min(element);
                    return true;
                }
            }
        } else {
            self.heap.push(element);
            return true;
        }
        false
    }

    pub fn pop_max(&mut self) -> Option<T> {
        self.heap.pop_max()
    }

    pub fn peek_min(&self) -> Option<&T> {
        self.heap.peek_min()
    }

    pub fn peek_max(&self) -> Option<&T> {
        self.heap.peek_max()
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn into_vec_desc(self) -> Vec<T> {
        self.heap.into_vec_desc()
    }
}

impl<T: Ord> IntoIterator for ConstrainedPriorityQueue<T> {
    type Item = T;
    type IntoIter = min_max_heap::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.heap.into_iter()
    }
}

#[inline]
pub fn relev_float_to_int(relev: f64) -> u8 {
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
pub fn relev_int_to_float(relev: u8) -> f64 {
    match relev {
        0 => 0.4,
        1 => 0.6,
        2 => 0.8,
        _ => 1.,
    }
}

// the groupby in itertools doesn't take ownership of the thing it's grouping, instead returning
// groups that reference an unowned buffer -- this is tricky for lifetime purposes if you want to
// return an iterator based on a groupby. This version makes a slightly different tradeoff -- it
// takes ownership, and eagerly collects each group into a vector as it goes. So it's still
// lazy-ish (in the sense that it doesn't advance beyond the current group), but more eager than
// the itertools version
pub fn somewhat_eager_groupby<T: Iterator, F, K>(
    mut it: T,
    mut key: F,
) -> impl Iterator<Item = (K, Vec<T::Item>)>
where
    K: Sized + Copy + PartialEq,
    F: FnMut(&T::Item) -> K,
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
