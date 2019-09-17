use crate::gridstore::gridstore_format::{Coord, UniformVec};
use itertools::Itertools;
use morton::{deinterleave_morton, interleave_morton};
use std::cmp::Ordering::{Equal, Greater, Less};

/// Generate a tuple of the (min, max) range of the Coord Vector that overlaps with the bounding box
///
/// Returns (Some(min,max)) if the Coord Vector morton order range overlaps with the bounding box,
/// [`None`] if the Coord Vector morton order range does not overlaps with the bounding box
pub fn bbox_range<'a>(coords: UniformVec<'a, Coord>, bbox: [u16; 4]) -> Option<(u32, u32)> {
    let min = interleave_morton(bbox[0], bbox[1]);
    let max = interleave_morton(bbox[2], bbox[3]);
    debug_assert!(min <= max, "Invalid bounding box");

    let len = coords.len();
    if len == 0 {
        return None;
    }

    let range_start = coords.get(0).coord;
    if min > range_start {
        return None;
    }
    let range_end = coords.get(len - 1).coord;
    if max < range_end {
        return None;
    }
    debug_assert!(range_start >= range_end, "Expected descending sort");

    let start = match coord_binary_search(&coords, max, 0) {
        Ok(v) => v,
        Err(_) => return None,
    };
    let mut end = match coord_binary_search(&coords, min, start) {
        Ok(v) => v,
        Err(_) => return None,
    };

    if end == (len as u32) {
        end -= 1;
    }
    debug_assert!(start <= end, "Start is before end");
    Some((start, end))
}

/// Generate an Iterator for a bounding box over a Coord Vector
///
/// Returns [`Some(Iterator<>`] if the Coord Vector morton order range overlaps with the bounding box,
/// [`None`] otherwise. May return an Iterator that yields no results if the morton order overlaps
/// but the actual elements are not in the bounding box.
pub fn bbox_filter<'a>(
    coords: UniformVec<'a, Coord>,
    bbox: [u16; 4],
) -> Option<impl Iterator<Item = Coord> + 'a> {
    let len = coords.len();
    if len == 0 {
        return None;
    }

    let range = bbox_range(coords, bbox)?;
    Some((range.0..=range.1).filter_map(move |idx| {
        let grid = coords.get(idx as usize);
        let (x, y) = deinterleave_morton(grid.coord); // TODO capture this so we don't have to do it again.
        if x >= bbox[0] && x <= bbox[2] && y >= bbox[1] && y <= bbox[3] {
            return Some(coords.get(idx as usize));
        }
        None
    }))
}

/// Generate an Iterator over a Coord Vector given a proximity point
///
/// Returns [`Some(Iterator<>`] which is a Coord Vector morton order range ordered by the z-order distance from the proximity point
/// [`None`] if the Coord Vector is empty
pub fn proximity<'a>(
    coords: UniformVec<'a, Coord>,
    proximity: [u16; 2],
) -> Option<impl Iterator<Item = Coord> + 'a> {
    let prox_pt = interleave_morton(proximity[0], proximity[1]) as i64;
    let len = coords.len() as u32;
    if len == 0 {
        return None;
    }

    let prox_mid = match coord_binary_search(&coords, prox_pt as u32, 0) {
        Ok(v) => v,
        Err(_) => return None,
    };

    let getter = move |i| coords.get(i as usize);
    let head = (0..prox_mid).rev().map(getter);
    let tail = (prox_mid..len).map(getter);
    let coord_sets = head.into_iter().merge_by(tail.into_iter(), move |a, b| {
        let d1 = (a.coord as i64 - prox_pt) as i64;
        let d2 = (b.coord as i64 - prox_pt) as i64;
        d1.abs().cmp(&d2.abs()) == Less
    });

    Some(coord_sets)
}

/// Generate an Iterator for a bounding box and proximity point over a Coord Vector
///
/// Returns [`Some(Iterator<>`] which is a Coord Vector morton order range that overlaps with a bounding box and is ordered by the z-order distance from the proximity point
/// [`None`] if the bounding box does not overlap with the morton order range
pub fn bbox_proximity_filter<'a>(
    coords: UniformVec<'a, Coord>,
    bbox: [u16; 4],
    proximity: [u16; 2],
) -> Option<impl Iterator<Item = Coord> + 'a> {
    let range = bbox_range(coords, bbox)?;
    let prox_pt = interleave_morton(proximity[0], proximity[1]) as i64;
    if coords.len() == 0 {
        return None;
    }

    let prox_mid = match coord_binary_search(&coords, prox_pt as u32, 0) {
        Ok(v) => v,
        Err(_) => return None,
    };

    let filtered_get = move |idx| {
        let grid = coords.get(idx as usize);
        let (x, y) = deinterleave_morton(grid.coord); // TODO capture this so we don't have to do it again.
        if x >= bbox[0] && x <= bbox[2] && y >= bbox[1] && y <= bbox[3] {
            return Some(coords.get(idx as usize));
        } else {
            return None;
        };
    };

    let head = (range.0..prox_mid).rev().filter_map(filtered_get);
    let tail = (prox_mid..=range.1).filter_map(filtered_get);
    let coord_sets = head.into_iter().merge_by(tail.into_iter(), move |a, b| {
        let d1 = (a.coord as i64 - prox_pt) as i64;
        let d2 = (b.coord as i64 - prox_pt) as i64;
        d1.abs().cmp(&d2.abs()) == Less
    });

    Some(coord_sets)
}
/// Binary search this FlatBuffers Coord Vector
///
/// Derived from binary_search_by in core/slice/mod.rs except this expects descending order.
///
/// If val is found within the range captured by Vector with given offset [`Result::Ok`] is returned, containing the
/// index of the matching element. If the value is less than the first element and greater than the last,
/// [`Result::Ok'] is returned containing either 0 or the length of the Vector. A ['Results:Err'] is
/// returned if the offset is greater to the vector length.
fn coord_binary_search<'a>(
    coords: &UniformVec<'a, Coord>,
    val: u32,
    offset: u32,
) -> Result<u32, &'a str> {
    let len = coords.len() as u32;

    if offset >= len {
        return Err("Offset greater than Vector");
    }

    let mut size = len - offset;

    if size == 0 {
        return Ok(0);
    }

    let mut base = offset;
    while size > 1 {
        let half = size / 2;
        let mid = base + half;
        let v = coords.get(mid as usize).coord;
        let cmp = v.cmp(&val);
        base = if cmp == Less { base } else { mid };
        size -= half;
    }
    if base.cmp(&(len - 1)) == Equal {
        return Ok(base);
    }
    let cmp = coords.get(base as usize).coord.cmp(&val);
    if cmp == Equal {
        Ok(base)
    } else {
        Ok(base + (cmp == Greater) as u32)
    }
}

// #[cfg(test)]
// fn flatbuffer_generator<T: Iterator<Item = u32>>(val: T) -> Vec<u8> {
//     let mut fb_builder = flatbuffers::FlatBufferBuilder::new_with_capacity(256);
//     let mut coords: Vec<_> = Vec::new();
//
//     for i in val {
//         let fb_coord = Coord::new(i as u32, 0);
//         coords.push(fb_coord);
//     }
//     let fb_coords = fb_builder.create_vector(&coords);
//
//     let fb_rs = RelevScore::create(
//         &mut fb_builder,
//         &RelevScoreArgs { relev_score: 1, coords: Some(fb_coords) },
//     );
//     fb_builder.finish(fb_rs, None);
//     let data = fb_builder.finished_data();
//     Vec::from(data)
// }
//
// #[cfg(test)]
// mod test {
//     use super::*;
//
//     #[test]
//     fn filter_bbox() {
//         let empty: Vec<u32> = vec![];
//         let buffer = flatbuffer_generator(empty.into_iter());
//         let rs = flatbuffers::get_root::<RelevScore>(&buffer);
//         let coords = rs.coords().unwrap();
//         assert_eq!(bbox_filter(coords, [0, 0, 0, 0]).is_none(), true);
//
//         let buffer = flatbuffer_generator((0..4).rev());
//         let rs = flatbuffers::get_root::<RelevScore>(&buffer);
//         let coords = rs.coords().unwrap();
//         let result = bbox_filter(coords, [0, 0, 1, 1]).unwrap().cloned().collect::<Vec<Coord>>();
//         assert_eq!(result.len(), 4);
//
//         let buffer = flatbuffer_generator((2..4).rev());
//         let rs = flatbuffers::get_root::<RelevScore>(&buffer);
//         let coords = rs.coords().unwrap();
//         let result = bbox_filter(coords, [0, 0, 1, 1]).unwrap().cloned().collect::<Vec<Coord>>();
//         assert_eq!(result.len(), 2, "starts before bbox and ends between the result set");
//
//         let buffer = flatbuffer_generator((2..4).rev());
//         let rs = flatbuffers::get_root::<RelevScore>(&buffer);
//         let coords = rs.coords().unwrap();
//         let result = bbox_filter(coords, [1, 1, 3, 1]).unwrap().cloned().collect::<Vec<Coord>>();
//         assert_eq!(result.len(), 1, "starts in the bbox and ends after the result set");
//
//         let buffer = flatbuffer_generator((1..4).rev());
//         let rs = flatbuffers::get_root::<RelevScore>(&buffer);
//         let coords = rs.coords().unwrap();
//         let result = bbox_filter(coords, [0, 1, 1, 1]).unwrap().cloned().collect::<Vec<Coord>>();
//         assert_eq!(result.len(), 2, "starts in the bbox and ends in the bbox");
//
//         let buffer = flatbuffer_generator((5..7).rev());
//         let rs = flatbuffers::get_root::<RelevScore>(&buffer);
//         let coords = rs.coords().unwrap();
//         assert_eq!(
//             bbox_filter(coords, [0, 0, 0, 1]).is_none(),
//             true,
//             "bbox ends before the range of coordinates"
//         );
//         assert_eq!(
//             bbox_filter(coords, [4, 0, 4, 1]).is_none(),
//             true,
//             "bbox starts after the range of coordinates"
//         );
//
//         let sparse: Vec<u32> = vec![24, 7];
//         let buffer = flatbuffer_generator(sparse.into_iter());
//         let rs = flatbuffers::get_root::<RelevScore>(&buffer);
//         let coords = rs.coords().unwrap();
//         let result = bbox_filter(coords, [3, 1, 4, 2]).unwrap().cloned().collect::<Vec<Coord>>();
//         assert_eq!(result.len(), 2, "sparse result set that spans z-order jumps");
//
//         let buffer = flatbuffer_generator((7..24).rev());
//         let rs = flatbuffers::get_root::<RelevScore>(&buffer);
//         let coords = rs.coords().unwrap();
//         let result = bbox_filter(coords, [3, 1, 4, 2]).unwrap().cloned().collect::<Vec<Coord>>();
//         assert_eq!(result.len(), 3, "continuous result set that spans z-order jumps");
//
//         let sparse: Vec<u32> = vec![8];
//         let buffer = flatbuffer_generator(sparse.into_iter());
//         let rs = flatbuffers::get_root::<RelevScore>(&buffer);
//         let coords = rs.coords().unwrap();
//         let result = bbox_filter(coords, [3, 1, 4, 2]).unwrap().cloned().collect::<Vec<Coord>>();
//         assert_eq!(result.len(), 0, "result is on the z-order curve but not in the bbox");
//     }
//
//     #[test]
//     fn proximity_search() {
//         let buffer = flatbuffer_generator((1..10).rev()); // [9,8,7,6,5,4,3,2,1]
//         let rs = flatbuffers::get_root::<RelevScore>(&buffer);
//         let coords = rs.coords().unwrap();
//
//         let result = proximity(coords, [3, 0]).unwrap().map(|x| x.coord).collect::<Vec<u32>>();
//         assert_eq!(
//             vec![5, 4, 6, 7, 3, 2, 8, 9, 1],
//             result,
//             "proximity point is in the middle of the result set - 5"
//         );
//
//         let result = proximity(coords, [0, 3]).unwrap().map(|x| x.coord).collect::<Vec<u32>>();
//         assert_eq!(
//             vec![9, 8, 7, 6, 5, 4, 3, 2, 1],
//             result,
//             "proximity point is greater than the result set - 10"
//         );
//
//         let result = proximity(coords, [1, 0]).unwrap().map(|x| x.coord).collect::<Vec<u32>>();
//         assert_eq!(
//             vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
//             result,
//             "proximity point is lesser than the result set - 1"
//         );
//
//         let empty: Vec<u32> = vec![];
//         let buffer = flatbuffer_generator(empty.into_iter());
//         let rs = flatbuffers::get_root::<RelevScore>(&buffer);
//         let coords = rs.coords().unwrap();
//         assert_eq!(proximity(coords, [3, 0]).is_none(), true);
//
//         let sparse: Vec<u32> = vec![24, 21, 13, 8, 7, 6, 1]; // 1 and 13 are at the same distance from 7
//         let buffer = flatbuffer_generator(sparse.into_iter());
//         let rs = flatbuffers::get_root::<RelevScore>(&buffer);
//         let coords = rs.coords().unwrap();
//         let result = proximity(coords, [3, 1]).unwrap().map(|x| x.coord).collect::<Vec<u32>>();
//         assert_eq!(
//             vec![7, 6, 8, 13, 1, 21, 24],
//             result,
//             "sparse result set sorted by z-order in the middle of the result set"
//         );
//     }
//
//     #[test]
//     fn bbox_proximity_search() {
//         let buffer = flatbuffer_generator((1..10).rev()); // [9,8,7,6,5,4,3,2,1]
//         let rs = flatbuffers::get_root::<RelevScore>(&buffer);
//         let coords = rs.coords().unwrap();
//         // bbox is from 1-7; proximity is 4
//         let result = bbox_proximity_filter(coords, [1, 0, 3, 1], [2, 0])
//             .unwrap()
//             .map(|x| x.coord)
//             .collect::<Vec<u32>>();
//         assert_eq!(
//             vec![4, 3, 5, 6, 7, 1],
//             result,
//             "bbox within the range of coordinates; proximity point within the result set"
//         );
//
//         assert_eq!(
//             bbox_proximity_filter(coords, [6, 4, 7, 5], [2, 0]).is_none(),
//             true,
//             "bbox outside list of coordinates; proximity within the result set"
//         );
//
//         let result = bbox_proximity_filter(coords, [1, 0, 3, 1], [0, 0])
//             .unwrap()
//             .map(|x| x.coord)
//             .collect::<Vec<u32>>();
//         assert_eq!(
//             vec![1, 3, 4, 5, 6, 7],
//             result,
//             "bbox within the range of coordinates; proximity point outside the result set"
//         );
//
//         let buffer = flatbuffer_generator((2..5).rev()); // [4,3,2]
//         let rs = flatbuffers::get_root::<RelevScore>(&buffer);
//         let coords = rs.coords().unwrap();
//         let result = bbox_proximity_filter(coords, [1, 1, 3, 1], [0, 0]) // bbox is 3-7; proximity is 0
//             .unwrap()
//             .map(|x| x.coord)
//             .collect::<Vec<u32>>();
//         assert_eq!(
//             vec![3],
//             result,
//             "bbox starts in between the list of coordinates and ends after; proximity point outside the result set"
//         );
//
//         let sparse: Vec<u32> = vec![24, 23, 13, 8, 7, 6, 1];
//         let buffer = flatbuffer_generator(sparse.into_iter());
//         let rs = flatbuffers::get_root::<RelevScore>(&buffer);
//         let coords = rs.coords().unwrap();
//         // bbox is 7-23; proximity is 7
//         let result = bbox_proximity_filter(coords, [3, 1, 7, 1], [3, 1])
//             .unwrap()
//             .map(|x| x.coord)
//             .collect::<Vec<u32>>();
//         assert_eq!(
//             vec![7, 23],
//             result,
//             "bbox within sparse result set; proximity within result set"
//         );
//     }
//
//     #[test]
//     fn binary_search() {
//         // Empty Coord list
//         let empty: Vec<u32> = vec![];
//         let buffer = flatbuffer_generator(empty.into_iter());
//         let rs = flatbuffers::get_root::<RelevScore>(&buffer);
//         let coords = rs.coords().unwrap();
//         assert_eq!(coord_binary_search(&coords, 0, 0), Err("Offset greater than Vector"));
//         assert_eq!(coord_binary_search(&coords, 1, 0), Err("Offset greater than Vector"));
//
//         // Single Coord list
//         let single: Vec<u32> = vec![0];
//         let buffer = flatbuffer_generator(single.into_iter());
//         let rs = flatbuffers::get_root::<RelevScore>(&buffer);
//         let coords = rs.coords().unwrap();
//
//         assert_eq!(coord_binary_search(&coords, 0, 0), Ok(0));
//         assert_eq!(coord_binary_search(&coords, 1, 0), Ok(0));
//
//         // Continuous Coord list
//         let buffer = flatbuffer_generator((4..8).rev()); // [7,6,5,4]
//         let rs = flatbuffers::get_root::<RelevScore>(&buffer);
//         let coords = rs.coords().unwrap();
//
//         assert_eq!(coord_binary_search(&coords, 0, 0), Ok(3));
//         assert_eq!(coord_binary_search(&coords, 4, 0), Ok(3));
//         assert_eq!(coord_binary_search(&coords, 4, 1), Ok(3));
//         assert_eq!(coord_binary_search(&coords, 5, 0), Ok(2));
//         assert_eq!(coord_binary_search(&coords, 6, 0), Ok(1));
//         assert_eq!(coord_binary_search(&coords, 7, 0), Ok(0));
//         assert_eq!(coord_binary_search(&coords, 7, 3), Ok(3));
//         assert_eq!(coord_binary_search(&coords, 7, 4), Err("Offset greater than Vector"));
//         assert_eq!(coord_binary_search(&coords, 8, 0), Ok(0));
//
//         // Sparse Coord list
//         let sparse: Vec<u32> = vec![7, 4, 2, 1];
//         let buffer = flatbuffer_generator(sparse.into_iter());
//         let rs = flatbuffers::get_root::<RelevScore>(&buffer);
//         let coords = rs.coords().unwrap();
//
//         assert_eq!(coord_binary_search(&coords, 0, 0), Ok(3));
//         assert_eq!(coord_binary_search(&coords, 1, 0), Ok(3));
//         assert_eq!(coord_binary_search(&coords, 1, 1), Ok(3));
//         assert_eq!(coord_binary_search(&coords, 2, 0), Ok(2));
//         assert_eq!(coord_binary_search(&coords, 3, 0), Ok(2));
//         assert_eq!(coord_binary_search(&coords, 4, 0), Ok(1));
//         assert_eq!(coord_binary_search(&coords, 5, 0), Ok(1));
//         assert_eq!(coord_binary_search(&coords, 7, 0), Ok(0));
//         assert_eq!(coord_binary_search(&coords, 7, 3), Ok(3));
//         assert_eq!(coord_binary_search(&coords, 7, 4), Err("Offset greater than Vector"));
//         assert_eq!(coord_binary_search(&coords, 8, 0), Ok(0));
//     }
// }

/// Calculates the tile distance between a proximity x and y and a grid x and y
pub fn tile_dist(proximity_x: u16, proximity_y: u16, grid_x: u16, grid_y: u16) -> f64 {
    let dx = (proximity_x as f64) - (grid_x as f64);
    let dy = (proximity_y as f64) - (grid_y as f64);
    ((dx * dx) + (dy * dy)).sqrt()
}

#[test]
fn tile_dist_test() {
    assert_eq!(
        tile_dist(1, 1, 1, 1),
        0.,
        "Grid with the same x and y as as the proximity x and y should have tile_dist 0"
    );
    assert_eq!(
        tile_dist(1, 1, 1, 0),
        1.,
        "Grid one tile away from proximity tile should have tile_dist 1"
    );
    assert_eq!(
        tile_dist(1, 1, 0, 0),
        1.4142135623730951,
        "Grid diagonal from proximity tile should have tile_dist between 0 and 1 "
    );
}

/// Returns the number of tiles per mile for a given zoom level
const fn tiles_per_mile_by_zoom(zoom: u16) -> f64 {
    // Array of the pre-calculated ratio of number of tiles per mile at each zoom level
    //
    // 32 tiles is about 40 miles at z14, use this as our mile <=> tile conversion.
    // The formula is (32 / 40 ) / 1.5^(14-zoom).
    // Pow functions are not supported in constant functions in rust,
    // and a custom constant pow function can't be implemented because if statements and loops are not yet supported in constant functions.
    //
    // Note: the formula uses 1.5^(14-zoom) instead of 2^(14-zoom) to be consistent with current behavior,
    // but a truly consistent radius scaled across zoom levels would use 2 as the base.
    // (See https://github.com/mapbox/carmen-cache/pull/110#discussion_r136497028)
    const TILES_PER_MILE_BY_ZOOM: [f64; 17] = [
        0.002740389912625401,
        0.004110584868938102,
        0.006165877303407152,
        0.009248815955110727,
        0.013873223932666092,
        0.020809835898999138,
        0.031214753848498707,
        0.046822130772748057,
        0.07023319615912209,
        0.10534979423868314,
        0.1580246913580247,
        0.23703703703703705,
        0.35555555555555557,
        0.5333333333333333,
        0.8,
        1.2000000000000002,
        1.8000000000000003,
    ];
    TILES_PER_MILE_BY_ZOOM[zoom as usize]
}

#[test]
fn tiles_per_mile_by_zoom_test() {
    assert_eq!(tiles_per_mile_by_zoom(14), 0.8, "Tiles per mile for zoom 14 should be 0.8");
    assert_eq!(
        tiles_per_mile_by_zoom(16),
        1.8000000000000003,
        "Tiles per mile should work for up to zoom 16"
    );
    assert_eq!(
        tiles_per_mile_by_zoom(6),
        0.031214753848498707,
        "Tiles per mile should work for down to zoom 6"
    );
}

/// Convert proximity radius from miles into scaled number of tiles
#[inline]
pub fn proximity_radius(zoom: u16, radius: f64) -> f64 {
    debug_assert!(zoom <= 16);
    // In carmen-cache, there's an array of pre-calculated values for zooms 6-14, otherwise it does the exact same calculation as zoomTileRadius (now tiles_per_mile)
    // Does this even need to be a function?
    radius * tiles_per_mile_by_zoom(zoom)
}

#[test]
fn proximity_radius_test() {
    assert_eq!(
        proximity_radius(14, 400.),
        320.,
        "Proximity radius in tiles for zoom 14, radius 400 is as expected"
    );
    assert_eq!(
        proximity_radius(16, 400.),
        720.0000000000001,
        "proximity_radius should work for zoom 14"
    );
    assert_eq!(proximity_radius(6, 0.), 0., "proximity_radius for a radius of 0 should be 0");
    assert_eq!(
        proximity_radius(6, 40.),
        1.2485901539399482,
        "proximity_radius in tiles for zoom 6, radius 40 is as expected"
    );
    // TODO: test zoom > 14?
}

// We don't know the scale of the axis we're modeling, but it doesn't really
// matter as we just need internal consistency.
const E_POW: [f64; 8] = [
    1.,
    2.718281828459045,
    7.38905609893065,
    20.085536923187668,
    54.598150033144236,
    148.4131591025766,
    403.4287934927351,
    1096.6331584284585,
];

pub fn scoredist(mut zoom: u16, mut distance: f64, mut score: u8, radius: f64) -> f64 {
    if zoom < 6 {
        zoom = 6;
    }
    if score > 7 {
        score = 7;
    }

    // If the distance is 0, set a minimum distance to avoid dividing by distratios that approach zero
    if distance < 1. {
        distance = 0.8;
    }

    let mut dist_ratio: f64 = distance / proximity_radius(zoom, radius);

    // Beyond the proximity radius just let scoredist be driven by score.
    if dist_ratio > 1.0 {
        dist_ratio = 1.00;
    }
    ((6. * E_POW[score as usize] / E_POW[7]) + 1.) / dist_ratio
}

#[test]
fn scoredist_test() {
    assert_eq!(scoredist(14, 1., 0, 400.), 321.7508133738646, "scoredist for a feature 1 tile away from proximity point with score 0 and radius 400 should be 321.7508133738646");
    assert_eq!(scoredist(14, 0., 0, 400.), 402.1885167173308, "scoredist for a feature on the same tile as the proximity point with score 0 and radius 400 should be 402.1885167173308,");
}
