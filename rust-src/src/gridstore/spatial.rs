use crate::gridstore::gridstore_generated::*;
use flatbuffers;
use itertools::Itertools;
use morton::{deinterleave_morton, interleave_morton};
use std::cmp::Ordering::{Equal, Greater, Less};

/// Generate an Iterator for a bounding box over a Coord Vector
///
/// Returns [`Some(Iterator<>`] if the Coord Vector morton order range overlaps with the bouding box,
/// [`None`] otherwise. May return an Iterator that yields no results if the morton order overlaps
/// but the actual elements are not in the bounding box.
pub fn bbox_range<'a>(
    coords: flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<Coord<'a>>>,
    bbox: [u16; 4],
) -> Option<(u32, u32)> {
    let min = interleave_morton(bbox[0], bbox[1]);
    let max = interleave_morton(bbox[2], bbox[3]);
    debug_assert!(min <= max, "Invalid bounding box");

    let len = coords.len();
    if len == 0 {
        return None;
    }

    let range_start = coords.get(0).coord();
    if min > range_start {
        return None;
    }
    let range_end = coords.get(len - 1).coord();
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

pub fn bbox_filter<'a>(
    coords: flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<Coord<'a>>>,
    bbox: [u16; 4],
) -> Option<impl Iterator<Item = Coord<'a>>> {
    let len = coords.len();
    if len == 0 {
        return None;
    }

    let range = bbox_range(coords, bbox)?;
    Some((range.0..(range.1 + 1)).filter_map(move |idx| {
        let grid = coords.get(idx as usize);
        let (x, y) = deinterleave_morton(grid.coord()); // TODO capture this so we don't have to do it again.
        if x >= bbox[0] && x <= bbox[2] && y >= bbox[1] && y <= bbox[3] {
            return Some(grid);
        }
        None
    }))
}

pub fn proximity<'a>(
    coords: flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<Coord<'a>>>,
    proximity: [u16; 2],
) -> Option<impl Iterator<Item = Coord<'a>>> {
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
    let head = Box::new((0..prox_mid).rev().map(getter)) as Box<Iterator<Item = Coord>>;
    let tail = Box::new((prox_mid..len).map(getter)) as Box<Iterator<Item = Coord>>;
    let coord_sets = vec![head, tail].into_iter().kmerge_by(move |a, b| {
        let d1 = (a.coord() as i64 - prox_pt) as i64;
        let d2 = (b.coord() as i64 - prox_pt) as i64;
        d1.abs().cmp(&d2.abs()) == Less
    });

    Some(coord_sets)
}

pub fn bbox_proximity_filter<'a>(
    coords: flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<Coord<'a>>>,
    bbox: [u16; 4],
    proximity: [u16; 2],
) -> Option<impl Iterator<Item = Coord<'a>>> {
    let range = bbox_range(coords, bbox)?;
    let prox_pt = interleave_morton(proximity[0], proximity[1]) as i64;
    if coords.len() == 0 {
        return None;
    }

    let prox_mid = match coord_binary_search(&coords, prox_pt as u32, 0) {
        Ok(v) => v,
        Err(_) => return None,
    };

    let getter = move |i| coords.get(i as usize);
    let head = Box::new((range.0..prox_mid).rev().map(getter)) as Box<Iterator<Item = Coord>>;
    let tail = Box::new((prox_mid..range.1 + 1).map(getter)) as Box<Iterator<Item = Coord>>;
    let coord_sets = vec![head, tail].into_iter().kmerge_by(move |a, b| {
        let d1 = (a.coord() as i64 - prox_pt) as i64;
        let d2 = (b.coord() as i64 - prox_pt) as i64;
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
    coords: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<Coord>>,
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
        let v = coords.get(mid as usize).coord();
        let cmp = v.cmp(&val);
        base = if cmp == Less { base } else { mid };
        size -= half;
    }
    if base.cmp(&(len - 1)) == Equal {
        return Ok(base);
    }
    let cmp = coords.get(base as usize).coord().cmp(&val);
    if cmp == Equal {
        Ok(base)
    } else {
        Ok(base + (cmp == Greater) as u32)
    }
}

#[cfg(test)]
fn flatbuffer_generator<T: Iterator<Item = u32>>(val: T) -> Vec<u8> {
    let mut fb_builder = flatbuffers::FlatBufferBuilder::new_with_capacity(256);
    let mut coords: Vec<_> = Vec::new();

    let ids: Vec<u32> = vec![0];
    for i in val {
        let fb_ids = fb_builder.create_vector(&ids);
        let fb_coord =
            Coord::create(&mut fb_builder, &CoordArgs { coord: i as u32, ids: Some(fb_ids) });
        coords.push(fb_coord);
    }
    let fb_coords = fb_builder.create_vector(&coords);

    let fb_rs = RelevScore::create(
        &mut fb_builder,
        &RelevScoreArgs { relev_score: 1, coords: Some(fb_coords) },
    );
    fb_builder.finish(fb_rs, None);
    let data = fb_builder.finished_data();
    Vec::from(data)
}

mod test {
    use super::*;

    #[test]
    fn filter_bbox() {
        let empty: Vec<u32> = vec![];
        let buffer = flatbuffer_generator(empty.into_iter());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        assert_eq!(bbox_filter(coords, [0, 0, 0, 0]).is_none(), true);

        let buffer = flatbuffer_generator((0..4).rev());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        let result = bbox_filter(coords, [0, 0, 1, 1]).unwrap().collect::<Vec<Coord>>();
        assert_eq!(result.len(), 4);

        let buffer = flatbuffer_generator((2..4).rev());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        let result = bbox_filter(coords, [0, 0, 1, 1]).unwrap().collect::<Vec<Coord>>();
        assert_eq!(result.len(), 2, "starts before bbox and ends between the result set");

        let buffer = flatbuffer_generator((2..4).rev());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        let result = bbox_filter(coords, [1, 1, 3, 1]).unwrap().collect::<Vec<Coord>>();
        assert_eq!(result.len(), 1, "starts in the bbox and ends after the result set");

        let buffer = flatbuffer_generator((1..4).rev());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        let result = bbox_filter(coords, [0, 1, 1, 1]).unwrap().collect::<Vec<Coord>>();
        assert_eq!(result.len(), 2, "starts in the bbox and ends in the bbox");

        let buffer = flatbuffer_generator((5..7).rev());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        assert_eq!(
            bbox_filter(coords, [0, 0, 0, 1]).is_none(),
            true,
            "bbox ends before the range of coordinates"
        );
        assert_eq!(
            bbox_filter(coords, [4, 0, 4, 1]).is_none(),
            true,
            "bbox starts after the range of coordinates"
        );

        let sparse: Vec<u32> = vec![24, 7];
        let buffer = flatbuffer_generator(sparse.into_iter());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        let result = bbox_filter(coords, [3, 1, 4, 2]).unwrap().collect::<Vec<Coord>>();
        assert_eq!(result.len(), 2, "sparse result set that spans z-order jumps");

        let buffer = flatbuffer_generator((7..24).rev());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        let result = bbox_filter(coords, [3, 1, 4, 2]).unwrap().collect::<Vec<Coord>>();
        assert_eq!(result.len(), 3, "continuous result set that spans z-order jumps");

        let sparse: Vec<u32> = vec![8];
        let buffer = flatbuffer_generator(sparse.into_iter());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        let result = bbox_filter(coords, [3, 1, 4, 2]).unwrap().collect::<Vec<Coord>>();
        assert_eq!(result.len(), 0, "result is on the z-order curve but not in the bbox");
    }

    #[test]
    fn proximity_search() {
        let buffer = flatbuffer_generator((1..10).rev()); // [9,8,7,6,5,4,3,2,1]
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();

        let result = proximity(coords, [3, 0]).unwrap().map(|x| x.coord()).collect::<Vec<u32>>();
        assert_eq!(
            vec![5, 4, 6, 7, 3, 2, 8, 9, 1],
            result,
            "proximity point is in the middle of the result set - 5"
        );

        let result = proximity(coords, [0, 3]).unwrap().map(|x| x.coord()).collect::<Vec<u32>>();
        assert_eq!(
            vec![9, 8, 7, 6, 5, 4, 3, 2, 1],
            result,
            "proximity point is greater than the result set - 10"
        );

        let result = proximity(coords, [1, 0]).unwrap().map(|x| x.coord()).collect::<Vec<u32>>();
        assert_eq!(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
            result,
            "proximity point is lesser than the result set - 1"
        );

        let empty: Vec<u32> = vec![];
        let buffer = flatbuffer_generator(empty.into_iter());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        assert_eq!(proximity(coords, [3, 0]).is_none(), true);

        let sparse: Vec<u32> = vec![24, 21, 13, 8, 7, 6, 1]; // 1 and 13 are the same distance from 7
        let buffer = flatbuffer_generator(sparse.into_iter());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        let result = proximity(coords, [3, 1]).unwrap().map(|x| x.coord()).collect::<Vec<u32>>();
        assert_eq!(
            vec![7, 6, 8, 13, 1, 21, 24],
            result,
            "sparse result set sorted by z-order in the middle of the result set"
        );
    }

    #[test]
    fn bbox_proximity_search() {
        let buffer = flatbuffer_generator((1..10).rev()); // [9,8,7,6,5,4,3,2,1]
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        // bbox is from 1-7; proximity is 4
        let result = bbox_proximity_filter(coords, [1, 0, 3, 1], [2, 0])
            .unwrap()
            .map(|x| x.coord())
            .collect::<Vec<u32>>();
        assert_eq!(
            vec![4, 3, 5, 6, 2, 1, 7],
            result,
            "bbox within the range of coordinates; proximity point within the result set"
        );
    }

    #[test]
    fn binary_search() {
        // Empty Coord list
        let empty: Vec<u32> = vec![];
        let buffer = flatbuffer_generator(empty.into_iter());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        assert_eq!(coord_binary_search(&coords, 0, 0), Err("Offset greater than Vector"));
        assert_eq!(coord_binary_search(&coords, 1, 0), Err("Offset greater than Vector"));

        // Single Coord list
        let single: Vec<u32> = vec![0];
        let buffer = flatbuffer_generator(single.into_iter());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();

        assert_eq!(coord_binary_search(&coords, 0, 0), Ok(0));
        assert_eq!(coord_binary_search(&coords, 1, 0), Ok(0));

        // Continuous Coord list
        let buffer = flatbuffer_generator((4..8).rev()); // [7,6,5,4]
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();

        assert_eq!(coord_binary_search(&coords, 0, 0), Ok(3));
        assert_eq!(coord_binary_search(&coords, 4, 0), Ok(3));
        assert_eq!(coord_binary_search(&coords, 4, 1), Ok(3));
        assert_eq!(coord_binary_search(&coords, 5, 0), Ok(2));
        assert_eq!(coord_binary_search(&coords, 6, 0), Ok(1));
        assert_eq!(coord_binary_search(&coords, 7, 0), Ok(0));
        assert_eq!(coord_binary_search(&coords, 7, 3), Ok(3));
        assert_eq!(coord_binary_search(&coords, 7, 4), Err("Offset greater than Vector"));
        assert_eq!(coord_binary_search(&coords, 8, 0), Ok(0));

        // Sparse Coord list
        let sparse: Vec<u32> = vec![7, 4, 2, 1];
        let buffer = flatbuffer_generator(sparse.into_iter());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();

        assert_eq!(coord_binary_search(&coords, 0, 0), Ok(3));
        assert_eq!(coord_binary_search(&coords, 1, 0), Ok(3));
        assert_eq!(coord_binary_search(&coords, 1, 1), Ok(3));
        assert_eq!(coord_binary_search(&coords, 2, 0), Ok(2));
        assert_eq!(coord_binary_search(&coords, 3, 0), Ok(2));
        assert_eq!(coord_binary_search(&coords, 4, 0), Ok(1));
        assert_eq!(coord_binary_search(&coords, 5, 0), Ok(1));
        assert_eq!(coord_binary_search(&coords, 7, 0), Ok(0));
        assert_eq!(coord_binary_search(&coords, 7, 3), Ok(3));
        assert_eq!(coord_binary_search(&coords, 7, 4), Err("Offset greater than Vector"));
        assert_eq!(coord_binary_search(&coords, 8, 0), Ok(0));
    }
}
