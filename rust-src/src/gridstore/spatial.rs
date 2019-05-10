use flatbuffers;
use crate::gridstore::gridstore_generated::*;
use morton::interleave_morton;
use std::cmp::Ordering::{Less, Equal, Greater};

pub fn bbox_filter<'a>(coords: flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<Coord<'a>>>, bbox: [u16; 4]) -> Option<impl Iterator<Item=Coord<'a>>> {
    let min = interleave_morton(bbox[0], bbox[1]);
    let max = interleave_morton(bbox[2], bbox[3]);
    debug_assert!(min.cmp(&max) != Greater, "Invalid bounding box");

    let len = coords.len();
    if len == 0 { return None; }

    let range_start = coords.get(0).coord();
    if min > range_start { return None; }
    let range_end = coords.get(len - 1).coord();
    if max < range_end { return None; }
    debug_assert!(range_start.cmp(&range_end) != Less, "Expected descending sort");

    let start = match bbox_binary_search(&coords, max, 0) {
        Ok(v) => v,
        Err(_) => return None,
    };
    let mut end = match bbox_binary_search(&coords, min, start) {
        Ok(v) => v,
        Err(_) => return None,
    };

    if end.cmp(&(len as u32)) == Equal { end -= 1; }
    debug_assert!(start.cmp(&end) != Greater, "Start is before end");

    Some((start..(end + 1)).map(move |idx| coords.get(idx as usize)))
}

/// Binary search this FlatBuffers Coord Vector
///
/// Derived from binary_search_by in core/slice/mod.rs
///
/// If val is found within the range captured by Vector with given offset [`Result::Ok`] is returned, containing the
/// index of the matching element. If the value is less than the first element and greater than the last,
/// [`Result::Err'] is returned containing either 0 or the length of the Vector.
fn bbox_binary_search<'a>(coords: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<Coord>>, val: u32, offset: u32) -> Result<u32, &'a str> {
    let len = coords.len() as u32;

    if offset.cmp(&len) != Less {
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
    if base.cmp(&(len - 1)) == Equal { return Ok(base); }
    let cmp = coords.get(base as usize).coord().cmp(&val);
    if cmp == Equal { Ok(base) } else { Ok(base + (cmp == Greater) as u32 ) }
}

#[cfg(test)]
fn flatbuffer_generator<T: Iterator<Item=u32>>(val: T) -> Vec<u8>{
    let mut fb_builder = flatbuffers::FlatBufferBuilder::new_with_capacity(256);
    let mut coords: Vec<_> = Vec::new();

    let ids: Vec<u32> = vec![0];
    for i in val {
        let fb_ids = fb_builder.create_vector(&ids);
        let fb_coord = Coord::create(&mut fb_builder, &CoordArgs{
            coord: i as u32,
            ids: Some(fb_ids)
        });
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
        assert_eq!(bbox_filter(coords, [0,0,0,0]).is_none(), true);

        let buffer = flatbuffer_generator((0..4).rev());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        let result = bbox_filter(coords, [0,0,1,1]).unwrap().collect::<Vec<Coord>>();
        assert_eq!(result.len(), 4);

        let buffer = flatbuffer_generator((2..4).rev());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        let result = bbox_filter(coords, [0,0,1,1]).unwrap().collect::<Vec<Coord>>();
        assert_eq!(result.len(), 2, "starts before bbox and ends between the result set");

        let buffer = flatbuffer_generator((2..4).rev());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        let result = bbox_filter(coords, [1,1,3,1]).unwrap().collect::<Vec<Coord>>();
        assert_eq!(result.len(), 1, "starts in the bbox and ends after the result set");

        let buffer = flatbuffer_generator((1..4).rev());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        let result = bbox_filter(coords, [0,1,1,1]).unwrap().collect::<Vec<Coord>>();
        assert_eq!(result.len(), 2, "starts in the bbox and ends in the bbox");

        let buffer = flatbuffer_generator((5..7).rev());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        assert_eq!(bbox_filter(coords, [0,0,0,1]).is_none(), true, "bbox ends before the range of coordinates");
        assert_eq!(bbox_filter(coords, [4,0,4,1]).is_none(), true, "bbox starts after the range of coordinates");

        let sparse: Vec<u32> = vec![24, 7];
        let buffer = flatbuffer_generator(sparse.into_iter());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        let result = bbox_filter(coords, [3,1,4,2]).unwrap().collect::<Vec<Coord>>();
        assert_eq!(result.len(), 2, "sparse result set that spans z-order jumps");

        let buffer = flatbuffer_generator((7..24).rev());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        let result = bbox_filter(coords, [3,1,4,2]).unwrap().collect::<Vec<Coord>>();
        assert_eq!(result.len(), 17, "continuous result set that spans z-order jumps"); // TODO this should probably be 2

        let sparse: Vec<u32> = vec![8];
        let buffer = flatbuffer_generator(sparse.into_iter());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        let result = bbox_filter(coords, [3,1,4,2]).unwrap().collect::<Vec<Coord>>();
        assert_eq!(result.len(), 1, "result is on the z-order curve but not in the bbox"); // TODO should return None
    }

    #[test]
    fn binary_search() {
        // TODO
        // - Determine if to return Result and how to handle out of bounds reads

        // Empty Coord list
        let empty: Vec<u32> = vec![];
        let buffer = flatbuffer_generator(empty.into_iter());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        assert_eq!(bbox_binary_search(&coords, 0, 0), Err("Offset greater than Vector"));
        assert_eq!(bbox_binary_search(&coords, 1, 0), Err("Offset greater than Vector"));

        // Single Coord list
        let single: Vec<u32> = vec![0];
        let buffer = flatbuffer_generator(single.into_iter());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();

        assert_eq!(bbox_binary_search(&coords, 0, 0), Ok(0));
        assert_eq!(bbox_binary_search(&coords, 1, 0), Ok(0));

        // Continuous Coord list
        let buffer = flatbuffer_generator((4..8).rev()); // [7,6,5,4]
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();

        assert_eq!(bbox_binary_search(&coords, 0, 0), Ok(3));
        assert_eq!(bbox_binary_search(&coords, 4, 0), Ok(3));
        assert_eq!(bbox_binary_search(&coords, 4, 1), Ok(3));
        assert_eq!(bbox_binary_search(&coords, 5, 0), Ok(2));
        assert_eq!(bbox_binary_search(&coords, 6, 0), Ok(1));
        assert_eq!(bbox_binary_search(&coords, 7, 0), Ok(0));
        assert_eq!(bbox_binary_search(&coords, 7, 3), Ok(3));
        assert_eq!(bbox_binary_search(&coords, 7, 4), Err("Offset greater than Vector")); // Offset is out of bounds
        assert_eq!(bbox_binary_search(&coords, 8, 0), Ok(0)); // Fails to find value, returns closest index

        // Sparse Coord list
        let sparse: Vec<u32> = vec![7,4,2,1];
        let buffer = flatbuffer_generator(sparse.into_iter());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();

        assert_eq!(bbox_binary_search(&coords, 0, 0), Ok(3));
        assert_eq!(bbox_binary_search(&coords, 1, 0), Ok(3));
        assert_eq!(bbox_binary_search(&coords, 1, 1), Ok(3));
        assert_eq!(bbox_binary_search(&coords, 2, 0), Ok(2));
        //assert_eq!(bbox_binary_search(&coords, 3, 0), Ok(2));
        assert_eq!(bbox_binary_search(&coords, 4, 0), Ok(1));
        //assert_eq!(bbox_binary_search(&coords, 5, 0), Ok(3));
        assert_eq!(bbox_binary_search(&coords, 7, 0), Ok(0));
        assert_eq!(bbox_binary_search(&coords, 7, 3), Ok(3));
        assert_eq!(bbox_binary_search(&coords, 7, 4), Err("Offset greater than Vector")); // Offset is out of bounds
        assert_eq!(bbox_binary_search(&coords, 8, 0), Ok(0)); // Fails to find value, returns closest index
    }
}
