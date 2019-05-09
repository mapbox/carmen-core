use flatbuffers;
use crate::gridstore::gridstore_generated::*;
use morton::interleave_morton;
use std::cmp::Ordering::{Less, Equal, Greater};

pub fn bbox_filter<'a>(coords: flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<Coord>>, bbox: [u16; 4]) -> Option<impl Iterator<Item=Coord<'a>>> {
    let min = interleave_morton(bbox[0], bbox[1]);
    let max = interleave_morton(bbox[2], bbox[3]);
    debug_assert!(min.cmp(&max) != Greater, "Invalid bounding box");

    let len = coords.len();
    if len == 0 { return None; }
    let range_min = coords.get(0).coord();
    if max < range_min { return None; }
    let range_max = coords.get(len - 1).coord();
    if min > range_max { return None; }

    let start = match bbox_binary_search(&coords, min, 0) {
        Ok(v) => v,
        Err(v) => v,
    };
    let mut end = match bbox_binary_search(&coords, max, start) {
        Ok(v) => v,
        Err(v) => v,
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
fn bbox_binary_search(coords: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<Coord>>, val: u32, offset: u32) -> Result<u32, u32> {
    let mut size = coords.len() as u32;
    assert!(size.cmp(&offset) != Less, "Offset is larger than Vector");
    size -= offset;

    if size == 0 {
        return Err(offset);
    }

    let mut base = offset;
    while size > 1 {
        let half = size / 2;
        let mid = base + half;
        let v = coords.get(mid as usize).coord();
        let cmp = v.cmp(&val);
        base = if cmp == Greater { base } else { mid };
        size -= half;
    }
    let cmp = coords.get(base as usize).coord().cmp(&val);
    if cmp == Equal { Ok(base) } else { Err(base + (cmp == Less) as u32 ) }
}

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

#[cfg(test)]
mod test {
    // TO DO:
    // case 5: when all the points are in the bbox
    // case 5: when bbox starts in the middle of the result set and ends beyond
    // case 6: when the bbox starts and ends in the middle of the result set
    // case 7: when it starts before the result set and ends in between
    use super::*;

    #[test]
    fn filter_bbox() {
        let empty: Vec<u32> = vec![];
        let buffer = flatbuffer_generator(empty.into_iter());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        assert_eq!(bbox_filter(coords, [0,0,0,0]).is_none(), true);

        let buffer = flatbuffer_generator(0..4);
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        let result = bbox_filter(coords, [0,0,1,1]).unwrap().collect::<Vec<Coord>>();
        assert_eq!(result.len(), 4);

        let buffer = flatbuffer_generator(2..4);
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        let result = bbox_filter(coords, [0,0,1,1]).unwrap().collect::<Vec<Coord>>();
        assert_eq!(result.len(), 3, "starts before bbox and ends between the result set");

        let buffer = flatbuffer_generator(5..7);
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        assert_eq!(bbox_filter(coords, [0,0,0,1]).is_none(), true, "bbox ends before the range of coordinates");
        assert_eq!(bbox_filter(coords, [4,0,4,1]).is_none(), true, "bbox starts after the range of coordinates");

        let sparse: Vec<u32> = vec![7, 24];
        let buffer = flatbuffer_generator(sparse.into_iter());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        let result = bbox_filter(coords, [3,1,4,2]).unwrap().collect::<Vec<Coord>>();
        assert_eq!(result.len(), 2, "sparse result set that spans z-order jumps");

        let buffer = flatbuffer_generator(7..24);
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
        assert_eq!(bbox_binary_search(&coords, 0, 0), Err(0));
        assert_eq!(bbox_binary_search(&coords, 1, 0), Err(0));

        // Single Coord list
        let single: Vec<u32> = vec![0];
        let buffer = flatbuffer_generator(single.into_iter());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();

        assert_eq!(bbox_binary_search(&coords, 0, 0), Ok(0));
        assert_eq!(bbox_binary_search(&coords, 1, 0), Err(1));

        // Continuous Coord list
        let buffer = flatbuffer_generator(4..8); // [4,5,6,7]
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();

        assert_eq!(bbox_binary_search(&coords, 0, 0), Err(0));
        assert_eq!(bbox_binary_search(&coords, 4, 0), Ok(0));
        assert_eq!(bbox_binary_search(&coords, 4, 1), Err(1));
        assert_eq!(bbox_binary_search(&coords, 5, 0), Ok(1));
        assert_eq!(bbox_binary_search(&coords, 6, 0), Ok(2));
        assert_eq!(bbox_binary_search(&coords, 7, 0), Ok(3));
        assert_eq!(bbox_binary_search(&coords, 7, 3), Ok(3));
        assert_eq!(bbox_binary_search(&coords, 7, 4), Err(4)); // Offset is out of bounds
        assert_eq!(bbox_binary_search(&coords, 8, 0), Err(4)); // Fails to find value, returns closes pos, the end

        // Sparse Coord list
        let sparse: Vec<u32> = vec![1,2,4,7];
        let buffer = flatbuffer_generator(sparse.into_iter());
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();

        assert_eq!(bbox_binary_search(&coords, 0, 0), Err(0));
        assert_eq!(bbox_binary_search(&coords, 1, 0), Ok(0));
        assert_eq!(bbox_binary_search(&coords, 1, 1), Err(1));
        assert_eq!(bbox_binary_search(&coords, 2, 0), Ok(1));
        //assert_eq!(bbox_binary_search(&coords, 3, 0), Ok(2));
        assert_eq!(bbox_binary_search(&coords, 4, 0), Ok(2));
        //assert_eq!(bbox_binary_search(&coords, 5, 0), Ok(3));
        assert_eq!(bbox_binary_search(&coords, 7, 0), Ok(3));
        assert_eq!(bbox_binary_search(&coords, 7, 3), Ok(3));
        assert_eq!(bbox_binary_search(&coords, 7, 4), Err(4)); // Offset is out of bounds
        assert_eq!(bbox_binary_search(&coords, 8, 0), Err(4)); // Fails to find value, returns closes pos, the end
    }
}
