use flatbuffers;
use crate::gridstore::gridstore_generated::*;
use morton::interleave_morton;
use std::cmp::Ordering::{Less, Equal, Greater};

pub fn bbox_filter<'a>(coords: flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<Coord>>, bbox: [u16; 4]) -> impl Iterator<Item=Coord<'a>> {
    let min = interleave_morton(bbox[0], bbox[1]);
    let max = interleave_morton(bbox[2], bbox[3]);
    let start = match bbox_binary_search(&coords, min, 0) {
        Ok(v) => v,
        Err(v) => v,
    };
    let end = match bbox_binary_search(&coords, max, start) {
        Ok(v) => v,
        Err(v) => v,
    };
    assert!(start <= end, "start is before end");
    (start..end).map(move |idx| coords.get(idx as usize))
}

// Essentially a copy paste from core/slice/mod.rs binary_search_by
fn bbox_binary_search(coords: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<Coord>>, pos: u32, offset: u32) -> Result<u32, u32> {
    let mut size = coords.len() as u32;
    size -= offset;

    // untested
    if size == 0 {
        return Err(offset);
    }

    let mut base = offset;
    while size > 1 {
        let half = size / 2;
        let mid = base + half;
        let v = coords.get(mid as usize).coord();
        let cmp = v.cmp(&pos);
        //println!("half {:?}, mid {:?}, v {:?}, cmp {:?}", half, mid, v, cmp);
        base = if cmp == Greater { base } else { mid };
        size -= half;
        //println!("base {:?}, size {:?}", base, size);
    }
    let cmp = coords.get(base as usize).coord().cmp(&pos);
    if cmp == Equal { Ok(base) } else { Err(base + (cmp == Less) as u32 ) }
}


fn flatbuffer_generator<'a>(min: u32, max: u32) -> Vec<u8> {
    let mut fb_builder = flatbuffers::FlatBufferBuilder::new_with_capacity(256);
    let mut coords: Vec<_> = Vec::new();

    let ids: Vec<u32> = vec![min; max as usize];
    for i in min..max {
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
    // move the generator into a helper -- should take an iterator and generate the flatbuffer, also takes min max and number of entries
    // case 1: when size is zero iterator over an empty vector
    // case 2: when the bbox is before the points should return iterator over an empty vector
    // case 3: when bbox is after the points should return iterator over an empty vector
    // case 4: when the z-order leaves the bbox should be captured (right now it's filtered out at the end)
    // case 5: when all the points are in the bbox
    // case 5: when bbox starts in the middle of the result set and ends beyond
    // case 6: when the bbox starts and ends in the middle of the result set
    // case 7: when it starts before the result set and ends in between
    // case 8: variation of case 4 where the z-order leaves but the bbox contains points to be returned
    use super::*;

    #[test]
    fn coords_within_bbox() {
        let buffer = flatbuffer_generator(0, 4);
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();
        let result = bbox_filter(coords, [0,0,1,1]).collect::<Vec<Coord>>();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn binary_search() {
        let buffer = flatbuffer_generator(4, 8); // [4,5,6,7]
        let rs = flatbuffers::get_root::<RelevScore>(&buffer);
        let coords = rs.coords().unwrap();

        let r = bbox_binary_search(&coords, 0, 0);
        assert_eq!(r, Err(0)); // locates first element

        let r = bbox_binary_search(&coords, 4, 0);
        assert_eq!(r, Ok(0)); // locates first value

        let r = bbox_binary_search(&coords, 4, 1);
        assert_eq!(r, Err(1)); // locates first element for given offset.

        let r = bbox_binary_search(&coords, 5, 0);
        assert_eq!(r, Ok(1));

        let r = bbox_binary_search(&coords, 6, 0);
        assert_eq!(r, Ok(2));

        let r = bbox_binary_search(&coords, 7, 0);
        assert_eq!(r, Ok(3));

        let r = bbox_binary_search(&coords, 7, 3);
        assert_eq!(r, Ok(3));

        let r = bbox_binary_search(&coords, 7, 4);
        assert_eq!(r, Err(4)); // Offset is out of bounds

        let r = bbox_binary_search(&coords, 8, 0);
        assert_eq!(r, Err(4)); // Fails to find value, returns closes pos, the end
    }
}
