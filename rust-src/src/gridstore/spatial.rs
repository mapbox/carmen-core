use flatbuffers;
use crate::gridstore::gridstore_generated::*;
use morton::interleave_morton;
use std::cmp::Ordering::{Less, Equal, Greater};

fn bbox_filter<'a>(coords: &'a flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<Coord>>, bbox: [u16; 4]) -> impl Iterator<Item=Coord<'a>> {
    let min = interleave_morton(bbox[0], bbox[1]);
    let max = interleave_morton(bbox[2], bbox[3]);
    let start = bbox_binary_search(&coords, min, 0).unwrap();
    let end = bbox_binary_search(&coords, max, start).unwrap();
    (start..end).map(move |idx| coords.get(idx as usize))
}

fn bbox_binary_search(coords: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<Coord>>, pos: u32, offset: u32) -> Result<u32, u32> {
    let mut size = coords.len() as u32;

    // untested
    if size == 0 {
        return Err(0);
    }

    let mut base = offset;
    while size > 1 {
        let half = size / 2;
        let mid = base + half;
            let v = coords.get(mid as usize).coord();
        let cmp = v.cmp(&pos);
        base = if cmp == Greater { base } else { mid };
        size -= half;
    }
    let cmp = coords.get(base as usize).coord().cmp(&pos);
    if cmp == Equal { Ok(base) } else { Err(base + (cmp == Less) as u32 ) }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn bbox_text() {
        let mut fb_builder = flatbuffers::FlatBufferBuilder::new_with_capacity(256);
        {
            let mut coords: Vec<_> = Vec::new();
            let ids: Vec<u32> = vec![0; 4];
            for i in 0..ids.len() {
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
        }
        let data = fb_builder.finished_data();

        let rs = flatbuffers::get_root::<RelevScore>(&data);

        let coords = rs.coords().unwrap();
        let result = bbox_filter(&coords, [0,0,1,1]).collect::<Vec<Coord>>();
        assert_eq!(result.len(), 3);
    }
}
