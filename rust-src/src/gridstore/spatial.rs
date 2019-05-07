use flatbuffers;
use crate::gridstore::gridstore_generated::*;
use morton::interleave_morton;

fn bbox_filter<'a>(coords: flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<Coord>>, bbox: (u16, u16, u16, u16)) -> Result<u32, u32> 
{
    let _min = interleave_morton(bbox.0, bbox.1);
    let _max = interleave_morton(bbox.2, bbox.3);

    let mut size = coords.len();

    // untested
    if size == 0 {
        return Err(0);
    } 

    let mut base = 0usize;
    while size > 1 {
    //    let half = size / 2;
    //    let mid =  base + half;
    //    base = if min < 
    }
    println!("{:?}", &size);

    let cmp = coords.get(base);
    println!("{:?}", cmp);

    Ok(42)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn bbox_text() {
        let mut fb_builder = flatbuffers::FlatBufferBuilder::new_with_capacity(256);
        {
            let mut coords: Vec<_> = Vec::new();
            {
                let ids: Vec<u32> = vec![0];
                let fb_ids = fb_builder.create_vector(&ids);
                let fb_coord = Coord::create(&mut fb_builder, &CoordArgs{
                    coord: 1,
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

        // This works fine here, but not in bbox_filter..
        //println!("{:?}", coords.get(0).coord());

        let _result = bbox_filter(coords, (0,0,1,1));
    }
}
