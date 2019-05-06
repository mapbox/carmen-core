use morton::interleave_morton;
fn bbox_filter<T: Iterator<Item=(u32, u32)>>(coords: T, bbox: (u32, u32, u32, u32)) -> impl Iterator<Item=(u32, u32)> {
    coords
}

#[test]
fn bbox_test() {
    let coords = vec![(1,0), (2,0), (3,0), (4,0)];
    let bbox = (0,0,1,1);
    let result = bbox_filter(coords.into_iter(), bbox).collect::<Vec<_>>();
}
