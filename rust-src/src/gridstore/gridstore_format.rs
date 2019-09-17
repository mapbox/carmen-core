use std::convert::TryInto;
use std::marker::PhantomData;

use integer_encoding::VarInt;

#[derive(Copy, Clone)]
pub struct VarScalarOffset<T: VarEncodable> {
    addr: usize,
    phantom: PhantomData<T>,
}

impl<T: VarEncodable> VarScalarOffset<T> {
    fn new(addr: usize) -> Self {
        Self { addr, phantom: PhantomData }
    }
}

#[derive(Copy, Clone)]
pub struct FixedScalarOffset<T: FixedEncodable> {
    addr: usize,
    phantom: PhantomData<T>,
}

impl<T: FixedEncodable> FixedScalarOffset<T> {
    fn new(addr: usize) -> Self {
        Self { addr, phantom: PhantomData }
    }
}

#[derive(Copy, Clone)]
pub struct UniformScalarOffset<T: UniformEncodable> {
    addr: usize,
    phantom: PhantomData<T>,
}

impl<T: UniformEncodable> UniformScalarOffset<T> {
    fn new(addr: usize) -> Self {
        Self { addr, phantom: PhantomData }
    }
}

#[derive(Copy, Clone)]
pub struct FixedVecOffset<T: FixedEncodable> {
    addr: usize,
    phantom: PhantomData<T>,
}

impl<T: FixedEncodable> FixedVecOffset<T> {
    fn new(addr: usize) -> Self {
        Self { addr, phantom: PhantomData }
    }

    fn from_fixed_pointer(data: &[u8], offset: usize) -> Self {
        let ptr = u32::from_le_bytes(data[offset..(offset + 4)].try_into().unwrap());
        Self::new(ptr as usize)
    }

    fn from_var_pointer(data: &[u8], offset: usize) -> (Self, usize) {
        let (ptr, len_len) = u32::decode_var(&data[offset..]);
        (Self::new(ptr as usize), len_len)
    }
}

#[derive(Copy, Clone)]
pub struct VarVecOffset<T: VarEncodable> {
    addr: usize,
    phantom: PhantomData<T>,
}

impl<T: VarEncodable> VarVecOffset<T> {
    fn new(addr: usize) -> Self {
        Self { addr, phantom: PhantomData }
    }

    fn from_fixed_pointer(data: &[u8], offset: usize) -> Self {
        let ptr = u32::from_le_bytes(data[offset..(offset + 4)].try_into().unwrap());
        Self::new(ptr as usize)
    }

    fn from_var_pointer(data: &[u8], offset: usize) -> (Self, usize) {
        let (ptr, len_len) = u32::decode_var(&data[offset..]);
        (Self::new(ptr as usize), len_len)
    }
}

#[derive(Copy, Clone)]
pub struct UniformVecOffset<T: UniformEncodable> {
    addr: usize,
    phantom: PhantomData<T>,
}

impl<T: UniformEncodable> UniformVecOffset<T> {
    fn new(addr: usize) -> Self {
        Self { addr, phantom: PhantomData }
    }

    fn from_fixed_pointer(data: &[u8], offset: usize) -> Self {
        let ptr = u32::from_le_bytes(data[offset..(offset + 4)].try_into().unwrap());
        Self::new(ptr as usize)
    }

    fn from_var_pointer(data: &[u8], offset: usize) -> (Self, usize) {
        let (ptr, len_len) = u32::decode_var(&data[offset..]);
        (Self::new(ptr as usize), len_len)
    }
}

pub trait VarEncodable: Sized {
    fn write_to(&self, buffer: &mut Vec<u8>) -> usize;
    fn read_from(buffer: &[u8], offset: VarScalarOffset<Self>) -> (Self, usize);
}

pub trait FixedEncodable: Sized {
    const SIZE: usize;
    fn write_fixed_to(&self, buffer: &mut Vec<u8>) -> ();
    fn read_fixed_from(buffer: &[u8], offset: FixedScalarOffset<Self>) -> Self;
}

pub trait UniformEncodable: Sized {
    const MAX_SIZE: usize;
    fn write_with_size_to(&self, size: usize, buffer: &mut Vec<u8>) -> ();
    fn read_with_size_from(buffer: &[u8], size: usize, offset: UniformScalarOffset<Self>) -> Self;
    fn get_min_size(&self) -> usize;
}

pub struct Writer {
    data: Vec<u8>,
}

impl Writer {
    pub fn new() -> Self {
        Writer { data: Vec::new() }
    }

    pub fn write_var_scalar<T: VarEncodable>(&mut self, s: T) -> VarScalarOffset<T> {
        let loc = self.data.len();
        s.write_to(&mut self.data);
        VarScalarOffset::new(loc)
    }

    pub fn write_fixed_scalar<T: FixedEncodable>(&mut self, s: T) -> FixedScalarOffset<T> {
        let loc = self.data.len();
        s.write_fixed_to(&mut self.data);
        FixedScalarOffset::new(loc)
    }

    pub fn write_uniform_scalar_with_size<T: UniformEncodable>(
        &mut self,
        s: T,
        size: usize,
    ) -> UniformScalarOffset<T> {
        let loc = self.data.len();
        s.write_with_size_to(size, &mut self.data);
        UniformScalarOffset::new(loc)
    }

    pub fn write_var_vec<T: VarEncodable>(&mut self, s: &[T]) -> VarVecOffset<T> {
        let loc = self.data.len();
        let mut len_buf = [0u8; 8];
        let len_len = (s.len() as u32).encode_var(&mut len_buf);
        self.data.extend_from_slice(&len_buf[..len_len]);
        for item in s {
            item.write_to(&mut self.data);
        }
        VarVecOffset::new(loc)
    }

    pub fn write_fixed_vec<T: FixedEncodable>(&mut self, s: &[T]) -> FixedVecOffset<T> {
        let loc = self.data.len();
        let mut len_buf = [0u8; 8];
        let len_len = (s.len() as u32).encode_var(&mut len_buf);
        self.data.extend_from_slice(&len_buf[..len_len]);
        for item in s {
            item.write_fixed_to(&mut self.data);
        }
        FixedVecOffset::new(loc)
    }

    pub fn write_uniform_vec<T: UniformEncodable>(&mut self, s: &[T]) -> UniformVecOffset<T> {
        let loc = self.data.len();
        let mut len_buf = [0u8; 8];
        let len_len = (s.len() as u32).encode_var(&mut len_buf);
        self.data.extend_from_slice(&len_buf[..len_len]);
        let rec_size: usize = s.iter().map(|obj| obj.get_min_size()).max().unwrap_or(255);
        debug_assert!(rec_size <= 255);
        self.data.push(rec_size as u8);
        for item in s {
            item.write_with_size_to(rec_size, &mut self.data);
        }
        UniformVecOffset::new(loc)
    }

    pub fn finish(self) -> Vec<u8> {
        self.data
    }
}

pub struct Reader<U: AsRef<[u8]>> {
    data: U,
}

impl<U: AsRef<[u8]>> Reader<U> {
    pub fn new(data: U) -> Self {
        Reader { data }
    }

    pub fn read_fixed_scalar<'a, T: FixedEncodable>(&'a self, offset: FixedScalarOffset<T>) -> T {
        T::read_fixed_from(self.data.as_ref(), offset)
    }

    pub fn read_var_scalar<'a, T: VarEncodable>(
        &'a self,
        offset: VarScalarOffset<T>,
    ) -> (T, usize) {
        T::read_from(self.data.as_ref(), offset)
    }

    pub fn read_uniform_scalar<'a, T: UniformEncodable>(
        &'a self,
        size: usize,
        offset: UniformScalarOffset<T>,
    ) -> T {
        T::read_with_size_from(self.data.as_ref(), size, offset)
    }

    pub fn read_fixed_vec<'a, T: FixedEncodable>(
        &'a self,
        offset: FixedVecOffset<T>,
    ) -> FixedVec<'a, T> {
        FixedVec::new(self.data.as_ref(), offset)
    }

    pub fn read_var_vec<'a, T: VarEncodable>(&'a self, offset: VarVecOffset<T>) -> VarVec<'a, T> {
        VarVec::new(self.data.as_ref(), offset)
    }

    pub fn read_uniform_vec<'a, T: UniformEncodable>(
        &'a self,
        offset: UniformVecOffset<T>,
    ) -> UniformVec<'a, T> {
        UniformVec::new(self.data.as_ref(), offset)
    }

    pub fn read_root<'a, T: FixedEncodable>(&'a self) -> T {
        let offset = FixedScalarOffset::new(self.data.as_ref().len() - T::SIZE);
        self.read_fixed_scalar(offset)
    }
}

pub fn read_fixed_vec_raw<'a, T: FixedEncodable>(
    buffer: &'a [u8],
    offset: FixedVecOffset<T>,
) -> FixedVec<'a, T> {
    FixedVec::new(buffer, offset)
}

pub fn read_var_vec_raw<'a, T: VarEncodable>(
    buffer: &'a [u8],
    offset: VarVecOffset<T>,
) -> VarVec<'a, T> {
    VarVec::new(buffer, offset)
}

pub fn read_uniform_vec_raw<'a, T: UniformEncodable>(
    buffer: &'a [u8],
    offset: UniformVecOffset<T>,
) -> UniformVec<'a, T> {
    UniformVec::new(buffer, offset)
}

#[derive(Copy, Clone)]
pub struct FixedVec<'a, T> {
    data: &'a [u8],
    start: usize,
    len: usize,
    phantom: PhantomData<&'a T>,
}

impl<'a, T: FixedEncodable> FixedVec<'a, T> {
    pub fn new(data: &'a [u8], offset: FixedVecOffset<T>) -> Self {
        let (len, len_len) = u32::decode_var(&data[(offset.addr as usize)..]);
        let start = (offset.addr + len_len) as usize;
        FixedVec { data, start, len: len.try_into().unwrap(), phantom: PhantomData }
    }

    pub fn get(&self, pos: usize) -> T {
        let offset = self.start + (pos * T::SIZE);
        T::read_fixed_from(self.data, FixedScalarOffset::new(offset))
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        (0..self.len).map(move |idx| self.get(idx))
    }

    pub fn into_iter(self) -> impl Iterator<Item = T> + 'a {
        (0..self.len).map(move |idx| self.get(idx))
    }
}

#[derive(Copy, Clone)]
pub struct VarVec<'a, T> {
    data: &'a [u8],
    start: usize,
    len: usize,
    phantom: PhantomData<&'a T>,
}

impl<'a, T: VarEncodable> VarVec<'a, T> {
    pub fn new(data: &'a [u8], offset: VarVecOffset<T>) -> Self {
        let (len, len_len) = u32::decode_var(&data[(offset.addr as usize)..]);
        let start = (offset.addr + len_len) as usize;
        VarVec { data, start, len: len.try_into().unwrap(), phantom: PhantomData }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        let mut loc: usize = self.start;
        let mut i: usize = 0;
        std::iter::from_fn(move || {
            if i < self.len {
                let (val, incr) = T::read_from(self.data, VarScalarOffset::new(loc));
                i += 1;
                loc += incr;
                Some(val)
            } else {
                None
            }
        })
    }

    pub fn into_iter(self) -> impl Iterator<Item = T> + 'a {
        let mut loc: usize = self.start;
        let mut i: usize = 0;
        std::iter::from_fn(move || {
            if i < self.len {
                let (val, incr) = T::read_from(self.data, VarScalarOffset::new(loc));
                i += 1;
                loc += incr;
                Some(val)
            } else {
                None
            }
        })
    }
}

#[derive(Copy, Clone)]
pub struct UniformVec<'a, T> {
    data: &'a [u8],
    start: usize,
    rec_size: usize,
    len: usize,
    phantom: PhantomData<&'a T>,
}

impl<'a, T: UniformEncodable> UniformVec<'a, T> {
    pub fn new(data: &'a [u8], offset: UniformVecOffset<T>) -> Self {
        let (len, len_len) = u32::decode_var(&data[(offset.addr as usize)..]);
        let rec_size = data[offset.addr + len_len] as usize;
        let start = (offset.addr + len_len + 1) as usize;
        UniformVec { data, start, rec_size, len: len.try_into().unwrap(), phantom: PhantomData }
    }

    pub fn get(&self, pos: usize) -> T {
        let offset = self.start + (pos * self.rec_size);
        T::read_with_size_from(self.data, self.rec_size, UniformScalarOffset::new(offset))
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        (0..self.len).map(move |idx| self.get(idx))
    }

    pub fn into_iter(self) -> impl Iterator<Item = T> + 'a {
        (0..self.len).map(move |idx| self.get(idx))
    }
}

pub struct RelevScore {
    pub relev_score: u8,
    pub coords: UniformVecOffset<Coord>,
}

impl VarEncodable for RelevScore {
    fn write_to(&self, buffer: &mut Vec<u8>) -> usize {
        buffer.push(self.relev_score);
        let mut addr_buf = [0u8; 8];
        let addr_len = (self.coords.addr as u32).encode_var(&mut addr_buf);
        buffer.extend_from_slice(&addr_buf[..addr_len]);
        1 + addr_len
    }

    fn read_from(buffer: &[u8], offset: VarScalarOffset<Self>) -> (Self, usize) {
        let relev_score = buffer[offset.addr];
        let (coords, addr_len) = UniformVecOffset::from_var_pointer(buffer, offset.addr + 1);
        (RelevScore { relev_score, coords }, 1 + addr_len)
    }
}

#[derive(Copy, Clone)]
pub struct Coord {
    pub coord: u32,
    pub ids: FixedVecOffset<u32>,
}

impl UniformEncodable for Coord {
    const MAX_SIZE: usize = 8;
    fn get_min_size(&self) -> usize {
        match self.ids.addr {
            0..=255 => 4 + 1,
            256..=65535 => 4 + 2,
            65536..=16777215 => 4 + 3,
            _ => 4 + 4,
        }
    }

    fn write_with_size_to(&self, size: usize, buffer: &mut Vec<u8>) {
        buffer.extend_from_slice(&(self.coord as u32).to_le_bytes());
        buffer.extend_from_slice(&(self.ids.addr as u32).to_le_bytes()[..(size - 4)]);
    }

    fn read_with_size_from(buffer: &[u8], size: usize, offset: UniformScalarOffset<Self>) -> Self {
        let coord = u32::from_le_bytes(buffer[offset.addr..(offset.addr + 4)].try_into().unwrap());
        let ptr_size = size - 4;
        let mut ptr_buf = [0u8; 4];
        ptr_buf[..ptr_size]
            .clone_from_slice(&buffer[(offset.addr + 4)..(offset.addr + 4 + ptr_size)]);
        let ptr = u32::from_le_bytes(ptr_buf);
        let ids = FixedVecOffset::<u32>::new(ptr as usize);
        Coord { coord, ids }
    }
}

impl FixedEncodable for u32 {
    const SIZE: usize = 4;
    fn write_fixed_to(&self, buffer: &mut Vec<u8>) {
        buffer.extend_from_slice(&self.to_le_bytes());
    }

    fn read_fixed_from(buffer: &[u8], offset: FixedScalarOffset<Self>) -> Self {
        u32::from_le_bytes(buffer[(offset.addr as usize)..offset.addr + 4].try_into().unwrap())
    }
}

pub struct PhraseRecord {
    pub relev_scores: VarVecOffset<RelevScore>,
}

pub fn read_phrase_record_from<U: AsRef<[u8]>>(reader: &Reader<U>) -> PhraseRecord {
    reader.read_root()
}

impl FixedEncodable for PhraseRecord {
    const SIZE: usize = 4;

    fn write_fixed_to(&self, buffer: &mut Vec<u8>) {
        buffer.extend_from_slice(&(self.relev_scores.addr as u32).to_le_bytes());
    }

    fn read_fixed_from(buffer: &[u8], offset: FixedScalarOffset<Self>) -> Self {
        let relev_scores = VarVecOffset::from_fixed_pointer(buffer, offset.addr);
        PhraseRecord { relev_scores }
    }
}

#[cfg(test)]
use itertools::Itertools;

#[test]
fn test_write() {
    #[derive(Debug, PartialOrd, PartialEq, Ord, Eq, Clone)]
    struct Grid {
        relev_score: u8,
        coord: u32,
        id: u32,
    }

    let mut grids = vec![
        Grid { relev_score: 50, coord: 421, id: 1235 },
        Grid { relev_score: 50, coord: 421, id: 1239 },
        Grid { relev_score: 50, coord: 842, id: 12 },
        Grid { relev_score: 100, coord: 42, id: 1235 },
        Grid { relev_score: 210, coord: 15, id: 12835 },
        Grid { relev_score: 250, coord: 842, id: 12 },
        Grid { relev_score: 106, coord: 2, id: 8 },
        Grid { relev_score: 210, coord: 15, id: 636 },
        Grid { relev_score: 250, coord: 8420, id: 1 },
        Grid { relev_score: 106, coord: 2, id: 8 },
    ];

    grids.sort_by(|a, b| b.cmp(&a));

    let mut writer = Writer::new();

    let mut rses = Vec::new();
    for (relev_score, rs_group) in &(&grids).into_iter().group_by(|g| g.relev_score) {
        let mut coords = Vec::new();
        for (coord, coord_group) in &rs_group.into_iter().group_by(|g| g.coord) {
            let ids: Vec<_> = coord_group.into_iter().map(|g| g.id).dedup().collect();
            let w_ids = writer.write_fixed_vec(&ids);
            coords.push(Coord { coord, ids: w_ids });
        }
        let w_coords = writer.write_uniform_vec(&coords);
        rses.push(RelevScore { relev_score, coords: w_coords });
    }
    let w_rses = writer.write_var_vec(&rses);

    let record = PhraseRecord { relev_scores: w_rses };
    writer.write_fixed_scalar(record);

    let reader = Reader::new(writer.data);
    let r_reader = read_phrase_record_from(&reader);

    let mut out_grids = Vec::new();
    for rs in reader.read_var_vec(r_reader.relev_scores).iter() {
        for coord in reader.read_uniform_vec(rs.coords).iter() {
            for id in reader.read_fixed_vec(coord.ids).iter() {
                out_grids.push(Grid { relev_score: rs.relev_score, coord: coord.coord, id })
            }
        }
    }

    let deduped_grids: Vec<_> = grids.iter().cloned().dedup().collect();
    assert_eq!(deduped_grids, out_grids);
}
