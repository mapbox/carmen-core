#[macro_use]
extern crate criterion;

use criterion::Criterion;

mod prod_data;

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = prod_data::benchmark
}
criterion_main!(benches);
