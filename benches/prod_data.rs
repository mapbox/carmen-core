use std::collections::HashSet;

use criterion::{Bencher, Benchmark, Criterion};

use carmen_core::gridstore::*;
use test_utils::*;

pub fn benchmark(c: &mut Criterion) {
    let to_bench = vec![
        ("coalesce_global", "gb_address_pm_global.ljson.lz4"),
        ("coalesce_prox", "gb_address_pm_with_proximity.ljson.lz4"),
        ("coalesce_ac_global", "gb_address_pm_ac_global.ljson.lz4"),
        ("coalesce_ac_prox", "gb_address_pm_ac_with_proximity.ljson.lz4"),
    ];

    for (label, file) in to_bench {
        c.bench(
            label,
            Benchmark::new(label, move |b: &mut Bencher| {
                let queries = prepare_phrasematches(file);
                let collapsed: Vec<_> = queries
                    .into_iter()
                    .map(|(query, opts)| (collapse_phrasematches(query), opts))
                    .collect();
                let trees: Vec<_> = collapsed
                    .iter()
                    .map(|(query, opts)| {
                        (stackable(query, None, 0, HashSet::new(), 0, 129, 0.0, 0), opts)
                    })
                    .collect();

                let mut cycle = trees.iter().cycle();

                b.iter(|| {
                    let (tree, opts) = cycle.next().unwrap();
                    tree_coalesce(&tree, &opts).unwrap()
                })
            })
            .sample_size(20),
        );
    }

    let to_bench = vec![
        ("stackable_us_address", "us-address-forward.ljson.lz4"),
        ("stackable_us_address_postcode", "us-address-postcode.ljson.lz4"),
        (
            "stackable_us-address_postcode_place_region",
            "us-address-postcode-place-region.ljson.lz4",
        ),
    ];

    for (label, file) in to_bench {
        c.bench(
            label,
            Benchmark::new(label, move |b: &mut Bencher| {
                let queries = prepare_stackable_phrasematches(file);
                let collapsed: Vec<_> =
                    queries.iter().map(|q| collapse_phrasematches(q.to_vec())).collect();
                let binned_phrasematches: Vec<_> = collapsed
                    .into_iter()
                    .map(|b| binned_phrasematches(b).into_iter().map(|(_k, v)| v).collect())
                    .collect();

                let mut cycle = binned_phrasematches.iter().cycle();

                b.iter(|| {
                    let pm = cycle.next().unwrap();
                    binned_stackable(&pm, None, HashSet::new(), 0, 129, 0.0, 0, 0)
                })
            })
            .sample_size(2),
        );
    }
}
