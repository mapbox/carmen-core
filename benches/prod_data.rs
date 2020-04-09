use std::collections::HashSet;

use criterion::{Bencher, Criterion, Benchmark};

use carmen_core::gridstore::*;
use test_utils::*;

pub fn benchmark(c: &mut Criterion) {
    let to_bench = vec![
        ("coalesce_global", "gb_address_pm_global.ljson.lz4"),
        ("coalesce_prox", "gb_address_pm_with_proximity.ljson.lz4"),
        ("coalesce_ac_global", "gb_address_pm_ac_global.ljson.lz4"),
        ("coalesce_ac_prox", "gb_address_pm_ac_with_proximity.ljson.lz4")
    ];

    for (label, file) in to_bench {
        c.bench(
            label,
            Benchmark::new(
                label,
                move |b: &mut Bencher| {
                    let queries = prepare_phrasematches(file);
                    let trees: Vec<_> = queries.into_iter().map(|(query, opts)| (stackable(&query, None, 0, HashSet::new(), 0, 129, 0.0, 0), opts)).collect();

                    let mut cycle = trees.iter().cycle();

                    b.iter(|| {
                        let (tree, opts) = cycle.next().unwrap();
                        tree_coalesce(&tree, &opts).unwrap()
                    })
                }
            ).sample_size(20)
        );
    }
}
