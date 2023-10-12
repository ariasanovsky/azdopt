#![allow(unused)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn r44_17_search_tree_with_trivial_predictions(c: &mut Criterion) {
    let mut group = c.benchmark_group("r44_17_search_tree_with_trivial_predictions");
    for n in [
        10_000i32,
        100_000,
        1_000_000,
        10_000_000,
        100_000_000,
        1_000_000_000,
    ] {
        group.bench_with_input(BenchmarkId::new("perform a search with a given length and ...?", n), &n, |b, &n| {
            b.iter(|| {
                // construct a root
                // construct a search tree
                // perform a search with a given length and ...?
                // calculate the observations
                // maybe loop the search -- root a new tree at a chosesn value from the search and repeat
                black_box(todo!("put some output here"))
            })
        });
    }
}

criterion_group!(benches, r44_17_search_tree_with_trivial_predictions);
criterion_main!(benches);
