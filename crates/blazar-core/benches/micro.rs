//! Micro-benchmarks for the hot catalog paths: model-name matching and
//! cache-key derivation. Run with `cargo bench -p blazar-core --bench micro`.

use std::hint::black_box;

use blazar_core::catalog::levenshtein;
use blazar_core::profile::path_safe;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

fn name_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("name_matching");
    for (case, pair) in [
        ("typo", ("qwen3.5:8b-instruct", "qwen3 5:8b-instruct")),
        (
            "divergent",
            ("deepseek-r1-distill-qwen-32b", "llama3.3-70b-instruct"),
        ),
    ] {
        group.bench_with_input(
            BenchmarkId::from_parameter(case),
            &pair,
            |bencher, &(a, b)| bencher.iter(|| black_box(levenshtein(a, b))),
        );
    }
    group.finish();
}

fn cache_keys(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_keys");
    for (case, name) in [
        ("model_tag", "qwen/qwen3.5-8b-instruct"),
        ("lora_tag", "meta/llama-3.3-70b:lora-esperanto"),
    ] {
        group.bench_with_input(BenchmarkId::from_parameter(case), &name, |bencher, &tag| {
            bencher.iter(|| black_box(path_safe(tag)));
        });
    }
    group.finish();
}

criterion_group!(benches, name_matching, cache_keys);
criterion_main!(benches);
