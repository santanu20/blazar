use blazar_core::catalog::levenshtein;
use blazar_core::profile::path_safe;
use iai_callgrind::{library_benchmark, library_benchmark_group, main};

#[library_benchmark]
#[bench::typo("qwen3.5:8b-instruct", "qwen3 5:8b-instruct")]
#[bench::divergent("deepseek-r1-distill-qwen-32b", "llama3.3-70b-instruct")]
fn bench_levenshtein(a: &str, b: &str) -> usize {
    levenshtein(a, b)
}

#[library_benchmark]
#[bench::model_tag("qwen/qwen3.5-8b-instruct")]
#[bench::lora_tag("meta/llama-3.3-70b:lora-esperanto")]
fn bench_path_safe(name: &str) -> String {
    path_safe(name)
}

library_benchmark_group!(
    name = name_matching;
    benchmarks = bench_levenshtein
);

library_benchmark_group!(
    name = cache_keys;
    benchmarks = bench_path_safe
);

main!(library_benchmark_groups = name_matching, cache_keys);
