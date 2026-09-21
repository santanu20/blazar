//! Test stub for `llama-quantize`: copies argv[1] to argv[2], printing
//! a couple of progress lines. `BLAZAR_STUB_QUANTIZE_FAIL=1` makes it
//! exit 1 with an error line instead (failure-path tests).

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: stub-llama-quantize <src> <dst> <type>");
        std::process::exit(2);
    }
    if std::env::var("BLAZAR_STUB_QUANTIZE_FAIL").is_ok() {
        eprintln!("error: unknown quantization type");
        std::process::exit(1);
    }
    std::fs::copy(&args[1], &args[2]).expect("copy");
    println!("[ 1/10] quantizing tensor 0 ...");
    println!("success: wrote {}", args[2]);
}
