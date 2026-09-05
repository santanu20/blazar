//! stub-llama-bench: deterministic test double for llama-bench.
//!
//! Emits the same JSON shape as `llama-bench -o json` and expands
//! comma-list grid args (`-p`/`-n`/`-c`/`-t`/`-ctk`/`-ctv`/`-fa`) into one result row
//! per combination, like the real tool. Throughput values are a
//! deterministic function of the config so `tune --search` tests can
//! predict the winner:
//!   `tg` `t/s` = `100 + threads + 2*ctx_bonus + 50*(k/v quantized)`
//!   `pp` `t/s` = `10 * tg`
//! `ctx_bonus`: any non-base ctx scores +3 `t/s` (predictable winner).

use std::io::Write as _;

/// Defaults mirror llama-bench so plain `-p`/`-n` runs still emit rows.
fn or_default(v: Vec<String>, d: &[&str]) -> Vec<String> {
    if v.is_empty() {
        d.iter().map(|s| (*s).to_string()).collect()
    } else {
        v
    }
}

#[allow(clippy::similar_names)] // cs/ts/ctks/ctvs mirror llama-bench flag names
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Ok(path) = std::env::var("STUB_ARGV_FILE") {
        let json = serde_json::to_string(&args).unwrap();
        if let Some(dir) = std::path::Path::new(&path).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(&path, json);
    }
    if args.iter().any(|a| a == "--version" || a == "-v") {
        println!("version: 9999 (deadbeef0000deadbeef0000deadbeef0000dead)");
        println!("built with cc (Linux) for Linux x64");
        return;
    }

    let flag_values = |name: &str| -> Vec<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .map(|v| v.split(',').map(str::to_string).collect())
            .unwrap_or_default()
    };

    let ps = or_default(flag_values("-p"), &["512"]);
    let ns = or_default(flag_values("-n"), &["128"]);
    let cs = or_default(flag_values("-c"), &["4096"]);
    let ts = or_default(flag_values("-t"), &["1"]);
    let ctks = or_default(flag_values("-ctk"), &["f16"]);
    let ctvs = or_default(flag_values("-ctv"), &["f16"]);
    let fas = or_default(flag_values("-fa"), &["on"]);
    let bs = or_default(flag_values("-b"), &["2048"]);

    let mut rows: Vec<serde_json::Value> = Vec::new();
    let default_ctx = std::env::var("STUB_BASE_CTX").ok().and_then(|v| v.parse::<u64>().ok());
    for c in &cs {
        let ctx: u64 = c.parse().unwrap_or(0);
        for t in &ts {
            let threads: u64 = t.parse().unwrap_or(1);
            for ctk in &ctks {
                for ctv in &ctvs {
                    let quant = ctk == "q8_0" && ctv == "q8_0";
                    for fa in &fas {
                        for b in &bs {
                            let batch: u64 = b.parse().unwrap_or(2048);
                            for n in &ns {
                                let ctx_bonus = match default_ctx {
                                    Some(base) if ctx == base => 0,
                                    Some(_) => 3,
                                    None => 0,
                                };
                                // Deterministic scoring: quant +50, fa-on
                                // +15, batch 1024 +8, +threads.
                                let fa_bonus = if fa == "on" { 15.0 } else { 0.0 };
                                let b_bonus = if batch == 1024 { 8.0 } else { 0.0 };
                                #[allow(clippy::cast_precision_loss)]
                                let tg = 100.0 + threads as f64 + f64::from(ctx_bonus)
                                    + f64::from(u8::from(quant)) * 50.0 + fa_bonus + b_bonus;
                                let fa_code: i64 = i64::from(fa == "on");
                                for p in &ps {
                                    let _ = p;
                                    rows.push(serde_json::json!({
                                        "model": "stub",
                                        "backend": "stub",
                                        "n_ctx": ctx,
                                        "n_threads": threads,
                                        "type_k": ctk,
                                        "type_v": ctv,
                                        "flash_attn": fa_code,
                                        "n_batch": batch,
                                        "test": format!("pp{n}"),
                                        "t/s": tg * 10.0,
                                    }));
                                }
                                rows.push(serde_json::json!({
                                    "model": "stub",
                                    "backend": "stub",
                                    "n_ctx": ctx,
                                    "n_threads": threads,
                                    "type_k": ctk,
                                    "type_v": ctv,
                                    "flash_attn": fa_code,
                                    "n_batch": batch,
                                    "test": format!("tg{n}"),
                                    "t/s": tg,
                                }));
                            }
                        }
                    }
                }
            }
        }
    }
    let mut out = std::io::stdout();
    let _ = serde_json::to_writer(&mut out, &rows);
    let _ = out.write_all(b"\n");
}
