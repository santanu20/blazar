//! `blazar upgrade` e2e against a fake GitHub Releases server.
//!
//! Covers: dry-run resolve+verify, tampered-digest rejection, and a real
//! self-replace on a *copy* of the binary (never the cargo artifact).
#![allow(non_snake_case)] // suite convention: unit__scenario__expected (§6b)
#![cfg(unix)]

use std::io::{Read, Write};
use std::net::TcpListener;

use assert_cmd::Command;

fn make_tarball(payload: &[u8]) -> Vec<u8> {
    let mut tar_builder = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    header.set_size(payload.len() as u64);
    header.set_mode(0o755);
    header.set_cksum();
    tar_builder
        .append_data(&mut header, "blazar", payload)
        .unwrap();
    let raw = tar_builder.into_inner().unwrap();
    let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    gz.write_all(&raw).unwrap();
    gz.finish().unwrap()
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest as _;
    format!("{:x}", sha2::Sha256::digest(bytes))
}

fn release_json(tag: &str, asset: &str, digest: &str, base: &str) -> Vec<u8> {
    format!(
        r#"{{"tag_name":"{tag}","assets":[{{"name":"{asset}","digest":"sha256:{digest}","browser_download_url":"{base}/download/{asset}"}}]}}"#
    )
    .into_bytes()
}

fn preferred_asset(tag: &str) -> String {
    blazar_runtime::upgrade::preferred_assets(tag)
        .into_iter()
        .next()
        .expect("host platform has a preferred asset")
}

/// Bind first (so the release JSON can embed the real URL), then serve
/// one-request-at-a-time on that SAME listener: `/releases/latest` and
/// `/download/<asset>`.
fn serve_loop(listener: &TcpListener, release_json: &[u8], asset: &[u8]) {
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else { break };
        let mut buf = [0u8; 4096];
        let n = stream.read(&mut buf).unwrap_or(0);
        let path = String::from_utf8_lossy(&buf[..n])
            .split_whitespace()
            .nth(1)
            .unwrap_or_default()
            .to_string();
        let (body, ctype) = if path.ends_with("/releases/latest") {
            (release_json.to_vec(), "application/json")
        } else if path.contains("/download/") {
            (asset.to_vec(), "application/octet-stream")
        } else {
            let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
            continue;
        };
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        let _ = stream.write_all(resp.as_bytes());
        let _ = stream.write_all(&body);
    }
}

/// Fake release server: one listener, JSON and downloads on the same port.
fn fake_release(tag: &str, payload: &[u8], digest_mangle: Option<fn(String) -> String>) -> String {
    let asset_name = preferred_asset(tag);
    let tarball = make_tarball(payload);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let digest = digest_mangle.map_or_else(|| sha256_hex(&tarball), |f| f(sha256_hex(&tarball)));
    let json = release_json(tag, &asset_name, &digest, &base);
    std::thread::spawn(move || serve_loop(&listener, &json, &tarball));
    base
}

#[test]
fn e2e__upgrade_dry_run_resolves_and_verifies() {
    let base = fake_release("v0.1.1", b"FAKE-BLAZAR-v0.1.1\n", None);
    Command::cargo_bin("blazar")
        .unwrap()
        .env("BLAZAR_INSTALL_BASE_URL", &base)
        .env("BLAZAR_REPO", "test/blazar")
        .args(["upgrade", "--dry-run"])
        .assert()
        .success()
        .stdout(predicates::str::contains("dry-run ok: v0.1.1"));
}

#[test]
fn e2e__upgrade_tampered_digest_rejected() {
    let mangle = |d: String| {
        let first = if d.starts_with('0') { '1' } else { '0' };
        format!("{first}{}", &d[1..])
    };
    let base = fake_release("v0.1.1", b"FAKE-BLAZAR-v0.1.1\n", Some(mangle));
    Command::cargo_bin("blazar")
        .unwrap()
        .env("BLAZAR_INSTALL_BASE_URL", &base)
        .env("BLAZAR_REPO", "test/blazar")
        .args(["upgrade", "--dry-run"])
        .assert()
        .failure()
        .stdout(predicates::str::contains("sha256 mismatch"));
}

#[test]
fn e2e__upgrade_replaces_a_copy_of_the_binary_atomically() {
    let payload = b"FAKE-BLAZAR-v0.1.2-REPLACED\n";
    let base = fake_release("v0.1.2", payload, None);

    // Run the upgrade from a COPY so the cargo artifact is never touched.
    let tmp = tempfile::tempdir().unwrap();
    let exe_copy = tmp.path().join("blazar");
    std::fs::copy(env!("CARGO_BIN_EXE_blazar"), &exe_copy).unwrap();

    let status = std::process::Command::new(&exe_copy)
        .env("BLAZAR_INSTALL_BASE_URL", &base)
        .env("BLAZAR_REPO", "test/blazar")
        .arg("upgrade")
        .status()
        .unwrap();
    assert!(status.success(), "upgrade on a copy must succeed");

    let on_disk = std::fs::read(&exe_copy).unwrap();
    assert_eq!(on_disk, payload, "exe replaced by verified fake payload");
}
