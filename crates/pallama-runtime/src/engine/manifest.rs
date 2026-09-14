//! Capability manifest: probe an installed llama-server binary once and
//! store what it actually supports. The profile compiler emits ONLY flags
//! present in `flags`; a needed-but-missing flag is a hard error naming
//! the flag and suggesting `pallama engine use <tag>` — never a guess.
//! `--list-devices` output is PLAIN TEXT (verified against upstream
//! common/arg.cpp): `  NAME: DESC (TOTAL MiB, FREE MiB free)`.

use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceDesc {
    pub name: String,
    pub description: String,
    pub total_mib: u64,
    pub free_mib: u64,
}

impl DeviceDesc {
    /// GPU vendor inferred from the device description substring (upstream
    /// descriptions: "NVIDIA CUDA", "AMD `ROCm`", "Intel SYCL", "Vulkan ...").
    #[must_use]
    pub fn vendor(&self) -> Vendor {
        // Vulkan-backend descriptions are often generic ("Vulkan"); the
        // device name carries the vendor instead.
        let d = format!("{} {}", self.name, self.description).to_lowercase();
        if d.contains("nvidia") || d.contains("cuda") {
            Vendor::Nvidia
        } else if d.contains("amd") || d.contains("rocm") {
            Vendor::Amd
        } else if d.contains("intel") || d.contains("sycl") {
            Vendor::Intel
        } else {
            Vendor::Other
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Vendor {
    Nvidia,
    Amd,
    Intel,
    Other,
}

/// Everything Pallama knows about one installed engine build.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub tag: String,
    pub build_number: u64,
    pub version_raw: String,
    pub devices: Vec<DeviceDesc>,
    /// Every long flag (`--ctx-size`) accepted by this binary, from `--help`.
    pub flags: BTreeSet<String>,
    /// Speculative-decoding types accepted by `--spec-type`, when advertised.
    pub spec_types: Vec<String>,
    /// Absolute path to the llama-server binary.
    pub server_path: String,
}

impl Manifest {
    #[must_use]
    pub fn has_flag(&self, flag: &str) -> bool {
        self.flags.contains(flag)
    }

    /// Assert every flag in `needed` exists; error names the first missing
    /// one plus remediation.
    pub fn require_flags(&self, needed: &[&str]) -> Result<()> {
        for f in needed {
            if !self.has_flag(f) {
                return Err(anyhow!(
                    "engine {tag} does not support {f} (from its --help); \
                     try `pallama engine update` for a newer build or \
                     `pallama engine use <tag>` for an older one",
                    tag = self.tag,
                    f = f
                ));
            }
        }
        Ok(())
    }
}

/// Probe a llama-server binary: version, devices, flags.
pub fn probe(server_path: &Path, tag: &str) -> Result<Manifest> {
    let server = server_path
        .to_str()
        .ok_or_else(|| anyhow!("non-UTF-8 engine path {}", server_path.display()))?;

    let out = crate::probe::probe_output(Command::new(server).arg("--version"), 30)
        .with_context(|| format!("run {server} --version (timed out or failed to spawn)"))?;
    if !out.status.success() {
        return Err(anyhow!(
            "{server} --version exited {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    // Upstream prints the version banner to STDERR (verified b10816);
    // tolerate either stream.
    let mut version_text = String::from_utf8_lossy(&out.stderr).to_string();
    if !version_text.contains("version") {
        version_text = String::from_utf8_lossy(&out.stdout).to_string();
    }
    let (parsed_build, version_raw) = parse_version(&version_text)?;
    // The install tag is the authoritative build identity: a source build
    // from a shallow clone reports a commit-count artifact ("build 1")
    // where the release binary reports the real count. The tag Pallama
    // registered (release asset or `engine build`) always carries the
    // true number; non-b tags ("local") keep the probed value.
    let build_number = super::gh::btag_number(tag).unwrap_or(parsed_build);

    let devices = run_list_devices(server_path);

    let out = crate::probe::probe_output(Command::new(server).arg("--help"), 30)
        .with_context(|| format!("run {server} --help (timed out or failed to spawn)"))?;
    let help = String::from_utf8_lossy(&out.stdout).to_string();
    let (flags, spec_types) = parse_help(&help);

    Ok(Manifest {
        tag: tag.to_string(),
        build_number,
        version_raw,
        devices,
        flags,
        spec_types,
        server_path: server.to_string(),
    })
}

/// Kind-aware probe entry: dispatches to the llamacpp parser (strict —
/// the banner shape is a verified upstream contract) or the mistralrs
/// parser (tolerant — its CLI surface is undocumented enough to only
/// trust what parses).
pub fn probe_kind(
    server_path: &Path,
    tag: &str,
    kind: &pallama_core::engine_kind::EngineKind,
) -> Result<Manifest> {
    match kind {
        pallama_core::engine_kind::EngineKind::LlamaCpp => probe(server_path, tag),
        pallama_core::engine_kind::EngineKind::MistralRs => probe_mistralrs(server_path, tag),
        pallama_core::engine_kind::EngineKind::Sglang => probe_sglang(server_path, tag),
    }
}

/// Probe a mistralrs binary. Divergences from llama-server (verified
/// against mistral.rs v0.9.x docs): no `--list-devices` equivalent;
/// `--version` output is not a documented contract, so the install tag
/// is the identity and the banner is best-effort; serve flags come from
/// `serve --help` (merged with the global `--help`).
fn probe_mistralrs(server_path: &Path, tag: &str) -> Result<Manifest> {
    let server = server_path
        .to_str()
        .ok_or_else(|| anyhow!("non-UTF-8 engine path {}", server_path.display()))?;

    // Best-effort banner; never fatal — the tag is authoritative.
    // F85: a hung `--version` counts as "no banner", not a wedge.
    let version_raw = match crate::probe::probe_output(Command::new(server).arg("--version"), 30) {
        Some(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let stdout = String::from_utf8_lossy(&out.stdout);
            let text = if stdout.contains("mistralrs") || stdout.contains("version") {
                stdout
            } else {
                stderr
            };
            text.lines()
                .map(str::trim)
                .find(|l| !l.is_empty())
                .unwrap_or("mistralrs")
                .to_string()
        }
        None => format!("mistralrs {tag}"),
    };
    // Display-only serial from the v-tag (v0.9.3 -> 0000009003): keeps
    // `engine list` sortable without implying llama.cpp build numbers.
    let build_number = super::gh::vtag_semver(tag)
        .map_or(0, |(maj, min, patch)| maj * 1_000_000 + min * 1_000 + patch);

    // Flags: global `--help` + `serve --help`, both best-effort. An empty
    // set downgrades argv gating to "emit the fixed dialect" — a stale
    // flag then fails at spawn, loudly.
    let mut flags = BTreeSet::new();
    // NEVER probe with zero args: a bare `mistralrs` drops into serving
    // mode and listens forever, wedging the sync Command::output() call
    // (and the tokio worker under it). Both help forms exit on their own.
    let help_invocations: [Vec<String>; 2] =
        [vec!["--help".into()], vec!["serve".into(), "--help".into()]];
    for args in help_invocations {
        if let Ok(out) = Command::new(server).args(&args).output() {
            let (mut f, _) = parse_help(&String::from_utf8_lossy(&out.stdout));
            flags.append(&mut f);
        }
    }

    Ok(Manifest {
        tag: tag.to_string(),
        build_number,
        version_raw,
        devices: Vec::new(),
        flags,
        spec_types: Vec::new(),
        server_path: server.to_string(),
    })
}

/// Probe a sglang venv install through its shim script. Layout contract
/// (`sglang_install.rs)`: `engines/<tag>/sglang-server` is an executable
/// shim `exec <dir>/venv/bin/python -m sglang.launch_server "$@"`, so
/// the shim's sibling `venv/` holds the interpreter.
///
/// Divergences from the other engines (verified against sglang v0.5.19):
/// - no `--version` flag on `launch_server`; the version comes from
///   `importlib.metadata` against the venv python (a metadata read — no
///   torch import, seconds even cold).
/// - flags come from `launch_server --help` (standard argparse renderer,
///   `server_args.py:345 add_cli_args`) — but the import chain behind it
///   pulls torch, so the FIRST cold run can take a minute: 120s budget.
/// - NEVER probe with zero args: a bare `launch_server` starts serving
///   and never exits (same wedge class as mistralrs).
fn probe_sglang(server_path: &Path, tag: &str) -> Result<Manifest> {
    let server = server_path
        .to_str()
        .ok_or_else(|| anyhow!("non-UTF-8 engine path {}", server_path.display()))?;
    let venv_python = server_path
        .parent()
        .ok_or_else(|| anyhow!("engine path {server} has no parent dir"))?
        .join("venv")
        .join("bin")
        .join("python");

    // Version: importlib metadata read. Strict — a broken venv must fail
    // the probe here with the venv path named, not at spawn.
    let version_out = crate::probe::probe_output(
        Command::new(&venv_python)
            .arg("-c")
            .arg("import importlib.metadata as m; print(m.version('sglang'))"),
        30,
    )
    .with_context(|| {
        format!(
            "run {} -c importlib.metadata (sglang venv broken?)",
            venv_python.display()
        )
    })?;
    if !version_out.status.success() {
        return Err(anyhow!(
            "sglang version probe exited {}: {}",
            version_out.status,
            String::from_utf8_lossy(&version_out.stderr)
        ));
    }
    let version_raw = format!(
        "sglang {}",
        String::from_utf8_lossy(&version_out.stdout).trim()
    );

    // Display-only serial from the install tag (sglang-0.5.19 ->
    // 0000005019): keeps `engine list` sortable, same convention as the
    // mistralrs lane.
    let tag_serial = tag
        .strip_prefix("sglang-")
        .and_then(|rest| rest.split(['-', '+']).next())
        .and_then(|v| {
            let mut it = v.split('.');
            let maj = it.next()?.parse::<u64>().ok()?;
            let min = it.next().unwrap_or("0").parse::<u64>().ok()?;
            let patch = it.next().unwrap_or("0").parse::<u64>().ok()?;
            Some(maj * 1_000_000 + min * 1_000 + patch)
        })
        .unwrap_or(0);

    // Flags: launch_server --help via the shim. Argparse renderer, so
    // parse_help applies. Torch import behind it: 120s cold budget. A
    // zero-token parse is a FORMAT change upstream — treat as probe
    // failure (fail fast) rather than degrading every gated emission.
    let help_out = crate::probe::probe_output(Command::new(server).arg("--help"), 120)
        .with_context(|| {
            format!("run {server} --help (sglang import chain can take a minute cold)")
        })?;
    let help = String::from_utf8_lossy(&help_out.stdout).to_string();
    let (flags, _) = parse_help(&help);
    if flags.is_empty() {
        return Err(anyhow!(
            "sglang --help parsed to zero flags (output format changed upstream?): {}",
            help.lines().take(3).collect::<Vec<_>>().join(" | ")
        ));
    }

    Ok(Manifest {
        tag: tag.to_string(),
        build_number: tag_serial,
        version_raw,
        devices: Vec::new(),
        flags,
        spec_types: Vec::new(),
        server_path: server.to_string(),
    })
}

/// Parse the version banner. Upstream shape (b10816, stderr):
/// `version: 0.4.0-dev (build 10816, commit 427291b5b)`.
/// The `build NNNN` token is authoritative; fall back to the first
/// integer after `version:`.
fn parse_version(text: &str) -> Result<(u64, String)> {
    let line = text
        .lines()
        .find(|l| l.trim_start().starts_with("version:"))
        .ok_or_else(|| anyhow!("no `version:` line in --version output: {text:?}"))?;
    let build = line
        .split("build")
        .nth(1)
        .and_then(|rest| {
            rest.split(|c: char| !c.is_ascii_digit())
                .find(|tok| !tok.is_empty())
                .and_then(|tok| tok.parse::<u64>().ok())
        })
        .or_else(|| {
            line.split(':')
                .nth(1)
                .and_then(|rest| {
                    rest.split(|c: char| !c.is_ascii_digit())
                        .find(|t| !t.is_empty())
                })
                .and_then(|tok| tok.parse::<u64>().ok())
        })
        .ok_or_else(|| anyhow!("cannot parse build number from {line:?}"))?;
    Ok((build, line.trim().to_string()))
}

/// Live device census: run `<server> --list-devices` and parse the
/// per-card TOTAL/FREE MiB straight from the engine binary. This is the
/// ONLY source of live VRAM numbers — the manifest's stored `devices`
/// are an install-day snapshot and go stale the moment any other
/// process (ollama, a desktop session) touches the card.
///
/// Failure-tolerant by design: a missing binary or a hung census
/// returns an empty list and callers fall back to the manifest
/// snapshot. Census runs take a few hundred milliseconds (backend
/// init), so callers throttle to spawn-time and ≥60s periodic.
#[must_use]
pub fn run_list_devices(server: &Path) -> Vec<DeviceDesc> {
    // F85: the doc's "a hung census returns an empty list" is now
    // literally true — a deadline-bounded probe replaces the blocking
    // `.output()` that would wedge forever.
    let Some(out) = crate::probe::probe_output(Command::new(server).arg("--list-devices"), 30)
    else {
        return Vec::new();
    };
    // Upstream exits 0 here even when listing; tolerate non-zero but parse stdout.
    parse_devices(&String::from_utf8_lossy(&out.stdout))
}

/// Parse `  NAME: DESC (TOTAL MiB, FREE MiB free)` device lines.
pub(crate) fn parse_devices(text: &str) -> Vec<DeviceDesc> {
    let mut out = Vec::new();
    for line in text
        .lines()
        .skip_while(|l| !l.contains("Available devices:"))
    {
        let line = line.trim();
        if line.is_empty() || line.contains("Available devices") || line == "(none)" {
            continue;
        }
        // NAME: DESC (TOTAL MiB, FREE MiB free)
        let Some((name, rest)) = line.split_once(':') else {
            continue;
        };
        let Some(open) = rest.rfind('(') else {
            continue;
        };
        let Some(close) = rest[open..].find(')') else {
            continue;
        };
        let desc = rest[..open].trim().to_string();
        let mem = &rest[open + 1..open + close];
        // "8192 MiB, 7000 MiB free"
        let nums: Vec<u64> = mem
            .split([',', ' '])
            .filter_map(|t| t.parse::<u64>().ok())
            .collect();
        if nums.len() < 2 {
            continue;
        }
        out.push(DeviceDesc {
            name: name.trim().to_string(),
            description: desc,
            total_mib: nums[0],
            free_mib: nums[1],
        });
    }
    out
}

/// Extract every long flag from `--help` text plus the `--spec-type`
/// value list when present.
fn parse_help(help: &str) -> (BTreeSet<String>, Vec<String>) {
    let mut flags = BTreeSet::new();
    let mut spec_types = Vec::new();
    for line in help.lines() {
        let trimmed = line.trim_start();
        // Option-table rows start with tokens like `-c, --ctx-size N` or
        // `--spec-type none,draft-simple,...`. Scan every whitespace token.
        for tok in trimmed.split_whitespace() {
            if let Some(long) = tok.strip_prefix("--") {
                let clean = long
                    .trim_end_matches(',')
                    .split('=')
                    .next()
                    .unwrap_or(long)
                    .to_string();
                // Reject value placeholders attached with spaces? Values are
                // separate tokens; keep alphabetic-dash names only.
                if clean
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
                    && !clean.is_empty()
                    && clean.len() > 1
                {
                    flags.insert(format!("--{clean}"));
                }
            }
        }
        if trimmed.contains("--spec-type") {
            // e.g. `--spec-type none,draft-simple,draft-eagle3,...`
            let list = trimmed
                .split_whitespace()
                .find(|t| t.starts_with("none,") || t.contains(",draft-") || t.contains(",ngram-"))
                .unwrap_or("");
            let values: Vec<String> = list
                .split(',')
                .map(|v| v.trim().to_string())
                .filter(|v| {
                    !v.is_empty()
                        && v.chars()
                            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
                })
                .collect();
            if values.len() > 1 {
                spec_types = values;
            }
        }
    }
    (flags, spec_types)
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__parse_version__upstream_shape() {
        let (n, raw) = parse_version("version: 10816 (deadbeef)\nbuilt with cc").unwrap();
        assert_eq!(n, 10816);
        assert!(raw.contains("10816"));
    }

    #[test]
    fn unit__parse_version__real_b10816_banner() {
        // Verified live: stderr, "build NNNN, commit X" inside parens.
        let text = "version: 0.4.0-dev (build 10816, commit 427291b5)\nbuilt with GNU 11.4.0 for Linux x86_64\n";
        let (n, raw) = parse_version(text).unwrap();
        assert_eq!(n, 10816);
        assert!(raw.starts_with("version: 0.4.0-dev"));
    }

    #[test]
    fn unit__parse_version__missing__error() {
        assert!(parse_version("llama.server\n").is_err());
    }

    #[test]
    fn unit__parse_devices__upstream_shape() {
        let text = "Available devices:\n  NVIDIA GeForce RTX 4070: NVIDIA CUDA (8188 MiB, 7000 MiB free)\n  Intel iGPU: Vulkan (32768 MiB, 24000 MiB free)\n";
        let d = parse_devices(text);
        assert_eq!(d.len(), 2);
        assert_eq!(d[0].name, "NVIDIA GeForce RTX 4070");
        assert_eq!(d[0].description, "NVIDIA CUDA");
        assert_eq!((d[0].total_mib, d[0].free_mib), (8188, 7000));
        assert_eq!(d[0].vendor(), Vendor::Nvidia);
        assert_eq!(d[1].vendor(), Vendor::Intel);
    }

    #[test]
    fn unit__parse_devices__none_case() {
        let d = parse_devices("Available devices:\n  (none)\n");
        assert!(d.is_empty());
    }

    #[test]
    fn unit__parse_help__flags_and_spec_types() {
        let help = "\
usage: llama-server [options]

options:
  -h, --help            show this help message and exit
  -v, --version         show version information and exit
  --mlock               force system to keep model in RAM etc.
  -c, --ctx-size N      size of the prompt context (default: 0)
  -ngl, --gpu-layers N  max. number of layers (default: auto)
  --spec-type none,draft-simple,draft-eagle3,ngram-simple  types of speculative decoding
  -fa, --flash-attn [on|off|auto]
";
        let (flags, spec) = parse_help(help);
        for expected in [
            "--help",
            "--version",
            "--mlock",
            "--ctx-size",
            "--gpu-layers",
            "--flash-attn",
            "--spec-type",
        ] {
            assert!(flags.contains(expected), "missing {expected} in {flags:?}");
        }
        assert_eq!(
            spec,
            vec![
                "none".to_string(),
                "draft-simple".to_string(),
                "draft-eagle3".to_string(),
                "ngram-simple".to_string()
            ]
        );
    }

    #[test]
    fn unit__manifest_require_flags__names_missing_flag_and_remediation() {
        let m = Manifest {
            tag: "b100".into(),
            build_number: 100,
            version_raw: "version: 100 (x)".into(),
            devices: vec![],
            flags: BTreeSet::from(["--jinja".to_string()]),
            spec_types: vec![],
            server_path: "/x".into(),
        };
        assert!(m.require_flags(&["--jinja"]).is_ok());
        let err = m
            .require_flags(&["--jinja", "--spec-draft-model"])
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("--spec-draft-model") && msg.contains("engine use"),
            "{msg}"
        );
    }

    // ------------------------------------------------------------------
    // sglang probe: venv-version + argparse --help contract
    // ------------------------------------------------------------------

    #[cfg(unix)]
    fn fake_sglang_install(dir: &std::path::Path) -> std::path::PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let root = dir.join("sglang-0.5.19");
        std::fs::create_dir_all(root.join("venv/bin")).expect("venv dir");
        // version stub: prints like importlib.metadata regardless of args
        std::fs::write(root.join("venv/bin/python"), "#!/bin/sh\necho 0.5.19\n")
            .expect("python stub");
        // shim stub: argparse-style help surface
        std::fs::write(
            root.join("sglang-server"),
            "#!/bin/sh\nprintf 'usage: launch_server [options]\\n\\noptions:\\n  --model-path MODEL_PATH\\n  --context-length N\\n  --mem-fraction-static F\\n  --kv-cache-dtype DTYPE\\n'\n",
        )
        .expect("shim stub");
        for f in ["venv/bin/python", "sglang-server"] {
            let p = root.join(f);
            let mut perm = std::fs::metadata(&p).expect("meta").permissions();
            perm.set_mode(0o755);
            std::fs::set_permissions(p, perm).expect("chmod");
        }
        root.join("sglang-server")
    }

    #[cfg(unix)]
    #[test]
    fn unit__probe_sglang__stub_venv_yields_manifest() {
        let dir = tempfile::tempdir().expect("tempdir");
        let shim = fake_sglang_install(dir.path());
        let m = probe_sglang(&shim, "sglang-0.5.19").expect("probe");
        assert_eq!(m.tag, "sglang-0.5.19");
        assert_eq!(m.version_raw, "sglang 0.5.19");
        assert_eq!(m.build_number, 5_019);
        assert!(m.flags.contains("--model-path"), "{:?}", m.flags);
        assert!(m.flags.contains("--context-length"), "{:?}", m.flags);
        assert!(m.flags.contains("--mem-fraction-static"), "{:?}", m.flags);
        assert!(m.flags.contains("--kv-cache-dtype"), "{:?}", m.flags);
        assert!(m.devices.is_empty());
        assert!(m.spec_types.is_empty());
        assert_eq!(m.server_path, shim);
    }

    #[cfg(unix)]
    #[test]
    fn unit__probe_sglang__broken_venv_fails_naming_it() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        // shim exists, venv does not: strict version probe must fail and
        // name the venv python path (fail at probe, not at first spawn).
        let root = dir.path().join("sglang-0.5.19");
        std::fs::create_dir_all(&root).expect("dir");
        let shim = root.join("sglang-server");
        std::fs::write(&shim, "#!/bin/sh\nexit 0\n").expect("shim");
        let mut perm = std::fs::metadata(&shim).unwrap().permissions();
        perm.set_mode(0o755);
        std::fs::set_permissions(&shim, perm).unwrap();
        let err = probe_sglang(&shim, "sglang-0.5.19").expect_err("must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("venv"), "{msg}");
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod live_census_tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn unit__run_list_devices__parses_live_census_output() {
        // tempfile guard: an earlier revision hand-rolled the fixture dir
        // and removed only the script file — every `cargo test` leaked an
        // empty /tmp/pallama-census-<pid> dir (200+ accumulated on the
        // dev box before this was noticed).
        let dir = tempfile::tempdir().expect("tempdir");
        let script = dir.path().join("fake-server");
        std::fs::write(
            &script,
            "#!/bin/sh\nprintf 'Available devices:\\n  CUDA0: NVIDIA CUDA (7805 MiB, 1200 MiB free)\\n'\n",
        )
        .expect("write script");
        make_executable(&script);
        let devices = run_list_devices(&script);
        assert_eq!(devices.len(), 1, "{devices:?}");
        assert_eq!(devices[0].name, "CUDA0");
        assert_eq!(devices[0].total_mib, 7805);
        assert_eq!(devices[0].free_mib, 1200);
    }

    #[test]
    fn unit__run_list_devices__missing_binary_is_empty_not_panic() {
        let devices = run_list_devices(std::path::Path::new("/nonexistent/pallama-census-probe"));
        assert!(devices.is_empty());
    }

    #[cfg(unix)]
    fn make_executable(path: &std::path::Path) {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(path).expect("stat").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(path, perms).expect("chmod");
    }
}
