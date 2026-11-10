//! Capability manifest: probe an installed llama-server binary once and
//! store what it actually supports. The profile compiler emits ONLY flags
//! present in `flags`; a needed-but-missing flag is a hard error naming
//! the flag and suggesting `blazar engine use <tag>` - never a guess.
//! `--list-devices` output is PLAIN TEXT (verified against upstream
//! common/arg.cpp): `  NAME: DESC (TOTAL MiB, FREE MiB free)`.

use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;
use std::process::Output;

use anyhow::{Context, Result, anyhow};
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

/// Where an installed engine's code came from. `Fork` marks a
/// capability lane built from an unmerged llama.cpp fork at a pinned
/// commit (see `engine build --fork`); routing treats forks as
/// capability shims (they lose same-kind ties to mainstream builds,
/// see `LaneClass`), and this field also colors provenance display and
/// retention policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EngineSource {
    /// Release asset or source build of the upstream repo.
    #[default]
    Upstream,
    /// Immutable `owner/repo@sha` capability lane.
    Fork,
    /// User-registered local binary (the `local` pseudo-tag).
    Local,
}

impl EngineSource {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Upstream => "upstream",
            Self::Fork => "fork",
            Self::Local => "local",
        }
    }
}

/// Who vouched for a fork lane's code. Curated lanes come from the
/// Blazar capability registry (pinned commit, recorded provenance) and
/// may be auto-retired once upstream covers their architectures; user
/// lanes were built explicitly with `engine build --fork` and are never
/// touched by lifecycle automation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrustTier {
    /// Built by explicit user command (default).
    #[default]
    User,
    /// Installed from a registry entry via `engine install --lane`.
    Curated,
}

impl TrustTier {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Curated => "curated",
        }
    }
}

/// Build-time provenance merged into a probed manifest when the engine
/// was compiled from source: everything needed to answer "which commit
/// of which repo produced this binary, and what did it claim to
/// support at build time".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneProvenance {
    pub source: EngineSource,
    /// e.g. `ggml-org/llama.cpp` or `acme/llama.cpp`.
    pub repo: Option<String>,
    /// Full commit SHA the tree was checked out at (forks are
    /// immutable: never a branch or tag name).
    pub ref_pin: Option<String>,
    /// Upstream anchor recorded when the lane was created (`bNNNN`
    /// base or `master`); provenance only - consumed by the Phase 3
    /// retire lifecycle, never by routing.
    pub base_ref: Option<String>,
    /// Architecture names mined from the built source's
    /// `llama-arch.cpp` (`LLM_ARCH_NAMES`). A CANDIDATE filter, not a
    /// guarantee: the runtime load verifies.
    pub architectures: BTreeSet<String>,
    /// Newest upstream llama.cpp release build number known at fork-build
    /// time. A mainstream lane can only graduate (supersede) this fork
    /// when its own build is NEWER than this floor - a lane at or below
    /// it predates the pin and cannot contain the fork's changes.
    /// `None` = unknown (offline build): legacy supersede behavior.
    pub floor_release: Option<u64>,
}

/// Everything Blazar knows about one installed engine build.
///
/// The v2 fields (`source`..`architectures`) are all
/// `#[serde(default)]`, so manifests serialized by older Blazar
/// versions (v1 shape) decode unchanged as upstream lanes with no
/// advertised architectures.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub tag: String,
    pub build_number: u64,
    pub version_raw: String,
    pub devices: Vec<DeviceDesc>,
    /// Every long flag (`--ctx-size`) accepted by this binary, from `--help`.
    pub flags: BTreeSet<String>,
    /// Speculative-decoding algorithms this lane can serve. Vocabulary
    /// is per-kind: llama.cpp rows carry `--spec-type` wire names
    /// (`draft-mtp`, `draft-eagle3`, ...); sglang rows carry
    /// `SpeculativeAlgorithm` names plus plugin-registered names
    /// (`NGRAM`, `EAGLE3`, `DFLASH`, `DSPINE`, ...) from the install
    /// probe. Empty = not probed: the profile compiler falls back to
    /// flag-presence gating.
    pub spec_types: Vec<String>,
    /// Quant loader methods this build actually registers - sglang
    /// lanes only, probed at install time from the engine's own
    /// registry (`sglang.srt.layers.quantization.BASE_QUANTIZATION_METHODS`,
    /// which appends entries beyond its source literal, so only a live
    /// probe is truthful). `None` = not probed (non-sglang lanes,
    /// manifests predating the probe); admission then falls back to
    /// the blazar-core const snapshot.
    #[serde(default)]
    pub quant_methods: Option<Vec<String>>,
    /// Path to the engine's server binary as recorded at install time.
    /// Stored RELATIVE to the data dir (`engines/<tag>/...`) so rows stay
    /// relocatable; the out-of-tree `local` lane (`BLAZAR_ENGINE_PATH`)
    /// keeps its user-given absolute path. Anchored to an absolute live
    /// path on load: see [`Manifest::anchor_server_path`].
    pub server_path: String,
    /// Where this build's code came from (v2; defaults to upstream for
    /// pre-v2 manifests).
    #[serde(default)]
    pub source: EngineSource,
    /// `owner/repo` for source-built lanes (v2).
    #[serde(default)]
    pub repo: Option<String>,
    /// Immutable commit pin for fork lanes (v2).
    #[serde(default)]
    pub ref_pin: Option<String>,
    /// Upstream anchor the fork lane was created against (v2;
    /// provenance only).
    #[serde(default)]
    pub base_ref: Option<String>,
    /// Newest upstream llama.cpp release build number at fork-build
    /// time (v2.2): mainstream lanes with `build_number <= floor` can
    /// never supersede this fork - they predate its commit pin. `None`
    /// = unstamped/legacy lane: arch coverage alone decides.
    #[serde(default)]
    pub floor_release: Option<u64>,
    /// Architecture names this build advertises (v2) - mined from the
    /// source tree at build time. Empty = unknown/unverified: such a
    /// lane is never picked by architecture-based re-routing.
    #[serde(default)]
    pub architectures: BTreeSet<String>,
    /// Who vouched for a fork lane (v2.1): user-built lanes are never
    /// auto-retired; curated registry lanes may be, after a grace
    /// period. Defaults to `user` for all pre-existing manifests.
    #[serde(default)]
    pub trust: TrustTier,
    /// Tag of the upstream lane whose architecture coverage superseded
    /// every architecture this fork lane was built for (v2.1). `None`
    /// = live lane. Set once by the supersede check; routing keeps the
    /// lane eligible as a rescue fallback (the runtime load is the
    /// truth - if upstream support turns out partial, the rescue path
    /// re-pins this lane).
    #[serde(default)]
    pub superseded_by: Option<String>,
    /// Unix epoch seconds when the lane was marked superseded (v2.1);
    /// starts the auto-retire grace clock for curated lanes.
    #[serde(default)]
    pub superseded_at_epoch: Option<i64>,
    /// Vocabulary-probe generation this manifest was captured with
    /// (see [`MANIFEST_PROBE_VERSION`]). `0` = legacy row from an
    /// older parser: the daemon boot backfill and `engine reprobe`
    /// refresh it; the admission gate keeps reading the stored vocab
    /// until then.
    #[serde(default)]
    pub probe_version: u32,
}

/// Generation of the manifest vocabulary probe. v1 (implicit - all
/// unstamped rows, `probe_version` absent) captured underscore-only
/// flag names and mistook `--help` separators for flags; v2 keeps
/// dotted flag names (`--fim-qwen-1.5b-default`) and rejects all-dash
/// tokens; v3 fills `spec_types` on sglang lanes (the
/// `SpeculativeAlgorithm` + plugin-registry probe), which v2 rows
/// recorded empty. Bump whenever `parse_help` (or another probe)
/// changes what the extracted vocabulary MEANS - stored rows stamped
/// below the current version self-heal via the daemon-boot backfill
/// and `blazar engine reprobe`.
pub const MANIFEST_PROBE_VERSION: u32 = 3;

/// Preview of the spawn-time capability rescue (the re-route in
/// `spawn_instance` after a lane dies rejecting an architecture): when
/// the lane the router picked provably cannot load this GGUF's arch -
/// its mined arch set is known and lacks it - the spawn path crashes
/// once, then re-routes to the installed lane that advertises the
/// architecture. Previews (list ENGINE cell, /api/tags, /v1/models)
/// call this so they show the lane that will ULTIMATELY serve. No
/// prediction under a user pin (the spawn rescue has the same gate),
/// for non-llama.cpp picked kinds, or when the picked lane's arch set
/// is unknown (honest unknown - the crash-time rescue still fires at
/// spawn, it just cannot be predicted here).
#[must_use]
pub fn predicted_rescue_lane(
    engine_rows: &[blazar_core::EngineRow],
    arch: Option<&str>,
    pin: Option<&str>,
    picked_tag: &str,
    picked_kind: blazar_core::engine_kind::EngineKind,
) -> Option<String> {
    let arch = arch?;
    if pin.is_some() || picked_kind != blazar_core::engine_kind::EngineKind::LlamaCpp {
        return None;
    }
    // Decoded manifests must outlive the borrowed lane views below.
    let decoded: Vec<(&blazar_core::EngineRow, Manifest)> = engine_rows
        .iter()
        .filter(|r| r.kind == blazar_core::engine_kind::EngineKind::LlamaCpp)
        .filter_map(|r| {
            serde_json::from_str::<Manifest>(&r.manifest)
                .ok()
                .map(|m| (r, m))
        })
        .collect();
    let lanes: Vec<blazar_core::engine_kind::LaneArchView> = decoded
        .iter()
        .map(|(r, m)| (r.tag.as_str(), r.lane_class(), m.advertised_archs()))
        .collect();
    let picked_lacks = lanes
        .iter()
        .find(|(t, _, _)| *t == picked_tag)
        .is_some_and(|(_, _, set)| set.is_some_and(|s| !s.contains(arch)));
    if !picked_lacks {
        return None;
    }
    blazar_core::engine_kind::advertising_lanes(arch, Some(picked_tag), &lanes)
        .first()
        .map(|t| (*t).to_string())
}

impl Manifest {
    #[must_use]
    pub fn has_flag(&self, flag: &str) -> bool {
        self.flags.contains(flag)
    }

    /// The engine-reported sglang version ("0.5.23"), preferring
    /// `version_raw` ("sglang 0.5.23") and falling back to the install
    /// tag ("sglang-0.5.23"). Empty when neither carries one - callers
    /// render that as "version unknown" rather than inventing one.
    #[must_use]
    pub fn sglang_version(&self) -> String {
        if let Some(v) = self.version_raw.strip_prefix("sglang ") {
            return v.trim().to_string();
        }
        self.tag
            .strip_prefix("sglang-")
            .map(str::to_string)
            .unwrap_or_default()
    }

    /// Lane facts for the sglang admission gate
    /// (`blazar_core::hfmeta::sglang_lane_teach`): this build's own
    /// version and probed quant registry. Manifests that predate the
    /// probe yield `quant_methods: None` - the gate falls back to its
    /// const snapshot.
    #[must_use]
    pub fn sglang_lane_facts(&self) -> blazar_core::hfmeta::SglangLaneFacts {
        blazar_core::hfmeta::SglangLaneFacts {
            version: self.sglang_version(),
            quant_methods: self.quant_methods.clone(),
        }
    }

    /// Does this lane advertise a GGUF architecture? An advertised name
    /// is a CANDIDATE filter for re-routing (the runtime load is the
    /// verification); lanes with no mined set never match.
    #[must_use]
    pub fn advertises_arch(&self, arch: &str) -> bool {
        self.architectures.contains(arch)
    }

    /// The mined arch set, `None` when nothing was mined (pre-v2 rows,
    /// non-llama.cpp kinds) - callers treat `None` as "advertises
    /// nothing" and "cannot be proven to lack" (honest unknown).
    #[must_use]
    pub fn advertised_archs(&self) -> Option<&std::collections::BTreeSet<String>> {
        (!self.architectures.is_empty()).then_some(&self.architectures)
    }

    /// Merge build-time provenance into a probed manifest (source
    /// builds only): marks the lane's origin and bakes in the
    /// architecture set mined from the built source tree.
    pub fn merge_provenance(&mut self, prov: &LaneProvenance) {
        self.source = prov.source;
        self.repo.clone_from(&prov.repo);
        self.ref_pin.clone_from(&prov.ref_pin);
        self.base_ref.clone_from(&prov.base_ref);
        self.architectures.clone_from(&prov.architectures);
        self.floor_release = prov.floor_release;
    }

    /// Short provenance label for tables and teaching strings, e.g.
    /// `fork acme/llama.cpp@7c81a9f0 (base b10980)` or `curated fork
    /// acme/llama.cpp@7c81a9f0`; empty for plain upstream builds.
    #[must_use]
    pub fn provenance_label(&self) -> String {
        match (self.source, &self.ref_pin) {
            (EngineSource::Fork, Some(pin)) => {
                let short = &pin[..pin.len().min(8)];
                let repo = self.repo.as_deref().unwrap_or("?");
                let base = self
                    .base_ref
                    .as_deref()
                    .map(|b| format!(" (base {b})"))
                    .unwrap_or_default();
                let tier = match self.trust {
                    TrustTier::User => "fork",
                    TrustTier::Curated => "curated fork",
                };
                format!("{tier} {repo}@{short}{base}")
            }
            (EngineSource::Local, _) => "local".to_string(),
            _ => String::new(),
        }
    }

    /// Assert every flag in `needed` exists; error names the first missing
    /// one plus remediation.
    pub fn require_flags(&self, needed: &[&str]) -> Result<()> {
        for f in needed {
            if !self.has_flag(f) {
                return Err(anyhow!(
                    "engine {tag} does not support {f} (from its --help); \
                     try `blazar engine update` for a newer build or \
                     `blazar engine use <tag>` for an older one",
                    tag = self.tag,
                    f = f
                ));
            }
        }
        Ok(())
    }

    /// Engine rows bake the absolute server path probed at install time,
    /// which makes a row non-relocatable: move `XDG_DATA_HOME` (or copy
    /// the DB into another data dir) and every pre-existing row points at
    /// the old root while the binaries sit intact under the new one.
    /// Re-anchor on load: when the recorded path is gone but the same
    /// `engines/<tag>/...` tail exists under the live engines dir, adopt
    /// it. Anything else stays untouched so a genuinely missing binary
    /// still fails loudly at spawn.
    pub fn re_root_server_path(&mut self, engines_dir: &Path) -> bool {
        let recorded = Path::new(&self.server_path);
        if recorded.exists() {
            return false;
        }
        let components: Vec<_> = recorded.components().collect();
        let Some(pos) = components
            .iter()
            .position(|c| c.as_os_str() == std::ffi::OsStr::new("engines"))
        else {
            return false;
        };
        let mut live = engines_dir.to_path_buf();
        for component in &components[pos + 1..] {
            live.push(component);
        }
        if !live.exists() {
            return false;
        }
        tracing::warn!(
            "engine {} was installed under a different data dir (recorded {}); \
             re-rooted to {}",
            self.tag,
            self.server_path,
            live.display()
        );
        self.server_path = live.display().to_string();
        true
    }

    /// Adopt the freshly probed vocabulary (flags, spec types, devices)
    /// and probe generation from `fresh`, preserving everything else -
    /// provenance, trust, pinned paths, the sglang quant registry.
    /// Shared by `engine reprobe` and the boot stale-vocab backfill so
    /// the two heals can never drift apart.
    pub fn merge_vocab_from(&mut self, fresh: &Manifest) {
        self.flags.clone_from(&fresh.flags);
        self.spec_types.clone_from(&fresh.spec_types);
        self.devices.clone_from(&fresh.devices);
        self.probe_version = fresh.probe_version;
    }

    /// Anchor a stored (possibly relative) `server_path` to an absolute
    /// live path - call at EVERY row decode before the manifest reaches
    /// consumers (spawns expect an executable path). Relative rows
    /// (`engines/<tag>/...`, the storage invariant) resolve against the
    /// live data dir; absolute rows are legacy installs or the
    /// out-of-tree `local` lane - legacy ones heal via
    /// [`Manifest::re_root_server_path`] when their recorded root moved,
    /// user-given local paths stay untouched. An empty path stays empty
    /// (guards the orphan sweep against a match-everything reference).
    pub fn anchor_server_path(&mut self, data_dir: &Path) {
        if self.server_path.is_empty() {
            return;
        }
        let recorded = Path::new(&self.server_path);
        if recorded.is_relative() {
            self.server_path = data_dir.join(recorded).display().to_string();
        } else {
            self.re_root_server_path(&data_dir.join("engines"));
        }
    }

    /// Fold an absolute in-tree `server_path` into its data-dir-relative
    /// storage form (`engines/<tag>/...`). Returns whether the manifest
    /// changed. Out-of-tree paths (the `local` lane's `BLAZAR_ENGINE_PATH`
    /// binary) and already-relative paths stay untouched.
    pub fn relativize_server_path(&mut self, data_dir: &Path) -> bool {
        let recorded = Path::new(&self.server_path);
        if !recorded.is_absolute() {
            return false;
        }
        let Ok(rel) = recorded.strip_prefix(data_dir) else {
            return false;
        };
        // Storage invariant: relative rows persist with forward slashes
        // (data dirs move between operating systems; a PathBuf stringify
        // would bake windows separators into the row). Rebuild from
        // components so the spelling is canonical everywhere.
        let folded = rel
            .components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        if folded.is_empty() {
            return false;
        }
        tracing::debug!(
            "engine {} server path stored relative: {}",
            self.tag,
            folded
        );
        self.server_path = folded;
        true
    }
}

/// Probe a llama-server binary: version, devices, flags.
pub fn probe(server_path: &Path, tag: &str) -> Result<Manifest> {
    probe_with_env(server_path, tag, &[])
}

/// `probe` with extra environment applied to every probe child, exactly
/// the way `engine_env` applies to a serving child (e.g. `GGML_BACKEND_PATH`
/// so a CUDA build discovers its GPU during probing). The process
/// environment itself is never mutated - edition 2024 makes that unsafe,
/// and a set-around-call dance would also leak the probe env into any
/// concurrently spawned child.
pub fn probe_with_env(
    server_path: &Path,
    tag: &str,
    extra_env: &[(String, String)],
) -> Result<Manifest> {
    let server = server_path
        .to_str()
        .ok_or_else(|| anyhow!("non-UTF-8 engine path {}", server_path.display()))?;

    let out = crate::probe::probe_output(
        Command::new(server)
            .arg("--version")
            .envs(extra_env.iter().map(|(k, v)| (k, v))),
        30,
    )
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
    // where the release binary reports the real count. The tag Blazar
    // registered (release asset or `engine build`) always carries the
    // true number; non-b tags ("local") keep the probed value.
    let build_number = super::gh::btag_number(tag).unwrap_or(parsed_build);

    let devices = run_list_devices_with_env(server_path, extra_env);

    let out = crate::probe::probe_output(
        Command::new(server)
            .arg("--help")
            .envs(extra_env.iter().map(|(k, v)| (k, v))),
        30,
    )
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
        source: EngineSource::Upstream,
        repo: None,
        ref_pin: None,
        base_ref: None,
        architectures: BTreeSet::new(),
        trust: TrustTier::default(),
        superseded_by: None,
        superseded_at_epoch: None,
        floor_release: None,
        quant_methods: None,
        probe_version: MANIFEST_PROBE_VERSION,
    })
}

/// Kind-aware probe entry: dispatches to the llamacpp parser (strict -
/// the banner shape is a verified upstream contract) or the mistralrs
/// parser (tolerant - its CLI surface is undocumented enough to only
/// trust what parses).
pub fn probe_kind(
    server_path: &Path,
    tag: &str,
    kind: &blazar_core::engine_kind::EngineKind,
) -> Result<Manifest> {
    match kind {
        blazar_core::engine_kind::EngineKind::LlamaCpp => probe(server_path, tag),
        blazar_core::engine_kind::EngineKind::MistralRs => probe_mistralrs(server_path, tag),
        blazar_core::engine_kind::EngineKind::Sglang => probe_sglang(server_path, tag),
        blazar_core::engine_kind::EngineKind::SdCpp => probe_sdcpp(server_path, tag),
        blazar_core::engine_kind::EngineKind::Whisper => probe_whisper(server_path, tag),
        blazar_core::engine_kind::EngineKind::Piper => probe_piper(server_path, tag),
        blazar_core::engine_kind::EngineKind::Kokoro => probe_kokoro(server_path, tag),
        blazar_core::engine_kind::EngineKind::Mlx => probe_mlx(server_path, tag),
        blazar_core::engine_kind::EngineKind::OpenAiHost => probe_openaihost(server_path, tag),
    }
}

/// Probe an mlx-lm venv lane. Same shape as sglang (the lane it
/// mirrors): importlib.metadata version read through the venv python,
/// display serial from the install tag, argparse `--help` flags via
/// the shim - the mlx import chain is torch-class cold, hence the
/// 120s budget. Zero parsed flags = upstream format change = probe
/// failure, not a silently degraded manifest.
fn probe_mlx(server_path: &Path, tag: &str) -> Result<Manifest> {
    let server = server_path
        .to_str()
        .ok_or_else(|| anyhow!("non-UTF-8 engine path {}", server_path.display()))?;
    let venv_python = server_path
        .parent()
        .ok_or_else(|| anyhow!("engine path {server} has no parent dir"))?
        .join("venv")
        .join("bin")
        .join("python");

    let version_out = crate::probe::probe_output(
        Command::new(&venv_python)
            .arg("-c")
            .arg("import importlib.metadata as m; print(m.version('mlx_lm'))"),
        30,
    )
    .with_context(|| {
        format!(
            "run {} -c importlib.metadata (mlx venv broken?)",
            venv_python.display()
        )
    })?;
    if !version_out.status.success() {
        return Err(anyhow!(
            "mlx-lm version probe exited {}: {}",
            version_out.status,
            String::from_utf8_lossy(&version_out.stderr)
        ));
    }
    let version_raw = format!(
        "mlx-lm {}",
        String::from_utf8_lossy(&version_out.stdout).trim()
    );

    // Display-only serial from the install tag (mlx-0.26.5 ->
    // 0000026005), same convention as the sglang/mistralrs lanes.
    let tag_serial = tag
        .strip_prefix("mlx-")
        .and_then(|rest| rest.split(['-', '+']).next())
        .and_then(|v| {
            let mut it = v.split('.');
            let maj = it.next()?.parse::<u64>().ok()?;
            let min = it.next().unwrap_or("0").parse::<u64>().ok()?;
            let patch = it.next().unwrap_or("0").parse::<u64>().ok()?;
            Some(maj * 1_000_000 + min * 1_000 + patch)
        })
        .unwrap_or(0);

    let help_out = crate::probe::probe_output(Command::new(server).arg("--help"), 120)
        .with_context(|| {
            format!("run {server} --help (mlx import chain can take a minute cold)")
        })?;
    let help = String::from_utf8_lossy(&help_out.stdout).to_string();
    let (flags, _) = parse_help(&help);
    if flags.is_empty() {
        return Err(anyhow!(
            "mlx-lm --help parsed to zero flags (output format changed upstream?): {}",
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
        probe_version: MANIFEST_PROBE_VERSION,
        ..Default::default()
    })
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

    // Best-effort banner; never fatal - the tag is authoritative.
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
    // set downgrades argv gating to "emit the fixed dialect" - a stale
    // flag then fails at spawn, loudly.
    let mut flags = BTreeSet::new();
    // NEVER probe with zero args: a bare `mistralrs` drops into serving
    // mode and listens forever, wedging the sync Command::output() call
    // (and the tokio worker under it). Both help forms exit on their own.
    let help_invocations: [Vec<String>; 2] =
        [vec!["--help".into()], vec!["serve".into(), "--help".into()]];
    for args in help_invocations {
        // Same bounded transient-retry policy as every other probe spawn:
        // an EAGAIN under load must not silently shrink the flag set.
        if let Ok(out) =
            crate::probe::with_spawn_retry(|| Command::new(server).args(&args).output())
        {
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
        probe_version: MANIFEST_PROBE_VERSION,
        ..Default::default()
    })
}

/// Probe a sglang venv install through its shim script. Layout contract
/// (`sglang_install.rs)`: `engines/<tag>/sglang-server` is an executable
/// shim `exec <dir>/venv/bin/python -m sglang.launch_server "$@"`, so
/// the shim's sibling `venv/` holds the interpreter.
///
/// Divergences from the other engines (verified against sglang v0.5.19):
/// - no `--version` flag on `launch_server`; the version comes from
///   `importlib.metadata` against the venv python (a metadata read - no
///   torch import, seconds even cold).
/// - flags come from `launch_server --help` (standard argparse renderer,
///   `server_args.py:345 add_cli_args`) - but the import chain behind it
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

    // Version: importlib metadata read. Strict - a broken venv must fail
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
    // zero-token parse is a FORMAT change upstream - treat as probe
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

    // Quant registry: the loader's own admission list, read through the
    // shared helper (the boot backfill reads the identical source).
    let quant_methods = probe_quant_registry(&venv_python)?;
    // Speculative-algorithm vocabulary: enum + plugin registry. LENIENT
    // (unlike the quant probe) - see probe_spec_algorithms.
    let spec_types = probe_spec_algorithms(&venv_python);

    Ok(Manifest {
        tag: tag.to_string(),
        build_number: tag_serial,
        version_raw,
        devices: Vec::new(),
        flags,
        spec_types,
        server_path: server.to_string(),
        quant_methods: Some(quant_methods),
        probe_version: MANIFEST_PROBE_VERSION,
        ..Default::default()
    })
}

/// Read the loader's quant admission registry from a sglang lane's venv
/// python. Shared by the install-time probe and the boot backfill so both
/// observe the identical runtime list. Strict - it runs after the import
/// chain proved healthy (--help at install, a recorded manifest at boot),
/// so a failure here means a broken venv or an upstream restructure, and a
/// silently missing list would quietly degrade every admission refusal
/// back to the const snapshot. Fresh interpreter = the torch import chain
/// pays again: 90s cold budget.
pub(crate) fn probe_quant_registry(venv_python: &Path) -> Result<Vec<String>> {
    let quant_out = crate::probe::probe_output(
        Command::new(venv_python).arg("-c").arg(
            "import json; from sglang.srt.layers.quantization import \
                 BASE_QUANTIZATION_METHODS as M; print(json.dumps(sorted(M)))",
        ),
        90,
    )
    .with_context(|| {
        format!(
            "run {} -c BASE_QUANTIZATION_METHODS (sglang venv broken?)",
            venv_python.display()
        )
    })?;
    if !quant_out.status.success() {
        return Err(anyhow!(
            "sglang quant-methods probe exited {}: {}",
            quant_out.status,
            String::from_utf8_lossy(&quant_out.stderr)
        ));
    }
    let quant_methods = serde_json::from_str::<Vec<String>>(&String::from_utf8_lossy(
        &quant_out.stdout,
    ))
    .map_err(|err| anyhow!("sglang quant-methods probe printed a non-list payload: {err}"))?;
    if quant_methods.is_empty() {
        return Err(anyhow!(
            "sglang quant-methods probe parsed to zero methods \
             (loader registry restructured upstream?)"
        ));
    }
    Ok(quant_methods)
}

/// Read the lane's speculative-decoding algorithm vocabulary from the
/// venv's own runtime: the `SpeculativeAlgorithm` enum (static
/// algorithms: NGRAM, EAGLE3, DFLASH, ...) merged with the plugin
/// registry (entry-point group `sglang.srt.plugins`, e.g. `DSpine`'s
/// DSPINE until upstream merges it). Deliberately LENIENT where the
/// quant probe is strict: any failure (module restructured upstream,
/// broken plugin import, junk payload) yields an empty list, which the
/// profile compiler treats as "not probed" and gates on flag presence
/// alone - a speculative-module restructure must not brick engine
/// installs; the worst case is losing the algorithm-presence hard
/// error. Fresh interpreter = the torch import chain again: 90s cold
/// budget.
fn probe_spec_algorithms(venv_python: &Path) -> Vec<String> {
    const PAYLOAD: &str = "\
import json
names = set()
try:
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm as A
    names = {a.name for a in A}
except Exception:
    pass
try:
    from sglang.srt.plugins import load_plugins
    load_plugins()
    from sglang.srt.speculative.spec_registry import _REGISTRY
    names |= set(_REGISTRY.keys())
except Exception:
    pass
names.discard('NONE')
print(json.dumps(sorted(names)))";
    let Some(probe) =
        crate::probe::probe_output(Command::new(venv_python).arg("-c").arg(PAYLOAD), 90)
    else {
        tracing::warn!(
            "sglang spec-algorithm probe could not run - capability \
             gate falls back to flag presence"
        );
        return Vec::new();
    };
    if !probe.status.success() {
        tracing::warn!(
            "sglang spec-algorithm probe exited {} - capability gate falls \
             back to flag presence",
            probe.status
        );
        return Vec::new();
    }
    match serde_json::from_str::<Vec<String>>(&String::from_utf8_lossy(&probe.stdout)) {
        Ok(list) if !list.is_empty() => list,
        parsed => {
            tracing::warn!(
                "sglang spec-algorithm probe returned no usable list ({parsed:?}) \
                 - capability gate falls back to flag presence"
            );
            Vec::new()
        }
    }
}

/// Probe a litert-lm venv lane (the openaihost engine). Same shape as
/// sglang/mlx (the lanes it mirrors): importlib.metadata version read
/// through the venv python, display serial from the install tag,
/// argparse flags via the shim's `serve --help` - `serve` is the only
/// subcommand whose flags gate this lane's argv, so that is the usage
/// surface the manifest mines. The python import chain is cold-boot
/// class, hence the 120s budget. Zero parsed flags = upstream format
/// change = probe failure, not a silently degraded manifest.
fn probe_openaihost(server_path: &Path, tag: &str) -> Result<Manifest> {
    let server = server_path
        .to_str()
        .ok_or_else(|| anyhow!("non-UTF-8 engine path {}", server_path.display()))?;
    // cpython venv layout: bin/python on unix, Scripts\python.exe on
    // Windows - one shared helper so probe and install never disagree.
    let venv_python = crate::engine::litert_install::venv_python_path(
        &server_path
            .parent()
            .ok_or_else(|| anyhow!("engine path {server} has no parent dir"))?
            .join("venv"),
    );

    // Version: importlib metadata read. Strict - a broken venv must
    // fail the probe here with the venv path named, not at spawn.
    let version_out = crate::probe::probe_output(
        Command::new(&venv_python)
            .arg("-c")
            .arg("import importlib.metadata as m; print(m.version('litert_lm'))"),
        30,
    )
    .with_context(|| {
        format!(
            "run {} -c importlib.metadata (litert-lm venv broken?)",
            venv_python.display()
        )
    })?;
    if !version_out.status.success() {
        return Err(anyhow!(
            "litert-lm version probe exited {}: {}",
            version_out.status,
            String::from_utf8_lossy(&version_out.stderr)
        ));
    }
    let version_raw = format!(
        "litert-lm {}",
        String::from_utf8_lossy(&version_out.stdout).trim()
    );

    // Display-only serial from the install tag (litert-0.19.1 ->
    // 0000000019): keeps `engine list` sortable, same convention as the
    // sglang/mistralrs lanes.
    let tag_serial = tag
        .strip_prefix("litert-")
        .and_then(|rest| rest.split(['-', '+']).next())
        .and_then(|v| {
            let mut it = v.split('.');
            let maj = it.next()?.parse::<u64>().ok()?;
            let min = it.next().unwrap_or("0").parse::<u64>().ok()?;
            let patch = it.next().unwrap_or("0").parse::<u64>().ok()?;
            Some(maj * 1_000_000 + min * 1_000 + patch)
        })
        .unwrap_or(0);

    // Flags: `serve --help` via the shim. Argparse renderer, so
    // parse_help applies. Python import behind it: 120s cold budget. A
    // zero-token parse is a FORMAT change upstream - treat as probe
    // failure (fail fast) rather than degrading every gated emission.
    let help_out = crate::probe::probe_output(Command::new(server).arg("serve").arg("--help"), 120)
        .with_context(|| {
            format!("run {server} serve --help (litert-lm import chain can take a minute cold)")
        })?;
    let help = String::from_utf8_lossy(&help_out.stdout).to_string();
    let (flags, _) = parse_help(&help);
    if flags.is_empty() {
        return Err(anyhow!(
            "litert-lm serve --help parsed to zero flags (output format changed upstream?): {}",
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
        probe_version: MANIFEST_PROBE_VERSION,
        ..Default::default()
    })
}

/// Explain a zero-flag `--help` parse honestly. Two very different
/// stories hide behind "no flags parsed":
/// - non-zero exit: the binary never reached its usage path. The
///   dynamic loader is the common culprit - a prebuilt linked against
///   a newer distro's libc (e.g. an Ubuntu 26.04 asset needing
///   `GLIBC_2.43`) exits 127 with `error while loading shared
///   libraries: ...` on stderr and prints NOTHING to stdout, which
///   used to masquerade as an upstream format change. Report the exit
///   code plus the first stderr line verbatim, with a libc teaching
///   when the loader signature is present.
/// - exit 0 with no parseable flags: genuinely an upstream usage
///   format change (or an unknown regression) - keep the original
///   message with a usage preview.
fn zero_flag_help_error(binary: &str, out: &Output, preview: &str) -> anyhow::Error {
    let stderr_text = String::from_utf8_lossy(&out.stderr);
    if !out.status.success() {
        let code = out
            .status
            .code()
            .map_or_else(|| "killed by signal".to_string(), |c| c.to_string());
        let first_nonempty = stderr_text
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .unwrap_or("<no stderr output>");
        let loader_signature = [
            "error while loading shared libraries",
            "GLIBC_",
            "cannot open shared object",
        ]
        .iter()
        .any(|needle| stderr_text.contains(needle));
        let mut msg = format!("{binary} --help exited {code}: {first_nonempty}");
        if loader_signature {
            msg.push_str(
                " (prebuilt targets a newer libc than this host - check the asset distro)",
            );
        }
        return anyhow!("{msg}");
    }
    anyhow!("{binary} --help parsed to zero flags (output format changed upstream?): {preview}")
}

/// Probe an sd.cpp `sd-server` binary. Divergences from llama-server
/// (verified against stable-diffusion.cpp master-890-74988b2):
/// - `--version` prints `stable-diffusion.cpp version unknown, commit
///   74988b2` and exits 0, but the version word is literally "unknown"
///   on rolling master builds - the banner is display-only; the commit
///   hash in the INSTALL TAG (`master-890-74988b2`) is the identity.
/// - `-h` prints the full usage to stdout and exits 0 (llama parity, so
///   `parse_help` applies). NEVER probe with zero args: a bare
///   `sd-server` demands `model_path/diffusion_model` and exits 1.
/// - `--list-devices` prints `NAME<TAB>description` per line - a
///   different shape from llama's `NAME: DESC (TOTAL MiB, FREE MiB
///   free)`, so it gets its own parser and no MiB numbers (sd.cpp does
///   not report VRAM there).
fn probe_sdcpp(server_path: &Path, tag: &str) -> Result<Manifest> {
    let server = server_path
        .to_str()
        .ok_or_else(|| anyhow!("non-UTF-8 engine path {}", server_path.display()))?;

    // Best-effort banner; never fatal - the tag is authoritative.
    let version_raw = match crate::probe::probe_output(Command::new(server).arg("--version"), 30) {
        Some(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            let text = if stdout.contains("stable-diffusion.cpp") {
                stdout
            } else {
                stderr
            };
            text.lines()
                .map(str::trim)
                .find(|l| !l.is_empty())
                .unwrap_or("stable-diffusion.cpp")
                .to_string()
        }
        _ => format!("stable-diffusion.cpp {tag}"),
    };

    // Display-only serial from the install tag (`master-890-74988b2` ->
    // 890, the upstream build counter): keeps `engine list` sortable.
    let build_number = tag
        .strip_prefix("master-")
        .and_then(|rest| rest.split('-').next())
        .and_then(|n| n.parse::<u64>().ok())
        .unwrap_or(0);

    // Flags: `-h`/`--help` both exit 0 with usage on stdout (verified
    // master-890). A zero-flag parse is an upstream format change -
    // fail the probe loudly rather than degrading argv gating.
    let help_out = crate::probe::probe_output(Command::new(server).arg("--help"), 30)
        .with_context(|| format!("run {server} --help (timed out or failed to spawn)"))?;
    let help = String::from_utf8_lossy(&help_out.stdout).to_string();
    let (flags, _) = parse_help(&help);
    if flags.is_empty() {
        let preview = help.lines().take(3).collect::<Vec<_>>().join(" | ");
        return Err(zero_flag_help_error("sd-server", &help_out, &preview));
    }

    let devices = run_sd_list_devices(Path::new(server));

    Ok(Manifest {
        tag: tag.to_string(),
        build_number,
        version_raw,
        devices,
        flags,
        spec_types: Vec::new(),
        server_path: server.to_string(),
        probe_version: MANIFEST_PROBE_VERSION,
        ..Default::default()
    })
}

/// `sd-server --list-devices` census. Failure-tolerant like
/// [`run_list_devices`]: a missing binary or a hung run yields an empty
/// list and callers fall back.
fn run_sd_list_devices(server: &Path) -> Vec<DeviceDesc> {
    let Some(out) = crate::probe::probe_output(Command::new(server).arg("--list-devices"), 30)
    else {
        return Vec::new();
    };
    parse_sd_devices(&String::from_utf8_lossy(&out.stdout))
}

/// Parse sd.cpp device lines: `NAME<TAB>description` (e.g.
/// `Vulkan1<TAB>NVIDIA GeForce RTX 4070 Laptop GPU`). Backend log lines
/// (`ggml_vulkan: Found 2 Vulkan devices`) carry no tab and drop out.
pub(crate) fn parse_sd_devices(text: &str) -> Vec<DeviceDesc> {
    text.lines()
        .filter_map(|line| {
            let (name, description) = line.trim().split_once('\t')?;
            if name.is_empty() || description.is_empty() {
                return None;
            }
            // sd-server also lists a `CPU<TAB>...` row (its --backend
            // accepts per-module cpu assignment). The host CPU is not an
            // accelerator: kept rows feed the GPU census, where a CPU
            // entry can only mislead placement (`--device CPU` won a
            // free-MiB tie on a CUDA box and killed llama spawns).
            if name.eq_ignore_ascii_case("cpu") {
                return None;
            }
            Some(DeviceDesc {
                name: name.to_string(),
                description: description.to_string(),
                total_mib: 0,
                free_mib: 0,
            })
        })
        .collect()
}
/// Probe a whisper-server binary. Divergences from llama-server
/// (verified against ggml-org/whisper.cpp b5130): no `--version` flag
/// at all (`error: unknown argument: --version`) - the b-tag is the
/// identity; `-h`/`--help` exit 0 with usage on stdout in the same
/// shape llama parses; no `--list-devices` (CPU-only serving contract).
fn probe_whisper(server_path: &Path, tag: &str) -> Result<Manifest> {
    let server = server_path
        .to_str()
        .ok_or_else(|| anyhow!("non-UTF-8 engine path {}", server_path.display()))?;

    // Tags are b-tags (`b5130`) - the upstream build counter is the
    // sortable identity, exactly like the llama lane.
    let build_number = super::gh::btag_number(tag).unwrap_or(0);
    let version_raw = format!("whisper.cpp {tag}");

    // Flags: `--help` exits 0 - but whisper.cpp prints its usage to
    // STDERR (live-verified b5130: 0 bytes stdout, ~4.8 KiB stderr),
    // unlike the llama lanes. Parse stderr first, stdout as fallback.
    // A zero-flag parse is an upstream format change - fail the probe
    // loudly rather than degrading argv gating.
    let help_out = crate::probe::probe_output(Command::new(server).arg("--help"), 30)
        .with_context(|| format!("run {server} --help (timed out or failed to spawn)"))?;
    let err_help = String::from_utf8_lossy(&help_out.stderr).to_string();
    let out_help = String::from_utf8_lossy(&help_out.stdout).to_string();
    let help_text = if err_help.trim().is_empty() {
        &out_help
    } else {
        &err_help
    };
    let (flags, _) = parse_help(help_text);
    if flags.is_empty() {
        let preview = err_help
            .lines()
            .chain(out_help.lines())
            .take(3)
            .collect::<Vec<_>>()
            .join(" | ");
        return Err(zero_flag_help_error("whisper-server", &help_out, &preview));
    }

    Ok(Manifest {
        tag: tag.to_string(),
        build_number,
        version_raw,
        devices: Vec::new(),
        flags,
        spec_types: Vec::new(),
        server_path: server.to_string(),
        probe_version: MANIFEST_PROBE_VERSION,
        ..Default::default()
    })
}

/// Probe a piper binary. Lazy audio lane like whisper, with two tag
/// divergences: releases are version tags (`v1.2.0`) or date tags
/// (`2023.11.14-2`), never b-tags - the numeric identity is the
/// concatenation of the tag's digit runs (`202311142`), monotonic
/// within a scheme and used only for listings (rollback orders by
/// installed rows, not build numbers). And the usage table's stream
/// varies across builds, so whichever stream is non-empty is parsed
/// (whisper-style tolerance); a zero-flag parse fails the probe
/// loudly rather than degrading argv gating.
fn probe_piper(server_path: &Path, tag: &str) -> Result<Manifest> {
    let server = server_path
        .to_str()
        .ok_or_else(|| anyhow!("non-UTF-8 engine path {}", server_path.display()))?;

    let build_number = tag
        .chars()
        .filter(char::is_ascii_digit)
        .take(18)
        .collect::<String>()
        .parse()
        .unwrap_or(0);
    let version_raw = format!("piper {tag}");

    let help_out = crate::probe::probe_output(Command::new(server).arg("--help"), 30)
        .with_context(|| format!("run {server} --help (timed out or failed to spawn)"))?;
    let err_help = String::from_utf8_lossy(&help_out.stderr).to_string();
    let out_help = String::from_utf8_lossy(&help_out.stdout).to_string();
    let help_text = if out_help.trim().is_empty() {
        &err_help
    } else {
        &out_help
    };
    let (flags, _) = parse_help(help_text);
    if flags.is_empty() {
        let preview = err_help
            .lines()
            .chain(out_help.lines())
            .take(3)
            .collect::<Vec<_>>()
            .join(" | ");
        return Err(zero_flag_help_error("piper", &help_out, &preview));
    }

    Ok(Manifest {
        tag: tag.to_string(),
        build_number,
        version_raw,
        devices: Vec::new(),
        flags,
        spec_types: Vec::new(),
        server_path: server.to_string(),
        probe_version: MANIFEST_PROBE_VERSION,
        ..Default::default()
    })
}

/// Probe a kokoro lane. The shim is blazar's own script, so there is no
/// upstream CLI banner or flag surface to parse: identity is the
/// kokoro-onnx pip version read through the venv python (the sglang/mlx
/// venv convention), the display serial is the model version in the tag
/// (`kokoro-1.0` -> 0000001000), and liveness is the install contract's
/// file set (shim, venv python, model, voices). Flags stay empty by
/// design: the lane speaks HTTP JSON, never CLI argv.
fn probe_kokoro(server_path: &Path, tag: &str) -> Result<Manifest> {
    let server = server_path
        .to_str()
        .ok_or_else(|| anyhow!("non-UTF-8 engine path {}", server_path.display()))?;
    let lane = server_path
        .parent()
        .ok_or_else(|| anyhow!("engine path {server} has no parent dir"))?;

    for (what, path) in [
        ("shim", server_path.to_path_buf()),
        ("venv python", lane.join("venv").join("bin").join("python")),
        ("model", lane.join(super::kokoro_install::MODEL_FILE)),
        ("voices", lane.join(super::kokoro_install::VOICES_FILE)),
    ] {
        if !path.is_file() {
            anyhow::bail!(
                "kokoro lane {} is missing (install contract broken?): {}",
                what,
                path.display()
            );
        }
    }

    let venv_python = lane.join("venv").join("bin").join("python");
    let version_out = crate::probe::probe_output(
        Command::new(&venv_python)
            .arg("-c")
            .arg("import importlib.metadata as m; print(m.version('kokoro_onnx'))"),
        30,
    )
    .with_context(|| {
        format!(
            "run {} -c importlib.metadata (kokoro venv broken?)",
            venv_python.display()
        )
    })?;
    if !version_out.status.success() {
        return Err(anyhow!(
            "kokoro-onnx version probe exited {}: {}",
            version_out.status,
            String::from_utf8_lossy(&version_out.stderr)
        ));
    }
    let version_raw = format!(
        "kokoro-onnx {} (model {})",
        String::from_utf8_lossy(&version_out.stdout).trim(),
        tag
    );

    // Display-only serial from the tag's model version (kokoro-1.0 ->
    // 0000001000), the mlx tag-serial convention.
    let build_number = tag
        .strip_prefix("kokoro-")
        .and_then(|rest| rest.split(['-', '+']).next())
        .and_then(|v| {
            let mut it = v.split('.');
            let maj = it.next()?.parse::<u64>().ok()?;
            let min = it.next().unwrap_or("0").parse::<u64>().ok()?;
            let patch = it.next().unwrap_or("0").parse::<u64>().ok()?;
            Some(maj * 1_000_000 + min * 1_000 + patch)
        })
        .unwrap_or(0);

    Ok(Manifest {
        tag: tag.to_string(),
        build_number,
        version_raw,
        devices: Vec::new(),
        flags: BTreeSet::new(),
        spec_types: Vec::new(),
        server_path: server.to_string(),
        probe_version: MANIFEST_PROBE_VERSION,
        ..Default::default()
    })
}

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
/// ONLY source of live VRAM numbers - the manifest's stored `devices`
/// are an install-day snapshot and go stale the moment any other
/// process (ollama, a desktop session) touches the card.
///
/// Failure-tolerant by design: a missing binary or a hung census
/// returns an empty list and callers fall back to the manifest
/// snapshot. Census runs take a few hundred milliseconds (backend
/// init), so callers throttle to spawn-time and >=60s periodic.
#[must_use]
pub fn run_list_devices(server: &Path) -> Vec<DeviceDesc> {
    run_list_devices_with_env(server, &[])
}

/// [`run_list_devices`] with extra environment for the census child (the
/// serving child's `engine_env`, so the census sees what a real spawn
/// would see). The process environment is never mutated.
#[must_use]
pub fn run_list_devices_with_env(server: &Path, extra_env: &[(String, String)]) -> Vec<DeviceDesc> {
    // F85: the doc's "a hung census returns an empty list" is now
    // literally true - a deadline-bounded probe replaces the blocking
    // `.output()` that would wedge forever.
    let Some(out) = crate::probe::probe_output(
        Command::new(server)
            .arg("--list-devices")
            .envs(extra_env.iter().map(|(k, v)| (k, v))),
        30,
    ) else {
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
                // Values are separate tokens; keep plain option names
                // only. Underscores included: sd-server documents
                // `--llm_vision`/`--clip_vision` style flags and the
                // strict probed-flags gate later refuses any argv flag
                // the manifest lacks - dropping them here disabled
                // image edits at spawn (verified live). Dots included:
                // llama-server ships version-qualified flags like
                // `--fim-qwen-1.5b-default`; all-dash tokens are help
                // section separators (`-----`), never flag names.
                if clean.chars().all(|c| {
                    c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_' || c == '.'
                }) && clean.chars().any(|c| c != '-')
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

    /// Platform-neutral `Output` seam for the zero-flag probe-error
    /// pins: `code` 0 renders a success status, anything else a
    /// non-zero exit.
    fn help_output(code: i32, stdout: &str, stderr: &str) -> Output {
        #[cfg(unix)]
        let status = {
            use std::os::unix::process::ExitStatusExt;
            std::process::ExitStatus::from_raw(code << 8)
        };
        #[cfg(not(unix))]
        let status = {
            use std::os::windows::process::ExitStatusExt;
            std::process::ExitStatus::from_raw(
                u32::try_from(code).expect("help_output exit codes are non-negative"),
            )
        };
        Output {
            status,
            stdout: stdout.as_bytes().to_vec(),
            stderr: stderr.as_bytes().to_vec(),
        }
    }

    #[test]
    fn unit__zero_flag_help_error__loader_failure_names_exit_stderr_and_teaches() {
        // Verified live: sd.cpp master-1050 on an Ubuntu 24.04 host -
        // exit 127, empty stdout, the dynamic-loader error on stderr.
        let out = help_output(
            127,
            "",
            "sd-server: error while loading shared libraries: libggml-base.so.0: \
             cannot open shared object file: No such file or directory\n",
        );
        let err = zero_flag_help_error("sd-server", &out, "").to_string();
        assert!(err.contains("exited 127"), "{err}");
        assert!(
            err.contains("error while loading shared libraries"),
            "{err}"
        );
        assert!(
            err.contains("newer libc than this host"),
            "missing the distro teaching: {err}"
        );
    }

    #[test]
    fn unit__zero_flag_help_error__nonzero_exit_without_loader_signature_stays_plain() {
        let out = help_output(2, "", "unknown flag --help\n");
        let err = zero_flag_help_error("piper", &out, "").to_string();
        assert!(err.contains("exited 2"), "{err}");
        assert!(err.contains("unknown flag --help"), "{err}");
        assert!(!err.contains("libc"), "{err}");
    }

    #[test]
    fn unit__zero_flag_help_error__exit_zero_keeps_format_change_message() {
        // A usage that parses to zero flags with a successful exit is
        // still the upstream-format-change story (unchanged behavior).
        let out = help_output(0, "usage: sd-server\n--model-path\n", "");
        let err =
            zero_flag_help_error("sd-server", &out, "usage: sd-server | --model-path").to_string();
        assert!(err.contains("output format changed upstream"), "{err}");
        assert!(err.contains("usage: sd-server"), "{err}");
    }

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
    fn unit__parse_sd_devices__tab_shape_drops_log_lines() {
        // Verified live against sd-server master-890-74988b2 on the
        // 2-GPU dev box: `NAME<TAB>description`, MiB columns absent.
        let text = "ggml_vulkan: Found 2 Vulkan devices\nVulkan0\tIntel(R) Iris Xe Graphics\nVulkan1\tNVIDIA GeForce RTX 4070 Laptop GPU\n\nCPU\tIntel(R) Core(TM) i7-14650HX\nCPU\t\n";
        let devices = parse_sd_devices(text);
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].name, "Vulkan0");
        assert!(devices[0].description.contains("Iris Xe"));
        assert_eq!(devices[1].name, "Vulkan1");
        assert!(devices[1].description.contains("4070"));
        assert_eq!(devices[0].total_mib, 0);
        // The host-CPU row is not an accelerator even with a populated
        // description (it won a placement tie and killed llama spawns);
        // backend log line carries no tab; empty-description line drops
    }

    #[test]
    fn unit__re_root_server_path__relocated_row_adopts_live_engines_dir() {
        // Sandbox regression (2026-09-15): a DB copied from another data
        // dir kept the old absolute server_path and failed ENOENT at
        // spawn while the binary sat intact under the live engines dir.
        let tmp = tempfile::tempdir().unwrap();
        let engines = tmp.path().join("data/blazar/engines");
        let live_bin = engines
            .join("b10970-cuda")
            .join("llama-b10970-cuda")
            .join("llama-server");
        std::fs::create_dir_all(live_bin.parent().unwrap()).unwrap();
        std::fs::write(&live_bin, b"#!/bin/sh\n").unwrap();

        let mut m = Manifest {
            tag: "b10970-cuda".into(),
            build_number: 10970,
            version_raw: "version: b10970".into(),
            devices: Vec::new(),
            flags: BTreeSet::new(),
            spec_types: Vec::new(),
            server_path:
                "/home/other/.local/share/blazar/engines/b10970-cuda/llama-b10970-cuda/llama-server"
                    .into(),
            ..Default::default()
        };
        assert!(m.re_root_server_path(&engines));
        assert_eq!(m.server_path, live_bin.display().to_string());
    }

    #[test]
    fn unit__re_root_server_path__recorded_path_wins_when_present() {
        let tmp = tempfile::tempdir().unwrap();
        let engines = tmp.path().join("engines");
        std::fs::create_dir_all(&engines).unwrap();

        let present = tmp.path().join("original/llama-server");
        std::fs::create_dir_all(present.parent().unwrap()).unwrap();
        std::fs::write(&present, b"#!/bin/sh\n").unwrap();

        let mut m = Manifest {
            tag: "local".into(),
            build_number: 0,
            version_raw: String::new(),
            devices: Vec::new(),
            flags: BTreeSet::new(),
            spec_types: Vec::new(),
            server_path: present.display().to_string(),
            ..Default::default()
        };
        assert!(!m.re_root_server_path(&engines));
        assert_eq!(m.server_path, present.display().to_string());
    }

    #[test]
    fn unit__re_root_server_path__no_tail_anywhere__fail_loud_unmodified() {
        // Neither the recorded path nor a live tail exists (and no
        // "engines" marker at all, e.g. the `local` pseudo-engine): the
        // manifest is left untouched so spawn fails loudly with ENOENT.
        let tmp = tempfile::tempdir().unwrap();
        let engines = tmp.path().join("engines");
        std::fs::create_dir_all(&engines).unwrap();

        let mut m = Manifest {
            tag: "local".into(),
            build_number: 0,
            version_raw: String::new(),
            devices: Vec::new(),
            flags: BTreeSet::new(),
            spec_types: Vec::new(),
            server_path: "/gone/custom/llama-server".into(),
            ..Default::default()
        };
        assert!(!m.re_root_server_path(&engines));
        assert_eq!(m.server_path, "/gone/custom/llama-server");
    }

    fn bare_manifest(server_path: &str) -> Manifest {
        Manifest {
            tag: "b1-cuda".into(),
            build_number: 1,
            version_raw: "version: b1".into(),
            devices: Vec::new(),
            flags: BTreeSet::new(),
            spec_types: Vec::new(),
            server_path: server_path.into(),
            ..Default::default()
        }
    }

    #[test]
    fn unit__relativize_server_path__in_tree_folds_to_data_dir_relative() {
        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path().join("data/blazar");
        let mut m = bare_manifest(
            &data
                .join("engines/b1-cuda/llama-b1-cuda/llama-server")
                .display()
                .to_string(),
        );
        assert!(m.relativize_server_path(&data));
        assert_eq!(m.server_path, "engines/b1-cuda/llama-b1-cuda/llama-server");
        // Already-relative is a no-op (idempotent).
        assert!(!m.relativize_server_path(&data));
    }

    #[test]
    fn unit__relativize_server_path__out_of_tree_stays_absolute() {
        // The `local` lane's BLAZAR_ENGINE_PATH binary never lives under
        // the data dir - relativizing it would corrupt a user-owned path.
        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path().join("data/blazar");
        let mut m = bare_manifest("/opt/llama.cpp/llama-server");
        assert!(!m.relativize_server_path(&data));
        assert_eq!(m.server_path, "/opt/llama.cpp/llama-server");
    }

    #[test]
    fn unit__anchor_server_path__relative_row_round_trips_to_absolute() {
        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path().join("data/blazar");
        let mut m = bare_manifest("engines/b1-cuda/llama-b1-cuda/llama-server");
        m.anchor_server_path(&data);
        assert_eq!(
            m.server_path,
            data.join("engines/b1-cuda/llama-b1-cuda/llama-server")
                .display()
                .to_string()
        );
    }

    #[test]
    fn unit__anchor_server_path__legacy_absolute_row_heals_to_live_engines_dir() {
        // A row recorded before the relative invariant, installed under a
        // data dir that has since moved: anchor adopts the live tail.
        // The legacy root derives from the tempdir so the row is a REAL
        // absolute path on every platform (a "/opt/..." literal is not
        // absolute on windows and never enters the heal path there).
        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path().join("data/blazar");
        let live_bin = data.join("engines/b1-cuda/llama-server");
        std::fs::create_dir_all(live_bin.parent().unwrap()).unwrap();
        std::fs::write(&live_bin, b"#!/bin/sh\n").unwrap();

        let legacy = tmp
            .path()
            .join("old-root/.local/share/blazar/engines/b1-cuda/llama-server");
        let mut m = bare_manifest(&legacy.display().to_string());
        m.anchor_server_path(&data);
        // Compare component-wise: the healer rebuilds the tail with the
        // platform separator while the staged literal keeps its forward
        // slashes - same file, different spelling on windows.
        assert_eq!(Path::new(&m.server_path), live_bin.as_path());
    }

    #[test]
    fn unit__anchor_server_path__external_absolute_path_left_untouched() {
        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path().join("data/blazar");
        std::fs::create_dir_all(&data).unwrap();
        // Absolute on the platform under test ("/gone/..." is not
        // absolute on windows and would take the re-rooting path).
        let external = if cfg!(windows) {
            "C:/gone/custom/llama-server"
        } else {
            "/gone/custom/llama-server"
        };
        let mut m = bare_manifest(external);
        m.anchor_server_path(&data);
        // No live tail anywhere: stays as recorded so spawn fails loudly
        // with ENOENT (the local-lane contract).
        assert_eq!(m.server_path, external);
    }

    #[test]
    fn unit__anchor_server_path__empty_path_stays_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path().join("data/blazar");
        let mut m = bare_manifest("");
        m.anchor_server_path(&data);
        assert_eq!(m.server_path, "");
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
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn unit__parse_help__underscore_flags_survive_the_charset_filter() {
        // sd-server documents `--llm_vision`/`--clip_vision` style flags.
        // The strict probed-flags gate refuses argv flags the manifest
        // lacks, so a charset filter that dropped '_' here disabled image
        // edits at spawn (verified live on master-890-74988b2).
        let help = "\
sd-server [options]
  --llm FNAME                text encoder
  --llm_vision FNAME         vision encoder for image edits
  --clip_vision FNAME        clip vision projector
  --qwen2vl_vision FNAME     qwen2vl projector
";
        let (flags, _) = parse_help(help);
        for expected in ["--llm", "--llm_vision", "--clip_vision", "--qwen2vl_vision"] {
            assert!(flags.contains(expected), "missing {expected} in {flags:?}");
        }
    }

    #[test]
    fn unit__parse_help__dotted_flags_kept_and_help_separators_dropped() {
        // llama-server b12202 documents `--fim-qwen-1.5b-default`
        // (help line 741). The strict probed-flags gate refused it as
        // extra_args until '.' joined the charset (F-S4-1). `-----`
        // section separators must never enter the vocab as noise.
        let help = "\
usage: llama-server [options]

options:
  --fim-qwen-1.5b-default   use Qwen 1.5B default FIM settings
  --fim-temp 0.0            temperature for FIM
----------------------------------------------------------
  --ctx-size N              size of the prompt context (default: 0)
";
        let (flags, _) = parse_help(help);
        assert!(
            flags.contains("--fim-qwen-1.5b-default"),
            "dotted flag must survive: {flags:?}"
        );
        assert!(flags.contains("--fim-temp"), "plain sibling: {flags:?}");
        assert!(flags.contains("--ctx-size"), "later flags: {flags:?}");
        for noise in ["-----", "---", "--"] {
            assert!(
                !flags.contains(noise),
                "separator noise {noise} in {flags:?}"
            );
        }
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

    fn rescue_row(tag: &str, archs: &[&str]) -> blazar_core::EngineRow {
        blazar_core::EngineRow {
            tag: tag.to_string(),
            asset: "asset".to_string(),
            sha256: "0".to_string(),
            installed_at: 0,
            active: false,
            manifest: serde_json::json!({
                "tag": tag,
                "build_number": 1,
                "version_raw": "version: 1",
                "devices": [],
                "flags": [],
                "spec_types": [],
                "server_path": "/x/llama-server",
                "architectures": archs,
            })
            .to_string(),
            kind: blazar_core::engine_kind::EngineKind::LlamaCpp,
        }
    }

    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__predicted_rescue_lane__marks_rescue_only_when_provable() {
        let rows = vec![
            rescue_row("b-main", &["qwen2"]),
            rescue_row("b-adv", &["instella-moe"]),
        ];
        let llama = blazar_core::engine_kind::EngineKind::LlamaCpp;
        // Picked provably lacks the arch, an advertiser exists.
        assert_eq!(
            predicted_rescue_lane(&rows, Some("instella-moe"), None, "b-main", llama),
            Some("b-adv".to_string())
        );
        // Arch advertised by the picked lane: no rescue.
        assert_eq!(
            predicted_rescue_lane(&rows, Some("qwen2"), None, "b-main", llama),
            None
        );
        // User pin blocks the prediction (spawn-rescue gate parity).
        assert_eq!(
            predicted_rescue_lane(&rows, Some("instella-moe"), Some("b-main"), "b-main", llama),
            None
        );
        // No advertiser for the arch.
        assert_eq!(
            predicted_rescue_lane(&rows, Some("mystery"), None, "b-main", llama),
            None
        );
        // Honest unknown: a v1 row (no architectures key) cannot prove a
        // lack, so it never triggers a prediction.
        let v1 = blazar_core::EngineRow {
            manifest: serde_json::json!({
                "tag": "b-v1",
                "build_number": 1,
                "version_raw": "version: 1",
                "devices": [],
                "flags": [],
                "spec_types": [],
                "server_path": "/x/llama-server"
            })
            .to_string(),
            ..rescue_row("b-v1", &[])
        };
        assert_eq!(
            predicted_rescue_lane(
                &[v1, rows[1].clone()],
                Some("instella-moe"),
                None,
                "b-v1",
                llama
            ),
            None
        );
        // Unknown model arch, and non-llamacpp picked kinds.
        assert_eq!(
            predicted_rescue_lane(&rows, None, None, "b-main", llama),
            None
        );
        assert_eq!(
            predicted_rescue_lane(
                &rows,
                Some("Qwen2ForCausalLM"),
                None,
                "b-main",
                blazar_core::engine_kind::EngineKind::Sglang
            ),
            None
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
            ..Default::default()
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
    // Manifest v2: provenance fields, v1 decode compat, capability set
    // ------------------------------------------------------------------

    #[test]
    fn unit__manifest__v1_json_decodes_as_upstream_with_no_archs() {
        // Rows written by pre-v2 Blazar carry only the seven v1 fields;
        // they must decode losslessly (serde defaults) as upstream lanes
        // that advertise nothing - never a fork, never a blocked lane.
        let v1 = r#"{
            "tag": "b10980-cuda",
            "build_number": 10980,
            "version_raw": "version: 0.4.0",
            "devices": [],
            "flags": ["--ctx-size"],
            "spec_types": [],
            "server_path": "/x/llama-server"
        }"#;
        let m: Manifest = serde_json::from_str(v1).expect("v1 decode");
        assert_eq!(m.source, EngineSource::Upstream);
        assert_eq!(m.repo, None);
        assert_eq!(m.ref_pin, None);
        assert_eq!(m.base_ref, None);
        assert_eq!(m.architectures.len(), 0);
        assert!(!m.advertises_arch("llama"));
        assert_eq!(m.provenance_label(), "");
    }

    #[test]
    fn unit__manifest__roundtrip_preserves_fork_provenance() {
        let mut m = Manifest {
            tag: "fork-acme_llama.cpp-7c81a9f0-cpu".into(),
            build_number: 0,
            version_raw: "version: dev".into(),
            devices: vec![],
            flags: BTreeSet::new(),
            spec_types: vec![],
            server_path: "/x/llama-server".into(),
            ..Default::default()
        };
        m.merge_provenance(&LaneProvenance {
            source: EngineSource::Fork,
            repo: Some("acme/llama.cpp".into()),
            ref_pin: Some("7c81a9f0123456789abcdef0123456789abcdef01".into()),
            base_ref: Some("b10980".into()),
            architectures: BTreeSet::from(["qwen35".into(), "llama".into()]),
            floor_release: Some(11_349),
        });
        let json = serde_json::to_string(&m).unwrap();
        let back: Manifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
        assert!(back.advertises_arch("qwen35"));
        assert!(!back.advertises_arch("qwen3"));
        assert_eq!(back.floor_release, Some(11_349));
        assert_eq!(
            back.provenance_label(),
            "fork acme/llama.cpp@7c81a9f0 (base b10980)"
        );
    }

    #[test]
    fn unit__manifest__local_label() {
        let m = Manifest {
            source: EngineSource::Local,
            ..Default::default()
        };
        assert_eq!(m.provenance_label(), "local");
    }

    // ------------------------------------------------------------------
    // sglang probe: venv-version + argparse --help contract
    // ------------------------------------------------------------------

    #[cfg(unix)]
    fn fake_sglang_install(dir: &std::path::Path) -> std::path::PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let root = dir.join("sglang-0.5.19");
        std::fs::create_dir_all(root.join("venv/bin")).expect("venv dir");
        // python stub: answers both -c probes by payload - the
        // importlib.metadata version read and the BASE_QUANTIZATION_
        // METHODS registry dump
        std::fs::write(
            root.join("venv/bin/python"),
            "#!/bin/sh\ncase \"$2\" in\n  *BASE_QUANTIZATION_METHODS*) echo '[\"awq\", \"fp8\"]' ;;\n  *SpeculativeAlgorithm*) echo '[\"DFLASH\", \"EAGLE3\", \"NGRAM\"]' ;;\n  *) echo 0.5.19 ;;\nesac\n",
        )
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
        assert_eq!(m.devices.len(), 0);
        // The probed algorithm vocabulary rides the manifest (enum +
        // plugin registry merged at probe time).
        assert_eq!(
            m.spec_types,
            vec![
                "DFLASH".to_string(),
                "EAGLE3".to_string(),
                "NGRAM".to_string()
            ]
        );
        assert_eq!(m.server_path, shim);
        // The probed loader registry rides the manifest.
        assert_eq!(
            m.quant_methods,
            Some(vec!["awq".to_string(), "fp8".to_string()])
        );
        assert_eq!(
            m.sglang_lane_facts(),
            blazar_core::hfmeta::SglangLaneFacts {
                version: "0.5.19".into(),
                quant_methods: Some(vec!["awq".into(), "fp8".into()]),
            }
        );
    }

    #[cfg(unix)]
    #[test]
    fn unit__probe_sglang__junk_spec_algorithm_payload_lenient_empty() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("sglang-0.5.19");
        std::fs::create_dir_all(root.join("venv/bin")).expect("venv dir");
        // A junk spec-algorithm payload must NOT fail the probe (unlike
        // the quant registry): the algorithm-presence gate degrades to
        // flag-presence gating with an empty list, never bricks the
        // install.
        std::fs::write(
            root.join("venv/bin/python"),
            "#!/bin/sh\ncase \"$2\" in\n  *BASE_QUANTIZATION_METHODS*) echo '[\"awq\"]' ;;\n  *SpeculativeAlgorithm*) echo 'garbage' ;;\n  *) echo 0.5.19 ;;\nesac\n",
        )
        .expect("python stub");
        std::fs::write(
            root.join("sglang-server"),
            "#!/bin/sh\nprintf 'usage: launch_server\\n\\n  --model-path M\\n'\n",
        )
        .expect("shim stub");
        for f in ["venv/bin/python", "sglang-server"] {
            let p = root.join(f);
            let mut perm = std::fs::metadata(&p).expect("meta").permissions();
            perm.set_mode(0o755);
            std::fs::set_permissions(p, perm).expect("chmod");
        }
        let m = probe_sglang(&root.join("sglang-server"), "sglang-0.5.19")
            .expect("lenient spec probe keeps the install alive");
        assert!(m.spec_types.is_empty());
        assert_eq!(m.quant_methods, Some(vec!["awq".to_string()]));
    }

    #[cfg(unix)]
    #[test]
    fn unit__probe_sglang__junk_quant_registry_fails_probe() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("sglang-0.5.19");
        std::fs::create_dir_all(root.join("venv/bin")).expect("venv dir");
        // Non-list payload for the registry probe: strict failure, not
        // a silently degraded const-fallback manifest.
        std::fs::write(
            root.join("venv/bin/python"),
            "#!/bin/sh\ncase \"$2\" in\n  *BASE_QUANTIZATION_METHODS*) echo 'not-json' ;;\n  *) echo 0.5.19 ;;\nesac\n",
        )
        .expect("python stub");
        std::fs::write(
            root.join("sglang-server"),
            "#!/bin/sh\nprintf 'usage: launch_server\\n\\n  --model-path M\\n'\n",
        )
        .expect("shim stub");
        for f in ["venv/bin/python", "sglang-server"] {
            let p = root.join(f);
            let mut perm = std::fs::metadata(&p).expect("meta").permissions();
            perm.set_mode(0o755);
            std::fs::set_permissions(p, perm).expect("chmod");
        }
        let err = probe_sglang(&root.join("sglang-server"), "sglang-0.5.19")
            .expect_err("junk registry fails the probe");
        assert!(format!("{err:#}").contains("quant-methods"), "{err:#}");
    }

    #[cfg(unix)]
    #[test]
    fn unit__probe_sglang__empty_quant_registry_fails_probe() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("sglang-0.5.19");
        std::fs::create_dir_all(root.join("venv/bin")).expect("venv dir");
        // An empty registry is a broken probe, not an engine that
        // loads nothing - refuse to record it.
        std::fs::write(
            root.join("venv/bin/python"),
            "#!/bin/sh\ncase \"$2\" in\n  *BASE_QUANTIZATION_METHODS*) echo '[]' ;;\n  *) echo 0.5.19 ;;\nesac\n",
        )
        .expect("python stub");
        std::fs::write(
            root.join("sglang-server"),
            "#!/bin/sh\nprintf 'usage: launch_server\\n\\n  --model-path M\\n'\n",
        )
        .expect("shim stub");
        for f in ["venv/bin/python", "sglang-server"] {
            let p = root.join(f);
            let mut perm = std::fs::metadata(&p).expect("meta").permissions();
            perm.set_mode(0o755);
            std::fs::set_permissions(p, perm).expect("chmod");
        }
        let err = probe_sglang(&root.join("sglang-server"), "sglang-0.5.19")
            .expect_err("empty registry fails the probe");
        assert!(format!("{err:#}").contains("zero methods"), "{err:#}");
    }

    #[test]
    fn unit__manifest__quant_methods_field_is_additive() {
        // Manifests serialized before the field existed decode with
        // `None` (const-fallback admission), not an error.
        let legacy = r#"{
            "tag": "sglang-0.5.21",
            "build_number": 5021,
            "version_raw": "sglang 0.5.21",
            "devices": [],
            "flags": [],
            "spec_types": [],
            "server_path": "engines/sglang-0.5.21/sglang-server"
        }"#;
        let m: Manifest = serde_json::from_str(legacy).expect("legacy manifest decodes");
        assert_eq!(m.quant_methods, None);
        let fresh = serde_json::json!({
            "tag": "sglang-0.5.23",
            "build_number": 5023,
            "version_raw": "sglang 0.5.23",
            "devices": [],
            "flags": [],
            "spec_types": [],
            "server_path": "engines/sglang-0.5.23/sglang-server",
            "quant_methods": ["awq", "fp8"],
        });
        let m: Manifest = serde_json::from_value(fresh).expect("fresh manifest decodes");
        assert_eq!(m.quant_methods, Some(vec!["awq".into(), "fp8".into()]));
    }

    #[test]
    fn unit__manifest__sglang_version_prefers_raw_then_tag() {
        let probed = Manifest {
            version_raw: "sglang 0.5.23".into(),
            tag: "sglang-0.5.23".into(),
            ..Default::default()
        };
        assert_eq!(probed.sglang_version(), "0.5.23");
        // No version_raw (legacy row): the tag still carries it.
        let tag_only = Manifest {
            tag: "sglang-0.5.23".into(),
            ..Default::default()
        };
        assert_eq!(tag_only.sglang_version(), "0.5.23");
        // Neither carries one: honest empty, never an invented version.
        assert_eq!(Manifest::default().sglang_version(), "");
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

    // ------------------------------------------------------------------
    // kokoro probe: venv-version + lane-file contract
    // ------------------------------------------------------------------

    #[cfg(unix)]
    fn fake_kokoro_install(dir: &std::path::Path) -> std::path::PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let root = dir.join("kokoro-1.0");
        std::fs::create_dir_all(root.join("venv/bin")).expect("venv dir");
        // python stub: answers the importlib.metadata version read
        std::fs::write(root.join("venv/bin/python"), "#!/bin/sh\necho 0.6.1\n")
            .expect("python stub");
        std::fs::write(root.join("kokoro.onnx"), "fake-onnx-bytes").expect("model file");
        std::fs::write(root.join("voices.bin"), "fake-voices-bytes").expect("voices file");
        let shim = root.join(super::super::kokoro_install::SHIM_NAME);
        std::fs::write(&shim, "#!/bin/sh\nexit 0\n").expect("shim stub");
        for f in ["venv/bin/python", super::super::kokoro_install::SHIM_NAME] {
            let p = root.join(f);
            let mut perm = std::fs::metadata(&p).expect("meta").permissions();
            perm.set_mode(0o755);
            std::fs::set_permissions(p, perm).expect("chmod");
        }
        shim
    }

    #[cfg(unix)]
    #[test]
    fn unit__probe_kokoro__stub_lane_yields_manifest() {
        let dir = tempfile::tempdir().expect("tempdir");
        let shim = fake_kokoro_install(dir.path());
        let m = probe_kokoro(&shim, "kokoro-1.0").expect("probe");
        assert_eq!(m.tag, "kokoro-1.0");
        assert_eq!(m.version_raw, "kokoro-onnx 0.6.1 (model kokoro-1.0)");
        // Model version serial: 1.0 -> 1_000_000 (mlx convention).
        assert_eq!(m.build_number, 1_000_000);
        // No upstream CLI to parse: the shim speaks HTTP, so no flags.
        assert_eq!(m.flags.len(), 0);
        assert_eq!(m.devices.len(), 0);
        assert_eq!(m.spec_types.len(), 0);
        assert_eq!(m.server_path, shim.to_str().unwrap());
    }

    #[cfg(unix)]
    #[test]
    fn unit__probe_kokoro__missing_model_file_fails_naming_it() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("kokoro-1.0");
        std::fs::create_dir_all(root.join("venv/bin")).expect("venv dir");
        std::fs::write(root.join("venv/bin/python"), "#!/bin/sh\necho 0.6.1\n").expect("python");
        std::fs::write(root.join("voices.bin"), "fake").expect("voices");
        let shim = root.join(super::super::kokoro_install::SHIM_NAME);
        std::fs::write(&shim, "#!/bin/sh\nexit 0\n").expect("shim");
        for f in ["venv/bin/python", super::super::kokoro_install::SHIM_NAME] {
            let p = root.join(f);
            let mut perm = std::fs::metadata(&p).unwrap().permissions();
            perm.set_mode(0o755);
            std::fs::set_permissions(p, perm).unwrap();
        }
        let err = probe_kokoro(&shim, "kokoro-1.0").expect_err("must fail");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("model") && msg.contains("kokoro.onnx"),
            "{msg}"
        );
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
        // and removed only the script file - every `cargo test` leaked an
        // empty /tmp/blazar-census-<pid> dir (200+ accumulated on the
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
        let devices = run_list_devices(std::path::Path::new("/nonexistent/blazar-census-probe"));
        assert_eq!(devices.len(), 0);
    }

    #[cfg(unix)]
    fn make_executable(path: &std::path::Path) {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(path).expect("stat").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(path, perms).expect("chmod");
    }
}
