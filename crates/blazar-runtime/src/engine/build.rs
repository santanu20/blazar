//! `blazar engine build`: compile llama.cpp's in-tree backends from
//! source into a normal installed engine. Closes the distribution gap
//! where upstream publishes no Linux-CUDA prebuilts: the CUDA backend
//! lives in the source tree (`ggml-cuda`: flash-attention with
//! quantized-KV `vec_dot`, `MMQ` quantized matmuls) and building it
//! locally yields the same probe/serve surface as a release asset,
//! tagged `bNNNN-cuda`.
//!
//! Authority rules: cmake and nvcc decide architecture compatibility —
//! Blazar only passes the GPU's compute capability (from `nvidia-smi`)
//! through and surfaces the compiler's own errors verbatim (tail).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};

use blazar_core::engine_kind::EngineKind;
use blazar_core::store::EngineRow;
use blazar_core::BlazarDirs;

use super::gh::LLAMA_CPP_REPO;
use super::manifest::{EngineSource, LaneProvenance};
use super::{
    discard_retired_engine, restore_retired_engine, retire_engine_dir, CancelledInstallGuard,
    EngineManager,
};

/// Where `engine build` compiles from. Upstream = a `bNNNN` tag of
/// `ggml-org/llama.cpp`; Fork = an immutable `owner/repo@sha` pin of a
/// llama.cpp fork (a temporary capability lane — see the README's
/// capability-lanes section).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildSource {
    Upstream,
    Fork {
        /// `owner/repo` slug (validated by [`validate_repo_slug`]; the
        /// `https://github.com/` prefix is constructed, never user input).
        repo: String,
        /// Full or abbreviated commit SHA (validated by
        /// [`validate_commit_sha`]); resolved to the full SHA after
        /// checkout.
        ref_sha: String,
        /// Upstream anchor to record as provenance (e.g. `b10980`);
        /// display-only in this phase.
        base_ref: Option<String>,
    },
}

/// `owner/repo` slug rules: exactly two non-empty segments of
/// `[A-Za-z0-9_.-]`. Rejects full URLs, `.git` suffixes, and anything
/// that could escape the github.com path we construct.
pub fn validate_repo_slug(repo: &str) -> Result<()> {
    let parts: Vec<&str> = repo.split('/').collect();
    let ok = parts.len() == 2
        && parts.iter().all(|p| {
            !p.is_empty()
                // "." and ".." are path-traversal segments: the host is
                // fixed so they cannot escape github.com, but the
                // constructed URL would silently normalize to a
                // different repo than the user named.
                && *p != "."
                && *p != ".."                // `repo.git` is a clone-URL spelling, not a slug — the
                // constructed github.com URL would double the suffix.
                && !p.rsplit_once('.')
                    .is_some_and(|(_, ext)| ext.eq_ignore_ascii_case("git"))
                && p.chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
        });
    if ok {
        Ok(())
    } else {
        Err(anyhow!(
            "invalid repo {repo:?} — pass an owner/name slug like ggml-org/llama.cpp \
             (letters, digits, _ . - only; no URL, no .git)"
        ))
    }
}

/// Commit SHA rules: 4-40 hex digits (full or abbreviated; the full
/// SHA is resolved and recorded after checkout).
pub fn validate_commit_sha(sha: &str) -> Result<()> {
    let ok = (4..=40).contains(&sha.len()) && sha.chars().all(|c| c.is_ascii_hexdigit());
    if ok {
        Ok(())
    } else {
        Err(anyhow!(
            "invalid commit {sha:?} — pass the fork's commit SHA (4-40 hex \
             digits, e.g. 7c81a9f0), never a branch or tag name: fork lanes \
             are immutable"
        ))
    }
}

/// Split the `--fork owner/repo@sha` shorthand into its parts.
pub fn parse_fork_spec(spec: &str) -> Result<(String, String)> {
    let Some((repo, sha)) = spec.rsplit_once('@') else {
        return Err(anyhow!(
            "invalid --fork {spec:?} — expected owner/repo@commit-sha"
        ));
    };
    validate_repo_slug(repo)?;
    validate_commit_sha(sha)?;
    Ok((repo.to_string(), sha.to_string()))
}

/// Engine tag for a fork lane: `fork-<owner>_<repo>-<sha8>-<backend>`.
/// Deterministic per (repo, commit, backend) so a rebuild replaces the
/// same row; the `fork-` prefix keeps it outside the `bNNNN` namespace
/// (`btag_number` yields None — safe from upstream currency checks).
#[must_use]
pub fn derive_fork_engine_tag(repo: &str, full_sha: &str, backend: BuildBackend) -> String {
    let slug = repo.replace('/', "_");
    let short = &full_sha[..full_sha.len().min(8)];
    format!("fork-{slug}-{short}-{}", backend.as_str())
}

/// Backends Blazar can build from source. Intentionally only the two
/// that upstream does not ship as Linux prebuilts worth building
/// (Vulkan/ROCm/SYCL prebuilts exist — use `engine update` for those).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildBackend {
    Cuda,
    Cpu,
}

impl BuildBackend {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
        }
    }
}

/// Binaries copied into the engine dir — exactly the sibling set the
/// release tarballs ship and the quantize/bench/perplexity/rpc lanes
/// discover. ggml-rpc-server is the one prebuilt assets ship that a
/// server-only build silently omits (F164/F167: the rpc override leg
/// dies without it).
pub const BUILD_TARGETS: [&str; 6] = [
    "llama-server",
    "llama-quantize",
    "llama-imatrix",
    "llama-perplexity",
    "llama-bench",
    "ggml-rpc-server",
];

pub const CONFIGURE_TIMEOUT: Duration = Duration::from_mins(5);
pub const CLONE_TIMEOUT: Duration = Duration::from_mins(10);
/// Full CUDA compile of the ggml template-instantiation matrix takes
/// 15-30 min on a typical box; 40 leaves headroom for slow disks.
pub const DEFAULT_BUILD_TIMEOUT: Duration = Duration::from_mins(40);
/// Lines of captured stderr embedded in a failed-step error.
pub const ERROR_TAIL_LINES: usize = 25;

/// Everything the build lane needs; fields the CLI leaves `None` are
/// auto-detected. `source_dir`/`clone_url` are seams for tests.
#[derive(Debug, Clone)]
pub struct BuildOpts {
    pub backend: BuildBackend,
    /// Concrete upstream `bNNNN` tag to compile (the Fork source
    /// ignores it for naming — the engine tag derives from the pinned
    /// SHA — but it still labels the build log).
    pub tag: String,
    /// Where the source comes from: an upstream `bNNNN` tag (default)
    /// or an immutable `owner/repo@sha` fork pin.
    pub source: BuildSource,
    /// Trust tier baked into the lane manifest: `Curated` marks a
    /// registry-installed lane (`engine install --lane`), eligible for
    /// auto-retirement once mainline covers it; the default `User`
    /// tier never auto-deletes.
    pub trust: super::manifest::TrustTier,
    /// Pre-fetched source tree (skips the clone entirely).
    pub source_dir: Option<PathBuf>,
    /// Git remote override (default: `github.com/<LLAMA_CPP_REPO>`).
    pub clone_url: Option<String>,
    /// CUDA architectures override, e.g. `"89"` or `"80;86"`
    /// (cross-builds; skips `nvidia-smi` detection).
    pub arch: Option<String>,
    /// Explicit CUDA host compiler (e.g. `/usr/bin/g++-12` for an nvcc
    /// older than the system gcc).
    pub cuda_host_compiler: Option<PathBuf>,
    /// Parallel compile jobs.
    pub jobs: usize,
    /// Whole-build deadline.
    pub timeout: Duration,
    /// Toolchain override (tests inject stub binaries; production
    /// auto-detects from PATH + the CUDA toolkit root).
    pub toolchain: Option<Toolchain>,
}

impl BuildOpts {
    #[must_use]
    pub fn new(backend: BuildBackend, tag: impl Into<String>) -> Self {
        Self {
            backend,
            tag: tag.into(),
            source: BuildSource::Upstream,
            trust: super::manifest::TrustTier::default(),
            source_dir: None,
            clone_url: None,
            arch: None,
            cuda_host_compiler: None,
            jobs: default_jobs(),
            timeout: DEFAULT_BUILD_TIMEOUT,
            toolchain: None,
        }
    }
}

fn default_jobs() -> usize {
    std::thread::available_parallelism().map_or(4, std::num::NonZeroUsize::get)
}

/// Toolchain probe result. Every member is the resolved absolute path.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Toolchain {
    pub git: Option<PathBuf>,
    pub cmake: Option<PathBuf>,
    pub cxx: Option<PathBuf>,
    pub nvcc: Option<PathBuf>,
    pub nvidia_smi: Option<PathBuf>,
    /// Optional compile-cache launcher (`ccache`/`sccache`). Purely an
    /// accelerator: absent means an uncached build, never an error, and
    /// `require_toolchain` deliberately never demands it.
    pub compiler_cache: Option<PathBuf>,
}

/// Find `names` in order, first hit wins, across the given search path.
/// Does `dir` contain executable `name` under either spelling (bare or
/// `.exe`)? Windows installs append `.exe` (System32's
/// `nvidia-smi.exe`); matching both on every OS keeps discovery
/// testable on POSIX.
#[must_use]
pub fn bin_on_path(dir: &Path, name: &str) -> bool {
    dir.join(name).is_file() || dir.join(format!("{name}.exe")).is_file()
}

/// Resolve a tool by name across PATH-style dirs.
fn find_bin(search: &[PathBuf], names: &[&str]) -> Option<PathBuf> {
    for dir in search {
        for name in names {
            if bin_on_path(dir, name) {
                return Some(dir.join(name));
            }
        }
    }
    None
}

/// Detect the build toolchain. `search` is a PATH-style list; nvcc is
/// additionally probed at the CUDA toolkit's canonical install root.
#[must_use]
pub fn detect_toolchain(search: &[PathBuf]) -> Toolchain {
    let mut search = search.to_vec();
    if let Ok(cuda_bin) = std::env::var("CUDA_PATH") {
        search.push(PathBuf::from(cuda_bin).join("bin"));
    }
    search.push(PathBuf::from("/usr/local/cuda/bin"));
    Toolchain {
        git: find_bin(&search, &["git"]),
        cmake: find_bin(&search, &["cmake"]),
        cxx: find_bin(&search, &["g++", "c++", "clang++"]),
        nvcc: find_bin(&search, &["nvcc"]),
        nvidia_smi: find_bin(&search, &["nvidia-smi"]),
        compiler_cache: find_bin(&search, &["ccache", "sccache"]),
    }
}

/// PATH split into dirs (the production search list).
#[must_use]
pub fn path_dirs() -> Vec<PathBuf> {
    std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).collect())
        .unwrap_or_default()
}

/// Missing-toolchain errors with install hints per platform family.
pub fn require_toolchain(tc: &Toolchain, backend: BuildBackend) -> Result<()> {
    let mut missing: Vec<String> = Vec::new();
    if tc.git.is_none() {
        missing.push("git (apt install git / dnf install git / brew install git)".to_string());
    }
    if tc.cmake.is_none() {
        missing.push(
            "cmake >= 3.14 (apt install cmake / dnf install cmake / brew install cmake)".into(),
        );
    }
    if tc.cxx.is_none() {
        missing.push(
            "C++ compiler (apt install g++ / dnf install gcc-c++ / xcode-select --install)".into(),
        );
    }
    if backend == BuildBackend::Cuda && tc.nvcc.is_none() {
        missing.push(
            "CUDA toolkit nvcc (https://developer.nvidia.com/cuda-downloads or \
             apt install nvidia-cuda-toolkit / dnf install cuda-toolkit)"
                .into(),
        );
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(anyhow!(
            "missing build toolchain: {} — install, then re-run blazar engine build",
            missing.join("; ")
        ))
    }
}

/// Map nvidia-smi compute caps ("8.9", "12.1") to cmake architecture
/// tokens ("89", "121"), deduplicated, first-seen order (first GPU wins
/// as primary). Malformed caps are an error, not a guess.
pub fn cuda_architectures(compute_caps: &[&str]) -> Result<String> {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut ordered: Vec<String> = Vec::new();
    for cap in compute_caps {
        let clean = cap.trim();
        let arch: String = clean.split('.').filter(|p| !p.is_empty()).collect();
        if clean.is_empty() || arch.is_empty() || !arch.chars().all(|c| c.is_ascii_digit()) {
            return Err(anyhow!(
                "cannot parse GPU compute capability {cap:?} into a CUDA architecture"
            ));
        }
        if seen.insert(arch.clone()) {
            ordered.push(arch);
        }
    }
    if ordered.is_empty() {
        return Err(anyhow!("no GPU compute capabilities provided"));
    }
    Ok(ordered.join(";"))
}

/// Major version of a `g++ --version` banner line, e.g.
/// "g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0" -> 13. The version is
/// the last whitespace token; its LEADING digit run is the major.
#[must_use]
pub fn parse_gnu_major(banner: &str) -> Option<u32> {
    let last = banner.lines().next()?.split_whitespace().last()?;
    let digits = last.split(|c: char| !c.is_ascii_digit()).next()?;
    (!digits.is_empty()).then(|| digits.parse().ok())?
}

/// Major version of an `nvcc --version` banner, e.g.
/// "Cuda compilation tools, release 12.0, V12.0.140" -> 12.
#[must_use]
pub fn parse_nvcc_major(banner: &str) -> Option<u32> {
    let line = banner.lines().find(|l| l.contains("release "))?;
    line.split("release ")
        .nth(1)?
        .split('.')
        .next()?
        .parse()
        .ok()
}

/// nvcc only supports gcc of its own major or older. When the default
/// C++ compiler is newer than nvcc, pick `g++-<nvcc major>` from the
/// candidate list (verified combo: nvcc 12 + system gcc 13 -> g++-12).
/// Returns the override; `None` = default compiler is fine.
#[must_use]
pub fn cuda_host_compiler_rule(
    default_cxx_major: u32,
    nvcc_major: u32,
    candidates: &[PathBuf],
) -> Option<PathBuf> {
    if default_cxx_major <= nvcc_major {
        return None;
    }
    let want = std::ffi::OsString::from(format!("g++-{nvcc_major}"));
    candidates
        .iter()
        .find(|p| p.file_name().is_some_and(|n| *n == want))
        .cloned()
}

/// Assemble the cmake configure arguments (without -S/-B).
#[must_use]
pub fn cmake_configure_args(
    backend: BuildBackend,
    arch: Option<&str>,
    host_compiler: Option<&Path>,
    compiler_cache: Option<&Path>,
) -> Vec<String> {
    let mut args = vec![
        "-DCMAKE_BUILD_TYPE=Release".to_string(),
        "-DLLAMA_CURL=ON".to_string(),
        // Parity with upstream prebuilt engines: they ship the
        // ggml-rpc-server tool (verified live b10896/b10902), and
        // upstream tools/CMakeLists.txt gates add_subdirectory(rpc)
        // behind GGML_RPC — without this flag the target does not
        // exist and BUILD_TARGETS' "ggml-rpc-server" entry fails the
        // whole build with "No rule to make target" (caught live
        // 2026-09-11, b10903). The rpc backend is inert unless an
        // engine is actually launched with --rpc.
        "-DGGML_RPC=ON".to_string(),
        // CMake build-tree binaries embed an ABSOLUTE build-dir RPATH by
        // default; copying them into the engine store leaves a dead path
        // and the loader fails with a misleading "cannot open shared
        // object file" for a library sitting right next to the binary
        // (caught live 2026-09-09: RUNPATH=/tmp/blazar-build-…/build/bin
        // vs upstream release binaries' $ORIGIN). BUILD_RPATH_USE_ORIGIN
        // makes build-tree RPATHs $ORIGIN-relative, matching the layout
        // we install (binary + shared libs in one directory).
        "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON".to_string(),
    ];
    if backend == BuildBackend::Cuda {
        args.push("-DGGML_CUDA=ON".to_string());
        // CI parity (.github/workflows/engine-cuda.yml): the prebuilt
        // overlay builds with half-precision CUDA math, which measured
        // +15% decode on sm89 (b10955 overlay 592 tok/s vs a local
        // b10970 build without this flag at 512 tok/s, same model).
        args.push("-DGGML_CUDA_F16=ON".to_string());
        if let Some(a) = arch {
            args.push(format!("-DCMAKE_CUDA_ARCHITECTURES={a}"));
        }
        if let Some(hc) = host_compiler {
            args.push(format!("-DCMAKE_CUDA_HOST_COMPILER={}", hc.display()));
        }
    }
    if let Some(cache) = compiler_cache {
        // A launcher cache gives repeat builds their speed WITHOUT a
        // persistent build dir: the fresh-tempdir-per-build discipline
        // (probe-after-teardown, see `build_and_install`) stays intact,
        // while unchanged translation units come from the cache.
        let c = cache.display();
        args.push(format!("-DCMAKE_C_COMPILER_LAUNCHER={c}"));
        args.push(format!("-DCMAKE_CXX_COMPILER_LAUNCHER={c}"));
        if backend == BuildBackend::Cuda {
            args.push(format!("-DCMAKE_CUDA_COMPILER_LAUNCHER={c}"));
        }
    }
    args
}

/// Engine tag for a built engine: `bNNNN-<backend>` (provenance suffix;
/// `btag_number` parses the leading build digits so currency checks work).
#[must_use]
pub fn derive_engine_tag(btag: &str, backend: BuildBackend) -> String {
    format!("{btag}-{}", backend.as_str())
}

/// One spawned build step: process-group isolation, hard timeout, stdout
/// relayed LIVE through `on_line` (mpsc + select against child.wait),
/// stderr captured for the error tail.
/// Configure and compile the build targets in `bld` from the fetched
/// source `src`, streaming progress through `on_line`.
// Argument list mirrors the cmake invocation inputs 1:1; a bundling
// struct would obscure that correspondence.
#[allow(clippy::too_many_arguments)]
async fn run_cmake_build(
    tc: &Toolchain,
    opts: &BuildOpts,
    src: &Path,
    bld: &Path,
    tag: &str,
    arch: Option<&str>,
    host_compiler: Option<&Path>,
    on_line: &mut dyn FnMut(&str),
) -> Result<()> {
    if let Some(cache) = &tc.compiler_cache {
        let name = cache.file_name().map_or_else(
            || cache.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        );
        (on_line)(&format!(
            "compiler cache: {name} — repeat builds skip recompiling"
        ));
    }
    let args = cmake_configure_args(
        opts.backend,
        arch,
        host_compiler,
        tc.compiler_cache.as_deref(),
    );
    (on_line)(&format!("configuring {tag} ({})", args.join(" ")));
    let mut cfg = tokio::process::Command::new(tc.cmake.clone().context("cmake gone")?);
    cfg.arg("-S").arg(src).arg("-B").arg(bld);
    for a in &args {
        cfg.arg(a);
    }
    run_step(cfg, "cmake configure", CONFIGURE_TIMEOUT, on_line).await?;

    let mut bldcmd = tokio::process::Command::new(tc.cmake.clone().context("cmake gone")?);
    bldcmd
        .arg("--build")
        .arg(bld)
        .arg("--target")
        .args(BUILD_TARGETS.iter().copied())
        .arg("-j")
        .arg(opts.jobs.to_string());
    (on_line)(&format!(
        "compiling {} targets with {} jobs (10-30 min for CUDA)",
        BUILD_TARGETS.len(),
        opts.jobs
    ));
    run_step(bldcmd, "cmake --build", opts.timeout, on_line).await
}

async fn run_step(
    mut cmd: tokio::process::Command,
    what: &str,
    timeout: Duration,
    on_line: &mut dyn FnMut(&str),
) -> Result<()> {
    use tokio::io::{AsyncBufReadExt as _, BufReader};

    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    {
        // tokio::process::Command has an inherent unix process_group —
        // new pgid so a killed build cannot take the daemon with it.
        cmd.process_group(0);
    }
    let mut child = cmd.spawn().with_context(|| format!("spawn {what}"))?;
    let pid = child.id();

    // Reader tasks own the pipes; the parent relays lines while waiting.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let stdout_pipe = child.stdout.take();
    let relay = tokio::spawn(async move {
        let Some(s) = stdout_pipe else { return };
        let mut reader = BufReader::new(s);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line).await {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    if tx.send(line.trim_end().to_string()).is_err() {
                        break;
                    }
                }
            }
        }
    });
    let stderr_pipe = child.stderr.take();
    let captured = tokio::spawn(async move {
        let mut tail: std::collections::VecDeque<String> = std::collections::VecDeque::default();
        let Some(s) = stderr_pipe else { return tail };
        let mut reader = BufReader::new(s);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line).await {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    tail.push_back(line.trim_end().to_string());
                    if tail.len() > ERROR_TAIL_LINES * 4 {
                        tail.pop_front();
                    }
                }
            }
        }
        tail
    });

    let waited = tokio::time::timeout(timeout, async {
        let status = loop {
            tokio::select! {
                line = rx.recv() => {
                    if let Some(l) = line {
                        on_line(&l);
                    }
                }
                st = child.wait() => break st,
            }
        };
        status
    })
    .await;

    let status = match waited {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => return Err(anyhow!("wait for {what}: {e}")),
        Err(_) => {
            let _ = child.start_kill();
            return Err(anyhow!(
                "{what} timed out after {}s (pid {pid:?}) — killed",
                timeout.as_secs()
            ));
        }
    };
    // The relay task owns the stdout pipe: only once it joins is the
    // channel guaranteed complete. child.wait() alone races lines still
    // buffered in the pipe — a fast child (stub builds, `cmake --build`
    // with a script backend) can exit before its final echo is read, and
    // an eager drain then loses it forever. Forward the remainder only
    // after the join, until the channel closes (sender dropped).
    let _ = relay.await;
    while let Some(l) = rx.recv().await {
        on_line(&l);
    }
    let tail: std::collections::VecDeque<String> = captured.await.unwrap_or_default();
    if !status.success() {
        let last: Vec<String> = tail
            .iter()
            .rev()
            .take(ERROR_TAIL_LINES)
            .rev()
            .cloned()
            .collect();
        return Err(anyhow!(
            "{what} exited {status} — last output:\n{}",
            last.join("\n")
        ));
    }
    Ok(())
}

/// Identity of what a build produced, resolved after fetch: a fork's
/// engine tag needs the FULL commit SHA only the fetch step can resolve.
struct BuiltLane {
    tag: String,
    dir: PathBuf,
    provenance: LaneProvenance,
}

/// Resolve the installed lane's identity: engine tag plus the provenance
/// recorded into its manifest (repo, commit pin, upstream anchor, and the
/// architecture set mined from the source).
fn lane_identity(
    source: &BuildSource,
    tag: &str,
    full_sha: Option<&str>,
    architectures: std::collections::BTreeSet<String>,
    backend: BuildBackend,
) -> (String, LaneProvenance) {
    match source {
        BuildSource::Upstream => (
            derive_engine_tag(tag, backend),
            LaneProvenance {
                source: EngineSource::Upstream,
                repo: Some(LLAMA_CPP_REPO.to_string()),
                ref_pin: full_sha.map(str::to_string),
                base_ref: None,
                architectures,
            },
        ),
        BuildSource::Fork {
            repo,
            ref_sha,
            base_ref,
        } => (
            derive_fork_engine_tag(repo, full_sha.unwrap_or(ref_sha), backend),
            LaneProvenance {
                source: EngineSource::Fork,
                repo: Some(repo.clone()),
                ref_pin: full_sha
                    .map(str::to_string)
                    .or_else(|| Some(ref_sha.clone())),
                base_ref: base_ref.clone(),
                architectures,
            },
        ),
    }
}

impl EngineManager {
    /// Effective build source, resolved before any fetch:
    ///
    /// - GitHub's smart-HTTP fetch only accepts full 40-char object
    ///   names as want-refs — an abbreviated pin (>= 4 hex, validated
    ///   at the CLI) is resolved to the full commit first.
    /// - Provenance honesty for caller-provided trees: a `source_dir`
    ///   whose git origin is NOT upstream llama.cpp is a fork
    ///   checkout, whatever the caller declared. Stamping it Upstream
    ///   would let the lane be treated as mainstream currency
    ///   (auto-retire windows, supersede mining) for code it is not.
    ///   The tree's own remote is the truth; plain fixture trees (no
    ///   .git) keep the declared source.
    async fn resolve_effective_source(
        &self,
        tc: &Toolchain,
        opts: &BuildOpts,
        on_line: &mut dyn FnMut(&str),
    ) -> Result<BuildOpts> {
        let resolved = match resolve_fork_pin(&self.gh, &opts.source, on_line).await? {
            Some(source) => BuildOpts {
                source,
                ..opts.clone()
            },
            None => opts.clone(),
        };
        if let (Some(dir), BuildSource::Upstream) = (&resolved.source_dir, &resolved.source) {
            let origin = git_remote_slug(tc, dir).await;
            let head = git_rev_parse(tc, dir).await.ok();
            if let (Some(slug), Some(sha)) = (origin, head) {
                if slug != *LLAMA_CPP_REPO {
                    (on_line)(&format!(
                        "source tree origin is {slug} (not upstream llama.cpp) — \
                         stamping this build as a fork lane"
                    ));
                    return Ok(BuildOpts {
                        source: BuildSource::Fork {
                            repo: slug,
                            ref_sha: sha,
                            base_ref: None,
                        },
                        ..resolved
                    });
                }
            }
        }
        Ok(resolved)
    }

    /// Clone (or take) the source tree, configure, build, install, probe,
    /// activate. The build tree is a tempfile — torn down on every exit
    /// path (Drop), success or failure.
    pub async fn build_and_install(
        &self,
        opts: &BuildOpts,
        on_line: &mut dyn FnMut(&str),
    ) -> Result<EngineRow> {
        let tc = opts
            .toolchain
            .clone()
            .unwrap_or_else(|| detect_toolchain(&path_dirs()));
        require_toolchain(&tc, opts.backend)?;

        let tag = opts.tag.trim().to_string();
        match &opts.source {
            BuildSource::Upstream => {
                if super::gh::btag_number(&tag).is_none() {
                    return Err(anyhow!(
                        "engine build needs a concrete upstream b-tag like b10816 (got {tag:?})"
                    ));
                }
            }
            // Fork lanes deliberately live outside the bNNNN currency
            // system: their identity is the pinned commit, not an
            // upstream tag.
            BuildSource::Fork { .. } => {}
        }

        // Effective source resolution (pin expansion + tree-origin
        // honesty) happens before the fetch: fetch, tag derivation,
        // and asset labels all consume the resolved source.
        let resolved = self.resolve_effective_source(&tc, opts, on_line).await?;
        let opts = &resolved;

        // CUDA specifics: architecture from the GPU (or override), host
        // compiler de-conflicted against nvcc's gcc support window.
        let (arch, host_compiler) = cuda_build_facts(&tc, opts).await?;

        // Build in a scoped tempfile so it is fully torn down BEFORE the
        // installed copy is probed. An absolute build-dir RPATH (the CMake
        // default this lane guards against via BUILD_RPATH_USE_ORIGIN)
        // would otherwise resolve through the still-alive build tree and
        // let a broken install pass its probe — exactly the live failure
        // of 2026-09-09. Probe-after-teardown keeps the install check
        // honest: the stored engine must stand on its own.
        let built: Result<(String, Option<PathBuf>, BuiltLane)> = {
            let build_root =
                tempfile::TempDir::with_prefix("blazar-build-").context("create build tempdir")?;
            let (src, full_sha) = fetch_source(build_root.path(), &tc, opts, &tag, on_line).await?;

            // Mine the architecture set the source advertises — the
            // capability currency the supervisor's unknown-arch re-route
            // consumes — and pin the lane's identity from it.
            let architectures = super::arch_miner::mine_architectures(&src);
            let (engine_tag, provenance) = lane_identity(
                &opts.source,
                &tag,
                full_sha.as_deref(),
                architectures,
                opts.backend,
            );
            (on_line)(&format!(
                "lane {engine_tag}: source advertises {} architectures",
                provenance.architectures.len()
            ));

            let bld = build_root.path().join("build");
            run_cmake_build(
                &tc,
                opts,
                &src,
                &bld,
                &tag,
                arch.as_deref(),
                host_compiler.as_deref(),
                on_line,
            )
            .await?;

            // Rollback-safe swap (same contract as the release lanes'
            // install_with_rollback): the old build is displaced only
            // AFTER the new one compiled — restored on any later
            // failure, discarded once the new row lands. Retire must
            // precede the extraction: install_built_binaries refuses to
            // merge into a live dir.
            let engine_dir = self.dirs.engines_dir().join(&engine_tag);
            let aside = retire_engine_dir(&engine_dir)?;
            // Panic safety for the extract below (future cancellation
            // cannot strike here — the region from retire to register is
            // await-free; the guard is what keeps that invariant honest
            // if an await ever sneaks in).
            let mut guard = CancelledInstallGuard {
                dir: engine_dir.clone(),
                aside: aside.clone(),
                armed: true,
            };
            let installed = install_built_binaries(&self.dirs, &bld, &engine_tag, on_line);
            // Registration owns the dir from here; a unwind past this
            // point must not delete a dir the store may reference.
            guard.disarm();
            match installed {
                Ok((_, digest)) => Ok((
                    digest,
                    aside,
                    BuiltLane {
                        tag: engine_tag,
                        dir: engine_dir,
                        provenance,
                    },
                )),
                Err(e) => {
                    restore_retired_engine(aside.as_deref(), &engine_dir);
                    Err(e)
                }
            }
        };
        let (digest, aside, lane) = built?;
        // The build tree is torn down above BEFORE registering so the
        // probe cannot resolve through it.
        let asset_label = match &opts.source {
            // Fork builds are opt-in capability lanes, not mainstream
            // currency; the backend rides in the tag.
            BuildSource::Fork { .. } => "built-fork".to_string(),
            BuildSource::Upstream => format!("built-{}", opts.backend.as_str()),
        };
        match self.register_engine_provenanced(
            &lane.dir,
            &lane.tag,
            &asset_label,
            &digest,
            EngineKind::LlamaCpp,
            &lane.provenance,
            opts.trust,
        ) {
            Ok(row) => {
                discard_retired_engine(aside.as_deref());
                Ok(row)
            }
            Err(e) => {
                restore_retired_engine(aside.as_deref(), &lane.dir);
                Err(e)
            }
        }
    }
}

/// CUDA source builds need two live facts the CPU lane ignores: the
/// GPU's compute capability (unless overridden) and a host compiler
/// inside nvcc's gcc support window. Both come back as cmake-ready
/// `Some(...)` only for the CUDA backend.
async fn cuda_build_facts(
    tc: &Toolchain,
    opts: &BuildOpts,
) -> Result<(Option<String>, Option<std::path::PathBuf>)> {
    let mut arch = opts.arch.clone();
    let mut host_compiler = opts.cuda_host_compiler.clone();
    if opts.backend == BuildBackend::Cuda && arch.is_none() {
        arch = Some(query_compute_caps(tc).await?);
    }
    if opts.backend == BuildBackend::Cuda && host_compiler.is_none() {
        host_compiler = resolve_host_compiler(tc).await?;
    }
    Ok((arch, host_compiler))
}

/// Expand an abbreviated fork pin to the full commit SHA. GitHub's
/// smart-HTTP fetch only accepts full 40-char object names as
/// want-refs; the commits API happily resolves short forms. Returns
/// `None` when the pin is already full-length (or not a fork at all),
/// so the caller can keep borrowing the original `BuildOpts`.
async fn resolve_fork_pin(
    gh: &super::gh::GhClient,
    source: &BuildSource,
    on_line: &mut dyn FnMut(&str),
) -> Result<Option<BuildSource>> {
    let BuildSource::Fork {
        repo,
        ref_sha,
        base_ref,
    } = source
    else {
        return Ok(None);
    };
    if ref_sha.len() >= 40 {
        return Ok(None);
    }
    let full = gh.resolve_commit(repo, ref_sha).await?;
    (on_line)(&format!("resolved fork pin {ref_sha} to {full}"));
    Ok(Some(BuildSource::Fork {
        repo: repo.clone(),
        ref_sha: full,
        base_ref: base_ref.clone(),
    }))
}

/// nvidia-smi compute-cap query -> cmake architectures token.
/// Extract `major.minor` from a version fragment even when padded with
/// table decoration (`nvidia-smi` banner: `CUDA Version: 13.0     |` —
/// the trailing pipe is what a plain trim-and-parse chokes on).
#[must_use]
pub(crate) fn parse_version_pair(s: &str) -> Option<(u32, u32)> {
    let digits = |part: &str| -> Option<u32> {
        // Banner fragments arrive space-padded (" 13.0     |") — skip
        // leading whitespace, then take the digit run; junk beyond it
        // (table border, "-rc") is ignored.
        let run: String = part
            .trim_start()
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        (!run.is_empty()).then(|| run.parse().ok())?
    };
    let mut it = s.split('.');
    Some((digits(it.next()?)?, digits(it.next()?)?))
}

/// Live NVIDIA facts for asset selection (mistralrs CUDA lane):
/// `(driver_cuda, first_gpu_compute_cap)` — `(None, None)` on any
/// probe failure (caller then falls back to CPU assets). Driver CUDA
/// comes from the `nvidia-smi` banner (`CUDA Version: 13.0`), which
/// reflects the driver's runtime capability — exactly the ceiling a
/// prebuilt CUDA binary must not exceed.
///
/// Latency contract: with driver persistence mode OFF (laptop default)
/// the GPU drops to P3 after ~45 s idle and the FIRST nvidia-smi of a
/// burst pays ~1.9 s re-waking it (measured 3/3 after idle vs 33-46 ms
/// back-to-back; CPU load does not reproduce it). Callers that time
/// this probe must budget seconds, not milliseconds — `sudo
/// nvidia-smi -pm 1` pins persistence for servers that want it.
pub async fn nvidia_gpu_facts() -> (Option<(u32, u32)>, Option<(u32, u32)>) {
    let Some(smi) = detect_toolchain(&path_dirs()).nvidia_smi else {
        return (None, None);
    };
    let driver_cuda = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tokio::process::Command::new(&smi).output(),
    )
    .await
    .ok()
    .and_then(std::result::Result::ok)
    .and_then(|out| {
        let text = String::from_utf8_lossy(&out.stdout);
        let line = text.lines().find(|l| l.contains("CUDA Version:"))?;
        parse_version_pair(line.split("CUDA Version:").nth(1)?)
    });
    let compute_cap = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tokio::process::Command::new(&smi)
            .arg("--query-gpu=compute_cap")
            .arg("--format=csv,noheader")
            .output(),
    )
    .await
    .ok()
    .and_then(std::result::Result::ok)
    .and_then(|out| {
        let csv = String::from_utf8_lossy(&out.stdout);
        let first = csv.lines().map(str::trim).find(|l| !l.is_empty())?;
        parse_version_pair(first)
    });
    (driver_cuda, compute_cap)
}

/// Runtime libs the cudart companion ships that a CUDA llama-server
/// actually dlopens at boot (verified against the b11011 companion
/// listing: libcublasLt, libcublas, libcudart). Checked per CUDA major.
const CUDA_RUNTIME_LIBS: [&str; 3] = ["libcudart", "libcublas", "libcublasLt"];

/// Decide from `ldconfig -p` output whether the system already exposes
/// the full CUDA runtime for `major`, so a 410-594 MiB companion
/// download can be skipped. Pure parser — unit-testable on any box.
#[must_use]
pub fn cuda_runtime_complete_from_ldconfig(text: &str, major: u32) -> bool {
    let suffix = format!(".so.{major} ");
    CUDA_RUNTIME_LIBS.iter().all(|lib| {
        text.lines()
            .any(|l| l.trim_start().starts_with(lib) && l.contains(&suffix))
    })
}

/// Live probe: does this system expose the full CUDA runtime for
/// `major`? Linux parses `ldconfig -p` (cudart + cublas + cublasLt for
/// the major); non-Linux always answers false — Windows resolves DLLs
/// from the exe dir, so the companion belongs beside the binary and the
/// probe would only risk a wrong skip.
pub async fn system_cuda_runtime_complete(major: u32) -> bool {
    if !cfg!(target_os = "linux") {
        return false;
    }
    let out = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tokio::process::Command::new("ldconfig").arg("-p").output(),
    )
    .await
    .ok()
    .and_then(std::result::Result::ok);
    match out {
        Some(o) => cuda_runtime_complete_from_ldconfig(&String::from_utf8_lossy(&o.stdout), major),
        // ldconfig missing (exotic distro): treat as incomplete — the
        // companion download is the fail-safe answer, never a skip.
        None => false,
    }
}

async fn query_compute_caps(tc: &Toolchain) -> Result<String> {
    let smi = tc.nvidia_smi.as_ref().ok_or_else(|| {
        anyhow!(
            "no nvidia-smi on PATH and no --arch given: cannot determine the \
         target CUDA architecture — pass --arch (e.g. 89 for RTX 40xx) or \
         install the NVIDIA driver tools"
        )
    })?;
    let out = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tokio::process::Command::new(smi)
            .arg("--query-gpu=compute_cap")
            .arg("--format=csv,noheader")
            .output(),
    )
    .await
    .with_context(|| {
        format!(
            "run {} compute_cap query (timed out or failed)",
            smi.display()
        )
    })??;
    if !out.status.success() {
        return Err(anyhow!(
            "nvidia-smi compute_cap query exited {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let csv = String::from_utf8_lossy(&out.stdout).to_string();
    let caps: Vec<&str> = csv
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    cuda_architectures(&caps)
}

/// Caller-provided source tree, or a fresh clone — plus the resolved
/// full commit SHA of the checked-out tree (`None` where no git history
/// exists, e.g. a test fixture tree). The SHA is recorded as lane
/// provenance so an engine row always names the exact code it runs.
///
/// Upstream: shallow recursive clone of the tag (recursive is a no-op
/// now that ggml is in-tree, and still correct for tags that carried it
/// as a submodule).
///
/// Fork: clone without checkout, fetch the pinned SHA straight from
/// origin (no local ref exists for an arbitrary commit), detach onto
/// `FETCH_HEAD`, verify the checkout matches the pin, then init
/// submodules — a no-op when the fork carries none.
async fn fetch_source(
    build_root: &Path,
    tc: &Toolchain,
    opts: &BuildOpts,
    tag: &str,
    on_line: &mut dyn FnMut(&str),
) -> Result<(PathBuf, Option<String>)> {
    if let Some(s) = &opts.source_dir {
        // Fixture trees often have no .git; provenance is best-effort.
        let full = git_rev_parse(tc, s).await.ok();
        return Ok((s.clone(), full));
    }
    let src = build_root.join("src");
    match &opts.source {
        BuildSource::Fork { repo, ref_sha, .. } => {
            let url = format!("https://github.com/{repo}.git");
            let git = tc.git.clone().context("git gone")?;
            (on_line)(&format!("fetching fork {repo}@{ref_sha}"));
            let mut clone = tokio::process::Command::new(&git);
            clone
                .arg("clone")
                .arg("--depth")
                .arg("1")
                .arg("--no-checkout")
                .arg(&url)
                .arg(&src);
            run_step(clone, &format!("git clone {repo}"), CLONE_TIMEOUT, on_line).await?;
            let mut fetch = tokio::process::Command::new(&git);
            fetch
                .arg("-C")
                .arg(&src)
                .arg("fetch")
                .arg("--depth")
                .arg("1")
                .arg("origin")
                .arg(ref_sha);
            run_step(
                fetch,
                &format!("git fetch origin {ref_sha}"),
                CLONE_TIMEOUT,
                on_line,
            )
            .await?;
            let mut checkout = tokio::process::Command::new(&git);
            checkout
                .arg("-C")
                .arg(&src)
                .arg("checkout")
                .arg("--detach")
                .arg("FETCH_HEAD");
            run_step(checkout, "git checkout FETCH_HEAD", CLONE_TIMEOUT, on_line).await?;
            let full = git_rev_parse(tc, &src).await?;
            anyhow::ensure!(
                full.to_lowercase().starts_with(&ref_sha.to_lowercase()),
                "checked out {full} but the pin asked for {ref_sha} — refusing an \
                 unverified fork lane (forks must build exactly the pinned commit)"
            );
            let mut subs = tokio::process::Command::new(&git);
            subs.arg("-C")
                .arg(&src)
                .arg("submodule")
                .arg("update")
                .arg("--init")
                .arg("--depth")
                .arg("1")
                .arg("--recursive");
            run_step(subs, "git submodule update", CLONE_TIMEOUT, on_line).await?;
            Ok((src, Some(full)))
        }
        BuildSource::Upstream => {
            let url = opts
                .clone_url
                .clone()
                .unwrap_or_else(|| format!("https://github.com/{LLAMA_CPP_REPO}.git"));
            (on_line)(&format!("cloning {tag} from {url}"));
            let mut cmd = tokio::process::Command::new(tc.git.clone().context("git gone")?);
            cmd.arg("clone")
                .arg("--depth")
                .arg("1")
                .arg("--branch")
                .arg(tag)
                .arg("--recursive")
                .arg(&url)
                .arg(&src);
            run_step(cmd, &format!("git clone {tag}"), CLONE_TIMEOUT, on_line).await?;
            let full = git_rev_parse(tc, &src).await.ok();
            Ok((src, full))
        }
    }
}

/// `git -C <dir> rev-parse HEAD` — full commit SHA of a checked-out
/// tree, for lane provenance.
async fn git_rev_parse(tc: &Toolchain, dir: &Path) -> Result<String> {
    let git = tc.git.as_ref().context("git gone")?;
    let out = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        tokio::process::Command::new(git)
            .arg("-C")
            .arg(dir)
            .arg("rev-parse")
            .arg("HEAD")
            .output(),
    )
    .await
    .context("git rev-parse HEAD timed out")??;
    anyhow::ensure!(
        out.status.success(),
        "git rev-parse HEAD in {} exited {}: {}",
        dir.display(),
        out.status,
        String::from_utf8_lossy(&out.stderr).trim()
    );
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// `git remote get-url origin` of a tree, normalized to an `owner/repo`
/// slug. `None` when the tree has no git history or no origin remote
/// (plain fixture trees) or the URL does not name a GitHub-style slug —
/// callers treat that as "origin unknown, keep the declared source".
async fn git_remote_slug(tc: &Toolchain, dir: &Path) -> Option<String> {
    let git = tc.git.as_ref()?;
    let out = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tokio::process::Command::new(git)
            .arg("-C")
            .arg(dir)
            .arg("remote")
            .arg("get-url")
            .arg("origin")
            .output(),
    )
    .await
    .ok()?
    .ok()?;
    if !out.status.success() {
        return None;
    }
    let url = String::from_utf8_lossy(&out.stdout).trim().to_string();
    // Both common remote forms: https://github.com/owner/repo.git and
    // git@github.com:owner/repo.git. Anything else is not a slug we can
    // vouch for.
    let base = url
        .strip_prefix("https://github.com/")
        .or_else(|| url.strip_prefix("git@github.com:"))?;
    let tail = base.strip_suffix(".git").unwrap_or(base);
    validate_repo_slug(tail).ok()?;
    Some(tail.to_string())
}
///
/// Upstream master links the tools against shared `libggml*/libllama*`
/// siblings in `build/bin` (verified live: the 17 KiB thin binaries
/// resolve them via an `$ORIGIN` rpath, so the whole directory must
/// travel together, symlink chains included).
fn install_built_binaries(
    dirs: &BlazarDirs,
    bld: &Path,
    engine_tag: &str,
    on_line: &mut dyn FnMut(&str),
) -> Result<(PathBuf, String)> {
    let dir = dirs.engines_dir().join(engine_tag);
    // The caller retires any existing copy first (rollback-safe swap);
    // merging into a live dir would leave stale files behind.
    anyhow::ensure!(
        !dir.exists(),
        "engine dir {} already exists — the caller must retire it first",
        dir.display()
    );
    let inner = dir.join(format!("llama-{engine_tag}"));
    std::fs::create_dir_all(&inner).with_context(|| format!("create {}", inner.display()))?;
    let bin = bld.join("bin");
    let server_name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    // The five tool binaries are the contract; a missing one is an
    // upstream layout change worth failing loudly on.
    for t in BUILD_TARGETS {
        let name = if cfg!(windows) {
            format!("{t}.exe")
        } else {
            t.to_string()
        };
        if !bin.join(&name).exists() {
            return Err(anyhow!(
                "build produced no {name} in {} — upstream layout change?",
                bin.display()
            ));
        }
    }
    let mut files = 0usize;
    let mut links = 0usize;
    copy_bin_tree(&bin, &inner, &mut files, &mut links)?;
    (on_line)(&format!(
        "installed {engine_tag}: {} tool binaries + {} libs/symlinks",
        BUILD_TARGETS.len(),
        files + links - BUILD_TARGETS.len()
    ));
    // sha256 of the built server for the engine row (self-computed:
    // source builds carry no upstream digest).
    let digest = {
        use sha2::{Digest as _, Sha256};
        let bytes =
            std::fs::read(inner.join(server_name)).context("read built llama-server for sha256")?;
        format!("{:x}", Sha256::digest(bytes))
    };
    Ok((dir, digest))
}

/// Copy `build/bin` contents: regular files verbatim, symlinks recreated
/// as symlinks (the loader needs the `lib*.so -> lib*.so.N` chains).
#[cfg_attr(not(unix), allow(clippy::only_used_in_recursion))] // link count only moves on unix
fn copy_bin_tree(from: &Path, to: &Path, files: &mut usize, links: &mut usize) -> Result<()> {
    for entry in std::fs::read_dir(from).with_context(|| format!("read {}", from.display()))? {
        let entry = entry.with_context(|| format!("stat {}", from.display()))?;
        let dest = to.join(entry.file_name());
        #[cfg(unix)]
        if entry.file_type().is_ok_and(|t| t.is_symlink()) {
            let target = std::fs::read_link(entry.path())?;
            let _ = std::fs::remove_file(&dest);
            std::os::unix::fs::symlink(&target, &dest)
                .with_context(|| format!("relink {} -> {}", dest.display(), target.display()))?;
            *links += 1;
            continue;
        }
        if entry.file_type().is_ok_and(|t| t.is_dir()) {
            std::fs::create_dir_all(&dest)?;
            copy_bin_tree(&entry.path(), &dest, files, links)?;
            continue;
        }
        std::fs::copy(entry.path(), &dest)
            .with_context(|| format!("copy {}", entry.path().display()))?;
        *files += 1;
    }
    Ok(())
}

/// Apply the nvcc/gcc support rule when the system compiler is too new.
async fn resolve_host_compiler(tc: &Toolchain) -> Result<Option<PathBuf>> {
    let (Some(nvcc), Some(cxx)) = (tc.nvcc.as_ref(), tc.cxx.as_ref()) else {
        return Ok(None);
    };
    let nvcc_banner = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tokio::process::Command::new(nvcc).arg("--version").output(),
    )
    .await
    .context("run nvcc --version (timed out or failed)")??;
    let cxx_banner = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tokio::process::Command::new(cxx).arg("--version").output(),
    )
    .await
    .context("run compiler --version (timed out or failed)")??;
    let (Some(nvcc_major), Some(cxx_major)) = (
        parse_nvcc_major(&String::from_utf8_lossy(&nvcc_banner.stdout)),
        parse_gnu_major(&String::from_utf8_lossy(&cxx_banner.stdout)),
    ) else {
        return Ok(None); // unknown versions: let cmake/nvcc arbitrate
    };
    if cxx_major <= nvcc_major {
        return Ok(None);
    }
    // g++-<nvcc major> normally sits next to the default compiler
    // (/usr/bin/g++-12 beside /usr/bin/g++); search its directory first,
    // then PATH.
    let mut search = path_dirs();
    if let Some(parent) = cxx.parent() {
        search.insert(0, parent.to_path_buf());
    }
    let want = format!("g++-{nvcc_major}");
    let candidates: Vec<PathBuf> = search
        .into_iter()
        .map(|d| d.join(&want))
        .filter(|p| p.is_file())
        .collect();
    match cuda_host_compiler_rule(cxx_major, nvcc_major, &candidates) {
        Some(p) => {
            tracing::info!(
                "system compiler is gcc {cxx_major}, newer than nvcc {nvcc_major} \
                 supports; using {} as the CUDA host compiler",
                p.display()
            );
            Ok(Some(p))
        }
        None => Err(anyhow!(
            "system C++ compiler is gcc {cxx_major} but nvcc {nvcc_major} supports \
             gcc <= {nvcc_major}; install a matching compiler (e.g. apt install \
             g++-{nvcc_major}) or pass --cuda-host-compiler"
        )),
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__validate_repo_slug__rejects_traversal_and_degenerate_segments() {
        // Path-traversal spellings: host-contained, but the constructed
        // URL would silently normalize to a different repo than named.
        for bad in [
            "../evil",
            "..",
            ".",
            "acme/..",
            "acme/.",
            "../acme/x",
            "/acme/x",
            "acme",
            "acme/llama.cpp/extra",
            "/",
        ] {
            assert!(
                validate_repo_slug(bad).is_err(),
                "slug {bad:?} must be rejected"
            );
        }
        // Dots *inside* a segment remain legitimate repo characters.
        for ok in ["acme/x..y", "acme/llama.cpp", "a-b_c/d.e-f"] {
            assert!(validate_repo_slug(ok).is_ok(), "slug {ok:?} must pass");
        }
        // The clone-URL spelling stays rejected via the fork spec path.
        assert!(parse_fork_spec("acme/llama.cpp.git@1234abcd").is_err());
    }

    #[test]
    fn unit__parse_version_pair__nvidia_smi_banner_shapes() {
        // Real banner row from an RTX 4070 laptop box: table borders pad
        // the version — the trailing pipe broke trim-and-parse (the live
        // CPU-fallback-on-CUDA-box bug).
        let banner =
            "| NVIDIA-SMI: 580.173.02    Driver Version: 580.173.02    CUDA Version: 13.0     |";
        let frag = banner.split("CUDA Version:").nth(1).unwrap();
        assert_eq!(parse_version_pair(frag), Some((13, 0)));
        assert_eq!(parse_version_pair(" 13.0"), Some((13, 0)));
        assert_eq!(parse_version_pair("8.9"), Some((8, 9)));
        assert_eq!(parse_version_pair("12.8-rc"), Some((12, 8)));
        assert_eq!(parse_version_pair(""), None);
        assert_eq!(parse_version_pair("x.y"), None);
        assert_eq!(parse_version_pair("13"), None);
    }

    #[test]
    fn unit__cuda_runtime_complete_from_ldconfig__needs_all_three_for_major() {
        // Real `ldconfig -p` row shape (indent + name (arch) => path).
        let full_12 = "\
        \tlibcudart.so.12 (libc6,x86-64) => /usr/local/cuda/lib64/libcudart.so.12\n\
        \tlibcublas.so.12 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcublas.so.12\n\
        \tlibcublasLt.so.12 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcublasLt.so.12\n\
        \tlibcuda.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcuda.so.1\n";
        assert!(cuda_runtime_complete_from_ldconfig(full_12, 12));
        // Missing libcublasLt: incomplete — the companion is the
        // fail-safe answer, never a half-skip.
        let no_lt = &full_12.replace("libcublasLt.so.12", "libother.so.12");
        assert!(!cuda_runtime_complete_from_ldconfig(no_lt, 12));
        // Wrong major present only: a 13 pick must not ride on 12 libs.
        assert!(!cuda_runtime_complete_from_ldconfig(full_12, 13));
        // Unversioned .so alone never satisfies a major query.
        let unversioned = "\tlibcudart.so (libc6,x86-64) => /opt/cuda/lib/libcudart.so\n";
        assert!(!cuda_runtime_complete_from_ldconfig(unversioned, 12));
    }

    #[test]
    fn unit__bin_on_path__exe_spelling_matches_windows_installs() {
        // Windows installs put `nvidia-smi.exe` in System32; both
        // spellings must resolve so vendor hints and driver probes work
        // cross-platform. Verified on a POSIX box via the .exe arm.
        let dir = std::env::temp_dir().join("blazar-bin-on-path-pin");
        std::fs::create_dir_all(&dir).unwrap();
        let exe = dir.join("nvidia-smi.exe");
        std::fs::write(&exe, b"").unwrap();
        assert!(bin_on_path(&dir, "nvidia-smi"));
        assert!(!bin_on_path(&dir, "rocminfo"));
        let bare = dir.join("nvidia-smi");
        std::fs::write(&bare, b"").unwrap();
        assert!(bin_on_path(&dir, "nvidia-smi"));
        drop(bare);
        drop(exe);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unit__cuda_architectures__caps_to_tokens_dedup() {
        assert_eq!(cuda_architectures(&["8.9"]).unwrap(), "89");
        assert_eq!(
            cuda_architectures(&["8.9", "12.1", "8.9"]).unwrap(),
            "89;121"
        );
        assert_eq!(cuda_architectures(&["9.0"]).unwrap(), "90");
        assert!(cuda_architectures(&["8.9", "bogus"]).is_err());
        assert!(cuda_architectures(&[]).is_err());
    }

    #[test]
    fn unit__parse_gnu_major__ubuntu_banner() {
        assert_eq!(
            parse_gnu_major("g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0\n"),
            Some(13)
        );
        assert_eq!(parse_gnu_major("g++ 12.3.0"), Some(12));
        assert_eq!(parse_gnu_major("garbage"), None);
    }

    #[test]
    fn unit__parse_nvcc_major__release_line() {
        assert_eq!(
            parse_nvcc_major(
                "nvcc: NVIDIA (R) Cuda compiler driver\n\
                 Cuda compilation tools, release 12.0, V12.0.140\n"
            ),
            Some(12)
        );
        assert_eq!(parse_nvcc_major("no banner"), None);
    }

    #[test]
    fn unit__host_compiler_rule__newer_gcc_picks_nvcc_major() {
        let gpp12 = PathBuf::from("/usr/bin/g++-12");
        assert_eq!(
            cuda_host_compiler_rule(13, 12, std::slice::from_ref(&gpp12)),
            Some(gpp12.clone())
        );
        assert_eq!(
            cuda_host_compiler_rule(12, 12, std::slice::from_ref(&gpp12)),
            None
        );
        assert_eq!(
            cuda_host_compiler_rule(11, 12, std::slice::from_ref(&gpp12)),
            None
        );
        assert_eq!(cuda_host_compiler_rule(13, 12, &[]), None);
    }

    #[test]
    fn unit__cmake_args__cuda_vs_cpu() {
        let cuda = cmake_configure_args(
            BuildBackend::Cuda,
            Some("89"),
            Some(Path::new("/usr/bin/g++-12")),
            None,
        );
        assert!(cuda.contains(&"-DGGML_CUDA=ON".to_string()));
        assert!(cuda.contains(&"-DCMAKE_CUDA_ARCHITECTURES=89".to_string()));
        assert!(cuda.contains(&"-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12".to_string()));
        assert!(cuda.contains(&"-DLLAMA_CURL=ON".to_string()));
        // rpc tool parity with prebuilts (gates ggml-rpc-server target)
        assert!(cuda.contains(&"-DGGML_RPC=ON".to_string()));
        let cpu = cmake_configure_args(BuildBackend::Cpu, None, None, None);
        assert!(cpu.contains(&"-DGGML_RPC=ON".to_string()));
        assert!(cpu.iter().all(|a| !a.contains("CUDA")));
    }

    #[test]
    fn unit__cmake_args__compiler_cache_launchers() {
        let cache = Path::new("/usr/bin/ccache");
        let cuda = cmake_configure_args(BuildBackend::Cuda, None, None, Some(cache));
        assert!(cuda.contains(&"-DCMAKE_C_COMPILER_LAUNCHER=/usr/bin/ccache".to_string()));
        assert!(cuda.contains(&"-DCMAKE_CXX_COMPILER_LAUNCHER=/usr/bin/ccache".to_string()));
        assert!(cuda.contains(&"-DCMAKE_CUDA_COMPILER_LAUNCHER=/usr/bin/ccache".to_string()));
        // CPU backend must not carry any CUDA flag, cache or otherwise.
        let cpu = cmake_configure_args(BuildBackend::Cpu, None, None, Some(cache));
        assert!(cpu.contains(&"-DCMAKE_C_COMPILER_LAUNCHER=/usr/bin/ccache".to_string()));
        assert!(cpu.iter().all(|a| !a.contains("CUDA")));
        // No cache on PATH -> no launcher flags at all.
        let bare = cmake_configure_args(BuildBackend::Cuda, None, None, None);
        assert!(bare.iter().all(|a| !a.contains("LAUNCHER")));
    }

    #[test]
    fn unit__cmake_args__cuda_f16_ci_parity() {
        // The overlay CI builds with half-precision CUDA math; a local
        // build without it measured -15% decode on sm89. CPU stays off.
        let cuda = cmake_configure_args(BuildBackend::Cuda, None, None, None);
        assert!(cuda.contains(&"-DGGML_CUDA_F16=ON".to_string()));
        let cpu = cmake_configure_args(BuildBackend::Cpu, None, None, None);
        assert!(cpu.iter().all(|a| !a.contains("F16")));
    }

    #[test]
    fn unit__derive_engine_tag__suffix() {
        assert_eq!(
            derive_engine_tag("b10816", BuildBackend::Cuda),
            "b10816-cuda"
        );
        assert_eq!(derive_engine_tag("b10816", BuildBackend::Cpu), "b10816-cpu");
    }

    #[test]
    fn unit__require_toolchain__teaching_errors() {
        let tc = Toolchain::default();
        let err = require_toolchain(&tc, BuildBackend::Cuda)
            .unwrap_err()
            .to_string();
        assert!(err.contains("git") && err.contains("cmake") && err.contains("nvcc"));
        let err_cpu = require_toolchain(&tc, BuildBackend::Cpu)
            .unwrap_err()
            .to_string();
        assert!(!err_cpu.contains("nvcc"));
        let ok = Toolchain {
            git: Some(PathBuf::from("/usr/bin/git")),
            cmake: Some(PathBuf::from("/usr/bin/cmake")),
            cxx: Some(PathBuf::from("/usr/bin/g++")),
            nvcc: Some(PathBuf::from("/usr/bin/nvcc")),
            nvidia_smi: None,
            compiler_cache: None,
        };
        assert!(require_toolchain(&ok, BuildBackend::Cuda).is_ok());
        // A ccache-only addition changes nothing: the cache is optional.
        let cached = Toolchain {
            compiler_cache: Some(PathBuf::from("/usr/bin/ccache")),
            ..ok
        };
        assert!(require_toolchain(&cached, BuildBackend::Cuda).is_ok());
        assert_eq!(
            detect_toolchain(&[PathBuf::from("/nonexistent")]).compiler_cache,
            None
        );
    }

    #[test]
    fn unit__detect_toolchain__cuda_path_probe() {
        // /usr/local/cuda/bin fallback is in the search list by construction.
        let search = vec![PathBuf::from("/nonexistent")];
        let tc = detect_toolchain(&search);
        assert!(tc.git.is_none());
        assert!(tc.cmake.is_none());
    }

    /// Remote-origin provenance: both GitHub URL forms normalize to a
    /// validated slug; foreign hosts and remote-less trees yield None
    /// (caller keeps the declared source). Skips silently where no git
    /// binary exists — the probe is about URL parsing, not the tool.
    #[tokio::test]
    async fn unit__git_remote_slug__parses_forms_and_ignores_foreign() {
        let Some(git) = std::env::var_os("PATH").and_then(|paths| {
            std::env::split_paths(&paths)
                .map(|p| p.join("git"))
                .find(|p| p.is_file())
        }) else {
            return;
        };
        let tc = Toolchain {
            git: Some(git.clone()),
            cmake: None,
            cxx: None,
            nvcc: None,
            nvidia_smi: None,
            compiler_cache: None,
        };
        let repo = tempfile::tempdir().unwrap();
        std::process::Command::new(&git)
            .arg("init")
            .arg("-q")
            .arg(repo.path())
            .status()
            .unwrap();
        let remote = |url: &str| {
            std::process::Command::new(&git)
                .arg("-C")
                .arg(repo.path())
                .args(["remote", "add", "origin", url])
                .status()
                .unwrap();
        };
        let rm = || {
            std::process::Command::new(&git)
                .arg("-C")
                .arg(repo.path())
                .args(["remote", "remove", "origin"])
                .status()
                .unwrap();
        };
        assert_eq!(
            git_remote_slug(&tc, repo.path()).await,
            None,
            "no remote configured"
        );
        remote("https://github.com/acme-forks/llama.cpp.git");
        assert_eq!(
            git_remote_slug(&tc, repo.path()).await,
            Some("acme-forks/llama.cpp".to_string())
        );
        rm();
        remote("git@github.com:acme-forks/llama.cpp.git");
        assert_eq!(
            git_remote_slug(&tc, repo.path()).await,
            Some("acme-forks/llama.cpp".to_string())
        );
        rm();
        remote("https://gitlab.com/acme/llama.cpp.git");
        assert_eq!(
            git_remote_slug(&tc, repo.path()).await,
            None,
            "foreign host"
        );
    }
}
