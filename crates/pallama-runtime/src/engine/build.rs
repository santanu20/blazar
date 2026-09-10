//! `pallama engine build`: compile llama.cpp's in-tree backends from
//! source into a normal installed engine. Closes the distribution gap
//! where upstream publishes no Linux-CUDA prebuilts: the CUDA backend
//! lives in the source tree (`ggml-cuda`: flash-attention with
//! quantized-KV `vec_dot`, `MMQ` quantized matmuls) and building it
//! locally yields the same probe/serve surface as a release asset,
//! tagged `bNNNN-cuda`.
//!
//! Authority rules: cmake and nvcc decide architecture compatibility —
//! Pallama only passes the GPU's compute capability (from `nvidia-smi`)
//! through and surfaces the compiler's own errors verbatim (tail).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};

use pallama_core::engine_kind::EngineKind;
use pallama_core::store::EngineRow;
use pallama_core::PallamaDirs;

use super::gh::LLAMA_CPP_REPO;
use super::EngineManager;

/// Backends Pallama can build from source. Intentionally only the two
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
/// release tarballs ship and the quantize/bench/perplexity lanes discover.
pub const BUILD_TARGETS: [&str; 5] = [
    "llama-server",
    "llama-quantize",
    "llama-imatrix",
    "llama-perplexity",
    "llama-bench",
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
    /// Concrete upstream `bNNNN` tag to compile.
    pub tag: String,
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
}

/// Find `names` in order, first hit wins, across the given search path.
fn find_bin(search: &[PathBuf], names: &[&str]) -> Option<PathBuf> {
    for dir in search {
        for name in names {
            let p = dir.join(name);
            if p.is_file() {
                return Some(p);
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
            "missing build toolchain: {} — install, then re-run pallama engine build",
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
) -> Vec<String> {
    let mut args = vec![
        "-DCMAKE_BUILD_TYPE=Release".to_string(),
        "-DLLAMA_CURL=ON".to_string(),
        // CMake build-tree binaries embed an ABSOLUTE build-dir RPATH by
        // default; copying them into the engine store leaves a dead path
        // and the loader fails with a misleading "cannot open shared
        // object file" for a library sitting right next to the binary
        // (caught live 2026-09-09: RUNPATH=/tmp/pallama-build-…/build/bin
        // vs upstream release binaries' $ORIGIN). BUILD_RPATH_USE_ORIGIN
        // makes build-tree RPATHs $ORIGIN-relative, matching the layout
        // we install (binary + shared libs in one directory).
        "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON".to_string(),
    ];
    if backend == BuildBackend::Cuda {
        args.push("-DGGML_CUDA=ON".to_string());
        if let Some(a) = arch {
            args.push(format!("-DCMAKE_CUDA_ARCHITECTURES={a}"));
        }
        if let Some(hc) = host_compiler {
            args.push(format!("-DCMAKE_CUDA_HOST_COMPILER={}", hc.display()));
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

impl EngineManager {
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
        if super::gh::btag_number(&tag).is_none() {
            return Err(anyhow!(
                "engine build needs a concrete upstream b-tag like b10816 (got {tag:?})"
            ));
        }

        // CUDA specifics: architecture from the GPU (or override), host
        // compiler de-conflicted against nvcc's gcc support window.
        let mut arch = opts.arch.clone();
        let mut host_compiler = opts.cuda_host_compiler.clone();
        if opts.backend == BuildBackend::Cuda && arch.is_none() {
            arch = Some(query_compute_caps(&tc).await?);
        }
        if opts.backend == BuildBackend::Cuda && host_compiler.is_none() {
            host_compiler = resolve_host_compiler(&tc).await?;
        }

        let engine_tag = derive_engine_tag(&tag, opts.backend);
        // Build in a scoped tempfile so it is fully torn down BEFORE the
        // installed copy is probed. An absolute build-dir RPATH (the CMake
        // default this lane guards against via BUILD_RPATH_USE_ORIGIN)
        // would otherwise resolve through the still-alive build tree and
        // let a broken install pass its probe — exactly the live failure
        // of 2026-09-09. Probe-after-teardown keeps the install check
        // honest: the stored engine must stand on its own.
        let (dir, digest) = {
            let build_root =
                tempfile::TempDir::with_prefix("pallama-build-").context("create build tempdir")?;
            let src = fetch_source(build_root.path(), &tc, opts, &tag, on_line).await?;

            let bld = build_root.path().join("build");
            let args =
                cmake_configure_args(opts.backend, arch.as_deref(), host_compiler.as_deref());
            (on_line)(&format!("configuring {tag} ({})", args.join(" ")));
            let mut cfg = tokio::process::Command::new(tc.cmake.clone().context("cmake gone")?);
            cfg.arg("-S").arg(&src).arg("-B").arg(&bld);
            for a in &args {
                cfg.arg(a);
            }
            run_step(cfg, "cmake configure", CONFIGURE_TIMEOUT, on_line).await?;

            let mut bldcmd = tokio::process::Command::new(tc.cmake.clone().context("cmake gone")?);
            bldcmd
                .arg("--build")
                .arg(&bld)
                .arg("--target")
                .args(BUILD_TARGETS.iter().copied())
                .arg("-j")
                .arg(opts.jobs.to_string());
            (on_line)(&format!(
                "compiling {} targets with {} jobs (10-30 min for CUDA)",
                BUILD_TARGETS.len(),
                opts.jobs
            ));
            run_step(bldcmd, "cmake --build", opts.timeout, on_line).await?;

            install_built_binaries(&self.dirs, &bld, &engine_tag, on_line)?
        };
        self.register_engine(
            &dir,
            &engine_tag,
            &format!("built-{}", opts.backend.as_str()),
            &digest,
            EngineKind::LlamaCpp,
        )
    }
}

/// nvidia-smi compute-cap query -> cmake architectures token.
/// Extract `major.minor` from a version fragment even when padded with
/// table decoration (`nvidia-smi` banner: `CUDA Version: 13.0     |` —
/// the trailing pipe is what a plain trim-and-parse chokes on).
#[must_use]
fn parse_version_pair(s: &str) -> Option<(u32, u32)> {
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

/// Caller-provided source tree, or a shallow recursive clone of the tag
/// (recursive is a no-op now that ggml is in-tree, and still correct for
/// tags that carried it as a submodule).
async fn fetch_source(
    build_root: &Path,
    tc: &Toolchain,
    opts: &BuildOpts,
    tag: &str,
    on_line: &mut dyn FnMut(&str),
) -> Result<PathBuf> {
    if let Some(s) = &opts.source_dir {
        return Ok(s.clone());
    }
    let url = opts
        .clone_url
        .clone()
        .unwrap_or_else(|| format!("https://github.com/{LLAMA_CPP_REPO}.git"));
    let src = build_root.join("src");
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
    Ok(src)
}

/// Copy the built output into the standard engine layout — the same
/// `llama-<tag>/` inner root as a release tarball, so the
/// quantize/bench/perplexity discovery lanes see the siblings — and
/// return the engine dir and the sha256 of the built server.
///
/// Upstream master links the tools against shared `libggml*/libllama*`
/// siblings in `build/bin` (verified live: the 17 KiB thin binaries
/// resolve them via an `$ORIGIN` rpath, so the whole directory must
/// travel together, symlink chains included).
fn install_built_binaries(
    dirs: &PallamaDirs,
    bld: &Path,
    engine_tag: &str,
    on_line: &mut dyn FnMut(&str),
) -> Result<(PathBuf, String)> {
    let dir = dirs.engines_dir().join(engine_tag);
    if dir.exists() {
        std::fs::remove_dir_all(&dir)
            .with_context(|| format!("replace existing engine dir {}", dir.display()))?;
    }
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
        );
        assert!(cuda.contains(&"-DGGML_CUDA=ON".to_string()));
        assert!(cuda.contains(&"-DCMAKE_CUDA_ARCHITECTURES=89".to_string()));
        assert!(cuda.contains(&"-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12".to_string()));
        assert!(cuda.contains(&"-DLLAMA_CURL=ON".to_string()));
        let cpu = cmake_configure_args(BuildBackend::Cpu, None, None);
        assert!(cpu.iter().all(|a| !a.contains("CUDA")));
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
        };
        assert!(require_toolchain(&ok, BuildBackend::Cuda).is_ok());
    }

    #[test]
    fn unit__detect_toolchain__cuda_path_probe() {
        // /usr/local/cuda/bin fallback is in the search list by construction.
        let search = vec![PathBuf::from("/nonexistent")];
        let tc = detect_toolchain(&search);
        assert!(tc.git.is_none());
        assert!(tc.cmake.is_none());
    }
}
