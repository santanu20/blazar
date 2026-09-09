//! Engine build-lane integration tests: a stub toolchain (cmake script
//! that "builds" the compiled stub-llama-server) drives the REAL lane —
//! source resolution, configure args, binary copyout, layout, probe,
//! store, activate — with zero network and zero compiler.

#![cfg(unix)]

use std::io::Write as _;
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};

use pallama_core::store::Store;
use pallama_core::PallamaDirs;
use pallama_runtime::engine::build::{BuildBackend, BuildOpts, Toolchain};
use pallama_runtime::engine::{EngineManager, LOCAL_TAG};
use pallama_runtime::EventBus;

fn stub_server_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_stub-llama-server"))
}

fn tmp_dirs() -> (tempfile::TempDir, PallamaDirs) {
    let tmp = tempfile::tempdir().unwrap();
    let dirs = PallamaDirs {
        config_dir: tmp.path().join("cfg"),
        data_dir: tmp.path().join("data"),
    };
    dirs.ensure().unwrap();
    (tmp, dirs)
}

/// Write an executable shell script.
fn stub_script(dir: &Path, name: &str, body: &str) -> PathBuf {
    let p = dir.join(name);
    let mut f = std::fs::File::create(&p).unwrap();
    writeln!(f, "#!/bin/sh\n{body}").unwrap();
    let mut perms = std::fs::metadata(&p).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&p, perms).unwrap();
    p
}

/// cmake stub: configure is a no-op; `--build <dir>` materializes all
/// five llama binaries in `<dir>/bin` as copies of the compiled stub
/// server (so probe exercises the genuine --version/--help path).
fn stub_cmake(dir: &Path, body_extra: &str) -> PathBuf {
    let body = format!(
        "if [ \"$1\" = \"--build\" ]; then
  bld=\"$2\"
  mkdir -p \"$bld/bin\"
  for t in llama-server llama-quantize llama-imatrix llama-perplexity llama-bench; do
    cp \"${{STUB_SERVER}}\" \"$bld/bin/$t\" || exit 9
  done
  echo fake-lib > \"$bld/bin/libggml.so.0\"
  ln -sf libggml.so.0 \"$bld/bin/libggml.so\"
  echo stub build ok
fi
{body_extra}
exit 0"
    );
    stub_script(dir, "cmake", &body)
}

fn stub_git(dir: &Path) -> PathBuf {
    stub_script(dir, "git", "exit 1")
}

fn manager(dirs: &PallamaDirs) -> EngineManager {
    EngineManager {
        dirs: dirs.clone(),
        gh: pallama_runtime::engine::gh::GhClient::new(None).unwrap(),
        bus: EventBus::default(),
        asset_override: "auto".into(),
    }
}

fn source_tree(tmp: &Path) -> PathBuf {
    let src = tmp.join("src");
    std::fs::create_dir_all(src.join("ggml")).unwrap();
    std::fs::write(src.join("CMakeLists.txt"), "# stub source\n").unwrap();
    src
}

fn opts(backend: BuildBackend, tag: &str) -> BuildOpts {
    let mut o = BuildOpts::new(backend, tag);
    o.timeout = std::time::Duration::from_mins(1);
    o
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__build_cpu__installs_probes_activates() {
    let (tmp, dirs) = tmp_dirs();
    let bin_dir = tmp.path().join("bins");
    std::fs::create_dir_all(&bin_dir).unwrap();
    let tc = Toolchain {
        git: Some(stub_git(&bin_dir)),
        cmake: Some(stub_cmake(&bin_dir, "")),
        cxx: Some(bin_dir.join("g++")),
        nvcc: None,
        nvidia_smi: None,
    };
    std::fs::write(bin_dir.join("g++"), "#!/bin/sh\nexit 0").unwrap();

    let mut o = opts(BuildBackend::Cpu, "b4242");
    o.source_dir = Some(source_tree(tmp.path()));
    o.toolchain = Some(tc);
    std::env::set_var("STUB_SERVER", stub_server_bin());

    let mut lines = Vec::new();
    let row = manager(&dirs)
        .build_and_install(&o, &mut |l| lines.push(l.to_string()))
        .await
        .unwrap();

    assert_eq!(row.tag, "b4242-cpu");
    assert_eq!(row.asset, "built-cpu");
    assert!(!row.sha256.is_empty());
    let store = Store::open(&dirs).unwrap();
    assert_eq!(store.active_engine().unwrap().unwrap().tag, "b4242-cpu");
    // Tarball layout: engines/<tag>/llama-<tag>/llama-server.
    let server = dirs
        .engines_dir()
        .join("b4242-cpu")
        .join("llama-b4242-cpu")
        .join("llama-server");
    assert!(server.exists(), "layout: {}", server.display());
    // Sibling binaries for the quantize/bench discovery lanes.
    for sib in [
        "llama-quantize",
        "llama-imatrix",
        "llama-perplexity",
        "llama-bench",
    ] {
        assert!(server.with_file_name(sib).exists(), "{sib} missing");
    }
    // Shared libs travel along, symlink chains preserved.
    let lib = server.with_file_name("libggml.so.0");
    let link = server.with_file_name("libggml.so");
    assert!(lib.exists(), "libggml.so.0 missing");
    assert!(
        link.symlink_metadata()
            .is_ok_and(|m| m.file_type().is_symlink()),
        "libggml.so must stay a symlink"
    );
    assert_eq!(
        std::fs::read_link(&link).unwrap(),
        Path::new("libggml.so.0")
    );
    // Real probe of the stub: the banner reports build 9999, but the
    // install tag (b4242) is the authoritative build identity — the
    // shallow-clone "build 1" artifact is overridden by exactly this rule.
    let m: pallama_runtime::Manifest = serde_json::from_str(&row.manifest).unwrap();
    assert_eq!(m.build_number, 4242);
    assert_eq!(m.devices.len(), 1);
    assert!(lines.iter().any(|l| l.contains("stub build ok")));
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__build_cuda__host_compiler_rule_and_arch_args() {
    let (tmp, dirs) = tmp_dirs();
    let bin_dir = tmp.path().join("bins");
    std::fs::create_dir_all(&bin_dir).unwrap();
    let tc = Toolchain {
        git: Some(stub_git(&bin_dir)),
        cmake: Some(stub_cmake(&bin_dir, "")),
        cxx: Some(bin_dir.join("g++")),
        nvcc: Some(stub_script(
            &bin_dir,
            "nvcc",
            "echo 'Cuda compilation tools, release 12.0, V12.0.140'",
        )),
        nvidia_smi: None,
    };
    // System g++ reports gcc 13 (newer than nvcc 12 supports); a
    // g++-12 sits next to it -> the rule must pick it up.
    stub_script(&bin_dir, "g++", "echo 'g++ (Ubuntu 13.3.0) 13.3.0'");
    stub_script(&bin_dir, "g++-12", "echo 'g++ 12.3.0'");

    let mut o = opts(BuildBackend::Cuda, "b4242");
    o.source_dir = Some(source_tree(tmp.path()));
    o.toolchain = Some(tc);
    o.arch = Some("89".into()); // cross-build: no nvidia-smi needed
    std::env::set_var("STUB_SERVER", stub_server_bin());

    let mut lines = Vec::new();
    let row = manager(&dirs)
        .build_and_install(&o, &mut |l| lines.push(l.to_string()))
        .await
        .unwrap();

    assert_eq!(row.tag, "b4242-cuda");
    let cfg_line = lines
        .iter()
        .find(|l| l.starts_with("configuring"))
        .expect("configure progress line");
    assert!(cfg_line.contains("-DGGML_CUDA=ON"), "{cfg_line}");
    assert!(
        cfg_line.contains("-DCMAKE_CUDA_ARCHITECTURES=89"),
        "{cfg_line}"
    );
    assert!(
        cfg_line.contains("-DCMAKE_CUDA_HOST_COMPILER=") && cfg_line.contains("g++-12"),
        "host-compiler rule must fire: {cfg_line}"
    );
    assert!(
        cfg_line.contains("-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON"),
        "build-tree RPATH must be $ORIGIN-relative (absolute build-dir \
         RPATH breaks the installed copy once the temp build tree drops): {cfg_line}"
    );
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__build_fail__error_carries_stderr_tail() {
    let (tmp, dirs) = tmp_dirs();
    let bin_dir = tmp.path().join("bins");
    std::fs::create_dir_all(&bin_dir).unwrap();
    let tc = Toolchain {
        git: Some(stub_git(&bin_dir)),
        cmake: Some(stub_script(
            &bin_dir,
            "cmake",
            "echo 'fatal: unsupported gcc' >&2\nexit 7",
        )),
        cxx: Some(bin_dir.join("g++")),
        nvcc: None,
        nvidia_smi: None,
    };
    std::fs::write(bin_dir.join("g++"), "#!/bin/sh\nexit 0").unwrap();

    let mut o = opts(BuildBackend::Cpu, "b4242");
    o.source_dir = Some(source_tree(tmp.path()));
    o.toolchain = Some(tc);

    let err = manager(&dirs)
        .build_and_install(&o, &mut |_| {})
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("exited"), "{err}");
    assert!(
        err.contains("unsupported gcc"),
        "stderr tail missing: {err}"
    );
    // Nothing registered on failure.
    let store = Store::open(&dirs).unwrap();
    assert!(store.active_engine().unwrap().is_none());
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__build_missing_binary__teaching_error() {
    let (tmp, dirs) = tmp_dirs();
    let bin_dir = tmp.path().join("bins");
    std::fs::create_dir_all(&bin_dir).unwrap();
    // Builds only llama-server: the copyout must catch the gap.
    let partial = "if [ \"$1\" = \"--build\" ]; then
  bld=\"$2\"
  mkdir -p \"$bld/bin\"
  cp \"${STUB_SERVER}\" \"$bld/bin/llama-server\"
fi
exit 0";
    let tc = Toolchain {
        git: Some(stub_git(&bin_dir)),
        cmake: Some(stub_script(&bin_dir, "cmake", partial)),
        cxx: Some(bin_dir.join("g++")),
        nvcc: None,
        nvidia_smi: None,
    };
    std::fs::write(bin_dir.join("g++"), "#!/bin/sh\nexit 0").unwrap();

    let mut o = opts(BuildBackend::Cpu, "b4242");
    o.source_dir = Some(source_tree(tmp.path()));
    o.toolchain = Some(tc);
    std::env::set_var("STUB_SERVER", stub_server_bin());

    let err = manager(&dirs)
        .build_and_install(&o, &mut |_| {})
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("llama-quantize"), "{err}");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__build_rejects__non_btag_and_local_tag_safety() {
    let (tmp, dirs) = tmp_dirs();
    let bin_dir = tmp.path().join("bins");
    std::fs::create_dir_all(&bin_dir).unwrap();
    let tc = Toolchain {
        git: Some(stub_git(&bin_dir)),
        cmake: Some(stub_cmake(&bin_dir, "")),
        cxx: Some(bin_dir.join("g++")),
        nvcc: None,
        nvidia_smi: None,
    };
    std::fs::write(bin_dir.join("g++"), "#!/bin/sh\nexit 0").unwrap();
    let mut o = opts(BuildBackend::Cpu, "v1.2.3");
    o.source_dir = Some(source_tree(tmp.path()));
    o.toolchain = Some(tc);

    let err = manager(&dirs)
        .build_and_install(&o, &mut |_| {})
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("b-tag"), "{err}");

    // Suffixed engine tags never collide with the never-pruned `local`.
    assert_ne!(LOCAL_TAG, "b4242-cpu");
}
