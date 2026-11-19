//! Supervised `lmcache server` lane: the shared persistent KV store
//! one daemon many engine instances dial. A prefix computed by any
//! `LMCache`-enabled sglang child restores into any later child of
//! the same lane whenever the model identity matches - the daemon is
//! the sharing platform, the supervisor only owns its lifecycle.
//! (Cross-engine reuse - `sglang` <-> `vllm` - is refused upstream of
//! here: `LMCache` ships no cross-layout converter, so foreign-layout
//! blocks would read as garbage; see `VllmKvOffload::validate`.)
//!
//! Lifecycle mirrors the whisper/kokoro lazy children: spawn on first
//! LMCache-lane engine spawn, healthcheck the HTTP frontend
//! (`/healthcheck`, loopback-pinned - engines ride the MP port),
//! idle-reap only when zero live instances still dial it (the L1 pool
//! is pinned CPU RAM; holding it after the last consumer leaves would
//! strand GiB), hard teardown on daemon shutdown. The binary is never
//! installed by blazar: it ships inside an engine venv's site-packages
//! (`<engines>/<lane>/venv/bin/lmcache`) and discovery teaches the
//! install command when no lane carries it.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use blazar_core::config::LmcacheDaemon;

/// Healthcheck + spawn budget. The daemon is a torch-importing Python
/// process: ~2 s warm, tens of seconds on a cold page cache, so the
/// bound is generous while still failing a wedged boot within one
/// engine spawn attempt.
const READY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// The lazy lane slot: `None` until the first LMCache-lane spawn.
pub struct LmcacheRuntime {
    child: tokio::sync::Mutex<Option<LmcacheChild>>,
}

/// One supervised `lmcache server` child plus the config fingerprint
/// it was spawned under - a config change (port, L1 size, eviction)
/// respawns the daemon rather than silently diverging from `[kv]`.
pub struct LmcacheChild {
    child: tokio::process::Child,
    bin: PathBuf,
    fingerprint: LmcacheFingerprint,
    last_used: tokio::time::Instant,
}

/// Every daemon-behavior knob from `[kv.lmcache]`: if any of these
/// changes, the running daemon no longer matches the config.
#[derive(Debug, Clone, PartialEq)]
struct LmcacheFingerprint {
    host: String,
    port: u16,
    http_port: u16,
    l1_size_gb: u32,
    eviction_policy: String,
}

impl LmcacheFingerprint {
    fn of(cfg: &LmcacheDaemon) -> Self {
        Self {
            host: cfg.host.clone(),
            port: cfg.port,
            http_port: cfg.http_port,
            l1_size_gb: cfg.l1_size_gb,
            eviction_policy: cfg.eviction_policy.clone(),
        }
    }
}

/// Whether a spawned argv dials the shared daemon. Applied to the
/// profile argv at spawn time (daemon ensure hook) and to each live
/// instance's stored argv at reap time (consumer count) - the same
/// predicate on both sides, so the daemon can never outlive (or miss)
/// an engine that needs it.
#[must_use]
pub fn argv_dials_lmcache(kind: blazar_core::engine_kind::EngineKind, argv: &[String]) -> bool {
    use blazar_core::engine_kind::EngineKind;
    match kind {
        // The flag pair is emitted atomically by the profile compiler;
        // user extra_args cannot reach `--enable-lmcache` without the
        // config opt-in that also armed this predicate.
        EngineKind::Sglang => argv
            .windows(2)
            .any(|w| w[0] == "--enable-lmcache" && w[1] == "--lmcache-config-file"),
        EngineKind::Vllm => argv
            .windows(2)
            .any(|w| w[0] == "--kv-transfer-config" && w[1].contains("LMCacheMPConnector")),
        _ => false,
    }
}

/// Find the `lmcache` console script across installed engine venvs.
/// Deterministic: lane directories in sorted order, first hit wins
/// (the MP protocol is transport-level - any lane's copy serves every
/// engine, so preference is stability, not affinity).
#[must_use]
pub fn discover_bin(engines_dir: &Path) -> Option<PathBuf> {
    let mut lanes: Vec<PathBuf> = std::fs::read_dir(engines_dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    lanes.sort();
    lanes.into_iter().find_map(|lane| {
        // venv layout is the installers' contract: bin/ on unix,
        // Scripts/ on windows.
        let unix = lane.join("venv").join("bin").join("lmcache");
        if unix.is_file() {
            return Some(unix);
        }
        let windows = lane.join("venv").join("Scripts").join("lmcache.exe");
        windows.is_file().then_some(windows)
    })
}

/// One bound GET against the daemon's HTTP frontend. Loopback only:
/// the frontend exists for this healthcheck, engines never dial it.
async fn health_ok(http_port: u16) -> bool {
    blazar_core::tls::ensure_tls_provider();
    let Ok(http) = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
    else {
        return false;
    };
    matches!(
        http
            .get(format!("http://127.0.0.1:{http_port}/healthcheck"))
            .send()
            .await,
        Ok(resp) if resp.status().is_success()
    )
}

impl LmcacheRuntime {
    #[must_use]
    pub fn new() -> Self {
        Self {
            child: tokio::sync::Mutex::new(None),
        }
    }

    /// Idempotent: a healthy daemon under the same config is reused,
    /// a config change respawns it, a dead one is replaced. Also
    /// (re)writes the sglang client YAML both lanes read - the file is
    /// derived state, rewritten on every ensure so it can never drift
    /// from the daemon it points at.
    pub async fn ensure(
        &self,
        cfg: &LmcacheDaemon,
        data_dir: &Path,
        engines_dir: &Path,
    ) -> Result<()> {
        let fingerprint = LmcacheFingerprint::of(cfg);
        let mut slot = self.child.lock().await;
        if let Some(live) = slot.as_mut() {
            if health_ok(live.fingerprint.http_port).await {
                if live.fingerprint == fingerprint {
                    live.last_used = tokio::time::Instant::now();
                    write_client_yaml(cfg, data_dir)?;
                    return Ok(());
                }
                // Config drifted from the running daemon: tear it down
                // so the respawn below serves the new fingerprint.
                tracing::info!(
                    port = live.fingerprint.port,
                    "lmcache daemon config changed - respawning under the new [kv.lmcache]"
                );
                let _ = live.child.kill().await;
                let _ = live.child.wait().await;
            } else {
                tracing::info!("lmcache daemon unhealthy - respawning");
            }
            *slot = None;
        }
        let bin = discover_bin(engines_dir).ok_or_else(|| {
            anyhow!(
                "no engine venv carries the `lmcache` server binary - install it into any \
                 lane venv, e.g. `uv pip install --python <engines>/<lane>/venv/bin/python \
                 lmcache`, then retry; blazar never installs it for you"
            )
        })?;
        let child = spawn_healthy(cfg, &bin, data_dir).await?;
        write_client_yaml(cfg, data_dir)?;
        *slot = Some(LmcacheChild {
            child,
            bin,
            fingerprint,
            last_used: tokio::time::Instant::now(),
        });
        Ok(())
    }

    /// Teardown: kill + reap. Never panics; a failed kill logs only.
    pub async fn shutdown(&self) {
        let mut slot = self.child.lock().await;
        if let Some(mut live) = slot.take() {
            let _ = live.child.kill().await;
            let _ = live.child.wait().await;
            tracing::info!(
                port = live.fingerprint.port,
                bin = %live.bin.display(),
                "lmcache daemon stopped"
            );
        }
    }

    /// Idle reaper with a consumer guard: unlike the whisper lane, the
    /// daemon holds a pinned GiB-scale L1 pool, so it must never
    /// outlive its last consumer - but it equally must never be reaped
    /// under one. `consumers` = live instances whose argv dials the
    /// daemon (same predicate as the spawn hook). A dead child is
    /// reaped regardless (hygiene; `ensure` would respawn anyway).
    pub async fn reap_idle(&self, max_idle: std::time::Duration, consumers: usize) {
        let mut slot = self.child.lock().await;
        let Some(live) = slot.as_ref() else { return };
        let dead = !health_ok(live.fingerprint.http_port).await;
        let idle = live.last_used.elapsed() >= max_idle;
        if consumers > 0 || (!dead && !idle) {
            return;
        }
        let mut live = slot.take().expect("checked Some above");
        let _ = live.child.kill().await;
        let _ = live.child.wait().await;
        if dead {
            tracing::info!(
                port = live.fingerprint.port,
                "lmcache daemon had died; reaped the slot"
            );
        } else {
            tracing::info!(
                port = live.fingerprint.port,
                idle_secs = live.last_used.elapsed().as_secs(),
                "lmcache daemon idle-reaped with zero LMCache lanes (respawns on next opt-in spawn)"
            );
        }
    }

    /// Live daemon port (MP transport) for status surfaces; `None`
    /// when the lane is down.
    pub async fn mp_port(&self) -> Option<u16> {
        self.child.lock().await.as_ref().map(|c| c.fingerprint.port)
    }
}

impl Default for LmcacheRuntime {
    fn default() -> Self {
        Self::new()
    }
}

/// The sglang client YAML (`--lmcache-config-file` target). `LMCache`'s
/// multiprocess client config is exactly `{mp_host, mp_port}` - the
/// daemon dial coordinates - and nothing else (lmcache 0.5.7
/// `v1/multiprocess/client_config.py`).
fn write_client_yaml(cfg: &LmcacheDaemon, data_dir: &Path) -> Result<()> {
    let dir = data_dir.join("kv").join("lmcache");
    std::fs::create_dir_all(&dir).context("create kv/lmcache dir")?;
    let yaml = format!("mp_host: {}\nmp_port: {}\n", cfg.host, cfg.port);
    std::fs::write(dir.join("sglang.yaml"), yaml).context("write sglang client yaml")
}

/// Spawn `lmcache server` and return it once the HTTP frontend passes
/// `/healthcheck` - callers only ever store a healthy child.
async fn spawn_healthy(
    cfg: &LmcacheDaemon,
    bin: &Path,
    data_dir: &Path,
) -> Result<tokio::process::Child> {
    // argv per the daemon CLI contract (lmcache 0.5.7 `server`
    // subcommand): MP bind is what lanes dial; HTTP frontend is
    // loopback-pinned for the healthcheck; L1 init == L1 size so the
    // lazy allocator starts at the operator's ceiling instead of the
    // 20 GiB default.
    let argv = [
        "server".to_string(),
        "--host".to_string(),
        cfg.host.clone(),
        "--port".to_string(),
        cfg.port.to_string(),
        "--http-host".to_string(),
        "127.0.0.1".to_string(),
        "--http-port".to_string(),
        cfg.http_port.to_string(),
        "--l1-size-gb".to_string(),
        cfg.l1_size_gb.to_string(),
        "--l1-init-size-gb".to_string(),
        cfg.l1_size_gb.to_string(),
        "--eviction-policy".to_string(),
        cfg.eviction_policy.clone(),
    ];
    let log_dir = data_dir.join("kv").join("lmcache");
    std::fs::create_dir_all(&log_dir).context("create kv/lmcache dir")?;
    let stderr = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_dir.join("daemon.log"))
        .context("open kv/lmcache/daemon.log")?;
    let mut std_cmd = std::process::Command::new(bin);
    std_cmd
        .args(&argv)
        .env("LMCACHE_DISABLE_BANNER", "1")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::from(stderr));
    #[cfg(unix)]
    {
        // Kernel lifetime tie: the daemon must not outlive the
        // supervisor that spawned it (a leaked daemon pins its whole
        // L1 pool).
        crate::probe::parent_death_tie(&mut std_cmd);
    }
    let mut cmd = tokio::process::Command::from(std_cmd);
    cmd.kill_on_drop(true);
    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawn {}", bin.display()))?;
    let deadline = tokio::time::Instant::now() + READY_TIMEOUT;
    loop {
        if health_ok(cfg.http_port).await {
            tracing::info!(
                mp_port = cfg.port,
                http_port = cfg.http_port,
                bin = %bin.display(),
                "lmcache daemon healthy"
            );
            return Ok(child);
        }
        // A daemon that died mid-boot fails fast with its log named,
        // instead of burning the whole ready timeout on a dead port.
        if let Ok(Some(status)) = child.try_wait() {
            return Err(anyhow!(
                "lmcache server exited before healthy: {status} - tail of \
                 kv/lmcache/daemon.log names the refusal"
            ));
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(anyhow!(
                "lmcache server did not pass /healthcheck on 127.0.0.1:{} within \
                 {} s - kv/lmcache/daemon.log has the boot trace",
                cfg.http_port,
                READY_TIMEOUT.as_secs()
            ));
        }
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
}

#[cfg(test)]
#[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
mod tests {
    use super::*;
    use blazar_core::engine_kind::EngineKind;

    fn argv(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn unit__argv_dials_lmcache__matches_both_lanes_and_ignores_the_rest() {
        // sglang: the compiler-emitted flag pair (atomic - the config
        // opt-in and the pair always travel together).
        assert!(argv_dials_lmcache(
            EngineKind::Sglang,
            &argv(&[
                "--enable-lmcache",
                "--lmcache-config-file",
                "/x/sglang.yaml"
            ])
        ));
        // The boolean alone (without its config-file sibling) is not a
        // dial this compiler ever emits - refuse it.
        assert!(!argv_dials_lmcache(
            EngineKind::Sglang,
            &argv(&["--enable-lmcache"])
        ));
        // vllm: the transfer config naming the MP connector.
        assert!(argv_dials_lmcache(
            EngineKind::Vllm,
            &argv(&[
                "--kv-transfer-config",
                r#"{"kv_connector":"LMCacheMPConnector"}"#
            ])
        ));
        // The built-in offload connector is NOT a daemon consumer - the
        // fs tier needs no daemon, so it must not pin one alive.
        assert!(!argv_dials_lmcache(
            EngineKind::Vllm,
            &argv(&[
                "--kv-transfer-config",
                r#"{"kv_connector":"OffloadingConnector"}"#
            ])
        ));
        // Other engines never dial the shared daemon.
        assert!(!argv_dials_lmcache(
            EngineKind::LlamaCpp,
            &argv(&["--enable-lmcache", "--lmcache-config-file"])
        ));
    }

    #[test]
    fn unit__discover_bin__first_sorted_lane_with_a_venv_lmcache_wins() {
        let engines = tempfile::tempdir().expect("temp engines dir");
        // Empty engines dir: no binary, caller teaches the install.
        assert_eq!(discover_bin(engines.path()), None);

        // Two lanes carrying the binary: sorted order decides, not
        // creation order (stability over affinity - any copy serves
        // every engine over the MP transport).
        for lane in ["vllm-0.34.0", "sglang-0.5.23"] {
            let bin = engines.path().join(lane).join("venv").join("bin");
            std::fs::create_dir_all(&bin).unwrap();
            std::fs::write(bin.join("lmcache"), "#!sh\n").unwrap();
        }
        let found = discover_bin(engines.path()).expect("bin found");
        assert_eq!(
            found,
            engines
                .path()
                .join("sglang-0.5.23")
                .join("venv")
                .join("bin")
                .join("lmcache"),
            "sorted-first lane wins"
        );

        // A lane without the binary never matches; a windows-layout
        // venv (Scripts/lmcache.exe) serves as the fallback hit.
        let engines = tempfile::tempdir().expect("temp engines dir 2");
        let scripts = engines
            .path()
            .join("vllm-0.34.0")
            .join("venv")
            .join("Scripts");
        std::fs::create_dir_all(&scripts).unwrap();
        std::fs::write(scripts.join("lmcache.exe"), "stub").unwrap();
        let found = discover_bin(engines.path()).expect("windows layout found");
        assert_eq!(
            found,
            engines
                .path()
                .join("vllm-0.34.0")
                .join("venv")
                .join("Scripts")
                .join("lmcache.exe")
        );
    }

    #[test]
    fn unit__write_client_yaml__emits_exactly_the_mp_dial_coordinates() {
        let data = tempfile::tempdir().expect("temp data dir");
        let cfg = LmcacheDaemon {
            enabled: true,
            host: "127.0.0.1".into(),
            port: 5556,
            ..Default::default()
        };
        write_client_yaml(&cfg, data.path()).expect("yaml written");
        let yaml =
            std::fs::read_to_string(data.path().join("kv").join("lmcache").join("sglang.yaml"))
                .expect("yaml readable");
        // LMCache's multiprocess client config is exactly these two
        // keys (v1/multiprocess/client_config.py): host bare (the
        // client prepends tcp:// itself), port bare.
        assert_eq!(yaml, "mp_host: 127.0.0.1\nmp_port: 5556\n");
    }
}
