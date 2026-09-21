//! Daemon lifecycle: pidfile guard and shutdown-signal handling.
//! One daemon per data dir; a second start fails fast naming the pid.
//! Refusals carry the [`LockHeld`] marker so the CLI can map them to the
//! singleton-conflict exit code the systemd unit refuses to restart on.

use anyhow::{anyhow, Context, Result};

use blazar_core::BlazarDirs;

/// Refusing to start because a live peer owns the daemon lock. Distinct
/// error type so `blazar serve` can exit with the same hard-conflict
/// code as a port bind conflict — `Restart=always` must not loop on it.
#[derive(Debug)]
pub struct LockHeld;

impl std::fmt::Display for LockHeld {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "daemon lock held by a live peer")
    }
}

impl std::error::Error for LockHeld {}

/// RAII daemon lock: `<run_dir>/blazar.pid`, removed on drop.
#[derive(Debug)]
pub struct DaemonLock {
    path: std::path::PathBuf,
    pid: u32,
}

impl DaemonLock {
    pub fn acquire(dirs: &BlazarDirs) -> Result<Self> {
        std::fs::create_dir_all(dirs.run_dir())?;
        let path = dirs.run_dir().join("blazar.pid");
        let pid = std::process::id();
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut f) => {
                use std::io::Write as _;
                writeln!(f, "{pid}").ok();
                Ok(Self { path, pid })
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                let existing: u32 = std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);
                if existing > 1 && process_alive(existing) {
                    return Err(anyhow::Error::new(LockHeld).context(format!(
                        "blazar already running (pid {existing}); if this is wrong, remove {}",
                        path.display()
                    )));
                }
                // Stale lock from a dead daemon: take over. F90: loop
                // back through `create_new` instead of remove+plain-write
                // — the old sequence had a window where a concurrent
                // successor's fresh pidfile landed between our remove and
                // our write, and we clobbered it (two daemons, both
                // convinced they hold the lock).
                tracing::warn!("removing stale pidfile for dead pid {existing}");
                std::fs::remove_file(&path)
                    .with_context(|| format!("remove {}", path.display()))?;
                match std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&path)
                {
                    Ok(mut f) => {
                        use std::io::Write as _;
                        writeln!(f, "{pid}").ok();
                        Ok(Self { path, pid })
                    }
                    // Lost the re-create race to a live successor: refuse.
                    Err(_) => Err(anyhow::Error::new(LockHeld).context(
                        "blazar already running (lock re-taken while replacing stale pidfile); retry",
                    )),
                }
            }
            Err(e) => Err(anyhow!("create {}: {e}", path.display())),
        }
    }

    #[must_use]
    pub fn pid(&self) -> u32 {
        self.pid
    }
}

impl Drop for DaemonLock {
    fn drop(&mut self) {
        // Only remove OUR pidfile (a successor may have replaced a stale one).
        let current = std::fs::read_to_string(&self.path)
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok());
        if current == Some(self.pid) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Shared with the supervisor's orphan sweep.
#[cfg(unix)]
#[allow(unsafe_code)] // existence probe via signal 0 to one exact pid
#[must_use]
pub fn process_alive_by_pid(pid: u32) -> bool {
    // SAFETY: signal 0 performs no action; single positive pid, no group.
    unsafe { libc::kill(i32::try_from(pid).unwrap_or(-1), 0) == 0 }
}

#[cfg(windows)]
#[must_use]
pub fn process_alive_by_pid(pid: u32) -> bool {
    // F89: `/proc` never exists on Windows, so the old probe made stale
    // takeover ALWAYS win — two daemons on one box. tasklist is the
    // dependency-free truth source (CSV rows quote the pid column).
    std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/NH", "/FO", "CSV"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains(&format!("\"{pid}\"")))
}

#[cfg(not(any(unix, windows)))]
#[must_use]
pub fn process_alive_by_pid(pid: u32) -> bool {
    std::path::Path::new(&format!("/proc/{pid}")).exists()
}

/// Validate-harness daemons (`BLAZAR_VALIDATE=1` in the environment)
/// must not outlive their harness: when the parent dies — a `timeout(1)`
/// SIGKILL bypasses every atexit sweep — the kernel delivers the signal
/// installed here. Belt to validate.py's marker sweep; Linux-only
/// because `PR_SET_PDEATHSIG` is a Linux prctl (other platforms fall
/// back to the harness sweep + signal-routed cleanup).
#[cfg(target_os = "linux")]
#[allow(unsafe_code)] // one prctl flag set + one ppid read on this thread
pub fn validate_parent_death_guard() {
    if std::env::var("BLAZAR_VALIDATE").ok().as_deref() != Some("1") {
        return;
    }
    // SAFETY: prctl(PR_SET_PDEATHSIG, SIGTERM) sets one kernel flag on
    // this thread; no pointers, no allocation. Failure leaves the
    // harness marker sweep as the covering net.
    unsafe {
        if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
            return;
        }
        // Race: the parent may have died between spawn and prctl — the
        // signal would never fire. Reparenting to init already happened.
        if libc::getppid() == 1 {
            std::process::exit(0);
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub fn validate_parent_death_guard() {}

#[cfg(unix)]
#[allow(unsafe_code)] // existence probe via signal 0 to one exact pid
fn process_alive(pid: u32) -> bool {
    // SAFETY: signal 0 performs no action; single positive pid, no group.
    unsafe { libc::kill(i32::try_from(pid).unwrap_or(-1), 0) == 0 }
}

#[cfg(windows)]
fn process_alive(pid: u32) -> bool {
    process_alive_by_pid(pid)
}

#[cfg(not(any(unix, windows)))]
fn process_alive(pid: u32) -> bool {
    std::path::Path::new(&format!("/proc/{pid}")).exists()
}

/// Resolve until either SIGINT or SIGTERM arrives (daemon stop paths:
/// Ctrl-C or `blazar stop`).
pub async fn wait_for_shutdown_signal() {
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler");
        let mut int = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
            .expect("install SIGINT handler");
        let mut hup = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::hangup())
            .expect("install SIGHUP handler");
        // SIGHUP = hot-reload hint: live registries (keys) re-read
        // config.toml without dropping children. Shutdown still drains.
        loop {
            tokio::select! {
                _ = term.recv() => {
                    tracing::info!("SIGTERM: draining");
                    break;
                }
                _ = int.recv() => {
                    tracing::info!("SIGINT: draining");
                    break;
                }
                _ = hup.recv() => {
                    if let Some(hook) = SIGHUP_HOOK.lock().expect("hook").as_ref() {
                        hook();
                        tracing::info!("SIGHUP: live registries reloaded");
                    }
                }
            }
        }
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c().await.expect("ctrl-c handler");
        tracing::info!("Ctrl-C: draining");
    }
}

/// J4: the daemon installs this in `serve` wiring — SIGHUP re-reads
/// config.toml into the live keys registry (single writer discipline:
/// file -> registry, never the reverse).
pub static SIGHUP_HOOK: std::sync::Mutex<Option<std::sync::Arc<dyn Fn() + Send + Sync>>> =
    std::sync::Mutex::new(None);

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    fn dirs() -> (tempfile::TempDir, BlazarDirs) {
        let tmp = tempfile::tempdir().unwrap();
        let d = BlazarDirs {
            config_dir: tmp.path().join("c"),
            data_dir: tmp.path().join("d"),
        };
        d.ensure().unwrap();
        (tmp, d)
    }

    #[test]
    fn unit__daemon_lock__second_acquire_errors_with_pid() {
        let (_t, d) = dirs();
        let a = DaemonLock::acquire(&d).unwrap();
        let err = DaemonLock::acquire(&d).unwrap_err();
        assert!(err.to_string().contains("already running"), "{err}");
        assert!(err.to_string().contains(&a.pid().to_string()));
        // The marker rides the chain: the CLI maps it to the exit code
        // the systemd unit refuses to restart on.
        assert!(err.downcast_ref::<LockHeld>().is_some());
        drop(a);
        assert!(DaemonLock::acquire(&d).is_ok(), "released on drop");
    }

    #[test]
    fn unit__daemon_lock__stale_pidfile_taken_over() {
        let (_t, d) = dirs();
        // A pid that certainly does not exist.
        std::fs::write(d.run_dir().join("blazar.pid"), "4194303\n").unwrap();
        let lock = DaemonLock::acquire(&d).unwrap();
        assert_ne!(lock.pid(), 4_190_403);
        drop(lock);
        assert!(!d.run_dir().join("blazar.pid").exists());
    }
}
