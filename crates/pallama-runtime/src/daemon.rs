//! Daemon lifecycle: pidfile guard and shutdown-signal handling.
//! One daemon per data dir; a second start fails fast naming the pid.

use anyhow::{anyhow, Context, Result};

use pallama_core::PallamaDirs;

/// RAII daemon lock: `<run_dir>/pallama.pid`, removed on drop.
#[derive(Debug)]
pub struct DaemonLock {
    path: std::path::PathBuf,
    pid: u32,
}

impl DaemonLock {
    pub fn acquire(dirs: &PallamaDirs) -> Result<Self> {
        std::fs::create_dir_all(dirs.run_dir())?;
        let path = dirs.run_dir().join("pallama.pid");
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
                    return Err(anyhow!(
                        "pallama already running (pid {existing}); if this is wrong, remove {}",
                        path.display()
                    ));
                }
                // Stale lock from a dead daemon: take over.
                tracing::warn!("removing stale pidfile for dead pid {existing}");
                std::fs::remove_file(&path)
                    .with_context(|| format!("remove {}", path.display()))?;
                std::fs::write(&path, format!("{pid}\n"))?;
                Ok(Self { path, pid })
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

#[cfg(not(unix))]
#[must_use]
pub fn process_alive_by_pid(pid: u32) -> bool {
    std::path::Path::new(&format!("/proc/{pid}")).exists()
}

#[cfg(unix)]
#[allow(unsafe_code)] // existence probe via signal 0 to one exact pid
fn process_alive(pid: u32) -> bool {
    // SAFETY: signal 0 performs no action; single positive pid, no group.
    unsafe { libc::kill(i32::try_from(pid).unwrap_or(-1), 0) == 0 }
}

#[cfg(not(unix))]
fn process_alive(pid: u32) -> bool {
    std::path::Path::new(&format!("/proc/{pid}")).exists()
}

/// Resolve until either SIGINT or SIGTERM arrives (daemon stop paths:
/// Ctrl-C or `pallama stop`).
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

    fn dirs() -> (tempfile::TempDir, PallamaDirs) {
        let tmp = tempfile::tempdir().unwrap();
        let d = PallamaDirs {
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
        drop(a);
        assert!(DaemonLock::acquire(&d).is_ok(), "released on drop");
    }

    #[test]
    fn unit__daemon_lock__stale_pidfile_taken_over() {
        let (_t, d) = dirs();
        // A pid that certainly does not exist.
        std::fs::write(d.run_dir().join("pallama.pid"), "4194303\n").unwrap();
        let lock = DaemonLock::acquire(&d).unwrap();
        assert_ne!(lock.pid(), 4_190_403);
        drop(lock);
        assert!(!d.run_dir().join("pallama.pid").exists());
    }
}
