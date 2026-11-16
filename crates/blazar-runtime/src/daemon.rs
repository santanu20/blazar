//! Daemon lifecycle: pidfile guard and shutdown-signal handling.
//! One daemon per data dir; a second start fails fast naming the pid.
//! Refusals carry the [`LockHeld`] marker so the CLI can map them to the
//! singleton-conflict exit code the systemd unit refuses to restart on.

use anyhow::{Context, Result, anyhow};

use blazar_core::BlazarDirs;

/// Refusing to start because a live peer owns the daemon lock. Distinct
/// error type so `blazar serve` can exit with the same hard-conflict
/// code as a port bind conflict - `Restart=always` must not loop on it.
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
                // An unreadable or unparseable pidfile is UNKNOWN state, not
                // evidence of a stale lock: guessing "stale" could fork a
                // second daemon under a live peer. Refuse and name the file.
                let existing = match std::fs::read_to_string(&path) {
                    Ok(s) => match s.trim().parse::<u32>() {
                        Ok(pid) => pid,
                        Err(why) => {
                            return Err(anyhow::anyhow!(
                                "pidfile {} is unparseable ({why}): no blazar can be \
                                 verified running - remove the file if you are certain \
                                 none is, then retry",
                                path.display()
                            ));
                        }
                    },
                    Err(why) => {
                        return Err(anyhow::anyhow!(
                            "pidfile {} is unreadable ({why}): no blazar can be verified \
                             running - fix permissions or remove the file if you are \
                             certain none is, then retry",
                            path.display()
                        ));
                    }
                };
                if existing > 1 && process_alive(existing) {
                    // Remedy lives at the CLI boundary (one place, user-
                    // facing); the chain stays fact-only so composed
                    // output never repeats the instruction.
                    return Err(anyhow::Error::new(LockHeld)
                        .context(format!("blazar already running (pid {existing})")));
                }
                // Stale lock from a dead daemon: take over. F90: loop
                // back through `create_new` instead of remove+plain-write
                // - the old sequence had a window where a concurrent
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
    // takeover ALWAYS win - two daemons on one box. tasklist is the
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
/// must not outlive their harness: when the parent dies - a `timeout(1)`
/// SIGKILL bypasses every atexit sweep - the kernel delivers the signal
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
        // Race: the parent may have died between spawn and prctl - the
        // signal would never fire. Reparenting to init already happened.
        if libc::getppid() == 1 {
            std::process::exit(0);
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub fn validate_parent_death_guard() {}

/// Environment stamp naming the data dir whose daemon spawns a process.
/// Every engine child inherits it at exec; the boot reclaim sweep matches
/// it to find children whose daemon died without teardown. The value is
/// the plain data-dir path (not secret): it identifies ownership, like a
/// pidfile that survives inside the child itself.
pub const ORPHAN_MARKER_ENV: &str = "BLAZAR_DAEMON_DATA_DIR";

/// Stamp this daemon's data dir into the process environment BEFORE any
/// engine child can be spawned, so every child (engine servers, whisper,
/// piper, and their subprocesses - env is inherited down the whole tree)
/// carries the marker at exec. Idempotent overwrite: a daemon re-exec'd
/// by a gateway auto-start chain inherits a stale marker from its parent
/// and must correct it before propagating it further.
///
/// The kernel keeps `/proc/<pid>/environ` at exec-time values, so this
/// set does NOT make the daemon itself visible to the sweep - only its
/// children, which is exactly the ownership question.
#[allow(unsafe_code)] // one env::set_var at serve() entry; SAFETY below
pub fn set_orphan_marker(data_dir: &std::path::Path) {
    // SAFETY: called once at serve() entry, before any task that reads
    // this key exists (only the boot reclaim reads it, on this same
    // thread, later). No other code reads or writes ORPHAN_MARKER_ENV
    // concurrently.
    unsafe { std::env::set_var(ORPHAN_MARKER_ENV, data_dir) };
}

/// Exact-token marker test over raw `/proc/<pid>/environ` bytes
/// (NUL-separated KEY=VALUE tokens). Exact token compare means a data
/// dir that is a prefix of another can never alias it.
#[cfg(target_os = "linux")]
fn environ_marks_owner(env: &[u8], data_dir: &std::path::Path) -> bool {
    let wanted = format!("{}={}", ORPHAN_MARKER_ENV, data_dir.display());
    env.split(|b| *b == 0).any(|tok| tok == wanted.as_bytes())
}

/// Read `/proc/<pid>/environ` and test the marker. `None` = unreadable
/// (not our process, or gone): never a match.
#[cfg(target_os = "linux")]
fn pid_marks_owner(pid: u32, data_dir: &std::path::Path) -> bool {
    std::fs::read(format!("/proc/{pid}/environ"))
        .is_ok_and(|env| environ_marks_owner(&env, data_dir))
}

/// Every live process whose exec-time env carries OUR data-dir marker.
/// The sweep runs while this daemon holds [`DaemonLock`], so a match can
/// only be a stray from a dead predecessor - never a sibling's child
/// (a live sibling would hold the lock). Linux-only: `/proc`-based, in
/// parity with PDEATHSIG being Linux-only.
#[cfg(target_os = "linux")]
fn marker_orphan_pids(data_dir: &std::path::Path) -> Vec<u32> {
    let own_pid = std::process::id();
    let Ok(dir) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };
    dir.flatten()
        .filter_map(|e| {
            let pid = e.file_name().to_str()?.parse::<u32>().ok()?;
            (pid > 1 && pid != own_pid).then_some(pid)
        })
        .filter(|pid| pid_marks_owner(*pid, data_dir))
        .collect()
}

/// Signal `pid` through a pidfd pinned BEFORE the identity check, so a
/// PID-reuse race between verification and the signal can never hit an
/// innocent process: the fd pins whichever process held the pid at
/// `pidfd_open` time, the identity check then verifies that same
/// process via `/proc`, and the signal travels the fd - not the pid.
///
/// Returns `None` when the kernel/libc has no pidfd lane (pre-5.3
/// kernels): the caller falls back to the classic re-read + `kill`
/// path. `Some(true)` = signalled; `Some(false)` = identity failed or
/// the process exited (skip, and do NOT fall back - re-reading the pid
/// would inspect a potentially different process).
#[cfg(target_os = "linux")]
#[allow(unsafe_code)] // three audited raw syscalls below
pub(crate) fn pidfd_guarded_signal(
    pid: u32,
    sig: i32,
    still_ours: impl FnOnce(u32) -> bool,
) -> Option<bool> {
    let Ok(raw_pid) = libc::pid_t::try_from(pid) else {
        return Some(false);
    };
    // SAFETY: pidfd_open(raw_pid, 0) takes a pid and flags by value;
    // a fresh fd or -1/errno comes back. No pointers involved.
    let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, raw_pid, 0u32) };
    if fd < 0 {
        return match std::io::Error::last_os_error().raw_os_error() {
            // No pidfd support on this kernel: caller falls back.
            Some(libc::ENOSYS | libc::EINVAL | libc::EPERM) => None,
            // Process gone (or anything else): not signalled, no fallback.
            _ => Some(false),
        };
    }
    let Ok(fd) = libc::c_int::try_from(fd) else {
        return Some(false);
    };
    // Verify identity AFTER the pin: /proc/<pid> now describes exactly
    // the process the fd holds (or nothing, if it died and was reaped).
    if !still_ours(pid) {
        // SAFETY: close on our own fresh fd.
        unsafe { libc::close(fd) };
        return Some(false);
    }
    // SAFETY: pidfd_send_signal(fd, sig, NULL, 0) - signals exactly the
    // pinned process; the info pointer is deliberately NULL (no siginfo).
    let rc = unsafe {
        libc::syscall(
            libc::SYS_pidfd_send_signal,
            fd,
            sig,
            std::ptr::null::<libc::c_void>(),
            0u32,
        )
    };
    // SAFETY: close on our own fresh fd.
    unsafe { libc::close(fd) };
    Some(rc == 0)
}

/// Escalate already-TERMed stray pids: poll liveness for `grace`, then
/// SIGKILL every survivor whose identity check still passes. Returns the
/// pids that needed the KILL. Shared by the marker reclaim and the
/// pidfile sweep - both signal SIGTERM without waiting and need the same
/// teeth against children that hang in graceful shutdown.
#[cfg(unix)]
#[allow(unsafe_code)] // audited signal calls on exact pids below
pub(crate) fn escalate_orphans(
    pids: &[u32],
    grace: std::time::Duration,
    still_ours: impl Fn(u32) -> bool,
) -> Vec<u32> {
    let deadline = std::time::Instant::now() + grace;
    let stragglers = loop {
        let alive: Vec<u32> = pids
            .iter()
            .copied()
            .filter(|pid| process_alive_by_pid(*pid))
            .collect();
        if alive.is_empty() || std::time::Instant::now() >= deadline {
            break alive;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    };
    for pid in &stragglers {
        // Identity re-check right before the irreversible signal: a pid
        // that exited and got recycled within the grace window must never
        // receive our KILL. On Linux the pidfd lane pins the process
        // before the check, closing the check-to-signal reuse race.
        #[cfg(target_os = "linux")]
        {
            match pidfd_guarded_signal(*pid, libc::SIGKILL, &still_ours) {
                Some(true) => {
                    tracing::warn!("orphan child pid {pid} ignored SIGTERM; SIGKILL");
                    continue;
                }
                Some(false) => continue,
                None => {} // no pidfd lane: classic path below
            }
        }
        if !still_ours(*pid) {
            continue;
        }
        tracing::warn!("orphan child pid {pid} ignored SIGTERM; SIGKILL");
        // SAFETY: single positive pid that passed the identity check one
        // line above; kill signals exactly that pid, no group.
        unsafe {
            libc::kill(i32::try_from(*pid).unwrap_or(-1), libc::SIGKILL);
        }
    }
    stragglers
}

/// Reclaim engine children orphaned by a dead predecessor daemon: every
/// live process carrying our data-dir marker gets SIGTERM, a bounded
/// grace window, then SIGKILL. Runs at `serve()` boot after
/// [`DaemonLock::acquire`] - the lock guarantees no live daemon owns
/// these processes. Returns one human-readable line per reclaimed pid.
#[cfg(target_os = "linux")]
#[must_use = "the caller prints each line as preflight evidence"]
pub fn reclaim_marker_orphans(data_dir: &std::path::Path) -> Vec<String> {
    reclaim_marker_orphans_with_grace(data_dir, std::time::Duration::from_secs(10))
}

#[cfg(target_os = "linux")]
#[allow(unsafe_code)] // two audited libc::kill calls below
fn reclaim_marker_orphans_with_grace(
    data_dir: &std::path::Path,
    grace: std::time::Duration,
) -> Vec<String> {
    let pids = marker_orphan_pids(data_dir);
    if pids.is_empty() {
        return Vec::new();
    }
    let mut lines = Vec::new();
    for pid in &pids {
        // Guard the signal with a fresh identity read: the scan
        // snapshot can be stale by one syscall. On Linux the pidfd
        // lane pins the process before the read, so even a PID reused
        // in that window cannot receive our TERM.
        #[cfg(target_os = "linux")]
        {
            match pidfd_guarded_signal(*pid, libc::SIGTERM, |p| pid_marks_owner(p, data_dir)) {
                Some(true) => {
                    tracing::warn!(
                        "reclaiming orphan engine child pid {pid} (marker {}) - its daemon died without teardown",
                        data_dir.display()
                    );
                    lines.push(format!("pid {pid} (marker {})", data_dir.display()));
                    continue;
                }
                Some(false) => continue,
                None => {} // no pidfd lane: classic path below
            }
        }
        if !pid_marks_owner(*pid, data_dir) {
            continue;
        }
        tracing::warn!(
            "reclaiming orphan engine child pid {pid} (marker {}) - its daemon died without teardown",
            data_dir.display()
        );
        // SAFETY: single positive pid whose exec-time env still carries
        // our marker, re-read above; kill signals exactly that pid.
        unsafe {
            libc::kill(i32::try_from(*pid).unwrap_or(-1), libc::SIGTERM);
        }
        lines.push(format!("pid {pid} (marker {})", data_dir.display()));
    }
    let killed = escalate_orphans(&pids, grace, |pid| pid_marks_owner(pid, data_dir));
    if !killed.is_empty() {
        lines.push(format!(
            "{} straggler(s) needed SIGKILL: {}",
            killed.len(),
            killed
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    lines
}

#[cfg(not(target_os = "linux"))]
#[must_use = "the caller prints each line as preflight evidence"]
pub fn reclaim_marker_orphans(_data_dir: &std::path::Path) -> Vec<String> {
    // Platform matrix for orphan reclamation:
    //   Linux   - /proc environ marker sweep + PDEATHSIG + pidfd-guarded
    //             signals (no PID-reuse window between identity check and
    //             signal on kernels >= 5.3; classic re-read + kill below).
    //   macOS   - no /proc environ lane: the pidfile sweep plus PDEATHSIG-
    //             adjacent child shutdown remain the covering nets.
    //   Windows - pidfile sweep + child kill on teardown. OS-level Job
    //             Objects would pin the whole process tree (survives
    //             daemon crash by kernel guarantee) but require a spawn-
    //             path rework plus a windows API dependency - deliberately
    //             deferred until a crash-recovery incident justifies it.
    Vec::new()
}

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

/// J4: the daemon installs this in `serve` wiring - SIGHUP re-reads
/// config.toml into the live keys registry (single writer discipline:
/// file -> registry, never the reverse).
pub static SIGHUP_HOOK: std::sync::Mutex<Option<std::sync::Arc<dyn Fn() + Send + Sync>>> =
    std::sync::Mutex::new(None);

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    /// A real child, signalled through the pidfd lane, dies by that
    /// signal - and a child the identity check refuses stays alive.
    #[cfg(target_os = "linux")]
    #[allow(unsafe_code)] // one teardown TERM on our own spawned child
    #[test]
    fn unit__pidfd_guarded_signal__signals_child_and_refuses_foreign() {
        use std::process::Command;
        // Legitimate target: our own sleeper, identity check passes.
        let mut mine = Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn sleep");
        let pid = mine.id();
        let signalled =
            pidfd_guarded_signal(pid, libc::SIGTERM, |_| true).expect("pidfd on a modern kernel");
        assert!(signalled, "own child must be signalled through the pin");
        let status = mine.wait().expect("reap child");
        assert!(
            std::os::unix::process::ExitStatusExt::signal(&status) == Some(libc::SIGTERM),
            "child died by SIGTERM, got {status:?}"
        );

        // Foreign process shape: identity check refuses -> no signal,
        // child survives untouched.
        let mut stranger = Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn sleep");
        let pid = stranger.id();
        let signalled =
            pidfd_guarded_signal(pid, libc::SIGTERM, |_| false).expect("pidfd lane available");
        assert!(!signalled, "refused identity must never signal");
        let alive = crate::daemon::process_alive_by_pid(pid);
        // Teardown regardless: TERM then reap so the test leaves nothing.
        unsafe { libc::kill(i32::try_from(pid).unwrap_or(-1), libc::SIGTERM) };
        let status = stranger.wait().expect("reap child");
        assert!(alive, "refused child must still be alive at check time");
        assert!(
            std::os::unix::process::ExitStatusExt::signal(&status) == Some(libc::SIGTERM),
            "child died by the teardown TERM, got {status:?}"
        );
    }

    /// A pid that exited and was reaped opens no pin: the helper must
    /// report not-signalled (and never fall back to a bare kill).
    #[cfg(target_os = "linux")]
    #[test]
    fn unit__pidfd_guarded_signal__dead_pid_is_not_signalled() {
        use std::process::Command;
        let mut gone = Command::new("true").spawn().expect("spawn true");
        let pid = gone.id();
        let _ = gone.wait();
        let signalled = pidfd_guarded_signal(pid, libc::SIGKILL, |_| true);
        assert_eq!(
            signalled,
            Some(false),
            "reaped pid: open fails ESRCH -> skip, never fallback-kill"
        );
    }

    /// The reclaim loop end-to-end against a REAL marker orphan: spawn
    /// a child carrying our data-dir marker, reclaim it, assert the
    /// teaching line and the child's death by SIGTERM. A concurrent
    /// reaper stands in for init (a reaped child stops answering
    /// signal-0 liveness, so the grace window ends the moment the TERM
    /// lands instead of polling zombies for the full window).
    #[cfg(target_os = "linux")]
    #[test]
    fn unit__reclaim_marker_orphans__terminates_real_marker_child() {
        use std::process::Command;
        let (tmp, d) = dirs();
        let mut child = Command::new("sleep")
            .arg("30")
            .env(ORPHAN_MARKER_ENV, d.data_dir.as_os_str())
            .spawn()
            .expect("spawn marked sleep");
        let pid = child.id();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(child.wait().expect("reap reclaimed child"));
        });
        let lines = reclaim_marker_orphans(&d.data_dir);
        assert!(
            lines.iter().any(|l| l.contains(&pid.to_string())),
            "reclaim must report the marked child: {lines:?}"
        );
        let status = rx
            .recv_timeout(std::time::Duration::from_secs(15))
            .expect("child must be reclaimed within the grace window");
        assert!(
            std::os::unix::process::ExitStatusExt::signal(&status) == Some(libc::SIGTERM)
                || status.code() == Some(0),
            "child stopped by the reclaim sweep, got {status:?}"
        );
        // A second sweep finds nothing: the marker child is gone.
        assert!(
            reclaim_marker_orphans(&d.data_dir).is_empty(),
            "second sweep must be clean"
        );
        drop(tmp);
    }

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

    #[test]
    fn unit__daemon_lock__garbage_pidfile_refuses_with_remedy() {
        let (_t, d) = dirs();
        std::fs::write(d.run_dir().join("blazar.pid"), "not-a-pid\n").unwrap();
        let err = DaemonLock::acquire(&d).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("unparseable"), "{msg}");
        // The remedy names the exact file so the operator can act on it.
        assert!(
            msg.contains(
                d.run_dir()
                    .join("blazar.pid")
                    .display()
                    .to_string()
                    .as_str()
            ),
            "{msg}"
        );
        // Refused, not taken over: the garbage file stays for inspection.
        assert!(d.run_dir().join("blazar.pid").exists());
    }

    // V3 (escalation): a child that ignores SIGTERM must die to the KILL
    // arm within the grace window.
    #[cfg(target_os = "linux")]
    #[test]
    fn unit__reclaim_marker_orphans__marked_decends_term_ignoring_killed() {
        let (tmp, d) = dirs();
        let marker = d.data_dir.display().to_string();
        // Single-process decoy: `exec` replaces bash with sleep in the
        // SAME pid, and an ignored signal disposition survives exec - so
        // exactly one TERM-immune marked process exists for the sweep to
        // kill (a forked sleep child would leak into the next sweep).
        let mut stubborn = std::process::Command::new("bash")
            .arg("-c")
            .arg("trap '' TERM; exec sleep 300")
            .env(ORPHAN_MARKER_ENV, &marker)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("stubborn decoy");
        // Unmarked control: the sweep must never touch it.
        let mut control = std::process::Command::new("sleep")
            .arg("300")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("control decoy");

        // The sweep can only see the marker once execve has stamped it
        // into /proc/<pid>/environ - between fork and exec the decoy
        // still shows this process's unmarked env, and under heavy CPU
        // load that window outlives the spawn call. Production never
        // races this (the boot sweep meets children exec'd long ago);
        // wait for the precondition the sweep actually depends on.
        let mut marker_visible = false;
        for _ in 0..500 {
            if pid_marks_owner(stubborn.id(), &d.data_dir) {
                marker_visible = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(
            marker_visible,
            "decoy never exposed its marker (exec blocked >5s?) - sweep would race it"
        );

        let lines =
            reclaim_marker_orphans_with_grace(&d.data_dir, std::time::Duration::from_secs(1));
        assert_eq!(
            lines.len(),
            2,
            "reclaim line + straggler summary: {lines:?}"
        );
        assert!(
            lines[1].contains("SIGKILL"),
            "escalation must be reported: {lines:?}"
        );
        // Reap before liveness asserts: a KILLed child of THIS test
        // process is a zombie until waited on, and kill(pid,0) happily
        // reports zombies alive (real orphans reparent to init, which
        // reaps them - this dance is test-only).
        let mut exited = None;
        for _ in 0..50 {
            if let Some(status) = stubborn.try_wait().unwrap() {
                exited = Some(status);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        assert!(
            exited.is_some(),
            "TERM-ignoring decoy must be KILLed (still running after grace + poll)"
        );
        assert!(
            process_alive_by_pid(control.id()),
            "unmarked control must survive"
        );
        let _ = control.kill();
        let _ = control.wait();

        // V2 (idempotency): nothing left to reclaim.
        let again =
            reclaim_marker_orphans_with_grace(&d.data_dir, std::time::Duration::from_secs(1));
        assert!(again.is_empty(), "second sweep must be a no-op: {again:?}");
        drop(tmp);
    }

    // Pure marker semantics: exact NUL-token equality, so a data dir that
    // is a prefix of another can never alias it.
    #[cfg(target_os = "linux")]
    #[test]
    fn unit__environ_marks_owner__exact_token_no_prefix_alias() {
        let env = format!("PATH=/usr/bin\0{ORPHAN_MARKER_ENV}=/srv/blazar\0HOME=/root\0");
        assert!(environ_marks_owner(
            env.as_bytes(),
            std::path::Path::new("/srv/blazar")
        ));
        assert!(!environ_marks_owner(
            env.as_bytes(),
            std::path::Path::new("/srv/blazar-team")
        ));
        assert!(!environ_marks_owner(
            env.as_bytes(),
            std::path::Path::new("/srv")
        ));
    }
}
