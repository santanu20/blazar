use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Layer;
use tracing_subscriber::fmt::writer::MakeWriter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Tracing sink that tees every event to stderr and, when armed, to a
/// daemon-owned log file. File-side failures (rotated-away file, full
/// disk) never take the daemon down and stderr still emits - but they
/// are no longer invisible: every arm is counted, the first failure of
/// an incident is announced on stderr, and a file replaced on disk is
/// re-opened so the lines that follow land in the current file.
struct TeeWriter {
    file: Option<Arc<Mutex<std::fs::File>>>,
    poison_warned: Arc<AtomicBool>,
}

impl MakeWriter<'_> for TeeWriter {
    type Writer = TeeWrite;

    fn make_writer(&self) -> Self::Writer {
        TeeWrite {
            file: self.file.clone(),
            poison_warned: Arc::clone(&self.poison_warned),
        }
    }
}

struct TeeWrite {
    file: Option<Arc<Mutex<std::fs::File>>>,
    poison_warned: Arc<AtomicBool>,
}

impl TeeWrite {
    #[cfg(test)]
    fn armed(file: Arc<Mutex<std::fs::File>>) -> Self {
        Self {
            file: Some(file),
            poison_warned: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Lock the file arm, recovering from poison exactly once with a
    /// stderr marker. `None` when the arm is unarmed.
    fn lock_file(&self) -> Option<std::sync::MutexGuard<'_, std::fs::File>> {
        let file = self.file.as_ref()?;
        match file.lock() {
            Ok(guard) => Some(guard),
            Err(poisoned) => {
                // The poison flag never clears once set, so a single
                // panic mid-write would otherwise silence the file arm
                // forever while stderr carries on - recover the guard
                // and name the moment exactly once.
                if !self.poison_warned.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "blazar: log file arm hit a poisoned lock (a writer panicked mid-write); recovering - subsequent lines are NOT lost"
                    );
                }
                Some(poisoned.into_inner())
            }
        }
    }
}

impl std::io::Write for TeeWrite {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if std::io::stderr().write_all(buf).is_ok() {
            STDERR_WRITES.fetch_add(1, Ordering::Relaxed);
        } else {
            STDERR_FAILURES.fetch_add(1, Ordering::Relaxed);
        }
        // The guard is dropped before any marker or re-open runs: the
        // file mutex is not re-entrant, and nothing in here may emit a
        // tracing event (it would come straight back into this writer).
        let outcome = self.lock_file().map(|mut guard| guard.write_all(buf));
        match outcome {
            Some(Ok(())) => {
                FILE_WRITES.fetch_add(1, Ordering::Relaxed);
                FILE_FAIL_WARNED.store(false, Ordering::Relaxed);
            }
            Some(Err(_)) => {
                FILE_FAILURES.fetch_add(1, Ordering::Relaxed);
                if !FILE_FAIL_WARNED.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "blazar: log file write failed - lines are reaching stderr only; the file arm is re-checked every detector tick and recovers on its own"
                    );
                    reopen_file_if_replaced();
                }
            }
            None => {}
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let _ = std::io::stderr().flush();
        if let Some(mut guard) = self.lock_file() {
            let _ = guard.flush();
        }
        Ok(())
    }
}

/// Counts the events the filter let through, whatever the sinks then
/// did with them. A sink that receives events but writes nothing is the
/// stall this counters exist to make visible.
struct EventCounter;

impl<S: tracing::Subscriber> Layer<S> for EventCounter {
    fn on_event(
        &self,
        _event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        EVENTS.fetch_add(1, Ordering::Relaxed);
    }
}

static EVENTS: AtomicU64 = AtomicU64::new(0);
static STDERR_WRITES: AtomicU64 = AtomicU64::new(0);
static STDERR_FAILURES: AtomicU64 = AtomicU64::new(0);
static FILE_WRITES: AtomicU64 = AtomicU64::new(0);
static FILE_FAILURES: AtomicU64 = AtomicU64::new(0);
static FILE_REOPENED: AtomicU64 = AtomicU64::new(0);
/// Set by the first file-arm failure of an incident, cleared by the
/// next successful write: one marker per incident, not one per line.
static FILE_FAIL_WARNED: AtomicBool = AtomicBool::new(false);
static REOPEN_FAIL_WARNED: AtomicBool = AtomicBool::new(false);

/// The live file arm, so the daemon's health tick can swap the handle
/// when the path on disk no longer names the file being written.
struct FileSink {
    path: PathBuf,
    file: Arc<Mutex<std::fs::File>>,
}

static FILE_SINK: Mutex<Option<FileSink>> = Mutex::new(None);

fn register_file_sink(path: &Path, file: Arc<Mutex<std::fs::File>>) {
    match FILE_SINK.lock() {
        Ok(mut slot) => {
            *slot = Some(FileSink {
                path: path.to_path_buf(),
                file,
            });
        }
        Err(poisoned) => {
            *poisoned.into_inner() = Some(FileSink {
                path: path.to_path_buf(),
                file,
            });
        }
    }
}

/// Per-sink write accounting since process start.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SinkStats {
    /// Events the filter enabled and handed to the sinks.
    pub events: u64,
    pub stderr_writes: u64,
    pub stderr_failures: u64,
    pub file_writes: u64,
    pub file_failures: u64,
    pub file_reopened: u64,
}

/// Current sink counters (process-wide; one subscriber per process).
#[must_use]
pub fn sink_stats() -> SinkStats {
    SinkStats {
        events: EVENTS.load(Ordering::Relaxed),
        stderr_writes: STDERR_WRITES.load(Ordering::Relaxed),
        stderr_failures: STDERR_FAILURES.load(Ordering::Relaxed),
        file_writes: FILE_WRITES.load(Ordering::Relaxed),
        file_failures: FILE_FAILURES.load(Ordering::Relaxed),
        file_reopened: FILE_REOPENED.load(Ordering::Relaxed),
    }
}

/// Whether the durable log file arm is active.
#[must_use]
pub fn file_sink_armed() -> bool {
    match FILE_SINK.lock() {
        Ok(slot) => slot.is_some(),
        Err(poisoned) => poisoned.into_inner().is_some(),
    }
}

/// Re-open the durable log when the path on disk no longer names the
/// file the daemon holds (rotated, replaced, deleted by something else
/// while the daemon kept writing to the old inode). Returns true when
/// the handle was swapped. Unix compares inode numbers; elsewhere the
/// check is a no-op.
pub fn reopen_file_if_replaced() -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let slot = match FILE_SINK.lock() {
            Ok(slot) => slot,
            Err(poisoned) => poisoned.into_inner(),
        };
        let Some(sink) = slot.as_ref() else {
            return false;
        };
        let held = sink.file.lock().ok().and_then(|file| file.metadata().ok());
        let on_disk = std::fs::metadata(&sink.path).ok();
        let replaced = match (&held, &on_disk) {
            (Some(held), Some(on_disk)) => {
                held.dev() != on_disk.dev() || held.ino() != on_disk.ino()
            }
            _ => true,
        };
        if !replaced {
            return false;
        }
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&sink.path)
        {
            Ok(mut new_file) => {
                if let Ok(mut file) = sink.file.lock() {
                    std::mem::swap(&mut *file, &mut new_file);
                } else {
                    return false;
                }
                FILE_REOPENED.fetch_add(1, Ordering::Relaxed);
                REOPEN_FAIL_WARNED.store(false, Ordering::Relaxed);
                eprintln!(
                    "blazar: log file {} was replaced on disk - re-opened, later lines land in the current file",
                    sink.path.display()
                );
                true
            }
            Err(e) => {
                if !REOPEN_FAIL_WARNED.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "blazar: log file {} was replaced on disk and cannot be re-opened ({e}) - file logging is detached until it can",
                        sink.path.display()
                    );
                }
                false
            }
        }
    }
    #[cfg(not(unix))]
    {
        false
    }
}

/// Sink counters as Prometheus text, for the gateway's `/metrics`.
#[must_use]
pub fn sink_metrics() -> String {
    let stats = sink_stats();
    format!(
        "# HELP blazar_log_events_total Log events the filter enabled and handed to the sinks\n\
         # TYPE blazar_log_events_total counter\n\
         blazar_log_events_total {}\n\
         # HELP blazar_log_sink_writes_total Successful log writes per sink\n\
         # TYPE blazar_log_sink_writes_total counter\n\
         blazar_log_sink_writes_total{{sink=\"stderr\"}} {}\n\
         blazar_log_sink_writes_total{{sink=\"file\"}} {}\n\
         # HELP blazar_log_sink_write_failures_total Failed log writes per sink (full disk, replaced file, closed stderr)\n\
         # TYPE blazar_log_sink_write_failures_total counter\n\
         blazar_log_sink_write_failures_total{{sink=\"stderr\"}} {}\n\
         blazar_log_sink_write_failures_total{{sink=\"file\"}} {}\n\
         # HELP blazar_log_sink_reopened_total Times the durable log handle was re-opened after the file was replaced on disk\n\
         # TYPE blazar_log_sink_reopened_total counter\n\
         blazar_log_sink_reopened_total {}\n",
        stats.events,
        stats.stderr_writes,
        stats.file_writes,
        stats.stderr_failures,
        stats.file_failures,
        stats.file_reopened,
    )
}

/// Initialize tracing. Called once by the daemon/CLI entrypoint.
/// `RUST_LOG` env var wins when set (standard tracing behavior);
/// then the config `log_level` directive (daemon escape hatch without
/// editing the systemd unit); otherwise `verbosity` selects the default.
/// An invalid directive warns on stderr and falls back - logs are never
/// silently dropped. `log_file` arms a durable daemon-owned sink
/// (the `serve` path passes `run/daemon.log`): under systemd the
/// journal is a secondary stream that rotation can strand - the file
/// is the one the daemon itself owns end to end.
#[allow(clippy::needless_pass_by_value)] // PathBuf-free: caller keeps ownership
pub fn init_tracing(verbosity: u8, log_level: Option<&str>, log_file: Option<&Path>) {
    let default = match verbosity {
        0 => "blazar=info,warn",
        1 => "blazar=debug,info",
        _ => "blazar=trace,debug",
    };
    let filter = match EnvFilter::try_from_default_env() {
        Ok(f) => f,
        Err(_) => match log_level.map(|d| EnvFilter::builder().parse(d)) {
            Some(Ok(f)) => f,
            Some(Err(e)) => {
                eprintln!("blazar: invalid log_level directive ({e}) - using built-in default");
                EnvFilter::new(default)
            }
            None => EnvFilter::new(default),
        },
    };
    let file = log_file.and_then(|path| {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            Ok(f) => {
                let shared = Arc::new(Mutex::new(f));
                register_file_sink(path, Arc::clone(&shared));
                Some(shared)
            }
            Err(e) => {
                eprintln!(
                    "blazar: cannot open log file {} ({e}) - logging to stderr only",
                    path.display()
                );
                None
            }
        }
    });
    let tee = TeeWriter {
        file,
        poison_warned: Arc::new(AtomicBool::new(false)),
    };
    // The filter wraps the whole stack (formatter AND the event
    // counter), so the counter sees exactly the events the formatter
    // was asked to write.
    // journald already stamps every line (with the machine's local time);
    // a second RFC3339 UTC stamp from tracing only adds noise and a
    // timezone mismatch. Foreground terminals and the daemon-owned log
    // file have no outer stamper, so they keep the tracing timestamp -
    // when the file sink is armed the stamp stays even under journald
    // (a duplicate stamp in the journal beats an untimestamped durable
    // log, which is the artifact post-mortems actually read).
    // JOURNAL_STREAM is systemd's documented marker for journald-attached
    // stdio.
    if std::env::var_os("JOURNAL_STREAM").is_some() && log_file.is_none() {
        tracing_subscriber::fmt()
            .with_target(false)
            .without_time()
            .with_writer(tee)
            .finish()
            .with(EventCounter)
            .with(filter)
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_target(false)
            .with_writer(tee)
            .finish()
            .with(EventCounter)
            .with(filter)
            .init();
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use super::*;
    use std::io::Write as _;

    // The file arm and the counters are process-wide: tests that touch
    // them run one at a time.
    static SINK_LOCK: Mutex<()> = Mutex::new(());

    fn sink_lock() -> std::sync::MutexGuard<'static, ()> {
        SINK_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn tee_for(file: std::fs::File) -> TeeWrite {
        TeeWrite::armed(Arc::new(Mutex::new(file)))
    }

    #[test]
    fn unit__tee_write__mirrors_bytes_into_the_file() {
        let tmp = tempfile::tempdir().unwrap();
        let log = tmp.path().join("tee.log");
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log)
            .unwrap();
        let mut tee = tee_for(file);
        tee.write_all(b"line one\n").unwrap();
        tee.write_all(b"line two\n").unwrap();
        tee.flush().unwrap();
        assert_eq!(
            std::fs::read_to_string(&log).unwrap(),
            "line one\nline two\n"
        );
    }

    #[test]
    fn unit__tee_write__unarmed_tee_still_writes() {
        let _serial = sink_lock();
        let mut tee = TeeWrite {
            file: None,
            poison_warned: Arc::new(AtomicBool::new(false)),
        };
        let before = sink_stats();
        // No file, and stderr is not observable here - the contract under
        // test is that the unarmed path returns success, not panics, and
        // that it invents neither a file write nor a file failure.
        assert_eq!(tee.write(b"x").unwrap(), 1);
        assert!(tee.flush().is_ok());
        let after = sink_stats();
        assert_eq!(after.file_writes, before.file_writes);
        assert_eq!(after.file_failures, before.file_failures);
    }

    #[test]
    fn unit__tee_write__poisoned_lock_recovers_instead_of_going_dark() {
        let tmp = tempfile::tempdir().unwrap();
        let log = tmp.path().join("tee.log");
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log)
            .unwrap();
        let shared = Arc::new(Mutex::new(file));
        // Poison the mutex the way it happens in production: a panic
        // while the guard is held.
        let boom = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = shared.lock().unwrap();
            panic!("writer died mid-write");
        }));
        assert!(boom.is_err());
        assert!(shared.is_poisoned());
        let mut tee = TeeWrite {
            file: Some(Arc::clone(&shared)),
            poison_warned: Arc::new(AtomicBool::new(false)),
        };
        // Pre-fix behavior: lock() errs, write silently skipped, file
        // stays empty forever. Post-fix: the guard is recovered and the
        // bytes land.
        tee.write_all(b"after poison\n").unwrap();
        tee.flush().unwrap();
        assert_eq!(std::fs::read_to_string(&log).unwrap(), "after poison\n");
        // The one-time marker fired.
        assert!(tee.poison_warned.load(Ordering::Relaxed));
    }

    #[test]
    fn unit__tee_write__healthy_file_arm_counts_the_write() {
        let _serial = sink_lock();
        let tmp = tempfile::tempdir().unwrap();
        let log = tmp.path().join("counted.log");
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log)
            .unwrap();
        let mut tee = tee_for(file);
        let before = sink_stats();
        tee.write_all(b"counted\n").unwrap();
        let after = sink_stats();
        assert_eq!(after.file_writes, before.file_writes + 1);
        assert_eq!(after.file_failures, before.file_failures);
        assert_eq!(std::fs::read_to_string(&log).unwrap(), "counted\n");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn unit__tee_write__full_disk_counts_the_file_failure_and_keeps_stderr() {
        let _serial = sink_lock();
        // /dev/full fails every write with ENOSPC: a real disk-full
        // condition, not a simulated one.
        let full = std::fs::OpenOptions::new()
            .append(true)
            .open("/dev/full")
            .unwrap();
        let mut tee = tee_for(full);
        let before = sink_stats();
        // The durable log must never take the caller down.
        tee.write_all(b"nowhere to go\n").unwrap();
        let after = sink_stats();
        assert_eq!(after.file_failures, before.file_failures + 1);
        assert_eq!(after.file_writes, before.file_writes);
        assert_eq!(after.stderr_writes, before.stderr_writes + 1);
    }

    #[cfg(unix)]
    #[test]
    fn unit__reopen_file_if_replaced__swaps_the_handle_to_the_current_file() {
        let _serial = sink_lock();
        let tmp = tempfile::tempdir().unwrap();
        let log = tmp.path().join("daemon.log");
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log)
            .unwrap();
        let shared = Arc::new(Mutex::new(file));
        register_file_sink(&log, Arc::clone(&shared));
        let mut tee = TeeWrite::armed(Arc::clone(&shared));
        tee.write_all(b"before rotation\n").unwrap();

        // Something outside the daemon moves the file away.
        let moved = tmp.path().join("daemon.log.moved");
        std::fs::rename(&log, &moved).unwrap();
        let before = sink_stats();

        assert!(reopen_file_if_replaced());
        assert_eq!(sink_stats().file_reopened, before.file_reopened + 1);
        assert!(log.exists(), "the replaced path is re-created");
        assert!(
            !reopen_file_if_replaced(),
            "the handle now names the current file - nothing left to swap"
        );

        // The very same writer keeps working and lands in the new file.
        tee.write_all(b"after rotation\n").unwrap();
        assert_eq!(std::fs::read_to_string(&log).unwrap(), "after rotation\n");
        assert_eq!(
            std::fs::read_to_string(&moved).unwrap(),
            "before rotation\n"
        );
    }

    #[test]
    fn unit__event_counter__counts_only_the_events_the_filter_enabled() {
        // Events land in a real file through the real tee, so the
        // counter and the writer are observed on the same pipeline.
        let tmp = tempfile::tempdir().unwrap();
        let log = tmp.path().join("events.log");
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log)
            .unwrap();
        let shared = Arc::new(Mutex::new(file));
        let tee = TeeWriter {
            file: Some(Arc::clone(&shared)),
            poison_warned: Arc::new(AtomicBool::new(false)),
        };
        let subscriber = tracing_subscriber::fmt()
            .with_target(false)
            .with_writer(tee)
            .finish()
            .with(EventCounter)
            .with(EnvFilter::new("count_probe=info,warn"));
        let before = sink_stats();
        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(target: "count_probe", "enabled by the filter");
            tracing::warn!(target: "count_probe", "enabled as well");
            tracing::debug!(target: "count_probe", "filtered out");
            tracing::info!(target: "other_crate", "filtered out too");
        });
        let after = sink_stats();
        assert_eq!(after.events, before.events + 2);
        let lines = std::fs::read_to_string(&log).unwrap();
        assert_eq!(lines.lines().count(), 2, "{lines}");
        assert!(lines.contains("enabled by the filter"));
        assert!(lines.contains("enabled as well"));
    }

    #[test]
    fn unit__sink_metrics__renders_every_sink_counter() {
        let text = sink_metrics();
        for needle in [
            "blazar_log_events_total ",
            "blazar_log_sink_writes_total{sink=\"stderr\"} ",
            "blazar_log_sink_writes_total{sink=\"file\"} ",
            "blazar_log_sink_write_failures_total{sink=\"stderr\"} ",
            "blazar_log_sink_write_failures_total{sink=\"file\"} ",
            "blazar_log_sink_reopened_total ",
        ] {
            assert!(text.contains(needle), "missing {needle} in:\n{text}");
        }
    }
}
