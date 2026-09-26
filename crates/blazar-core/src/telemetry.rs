use std::path::Path;
use std::sync::{Arc, Mutex};
use tracing_subscriber::fmt::writer::MakeWriter;
use tracing_subscriber::EnvFilter;

/// Tracing sink that tees every event to stderr and, when armed, to a
/// daemon-owned log file. File-side failures (rotated-away file, full
/// disk) are swallowed on purpose: the durable log must never take the
/// daemon down, and stderr still emits — an `eprintln!` here would be
/// the only stderr writer left standing mid-shutdown, so the write is
/// simply dropped and the next rotation/daemon start recovers.
struct TeeWriter {
    file: Option<Arc<Mutex<std::fs::File>>>,
}

impl MakeWriter<'_> for TeeWriter {
    type Writer = TeeWrite;

    fn make_writer(&self) -> Self::Writer {
        TeeWrite {
            file: self.file.clone(),
        }
    }
}

struct TeeWrite {
    file: Option<Arc<Mutex<std::fs::File>>>,
}

impl std::io::Write for TeeWrite {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let _ = std::io::stderr().write_all(buf);
        if let Some(file) = &self.file {
            if let Ok(mut guard) = file.lock() {
                let _ = guard.write_all(buf);
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let _ = std::io::stderr().flush();
        if let Some(file) = &self.file {
            if let Ok(mut guard) = file.lock() {
                let _ = guard.flush();
            }
        }
        Ok(())
    }
}

/// Initialize tracing. Called once by the daemon/CLI entrypoint.
/// `RUST_LOG` env var wins when set (standard tracing behavior);
/// then the config `log_level` directive (daemon escape hatch without
/// editing the systemd unit); otherwise `verbosity` selects the default.
/// An invalid directive warns on stderr and falls back — logs are never
/// silently dropped. `log_file` arms a durable daemon-owned sink
/// (the `serve` path passes `run/daemon.log`): under systemd the
/// journal is a secondary stream that rotation can strand — the file
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
                eprintln!("blazar: invalid log_level directive ({e}) — using built-in default");
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
            Ok(f) => Some(Arc::new(Mutex::new(f))),
            Err(e) => {
                eprintln!(
                    "blazar: cannot open log file {} ({e}) — logging to stderr only",
                    path.display()
                );
                None
            }
        }
    });
    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(TeeWriter { file });
    // journald already stamps every line (with the machine's local time);
    // a second RFC3339 UTC stamp from tracing only adds noise and a
    // timezone mismatch. Foreground terminals and the daemon-owned log
    // file have no outer stamper, so they keep the tracing timestamp —
    // when the file sink is armed the stamp stays even under journald
    // (a duplicate stamp in the journal beats an untimestamped durable
    // log, which is the artifact post-mortems actually read).
    // JOURNAL_STREAM is systemd's documented marker for journald-attached
    // stdio.
    if std::env::var_os("JOURNAL_STREAM").is_some() && log_file.is_none() {
        builder.without_time().init();
    } else {
        builder.init();
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use super::*;
    use std::io::Write as _;

    #[test]
    fn unit__tee_write__mirrors_bytes_into_the_file() {
        let tmp = tempfile::tempdir().unwrap();
        let log = tmp.path().join("tee.log");
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log)
            .unwrap();
        let mut tee = TeeWrite {
            file: Some(Arc::new(Mutex::new(file))),
        };
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
        let mut tee = TeeWrite { file: None };
        // No file, and stderr is not observable here — the contract under
        // test is that the unarmed path returns success, not panics.
        assert_eq!(tee.write(b"x").unwrap(), 1);
        assert!(tee.flush().is_ok());
    }
}
