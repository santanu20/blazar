use tracing_subscriber::EnvFilter;

/// Initialize tracing. Called once by the daemon/CLI entrypoint.
/// `RUST_LOG` env var wins when set (standard tracing behavior);
/// then the config `log_level` directive (daemon escape hatch without
/// editing the systemd unit); otherwise `verbosity` selects the default.
/// An invalid directive warns on stderr and falls back — logs are never
/// silently dropped.
pub fn init_tracing(verbosity: u8, log_level: Option<&str>) {
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
    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr);
    // journald already stamps every line (with the machine's local time);
    // a second RFC3339 UTC stamp from tracing only adds noise and a
    // timezone mismatch. Foreground terminals have no outer stamper, so
    // they keep the tracing timestamp. JOURNAL_STREAM is systemd's
    // documented marker for journald-attached stdio.
    if std::env::var_os("JOURNAL_STREAM").is_some() {
        builder.without_time().init();
    } else {
        builder.init();
    }
}
