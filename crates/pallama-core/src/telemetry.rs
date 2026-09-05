use tracing_subscriber::EnvFilter;

/// Initialize tracing. Called once by the daemon/CLI entrypoint.
/// `RUST_LOG` env var wins when set (standard tracing behavior);
/// otherwise `verbosity` selects the default directive.
pub fn init_tracing(verbosity: u8) {
    let default = match verbosity {
        0 => "pallama=info,warn",
        1 => "pallama=debug,info",
        _ => "pallama=trace,debug",
    };
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();
}
