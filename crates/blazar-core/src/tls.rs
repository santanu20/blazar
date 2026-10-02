//! Process-wide TLS provider bootstrap.
//!
//! The workspace builds its whole rustls stack provider-less on purpose:
//! exactly one crypto provider (ring) is compiled in via the workspace
//! `rustls` dependency, keeping aws-lc-sys — and its cmake/nasm build
//! prerequisites — out of every release target. The flip side is that
//! anything constructing a TLS client (reqwest in particular panics by
//! design in this configuration) needs a process-default provider
//! installed first. Call [`ensure_tls_provider`] before the first
//! `reqwest::Client` construction; it is idempotent and race-safe, so
//! every entry point can call it unconditionally.

/// Install the process-default rustls crypto provider (ring).
///
/// `install_default` fails only when a provider is already installed —
/// by an earlier caller here or by rustls's own lazy resolution — which
/// is exactly the success condition we want, so the result is ignored.
pub fn ensure_tls_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit__tls__provider_installs_and_repeats_cleanly() {
        // First call installs ring; the second proves idempotence (an
        // already-installed provider must not surface an error callers
        // would have to handle).
        ensure_tls_provider();
        ensure_tls_provider();
        assert!(rustls::crypto::CryptoProvider::get_default().is_some());
    }
}
