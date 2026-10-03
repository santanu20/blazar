//! Test-only process-env mutation, shared by this crate's lib tests.
//!
//! Edition 2024 made `std::env::set_var` / `std::env::remove_var` unsafe:
//! std requires that no other thread read or write the environment while a
//! mutation is in flight. These tests run one-test-per-process under
//! nextest, and the env-reading code under test executes on the calling
//! thread between mutation and restore — the one condition std demands.
//! The single audited `unsafe` block for the whole crate lives here.

/// `std::env::set_var` for tests (see module docs).
pub(crate) fn set_env<K: AsRef<std::ffi::OsStr>, V: AsRef<std::ffi::OsStr>>(key: K, value: V) {
    #[expect(unsafe_code)]
    unsafe {
        std::env::set_var(key, value);
    }
}

/// `std::env::remove_var` for tests (see module docs).
pub(crate) fn remove_env<K: AsRef<std::ffi::OsStr>>(key: K) {
    #[expect(unsafe_code)]
    unsafe {
        std::env::remove_var(key);
    }
}
