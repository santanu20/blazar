//! Loading vendor libraries at run time.
//!
//! The GPU census calls NVML and the Vulkan loader. Neither is guaranteed
//! to exist on a machine: no NVIDIA driver means no NVML, a minimal install
//! has no Vulkan loader. Linking either at build time would stop the daemon
//! from starting there, so both are loaded on demand and a missing library
//! simply contributes no GPU rows.
//!
//! A loaded library is never unloaded. Driver libraries keep process-wide
//! state and the daemon is long lived, so releasing the mapping could only
//! ever hand a later census a library that is being torn down.

use std::ffi::{CStr, c_void};

/// Binds a symbol to a function prototype.
///
/// The prototype has to be the native signature exactly; nothing here can
/// verify it. Every bind site therefore names a prototype that is declared
/// next to the layouts and constants its tests pin.
macro_rules! bind {
    ($library:expr, $prototype:ty, $name:expr) => {{
        let library = &$library;
        let address = library.symbol($name)?;
        // SAFETY: the platform symbol lookup returned this address, so it
        // is a function entry point. The prototype is the caller's, see
        // the macro documentation.
        Some(unsafe { std::mem::transmute::<*mut std::ffi::c_void, $prototype>(address) })
    }};
}
pub(super) use bind;

/// A library opened with the platform loader. The handle is an integer
/// because a raw pointer cannot be stored in a static (it is not `Sync`).
pub(super) struct LoadedLib(usize);

impl LoadedLib {
    /// The first of `names` the platform loader finds.
    pub(super) fn open(names: &[&str]) -> Option<Self> {
        names
            .iter()
            .find_map(|name| platform_open(name).map(LoadedLib))
    }

    /// Address of the exported symbol, `None` when the library lacks it.
    pub(super) fn symbol(&self, name: &CStr) -> Option<*mut c_void> {
        platform_symbol(self.0, name)
    }
}

#[cfg(not(windows))]
fn platform_open(name: &str) -> Option<usize> {
    let name = std::ffi::CString::new(name).ok()?;
    // SAFETY: `name` is a valid NUL-terminated string. The handle outlives
    // this call because the library is never unloaded.
    let handle = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_LOCAL | libc::RTLD_NOW) };
    if handle.is_null() {
        None
    } else {
        Some(handle as usize)
    }
}

#[cfg(windows)]
fn platform_open(name: &str) -> Option<usize> {
    let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
    // SAFETY: `wide` is NUL-terminated UTF-16 and alive for the call. The
    // handle outlives it because the library is never unloaded.
    let handle = unsafe { windows_sys::Win32::System::LibraryLoader::LoadLibraryW(wide.as_ptr()) };
    if handle.is_null() {
        None
    } else {
        Some(handle as usize)
    }
}

#[cfg(not(windows))]
fn platform_symbol(handle: usize, name: &CStr) -> Option<*mut c_void> {
    // SAFETY: `handle` comes from a successful `dlopen` on a library that
    // is never unloaded; `name` is a valid C string.
    let address = unsafe { libc::dlsym(handle as *mut c_void, name.as_ptr()) };
    if address.is_null() {
        None
    } else {
        Some(address)
    }
}

#[cfg(windows)]
fn platform_symbol(handle: usize, name: &CStr) -> Option<*mut c_void> {
    // SAFETY: `handle` comes from a successful `LoadLibraryW` on a library
    // that is never unloaded; `name` is a valid C string.
    let function = unsafe {
        windows_sys::Win32::System::LibraryLoader::GetProcAddress(
            handle as *mut c_void,
            name.as_ptr().cast(),
        )
    }?;
    Some(function as *mut c_void)
}
