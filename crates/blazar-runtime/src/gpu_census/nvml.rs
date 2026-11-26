//! NVIDIA GPUs through NVML.
//!
//! `nvidia-smi` is a separate package on several distributions and is not
//! part of a minimal driver install, while the driver's management library
//! always is. Reading it directly gives the census the same rows
//! `nvidia-smi` gives, so nothing downstream can tell the two apart.

use std::ffi::{c_char, c_void};
use std::sync::OnceLock;

use blazar_core::Hardware;

use super::CensusGpu;
use super::dynlib::{LoadedLib, bind};

#[cfg(windows)]
const LIBRARY_NAMES: &[&str] = &["nvml.dll"];
#[cfg(not(windows))]
const LIBRARY_NAMES: &[&str] = &["libnvidia-ml.so.1", "libnvidia-ml.so"];

const VENDOR_NVIDIA: u32 = 0x10de;

/// `NVML_SUCCESS`.
const NVML_SUCCESS: i32 = 0;
/// `NVML_ERROR_ALREADY_INITIALIZED`: another user in this process already
/// initialized NVML. The library is usable and still expects one shutdown
/// for this init.
const NVML_ALREADY_INITIALIZED: i32 = 5;
/// `NVML_DEVICE_NAME_V2_BUFFER_SIZE`.
const NAME_BUFFER_BYTES: u32 = 96;

/// `nvmlReturn_t`. Every value other than success is a failure whose exact
/// reason the census has no use for.
type NvmlStatus = i32;

/// `nvmlMemory_t`, in bytes.
#[repr(C)]
struct MemoryInfo {
    total: u64,
    free: u64,
    used: u64,
}

type NvmlInit = unsafe extern "C" fn() -> NvmlStatus;
type NvmlShutdown = unsafe extern "C" fn() -> NvmlStatus;
type DeviceCount = unsafe extern "C" fn(*mut u32) -> NvmlStatus;
type DeviceHandleByIndex = unsafe extern "C" fn(u32, *mut *mut c_void) -> NvmlStatus;
type DeviceName = unsafe extern "C" fn(*mut c_void, *mut c_char, u32) -> NvmlStatus;
type DeviceMemoryInfo = unsafe extern "C" fn(*mut c_void, *mut MemoryInfo) -> NvmlStatus;

struct NvmlApi {
    init: NvmlInit,
    shutdown: NvmlShutdown,
    count: DeviceCount,
    handle_by_index: DeviceHandleByIndex,
    name: DeviceName,
    memory_info: DeviceMemoryInfo,
}

/// Resolved once per process. A driver installed after the daemon started
/// needs a daemon restart, like any other hardware change.
static API: OnceLock<Option<NvmlApi>> = OnceLock::new();

pub(super) fn gpus() -> Vec<CensusGpu> {
    let Some(api) = API.get_or_init(bind_api).as_ref() else {
        return Vec::new();
    };
    // SAFETY: `nvmlInit_v2` takes no arguments.
    let status = unsafe { (api.init)() };
    if status != NVML_SUCCESS && status != NVML_ALREADY_INITIALIZED {
        return Vec::new();
    }
    let _shutdown = ShutdownOnDrop(api.shutdown);
    list_devices(api)
}

/// Calls `nvmlShutdown` when dropped, so an early return or a panic in the
/// middle of a census never leaves NVML initialized.
struct ShutdownOnDrop(NvmlShutdown);

impl Drop for ShutdownOnDrop {
    fn drop(&mut self) {
        // SAFETY: the caller initialized NVML before creating this guard.
        unsafe { (self.0)() };
    }
}

fn bind_api() -> Option<NvmlApi> {
    let library = LoadedLib::open(LIBRARY_NAMES)?;
    Some(NvmlApi {
        init: bind!(library, NvmlInit, c"nvmlInit_v2")?,
        shutdown: bind!(library, NvmlShutdown, c"nvmlShutdown")?,
        count: bind!(library, DeviceCount, c"nvmlDeviceGetCount_v2")?,
        handle_by_index: bind!(
            library,
            DeviceHandleByIndex,
            c"nvmlDeviceGetHandleByIndex_v2"
        )?,
        name: bind!(library, DeviceName, c"nvmlDeviceGetName")?,
        memory_info: bind!(library, DeviceMemoryInfo, c"nvmlDeviceGetMemoryInfo")?,
    })
}

/// All cards or none: a partial list would read as a smaller machine and
/// silently change placement decisions.
fn list_devices(api: &NvmlApi) -> Vec<CensusGpu> {
    let mut count: u32 = 0;
    // SAFETY: `count` is a writable `unsigned int`.
    if unsafe { (api.count)(&raw mut count) } != NVML_SUCCESS {
        return Vec::new();
    }
    (0..count)
        .map(|index| device_row(api, index))
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default()
}

fn device_row(api: &NvmlApi, index: u32) -> Option<CensusGpu> {
    let mut device: *mut c_void = std::ptr::null_mut();
    // SAFETY: `index` is below the count NVML reported and `device` is
    // writable.
    if unsafe { (api.handle_by_index)(index, &raw mut device) } != NVML_SUCCESS {
        return None;
    }
    let mut name = [0u8; NAME_BUFFER_BYTES as usize];
    // SAFETY: `device` is a handle NVML just returned; `name` has
    // `NAME_BUFFER_BYTES` bytes and NVML writes at most the length it is
    // given.
    if unsafe { (api.name)(device, name.as_mut_ptr().cast(), NAME_BUFFER_BYTES) } != NVML_SUCCESS {
        return None;
    }
    let mut memory = MemoryInfo {
        total: 0,
        free: 0,
        used: 0,
    };
    // SAFETY: `device` is a handle NVML just returned; `memory` is writable
    // and has the layout of `nvmlMemory_t`.
    if unsafe { (api.memory_info)(device, &raw mut memory) } != NVML_SUCCESS {
        return None;
    }
    let name = super::c_string(&name);
    Some(CensusGpu {
        description: format!("NVIDIA {name}"),
        name,
        total_mib: Hardware::mib(memory.total),
        free_mib: Hardware::mib(memory.free),
        vendor_id: VENDOR_NVIDIA,
        integrated: false,
    })
}

#[cfg(all(test, target_pointer_width = "64"))]
#[allow(non_snake_case)]
mod tests {
    use std::mem::{offset_of, size_of};

    use super::MemoryInfo;

    #[test]
    fn unit__nvml_memory_info__layout_is_three_u64_fields() {
        assert_eq!(size_of::<MemoryInfo>(), 24);
        assert_eq!(offset_of!(MemoryInfo, total), 0);
        assert_eq!(offset_of!(MemoryInfo, free), 8);
        assert_eq!(offset_of!(MemoryInfo, used), 16);
    }
}
