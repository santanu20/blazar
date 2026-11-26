//! GPUs through the Vulkan loader.
//!
//! The loader is the one vendor-neutral inventory that every desktop Linux
//! distribution ships: it enumerates whichever ICDs are installed (NVIDIA,
//! AMD, Intel) without any vendor tool. That is also why it runs last.
//! Creating an instance loads every ICD on the machine (measured 52 ms
//! with eight installed), and what it lists is a superset of what any one
//! engine can drive. Its rows describe the machine; they are never used as
//! an engine's device names.

use std::ffi::{c_char, c_void};
use std::sync::OnceLock;

use blazar_core::Hardware;

use super::CensusGpu;
use super::dynlib::{LoadedLib, bind};

#[cfg(windows)]
const LIBRARY_NAMES: &[&str] = &["vulkan-1.dll"];
#[cfg(not(windows))]
const LIBRARY_NAMES: &[&str] = &["libvulkan.so.1"];

/// `VK_STRUCTURE_TYPE_APPLICATION_INFO`.
const STRUCTURE_TYPE_APPLICATION_INFO: i32 = 0;
/// `VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO`.
const STRUCTURE_TYPE_INSTANCE_CREATE_INFO: i32 = 1;
/// `VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2`.
const STRUCTURE_TYPE_MEMORY_PROPERTIES_2: i32 = 1_000_059_006;
/// `VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT`.
const STRUCTURE_TYPE_MEMORY_BUDGET: i32 = 1_000_237_000;
/// `VK_MEMORY_HEAP_DEVICE_LOCAL_BIT`.
const HEAP_DEVICE_LOCAL: u32 = 0x1;
/// `VK_MAX_MEMORY_TYPES`.
const MAX_MEMORY_TYPES: usize = 32;
/// `VK_MAX_MEMORY_HEAPS`.
const MAX_MEMORY_HEAPS: usize = 16;
/// `VK_MAX_PHYSICAL_DEVICE_NAME_SIZE`.
const NAME_BYTES: usize = 256;
/// `VK_API_VERSION_1_1`: `VK_MAKE_API_VERSION(0, 1, 1, 0)`.
const API_VERSION_PREFERRED: u32 = (1 << 22) | (1 << 12);
/// `VK_API_VERSION_1_0`, for loaders older than the preferred version.
const API_VERSION_FALLBACK: u32 = 1 << 12;
/// `VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU`.
const DEVICE_TYPE_INTEGRATED: i32 = 1;
/// `VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`.
const DEVICE_TYPE_DISCRETE: i32 = 2;
/// `VK_SUCCESS`.
const VK_SUCCESS: i32 = 0;
/// `VK_INCOMPLETE`: more devices exist than the caller made room for.
const VK_INCOMPLETE: i32 = 5;
/// More GPUs than this is not a machine this census sizes.
const MAX_DEVICES: usize = 64;
/// `vkGetPhysicalDeviceProperties` writes a struct whose tail grew with
/// every Vulkan version. 4 KiB in 8-byte words is larger than any of them,
/// so no exact size has to be known to receive it.
const PROPERTIES_BUFFER_WORDS: usize = 512;

/// `VkApplicationInfo`. Filled in for the call, never read back.
#[allow(dead_code)]
#[repr(C)]
struct ApplicationInfo {
    s_type: i32,
    next: *mut c_void,
    application_name: *const c_char,
    application_version: u32,
    engine_name: *const c_char,
    engine_version: u32,
    api_version: u32,
}

/// `VkInstanceCreateInfo`. Filled in for the call, never read back.
#[allow(dead_code)]
#[repr(C)]
struct InstanceCreateInfo {
    s_type: i32,
    next: *mut c_void,
    flags: u32,
    application_info: *const ApplicationInfo,
    enabled_layer_count: u32,
    enabled_layer_names: *const *const c_char,
    enabled_extension_count: u32,
    enabled_extension_names: *const *const c_char,
}

/// `VkMemoryHeap`.
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct MemoryHeap {
    size: u64,
    flags: u32,
}

/// `VkMemoryType`. Only the heap list is read; the types pad the struct.
#[allow(dead_code)]
#[derive(Default)]
#[repr(C)]
struct MemoryType {
    property_flags: u32,
    heap_index: u32,
}

/// `VkPhysicalDeviceMemoryProperties`. The memory type list is never read.
#[allow(dead_code)]
#[derive(Default)]
#[repr(C)]
struct MemoryProperties {
    type_count: u32,
    types: [MemoryType; MAX_MEMORY_TYPES],
    heap_count: u32,
    heaps: [MemoryHeap; MAX_MEMORY_HEAPS],
}

/// `VkPhysicalDeviceMemoryBudgetPropertiesEXT`. The budget is the memory
/// this process may still allocate from a heap; the usage it is computed
/// from is not needed.
#[allow(dead_code)]
#[derive(Default)]
#[repr(C)]
struct MemoryBudget {
    s_type: i32,
    next: *mut c_void,
    heap_budget: [u64; MAX_MEMORY_HEAPS],
    heap_usage: [u64; MAX_MEMORY_HEAPS],
}

/// `VkPhysicalDeviceMemoryProperties2`, chaining the budget struct.
#[allow(dead_code)]
#[repr(C)]
struct MemoryProperties2 {
    s_type: i32,
    next: *mut c_void,
    memory_properties: MemoryProperties,
}

/// The leading fields of `VkPhysicalDeviceProperties`. The full struct
/// grew with every Vulkan version, so it is received in an oversized
/// buffer and only this head is read.
#[allow(dead_code)]
#[repr(C)]
struct PropertiesHead {
    api_version: u32,
    driver_version: u32,
    vendor_id: u32,
    device_id: u32,
    device_type: i32,
    device_name: [u8; NAME_BYTES],
}

type CreateInstance =
    unsafe extern "system" fn(*const InstanceCreateInfo, *const c_void, *mut *mut c_void) -> i32;
type DestroyInstance = unsafe extern "system" fn(*mut c_void, *const c_void);
type GetInstanceProcAddr = unsafe extern "system" fn(*mut c_void, *const c_char) -> *mut c_void;
type EnumeratePhysicalDevices =
    unsafe extern "system" fn(*mut c_void, *mut u32, *mut *mut c_void) -> i32;
type GetPhysicalDeviceProperties = unsafe extern "system" fn(*mut c_void, *mut c_void);
type GetPhysicalDeviceMemoryProperties2 =
    unsafe extern "system" fn(*mut c_void, *mut MemoryProperties2);
type GetPhysicalDeviceMemoryProperties =
    unsafe extern "system" fn(*mut c_void, *mut MemoryProperties);

/// Resolves an instance-level entry point. These dispatch per instance, so
/// the live instance is always passed, and a missing entry point is a
/// clean `None`.
macro_rules! instance_proc {
    ($resolver:expr, $instance:expr, $prototype:ty, $name:expr) => {{
        let resolver = $resolver;
        let instance = $instance;
        // SAFETY: `instance` was created by this census and `name` is a
        // valid C string; the call only looks up an address.
        let address = unsafe { resolver(instance, $name.as_ptr()) };
        if address.is_null() {
            None
        } else {
            // SAFETY: the loader returned this address for exactly this
            // instance. The prototype is pinned next to its bind site.
            Some(unsafe { std::mem::transmute::<*mut std::ffi::c_void, $prototype>(address) })
        }
    }};
}

/// Resolved once per process, like the NVML library next to it.
static LOADER: OnceLock<Option<LoadedLib>> = OnceLock::new();

/// Every GPU the Vulkan loader enumerates on this host.
pub(super) fn gpus() -> Vec<CensusGpu> {
    enumerate().unwrap_or_default()
}

/// Vulkan is consulted only when no earlier source saw a discrete card:
/// at that point it could only add integrated rows (dropped anyway when a
/// discrete card exists) or a second vendor's card.
pub(super) fn gpus_if_needed(earlier: &[&[CensusGpu]]) -> Vec<CensusGpu> {
    if earlier.iter().copied().flatten().any(|gpu| !gpu.integrated) {
        return Vec::new();
    }
    gpus()
}

/// One instance per census: it is created and destroyed within this call.
fn enumerate() -> Option<Vec<CensusGpu>> {
    let loader = LOADER
        .get_or_init(|| LoadedLib::open(LIBRARY_NAMES))
        .as_ref()?;
    let resolve: GetInstanceProcAddr =
        bind!(loader, GetInstanceProcAddr, c"vkGetInstanceProcAddr")?;
    let destroy: DestroyInstance = bind!(loader, DestroyInstance, c"vkDestroyInstance")?;
    let instance = create_instance(loader, destroy)?;
    let handle = instance.handle;
    let enumerate_devices: EnumeratePhysicalDevices = instance_proc!(
        resolve,
        handle,
        EnumeratePhysicalDevices,
        c"vkEnumeratePhysicalDevices"
    )?;
    let properties: GetPhysicalDeviceProperties = instance_proc!(
        resolve,
        handle,
        GetPhysicalDeviceProperties,
        c"vkGetPhysicalDeviceProperties"
    )?;
    let budgeted: Option<GetPhysicalDeviceMemoryProperties2> = instance_proc!(
        resolve,
        handle,
        GetPhysicalDeviceMemoryProperties2,
        c"vkGetPhysicalDeviceMemoryProperties2"
    );
    let plain: Option<GetPhysicalDeviceMemoryProperties> = if budgeted.is_some() {
        None
    } else {
        instance_proc!(
            resolve,
            handle,
            GetPhysicalDeviceMemoryProperties,
            c"vkGetPhysicalDeviceMemoryProperties"
        )
    };

    let mut count: u32 = 0;
    // SAFETY: `handle` is a live instance and `count` is writable. With no
    // array passed the call only reports how many devices exist.
    if unsafe { enumerate_devices(handle, &raw mut count, std::ptr::null_mut()) } != VK_SUCCESS {
        return None;
    }
    let mut devices = vec![std::ptr::null_mut(); (count as usize).min(MAX_DEVICES)];
    // SAFETY: `handle` is a live instance; `devices` has room for as many
    // handles as the first call reported; `count` is rewritten to the
    // number actually written.
    let status = unsafe { enumerate_devices(handle, &raw mut count, devices.as_mut_ptr()) };
    if status != VK_SUCCESS && status != VK_INCOMPLETE {
        return None;
    }
    devices.truncate(count as usize);
    Some(
        devices
            .into_iter()
            .filter_map(|device| {
                raw_device(device, properties, budgeted, plain).and_then(device_row)
            })
            .collect(),
    )
}

/// Destroys the Vulkan instance on every exit path.
struct InstanceGuard {
    handle: *mut c_void,
    destroy: DestroyInstance,
}

impl Drop for InstanceGuard {
    fn drop(&mut self) {
        // SAFETY: `handle` was created by the matching `vkCreateInstance`
        // and never leaves this census.
        unsafe { (self.destroy)(self.handle, std::ptr::null()) };
    }
}

fn create_instance(loader: &LoadedLib, destroy: DestroyInstance) -> Option<InstanceGuard> {
    let create: CreateInstance = bind!(loader, CreateInstance, c"vkCreateInstance")?;
    for api_version in [API_VERSION_PREFERRED, API_VERSION_FALLBACK] {
        let application = ApplicationInfo {
            s_type: STRUCTURE_TYPE_APPLICATION_INFO,
            next: std::ptr::null_mut(),
            application_name: std::ptr::null(),
            application_version: 0,
            engine_name: std::ptr::null(),
            engine_version: 0,
            api_version,
        };
        let create_info = InstanceCreateInfo {
            s_type: STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            next: std::ptr::null_mut(),
            flags: 0,
            application_info: &raw const application,
            enabled_layer_count: 0,
            enabled_layer_names: std::ptr::null(),
            enabled_extension_count: 0,
            enabled_extension_names: std::ptr::null(),
        };
        let mut handle: *mut c_void = std::ptr::null_mut();
        // SAFETY: `create_info` is fully initialized and requests no layers
        // and no extensions; `handle` is writable.
        let status = unsafe { create(&raw const create_info, std::ptr::null(), &raw mut handle) };
        if status == VK_SUCCESS && !handle.is_null() {
            return Some(InstanceGuard { handle, destroy });
        }
    }
    None
}

/// What one device reports, before the rules decide whether it is a GPU.
struct RawDevice {
    name: String,
    vendor_id: u32,
    device_type: i32,
    /// Device-local heaps as (size, budget) in bytes. A budget of 0 means
    /// the driver reported none.
    local_heaps: Vec<(u64, u64)>,
}

fn raw_device(
    device: *mut c_void,
    properties: GetPhysicalDeviceProperties,
    budgeted: Option<GetPhysicalDeviceMemoryProperties2>,
    plain: Option<GetPhysicalDeviceMemoryProperties>,
) -> Option<RawDevice> {
    let mut buffer = [0u64; PROPERTIES_BUFFER_WORDS];
    // SAFETY: Vulkan writes its properties struct into `buffer`, which is
    // 8-byte aligned and larger than any version of that struct.
    unsafe { properties(device, buffer.as_mut_ptr().cast()) };
    // SAFETY: Vulkan has just filled `buffer`; its first bytes have the
    // pinned head layout and the reference stays inside the buffer.
    let head = unsafe { buffer.as_ptr().cast::<PropertiesHead>().as_ref()? };

    let mut budget = MemoryBudget {
        s_type: STRUCTURE_TYPE_MEMORY_BUDGET,
        next: std::ptr::null_mut(),
        ..MemoryBudget::default()
    };
    let mut memory = MemoryProperties2 {
        s_type: STRUCTURE_TYPE_MEMORY_PROPERTIES_2,
        next: std::ptr::addr_of_mut!(budget).cast(),
        memory_properties: MemoryProperties::default(),
    };
    if let Some(query) = budgeted {
        // SAFETY: `device` is a live physical device; `memory` is writable
        // and its chain is fully initialized.
        unsafe { query(device, &raw mut memory) };
    } else {
        let query = plain?;
        // SAFETY: `device` is a live physical device; the memory properties
        // struct at the tail of `memory` is writable on its own.
        unsafe { query(device, &raw mut memory.memory_properties) };
    }

    let heaps = &memory.memory_properties;
    let heap_count = (heaps.heap_count as usize).min(MAX_MEMORY_HEAPS);
    let local_heaps: Vec<(u64, u64)> = heaps.heaps[..heap_count]
        .iter()
        .zip(&budget.heap_budget[..heap_count])
        .filter(|(heap, _)| (heap.flags & HEAP_DEVICE_LOCAL) != 0)
        .map(|(heap, budget)| (heap.size, *budget))
        .collect();
    Some(RawDevice {
        name: super::c_string(&head.device_name),
        vendor_id: head.vendor_id,
        device_type: head.device_type,
        local_heaps,
    })
}

/// Turns one device into a row, or into nothing when it is not a GPU.
///
/// Memory numbers follow what the driver budgets for this process, which
/// is what a model load can actually allocate:
///
/// * A discrete card owns its device-local heap; the budget of the largest
///   one is its free memory.
/// * An integrated GPU allocates from system RAM. Its heap reads as most
///   of the machine's RAM (10 GiB on a 16 GiB laptop) while the budget is
///   the share it may really use, so the budget is both total and free.
/// * Without a budget the heap size is all the driver says.
fn device_row(device: RawDevice) -> Option<CensusGpu> {
    let integrated = match device.device_type {
        DEVICE_TYPE_DISCRETE => false,
        DEVICE_TYPE_INTEGRATED => true,
        // OTHER, VIRTUAL and CPU. lavapipe, the software rasterizer,
        // reports a device-local heap that is plain system RAM.
        _ => return None,
    };
    let (heap, budget) = *device.local_heaps.iter().max_by_key(|(heap, _)| *heap)?;
    let heap_mib = Hardware::mib(heap);
    let budget_mib = Hardware::mib(budget);
    let (total_mib, free_mib) = match (integrated, budget) {
        (_, 0) => (heap_mib, heap_mib),
        (true, _) => (budget_mib, budget_mib),
        (false, _) => (heap_mib, budget_mib.min(heap_mib)),
    };
    Some(CensusGpu {
        description: device.name.clone(),
        name: device.name,
        total_mib,
        free_mib,
        vendor_id: device.vendor_id,
        integrated,
    })
}

#[cfg(all(test, target_pointer_width = "64"))]
#[allow(non_snake_case)]
mod tests {
    use std::mem::{offset_of, size_of};

    use super::{
        DEVICE_TYPE_DISCRETE, DEVICE_TYPE_INTEGRATED, PROPERTIES_BUFFER_WORDS, RawDevice,
        device_row,
    };

    const MIB: u64 = 1_048_576;

    fn raw(device_type: i32, heaps: &[(u64, u64)]) -> RawDevice {
        RawDevice {
            name: "Card".into(),
            vendor_id: 0x10de,
            device_type,
            local_heaps: heaps.to_vec(),
        }
    }

    #[test]
    fn unit__vulkan_structs__layouts_match_the_vulkan_header() {
        use super::{
            ApplicationInfo, InstanceCreateInfo, MemoryBudget, MemoryHeap, MemoryProperties,
            MemoryProperties2, PropertiesHead,
        };

        assert_eq!(size_of::<ApplicationInfo>(), 48);
        assert_eq!(offset_of!(ApplicationInfo, next), 8);
        assert_eq!(offset_of!(ApplicationInfo, api_version), 44);

        assert_eq!(size_of::<InstanceCreateInfo>(), 64);
        assert_eq!(offset_of!(InstanceCreateInfo, application_info), 24);
        assert_eq!(offset_of!(InstanceCreateInfo, enabled_extension_names), 56);

        assert_eq!(size_of::<MemoryHeap>(), 16);
        assert_eq!(offset_of!(MemoryHeap, flags), 8);

        assert_eq!(size_of::<MemoryProperties>(), 520);
        assert_eq!(offset_of!(MemoryProperties, heap_count), 260);
        assert_eq!(offset_of!(MemoryProperties, heaps), 264);

        assert_eq!(size_of::<MemoryProperties2>(), 536);
        assert_eq!(offset_of!(MemoryProperties2, memory_properties), 16);

        assert_eq!(size_of::<MemoryBudget>(), 272);
        assert_eq!(offset_of!(MemoryBudget, heap_budget), 16);
        assert_eq!(offset_of!(MemoryBudget, heap_usage), 144);

        assert_eq!(size_of::<PropertiesHead>(), 276);
        assert_eq!(offset_of!(PropertiesHead, vendor_id), 8);
        assert_eq!(offset_of!(PropertiesHead, device_type), 16);
        assert_eq!(offset_of!(PropertiesHead, device_name), 20);

        // The oversized receive buffer must hold the head with room to
        // spare for the struct's version-dependent tail.
        assert!(
            size_of::<[u64; PROPERTIES_BUFFER_WORDS]>() > size_of::<PropertiesHead>(),
            "properties buffer no longer larger than the properties head"
        );
    }

    #[test]
    fn unit__vulkan_device_row__discrete_card_with_budget_reports_budgeted_free_memory() {
        let row = device_row(raw(DEVICE_TYPE_DISCRETE, &[(8_188 * MIB, 6_842 * MIB)]))
            .expect("discrete card with a device-local heap");
        assert_eq!(row.name, "Card");
        assert_eq!(row.description, "Card");
        assert_eq!(row.vendor_id, 0x10de);
        assert!(!row.integrated);
        assert_eq!((row.total_mib, row.free_mib), (8_188, 6_842));
    }

    #[test]
    fn unit__vulkan_device_row__discrete_card_without_budget_reports_all_memory_free() {
        let row = device_row(raw(DEVICE_TYPE_DISCRETE, &[(8_188 * MIB, 0)]))
            .expect("discrete card with a device-local heap");
        assert_eq!((row.total_mib, row.free_mib), (8_188, 8_188));
    }

    #[test]
    fn unit__vulkan_device_row__discrete_card_reads_its_largest_local_heap() {
        let heaps = [(512 * MIB, 100 * MIB), (8_188 * MIB, 6_842 * MIB)];
        let row = device_row(raw(DEVICE_TYPE_DISCRETE, &heaps)).expect("two local heaps");
        assert_eq!((row.total_mib, row.free_mib), (8_188, 6_842));
    }

    #[test]
    fn unit__vulkan_device_row__integrated_card_is_capped_to_its_budget() {
        let row = device_row(raw(DEVICE_TYPE_INTEGRATED, &[(10_256 * MIB, 4_236 * MIB)]))
            .expect("integrated GPU with a budget");
        assert!(row.integrated);
        assert_eq!((row.total_mib, row.free_mib), (4_236, 4_236));
    }

    #[test]
    fn unit__vulkan_device_row__integrated_card_without_budget_reports_its_heap() {
        let row = device_row(raw(DEVICE_TYPE_INTEGRATED, &[(10_256 * MIB, 0)]))
            .expect("integrated GPU with a device-local heap");
        assert_eq!((row.total_mib, row.free_mib), (10_256, 10_256));
    }

    #[test]
    fn unit__vulkan_device_row__software_virtual_and_other_devices_are_not_gpus() {
        // OTHER = 0, VIRTUAL = 3, CPU = 4 (lavapipe: a 13 GiB heap of RAM).
        for device_type in [0, 3, 4] {
            assert!(
                device_row(raw(device_type, &[(13_674 * MIB, 13_674 * MIB)])).is_none(),
                "device type {device_type} must not become a GPU row"
            );
        }
    }

    #[test]
    fn unit__vulkan_device_row__card_without_local_heap_is_dropped() {
        // A device whose heaps are all system RAM has nothing to size against.
        assert!(device_row(raw(DEVICE_TYPE_DISCRETE, &[])).is_none());
    }
}
