//! Apple GPUs through `system_profiler`.
//!
//! macOS has no sysfs counters and ships neither NVML nor a Vulkan loader,
//! so the display inventory is the only list of GPUs the system publishes.
//! It is slow (one to three seconds per reading), which is why it is cached
//! and why every other platform answers without it.
//!
//! Memory means something different on every Mac class, so a report entry
//! is classified before any number is used:
//!
//! * Apple Silicon has no dedicated video memory. The GPU works in a slice
//!   of unified memory that `system_profiler` states on some releases only.
//!   The slice is read as memory a model may use, never as free VRAM: the
//!   row's free side follows the system's available memory.
//! * A discrete card owns its video memory.
//! * An integrated GPU of the Intel era has a fixed slice of system memory.

use serde_json::Value;

const VENDOR_AMD: u32 = 0x1002;
const VENDOR_INTEL: u32 = 0x8086;
const VENDOR_NVIDIA: u32 = 0x10de;
const VENDOR_APPLE: u32 = 0x106b;

/// The memory figure of one GPU entry, in MiB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DisplayMemory {
    /// Apple Silicon unified memory. The figure is what the release
    /// reports for the GPU's slice, when it reports one at all.
    Unified(Option<u64>),
    /// Video memory the card owns.
    Dedicated(u64),
    /// A fixed slice of system memory (Intel-era integrated GPUs).
    SystemSlice(u64),
}

/// One GPU as `system_profiler -json` lists it under `SPDisplaysDataType`.
#[derive(Debug, Clone, PartialEq)]
struct Display {
    name: String,
    memory: DisplayMemory,
    /// PCI vendor id, `0` when the report does not name the vendor.
    vendor_id: u32,
}

fn parse_displays(json: &str) -> Vec<Display> {
    const DATA_TYPE: &str = "SPDisplaysDataType";

    let Ok(root) = serde_json::from_str::<Value>(json) else {
        return Vec::new();
    };
    let items: &[Value] = root
        .get(DATA_TYPE)
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice);
    items.iter().filter_map(display_row).collect()
}

fn display_row(item: &Value) -> Option<Display> {
    let name = ["sppci_model", "_name"]
        .iter()
        .find_map(|key| item.get(key).and_then(Value::as_str))
        .map(str::trim)?
        .to_string();
    if name.is_empty() {
        return None;
    }
    let vendor_id = vendor_id(item);
    let dedicated = reported_mib(item, "spdisplays_vram");
    let shared = reported_mib(item, "spdisplays_vram_shared");
    let on_builtin_bus =
        item.get("sppci_bus").and_then(Value::as_str) == Some("spdisplays_builtin");
    // Every Apple Silicon system-on-chip is named `Apple M...`. The model
    // name alone names the class on a release that omits the vendor key;
    // nothing else may claim unified memory.
    let memory = if vendor_id == VENDOR_APPLE || name.starts_with("Apple M") {
        DisplayMemory::Unified(shared.or(dedicated))
    } else if let Some(total_mib) = dedicated
        && !on_builtin_bus
        && shared.is_none()
        && vendor_id != VENDOR_INTEL
    {
        // A card is proven discrete by memory it owns alone: no shared
        // figure, not on the builtin bus, and not from Intel, whose Mac
        // GPUs were all integrated. A missing bus key is not builtin;
        // older releases print none.
        DisplayMemory::Dedicated(total_mib)
    } else {
        DisplayMemory::SystemSlice(shared.or(dedicated)?)
    };
    Some(Display {
        name,
        memory,
        vendor_id,
    })
}

/// The memory an entry states under `key`, in MiB. A zero figure means
/// "not reported" and reads as absent.
fn reported_mib(item: &Value, key: &str) -> Option<u64> {
    item.get(key)
        .and_then(size_mib)
        .filter(|total_mib| *total_mib > 0)
}

/// `"16 GB"`, `"1536MB"`, or a plain byte count.
fn size_mib(value: &Value) -> Option<u64> {
    if let Some(bytes) = value.as_u64() {
        return Some(blazar_core::Hardware::mib(bytes));
    }
    let text = value.as_str()?.trim();
    let digits = text.bytes().take_while(u8::is_ascii_digit).count();
    let amount = text.get(..digits)?.parse::<u64>().ok()?;
    let scale: u64 = match text[digits..]
        .replace(' ', "")
        .to_ascii_uppercase()
        .as_str()
    {
        "KB" => 1 << 10,
        "MB" => 1 << 20,
        "GB" => 1 << 30,
        "TB" => 1 << 40,
        _ => return None,
    };
    amount.checked_mul(scale).map(blazar_core::Hardware::mib)
}

/// The PCI vendor id of an entry: a `0x` id stated in a vendor field
/// (`"0x1002"`, `"Intel (0x8086)"`), else the vendor name.
fn vendor_id(item: &Value) -> u32 {
    let stated = ["spdisplays_vendor", "spdisplays_vendor-id"]
        .iter()
        .find_map(|key| {
            item.get(key)
                .and_then(Value::as_str)
                .and_then(stated_vendor_id)
        });
    if let Some(vendor_id) = stated {
        return vendor_id;
    }
    let vendor = item
        .get("spdisplays_vendor")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    match vendor.as_str() {
        v if v.contains("apple") => VENDOR_APPLE,
        v if v.contains("amd") => VENDOR_AMD,
        v if v.contains("intel") => VENDOR_INTEL,
        v if v.contains("nvidia") => VENDOR_NVIDIA,
        _ => 0,
    }
}

fn stated_vendor_id(text: &str) -> Option<u32> {
    let (_, after_prefix) = text.split_once("0x")?;
    let digits: String = after_prefix
        .chars()
        .take_while(char::is_ascii_hexdigit)
        .collect();
    if digits.is_empty() || digits.len() > 4 {
        return None;
    }
    u32::from_str_radix(&digits, 16).ok()
}

/// The memory an Apple Silicon GPU may use, in MiB. `reported_mib` is the
/// figure the release publishes for the GPU's slice; when it publishes
/// none, the GPU gets a share of system RAM. The smaller figure wins, so
/// the share understates a host whose GPU limit was raised and never
/// overstates the pool.
fn unified_total_mib(reported_mib: Option<u64>, ram_mib: Option<u64>) -> Option<u64> {
    // About three quarters of unified memory is what macOS lets its GPU
    // address by default; this is the share other hardware reporters fall
    // back on for releases that print no figure.
    const GPU_RAM_SHARE: (u64, u64) = (3, 4);

    let share_mib = ram_mib.map(|ram_mib| ram_mib / GPU_RAM_SHARE.1 * GPU_RAM_SHARE.0);
    match (reported_mib, share_mib) {
        (Some(reported), Some(share)) => Some(reported.min(share)),
        (Some(figure), None) | (None, Some(figure)) => Some(figure),
        (None, None) => None,
    }
}

#[cfg(target_os = "macos")]
use super::CensusGpu;

#[cfg(target_os = "macos")]
pub(super) fn gpus() -> Vec<CensusGpu> {
    use DisplayMemory::{Dedicated, SystemSlice, Unified};

    let memory = memory_mib();
    inventory()
        .into_iter()
        .filter_map(|display| {
            let (total_mib, integrated) = match display.memory {
                Unified(reported_mib) => (
                    unified_total_mib(reported_mib, memory.map(|(ram_mib, _)| ram_mib))?,
                    true,
                ),
                Dedicated(total_mib) => (total_mib, false),
                SystemSlice(total_mib) => (total_mib, true),
            };
            // Shared memory is never free VRAM: what the system has
            // available right now is what a model may take.
            let free_mib = if integrated {
                memory.map_or(total_mib, |(_, available_mib)| available_mib.min(total_mib))
            } else {
                total_mib
            };
            Some(CensusGpu {
                description: display.name.clone(),
                name: display.name,
                total_mib,
                free_mib,
                vendor_id: display.vendor_id,
                integrated,
            })
        })
        .collect()
}

/// `system_profiler` needs one to three seconds and gateway fabric
/// requests repeat the hardware probe. The display inventory does not
/// change while the daemon runs, so one reading is trusted for a minute.
#[cfg(target_os = "macos")]
fn inventory() -> Vec<Display> {
    use std::process::Command;
    use std::sync::{Mutex, PoisonError};
    use std::time::{Duration, Instant};

    const INVENTORY_TTL: Duration = Duration::from_secs(60);
    const QUERY_TIMEOUT_SECS: u64 = 30;
    static INVENTORY: Mutex<Option<(Instant, Vec<Display>)>> = Mutex::new(None);

    if let Some((read_at, displays)) = INVENTORY
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .as_ref()
        && read_at.elapsed() < INVENTORY_TTL
    {
        return displays.clone();
    }
    let output = crate::probe::probe_output(
        Command::new("system_profiler").args(["SPDisplaysDataType", "-json"]),
        QUERY_TIMEOUT_SECS,
    );
    let displays = output.map_or_else(Vec::new, |out| {
        parse_displays(&String::from_utf8_lossy(&out.stdout))
    });
    *INVENTORY.lock().unwrap_or_else(PoisonError::into_inner) =
        Some((Instant::now(), displays.clone()));
    displays
}

/// System memory as (total, available) in MiB, `None` when sysinfo cannot
/// say.
#[cfg(target_os = "macos")]
fn memory_mib() -> Option<(u64, u64)> {
    use blazar_core::Hardware;

    let mut system = sysinfo::System::new().ok()?;
    system.refresh_memory();
    Some((
        Hardware::mib(system.total_memory()),
        Hardware::mib(system.available_memory()),
    ))
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::{
        Display, DisplayMemory, parse_displays, reported_mib, size_mib, unified_total_mib,
        vendor_id,
    };
    use DisplayMemory::{Dedicated, SystemSlice, Unified};

    // Real reports first. Each of these entries was captured on the Mac it
    // names and published in a public bug report or test fixture of a
    // hardware detector that reads the same `system_profiler` report.
    // Constructed entries say so.

    #[test]
    fn unit__parse_displays__apple_silicon_with_a_shared_figure() {
        // Apple M4 Pro: the release prints the shared pool and no model key.
        let json = r#"{"SPDisplaysDataType":[{"_name":"Apple M4 Pro","spdisplays_cores":"20","spdisplays_metal":"spdisplays_metal_featuresetfamily_mac_2","spdisplays_vendor":"sppci_vendor_Apple","spdisplays_vram_shared":"24 GB","spdisplays_ndrvs":[{"_name":"Color LCD"}]}]}"#;
        assert_eq!(
            parse_displays(json),
            vec![Display {
                name: "Apple M4 Pro".into(),
                memory: Unified(Some(24_576)),
                vendor_id: 0x106b,
            }]
        );
    }

    #[test]
    fn unit__parse_displays__apple_silicon_without_a_vram_key_is_still_a_gpu() {
        // Apple M5 Pro on a release that prints no memory key at all.
        let json = r#"{"SPDisplaysDataType":[{"_name":"Apple M5 Pro","sppci_model":"Apple M5 Pro","spdisplays_cores":"20","spdisplays_vendor":"sppci_vendor_Apple"}]}"#;
        assert_eq!(
            parse_displays(json),
            vec![Display {
                name: "Apple M5 Pro".into(),
                memory: Unified(None),
                vendor_id: 0x106b,
            }],
            "an Apple Silicon Mac has one GPU even when no memory key is reported"
        );
    }

    #[test]
    fn unit__parse_displays__vendor_arriving_as_a_hex_id_is_read() {
        // Intel-era Macs: a Radeon Pro 580X on the PCIe bus and a GeForce
        // GT 750M from an older release that prints no bus key. The device
        // id `0x67df` must not be mistaken for the vendor.
        let json = r#"{"SPDisplaysDataType":[{"spdisplays_device-id":"0x67df","spdisplays_metal":"spdisplays_metal_supported","spdisplays_vendor":"0x1002","spdisplays_vram":"8 GB","sppci_bus":"sppci_pcie_device","sppci_model":"Radeon Pro 580X"},{"spdisplays_device-id":"0x0fe9","spdisplays_vendor":"0x10de","spdisplays_vram":"2 GB","sppci_model":"NVIDIA GeForce GT 750M"}]}"#;
        assert_eq!(
            parse_displays(json),
            vec![
                Display {
                    name: "Radeon Pro 580X".into(),
                    memory: Dedicated(8_192),
                    vendor_id: 0x1002,
                },
                Display {
                    name: "NVIDIA GeForce GT 750M".into(),
                    memory: Dedicated(2_048),
                    vendor_id: 0x10de,
                },
            ]
        );
    }

    #[test]
    fn unit__parse_displays__builtin_bus_gpu_with_a_vram_key_is_integrated() {
        // Intel HD Graphics 4000: the same 1024 MB appears as dedicated
        // and as shared. munkireport issue 134 published it as an XML
        // report; the JSON keys here are the XML names without their
        // leading underscore.
        let json = r#"{"SPDisplaysDataType":[{"_name":"kHW_IntelHD4000Item","spdisplays_vram":"1024 MB","spdisplays_vendor":"Intel (0x8086)","spdisplays_vram_shared":"1024 MB","sppci_bus":"spdisplays_builtin"}]}"#;
        assert_eq!(
            parse_displays(json),
            vec![Display {
                name: "kHW_IntelHD4000Item".into(),
                memory: SystemSlice(1_024),
                vendor_id: 0x8086,
            }]
        );
    }

    #[test]
    fn unit__parse_displays__intel_mac_reports_only_the_shared_slice() {
        // A 16 GiB Intel Mac with an Iris Pro. Model, builtin bus, the
        // 1536 MB shared slice and the absent dedicated key come from the
        // capture published in opensourcesai-cmdcenter PR 81; the vendor
        // is written as the report's plain name.
        let json = r#"{"SPDisplaysDataType":[{"sppci_model":"Intel Iris Pro","spdisplays_vendor":"Intel","sppci_bus":"spdisplays_builtin","spdisplays_vram_shared":"1536 MB"}]}"#;
        assert_eq!(
            parse_displays(json),
            vec![Display {
                name: "Intel Iris Pro".into(),
                memory: SystemSlice(1_536),
                vendor_id: 0x8086,
            }]
        );
    }

    #[test]
    fn unit__parse_displays__apple_silicon_named_by_model_only() {
        // Constructed: no vendor key, so the model name names the class.
        let json = r#"{"SPDisplaysDataType":[{"sppci_model":"Apple M1","spdisplays_vram_shared":"8 GB"}]}"#;
        assert_eq!(
            parse_displays(json),
            vec![Display {
                name: "Apple M1".into(),
                memory: Unified(Some(8_192)),
                vendor_id: 0,
            }]
        );
    }

    #[test]
    fn unit__parse_displays__intel_gpu_with_only_a_vram_key_stays_integrated() {
        // Constructed: an Intel entry that states one figure and no bus.
        let json = r#"{"SPDisplaysDataType":[{"sppci_model":"Intel Iris Plus Graphics","spdisplays_vendor":"sppci_vendor_Intel","spdisplays_vram":"1536 MB"}]}"#;
        assert_eq!(
            parse_displays(json),
            vec![Display {
                name: "Intel Iris Plus Graphics".into(),
                memory: SystemSlice(1_536),
                vendor_id: 0x8086,
            }]
        );
    }

    #[test]
    fn unit__parse_displays__entries_that_state_no_memory_are_dropped() {
        // Constructed: nothing here can be sized and none is Apple Silicon.
        let json = r#"{"SPDisplaysDataType":[{"_name":"No memory reported"},{"sppci_model":"Odd unit","spdisplays_vram":"16 pebbles"},{"spdisplays_ndrvs":[{"_name":"Color LCD"}]}],"SPOtherDataType":[{"_name":"Not a display"}]}"#;
        assert_eq!(
            parse_displays(json),
            [] as [crate::gpu_census::mac_displays::Display; 0]
        );
    }

    #[test]
    fn unit__parse_displays__not_json_is_empty() {
        assert_eq!(
            parse_displays("system_profiler: not json"),
            [] as [crate::gpu_census::mac_displays::Display; 0]
        );
        assert_eq!(
            parse_displays(""),
            [] as [crate::gpu_census::mac_displays::Display; 0]
        );
        assert_eq!(
            parse_displays(r#"{"SPDisplaysDataType":"none"}"#),
            [] as [crate::gpu_census::mac_displays::Display; 0]
        );
    }

    #[test]
    fn unit__size_mib__reads_units_and_byte_counts() {
        assert_eq!(size_mib(&serde_json::json!("16 GB")), Some(16_384));
        assert_eq!(size_mib(&serde_json::json!("1536 MB")), Some(1_536));
        assert_eq!(size_mib(&serde_json::json!("2 TB")), Some(2 * 1_048_576));
        assert_eq!(size_mib(&serde_json::json!(1_048_576)), Some(1));
        assert_eq!(size_mib(&serde_json::json!("16")), None);
        assert_eq!(size_mib(&serde_json::json!(true)), None);
    }

    #[test]
    fn unit__size_mib__reads_a_unit_written_without_a_space() {
        // The spacing of this figure is not stable across releases, so the
        // leading number is read whatever follows it.
        assert_eq!(size_mib(&serde_json::json!("1536MB")), Some(1_536));
    }

    #[test]
    fn unit__reported_mib__zero_reads_as_not_reported() {
        let zero = serde_json::json!({"spdisplays_vram": "0 MB"});
        assert_eq!(reported_mib(&zero, "spdisplays_vram"), None);
        let sized = serde_json::json!({"spdisplays_vram": "8 GB"});
        assert_eq!(reported_mib(&sized, "spdisplays_vram"), Some(8_192));
        assert_eq!(reported_mib(&sized, "spdisplays_vram_shared"), None);
    }

    #[test]
    fn unit__unified_total_mib__reported_figure_and_ram_share_combine() {
        assert_eq!(unified_total_mib(None, Some(16_384)), Some(12_288));
        assert_eq!(unified_total_mib(Some(24_576), Some(24_576)), Some(18_432));
        assert_eq!(unified_total_mib(Some(8_192), None), Some(8_192));
        assert_eq!(unified_total_mib(None, None), None);
        assert_eq!(unified_total_mib(Some(12_288), Some(16_384)), Some(12_288));
    }

    #[test]
    fn unit__vendor_id__hex_ids_and_names() {
        let vendor =
            |json: &str| vendor_id(&serde_json::from_str::<serde_json::Value>(json).unwrap());
        assert_eq!(
            vendor(r#"{"spdisplays_vendor":"sppci_vendor_Apple"}"#),
            0x106b
        );
        assert_eq!(vendor(r#"{"spdisplays_vendor":"0x1002"}"#), 0x1002);
        assert_eq!(vendor(r#"{"spdisplays_vendor":"0x10de"}"#), 0x10de);
        assert_eq!(vendor(r#"{"spdisplays_vendor":"Intel"}"#), 0x8086);
        assert_eq!(vendor(r#"{"spdisplays_vendor":"Intel (0x8086)"}"#), 0x8086);
        assert_eq!(vendor(r#"{"spdisplays_vendor":"0x12345"}"#), 0);
        assert_eq!(vendor(r#"{"_name":"No vendor"}"#), 0);
    }
}
