//! Blazar runtime: async engine management, HF client, process
//! supervision, event bus, bench runner.

/// File name of a llama.cpp tool binary on the current platform
/// (`llama-quantize` vs `llama-quantize.exe`) — F91: every discovery
/// site must go through this, not hand-join the unix name.
#[must_use]
pub fn tool_file_name(base: &str) -> String {
    if cfg!(windows) {
        format!("{base}.exe")
    } else {
        base.to_string()
    }
}

pub mod bench;
pub mod daemon;
pub mod diffusion;
pub mod engine;
pub mod engine_impl;
pub mod events;
pub mod hf;
pub mod hf_parallel;
pub mod models;
pub mod piper;
pub mod probe;
pub mod quantize;
pub mod registry;
pub mod sessionreg;
pub mod storage;
pub mod supervisor;
pub mod throttle;
pub mod uds;
pub mod upgrade;
pub mod verify;
pub mod whisper;

#[cfg(test)]
mod test_env;

pub use bench::{BenchRow, Tuner, parse_bench_json};
pub use daemon::{
    DaemonLock, LockHeld, process_alive_by_pid, validate_parent_death_guard,
    wait_for_shutdown_signal,
};
pub use engine::gh::GhClient;
pub use engine::manifest::{Manifest, Vendor, predicted_rescue_lane, probe as probe_manifest};
pub use engine::{EngineManager, LOCAL_TAG, system_vendor_hint};
pub use engine_impl::{
    ChildHandle, Engine, LlamaCppEngine, MistralRsEngine, MlxEngine, SdCppEngine, SglangEngine,
};
pub use events::{BlazarEvent, EventBus, InstanceState};
pub use hf::{PullOutcome, PullTarget, Puller, parse_pull_target, registry_name};
pub use models::{instance_running, remove_model};
pub use probe::parent_death_tie;
pub use probe::probe_hardware;
pub use supervisor::{
    EngineRef, PrefixKey, PsRow, ROUTER_KEY, SupervisionError, Supervisor, resolve_draft_path,
    resolve_spec_mode,
};
pub use verify::{VerifyReport, verify_model};
