//! Pallama runtime: async engine management, HF client, process
//! supervision, event bus, bench runner.

pub mod bench;
pub mod engine;
pub mod events;
pub mod models;
pub mod probe;
pub mod hf;

pub use engine::gh::GhClient;
pub use engine::manifest::{probe as probe_manifest, Manifest, Vendor};
pub use engine::{system_vendor_hint, EngineManager, LOCAL_TAG};
pub use events::{EventBus, InstanceState, PallamaEvent};
pub use hf::{parse_pull_target, registry_name, Puller, PullTarget};
pub use bench::{parse_bench_json, BenchRow, Tuner};
pub use probe::probe_hardware;
