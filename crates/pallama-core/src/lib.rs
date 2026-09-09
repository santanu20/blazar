//! Pallama core: pure, synchronous domain logic. No tokio, no axum, no
//! network — exhaustively unit-testable. Async lifetimes (process
//! supervision, downloads, HTTP) live in `pallama-runtime`.

pub mod catalog;
pub mod config;
pub mod coreside;
pub mod dirs;
pub mod engine_kind;
pub mod error;
pub mod gguf;
pub mod hardware;
pub mod profile;
pub mod session_identity;
pub mod store;
pub mod telemetry;

pub use catalog::{catalog, resolve, spec_pair_for};
pub use config::{ApiKey, Config, ModelOverride, Remote, SemanticCacheConfig};
pub use dirs::PallamaDirs;
pub use error::{CoreError, CoreResult};
pub use gguf::{read_metadata_file, GgufMeta, GgufValue};
pub use hardware::{GpuInfo, Hardware};
pub use profile::{compile as compile_profile, Endpoint, Profile, ProfileInput, TuningOverrides};
pub use store::{EngineRow, KeyUsageRow, LoraRow, ModelRow, ProfileRow, Store};
