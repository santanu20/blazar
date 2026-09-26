//! Blazar core: pure, synchronous domain logic. No tokio, no axum, no
//! network — exhaustively unit-testable. Async lifetimes (process
//! supervision, downloads, HTTP) live in `blazar-runtime`.

pub mod catalog;
pub mod config;
pub mod coreside;
pub mod dirs;
pub mod engine_kind;
pub mod error;
pub mod gguf;
pub mod hardware;
pub mod hfmeta;
pub mod profile;
pub mod session_identity;
pub mod store;
pub mod telemetry;

pub use catalog::{catalog, pair_for_spec_mode, resolve, spec_pair_for, spec_pair_for_typed};
pub use config::{
    is_valid_spec_mode, persist_config, ApiKey, Config, ModelOverride, Remote, SemanticCacheConfig,
};
pub use dirs::BlazarDirs;
pub use error::{CoreError, CoreResult};
pub use gguf::{is_gguf_container, read_metadata_file, GgufMeta, GgufValue};
pub use hardware::{GpuInfo, Hardware};
pub use hfmeta::{read_hf_config, HfMeta, KvGeom, ModelMeta};
pub use profile::{compile as compile_profile, Endpoint, Profile, ProfileInput, TuningOverrides};
pub use store::{EngineRow, KeyUsageRow, LoraRow, ModelRow, ProfileRow, Store};
