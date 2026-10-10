//! Blazar core: pure, synchronous domain logic. No tokio, no axum, no
//! network — exhaustively unit-testable. Async lifetimes (process
//! supervision, downloads, HTTP) live in `blazar-runtime`.

pub mod catalog;
pub mod config;
pub mod coreside;
pub mod dirs;
pub mod engine_kind;
pub mod error;
pub mod fs_safety;
pub mod gguf;
pub mod hardware;
pub mod hfmeta;
pub mod knob_registry;
pub mod predict;
pub mod profile;
pub mod session_identity;
pub mod store;
pub mod telemetry;
pub mod tls;

pub use catalog::{catalog, pair_for_spec_mode, resolve, spec_pair_for, spec_pair_for_typed};
pub use config::{
    ApiKey, Config, ModelOverride, Remote, SemanticCacheConfig, is_valid_spec_mode, persist_config,
};
pub use dirs::BlazarDirs;
pub use error::{CoreError, CoreResult};
pub use gguf::{
    DIFFUSION_PARADIGM_ARCHS, GgufMeta, GgufValue, is_gguf_container, read_metadata_file,
};
pub use hardware::{GpuInfo, Hardware};
pub use hfmeta::{
    HfMeta, KvGeom, ModelMeta, read_hf_config, root_safetensors, root_safetensors_bytes,
};
pub use profile::{Endpoint, Profile, ProfileInput, TuningOverrides, compile as compile_profile};
pub use store::{
    BenchResultRow, CompletionCardRow, EngineRow, JobEventRow, JobRow, KeyUsageRow, LoraRow,
    ModelRow, ProfileRow, Store, StoredResponseRow,
};
