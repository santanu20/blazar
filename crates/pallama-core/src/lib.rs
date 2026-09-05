//! Pallama core: pure, synchronous domain logic. No tokio, no axum, no
//! network — exhaustively unit-testable. Async lifetimes (process
//! supervision, downloads, HTTP) live in `pallama-runtime`.

pub mod catalog;
pub mod config;
pub mod dirs;
pub mod error;
pub mod store;
pub mod telemetry;

pub use catalog::{catalog, resolve, spec_pair_for};
pub use config::{Config, ModelOverride};
pub use dirs::PallamaDirs;
pub use error::{CoreError, CoreResult};
pub use store::{EngineRow, LoraRow, ModelRow, ProfileRow, Store};
