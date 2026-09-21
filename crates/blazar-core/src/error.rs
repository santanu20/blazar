use thiserror::Error;

/// Errors raised by the pure core: config parsing, store access, catalog lookup.
#[derive(Debug, Error)]
pub enum CoreError {
    #[error("config: {0}")]
    Config(String),

    #[error("store: {0}")]
    Store(#[from] rusqlite::Error),

    #[error("catalog: {0}")]
    Catalog(String),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

pub type CoreResult<T> = Result<T, CoreError>;
