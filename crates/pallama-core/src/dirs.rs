use std::path::PathBuf;

/// Filesystem layout for Pallama. Every path Pallama touches is derived from
/// this struct so tests can point it at a tempdir instead of the real home.
///
/// Production layout (XDG):
///   config: ~/.config/pallama/config.toml
///   data:   ~/.local/share/pallama/{models,engines,run,pallama.db}
#[derive(Debug, Clone)]
pub struct PallamaDirs {
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
}

impl PallamaDirs {
    /// Real user directories via the `dirs` crate (XDG on Linux).
    #[must_use] 
    pub fn from_env() -> Self {
        let config_dir = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("pallama");
        let data_dir = dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("pallama");
        Self { config_dir, data_dir }
    }

    #[must_use] 
    pub fn config_file(&self) -> PathBuf {
        self.config_dir.join("config.toml")
    }

    #[must_use] 
    pub fn db_file(&self) -> PathBuf {
        self.data_dir.join("pallama.db")
    }

    #[must_use] 
    pub fn models_dir(&self) -> PathBuf {
        self.data_dir.join("models")
    }

    #[must_use] 
    pub fn engines_dir(&self) -> PathBuf {
        self.data_dir.join("engines")
    }

    #[must_use] 
    pub fn run_dir(&self) -> PathBuf {
        self.data_dir.join("run")
    }

    /// Persistent n-gram speculative caches (`--lookup-cache-dynamic`),
    /// one file per model; survives restarts.
    #[must_use]
    pub fn speccache_dir(&self) -> PathBuf {
        self.data_dir.join("speccache")
    }

    /// Slot KV-cache checkpoints (`--slot-save-path`) for `pallama session`.
    #[must_use]
    pub fn sessions_dir(&self) -> PathBuf {
        self.data_dir.join("sessions")
    }

    /// Ensure all data subdirectories exist (config dir included).
    pub fn ensure(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.config_dir)?;
        std::fs::create_dir_all(self.models_dir())?;
        std::fs::create_dir_all(self.engines_dir())?;
        std::fs::create_dir_all(self.run_dir())?;
        std::fs::create_dir_all(self.speccache_dir())?;
        std::fs::create_dir_all(self.sessions_dir())?;
        Ok(())
    }
}
