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
    pub fn from_env() -> Self {
        let config_dir = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("pallama");
        let data_dir = dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("pallama");
        Self { config_dir, data_dir }
    }

    pub fn config_file(&self) -> PathBuf {
        self.config_dir.join("config.toml")
    }

    pub fn db_file(&self) -> PathBuf {
        self.data_dir.join("pallama.db")
    }

    pub fn models_dir(&self) -> PathBuf {
        self.data_dir.join("models")
    }

    pub fn engines_dir(&self) -> PathBuf {
        self.data_dir.join("engines")
    }

    pub fn run_dir(&self) -> PathBuf {
        self.data_dir.join("run")
    }

    /// Ensure all data subdirectories exist (config dir included).
    pub fn ensure(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.config_dir)?;
        std::fs::create_dir_all(self.models_dir())?;
        std::fs::create_dir_all(self.engines_dir())?;
        std::fs::create_dir_all(self.run_dir())?;
        Ok(())
    }
}
