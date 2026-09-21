use std::path::PathBuf;

/// One-shot stderr warning for the CWD fallback (`from_env` is sync and
/// called before tracing is up; eprintln is the only channel that
/// exists at that point).
fn warn_fallback(what: &str) {
    eprintln!(
        "blazar: {what} not found (HOME unset?); falling back to the \
         current directory — set HOME or XDG_* to pin the layout"
    );
}

/// Filesystem layout for Blazar. Every path Blazar touches is derived from
/// this struct so tests can point it at a tempdir instead of the real home.
///
/// Production layout (XDG):
///   config: ~/.config/blazar/config.toml
///   data:   ~/.local/share/blazar/{models,engines,run,blazar.db}
#[derive(Debug, Clone)]
pub struct BlazarDirs {
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
}

impl BlazarDirs {
    /// Real user directories via the `dirs` crate (XDG on Linux).
    ///
    /// An explicitly exported `XDG_CONFIG_HOME` / `XDG_DATA_HOME` wins on
    /// EVERY platform, ahead of the platform default the `dirs` crate
    /// would pick (macOS `~/Library/Application Support`, Windows
    /// `%APPDATA%`). The override is deliberate user intent; the default
    /// stays whatever the platform expects when the env is unset.
    /// F123: a missing HOME/XDG root falls back to CWD — loudly. The old
    /// silent `.` fallback hid a broken env until files started landing
    /// in whatever directory the daemon happened to start from.
    #[must_use]
    pub fn from_env() -> Self {
        let xdg =
            |var: &str, label: &str, platform_default: Option<PathBuf>| match std::env::var_os(var)
            {
                Some(v) if !v.is_empty() => PathBuf::from(v),
                _ => platform_default.unwrap_or_else(|| {
                    warn_fallback(label);
                    PathBuf::from(".")
                }),
            };
        let config_root = xdg("XDG_CONFIG_HOME", "XDG config root", dirs::config_dir());
        let data_root = xdg("XDG_DATA_HOME", "XDG data root", dirs::data_dir());
        Self {
            config_dir: config_root.join("blazar"),
            data_dir: data_root.join("blazar"),
        }
    }

    #[must_use]
    pub fn config_file(&self) -> PathBuf {
        self.config_dir.join("config.toml")
    }

    #[must_use]
    pub fn db_file(&self) -> PathBuf {
        self.data_dir.join("blazar.db")
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

    /// Slot KV-cache checkpoints (`--slot-save-path`) for `blazar session`.
    #[must_use]
    pub fn sessions_dir(&self) -> PathBuf {
        self.data_dir.join("sessions")
    }

    /// Piper TTS voices (`<voice>/<voice>.onnx` + `.onnx.json`), from
    /// `rhasspy/piper-voices` on HF.
    #[must_use]
    pub fn voices_dir(&self) -> PathBuf {
        self.data_dir.join("voices")
    }

    /// Ensure all data subdirectories exist (config dir included).
    pub fn ensure(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.config_dir)?;
        std::fs::create_dir_all(self.models_dir())?;
        std::fs::create_dir_all(self.engines_dir())?;
        std::fs::create_dir_all(self.run_dir())?;
        std::fs::create_dir_all(self.speccache_dir())?;
        std::fs::create_dir_all(self.sessions_dir())?;
        std::fs::create_dir_all(self.voices_dir())?;
        Ok(())
    }
}
