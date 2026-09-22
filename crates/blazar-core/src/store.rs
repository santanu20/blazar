use rusqlite::{params, Connection};

use crate::dirs::BlazarDirs;
use crate::error::{CoreError, CoreResult};

/// SQLite-backed persistent state (WAL mode). One DB file at
/// `<data_dir>/blazar.db`. Schema versioned via `PRAGMA user_version`.
///
/// Tables:
/// - engines:  installed llama-server builds + capability manifest JSON
/// - models:   pulled GGUFs (plain files, no blob store)
/// - profiles: per-(model, engine) launch argv + benchmark results
/// - loras:    `LoRA` adapters per model
#[derive(Debug)]
pub struct Store {
    conn: Connection,
}

const SCHEMA_VERSION: i32 = 6;

const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS engines (
    tag          TEXT PRIMARY KEY,
    asset        TEXT NOT NULL,
    sha256       TEXT NOT NULL,
    installed_at INTEGER NOT NULL,
    active       INTEGER NOT NULL DEFAULT 0,
    manifest     TEXT NOT NULL,
    kind         TEXT NOT NULL DEFAULT 'llamacpp'
);
CREATE TABLE IF NOT EXISTS models (
    name       TEXT PRIMARY KEY,
    repo       TEXT NOT NULL,
    quant      TEXT NOT NULL,
    path       TEXT NOT NULL,
    bytes      INTEGER NOT NULL,
    sha256     TEXT,
    mmproj_path TEXT,
    components  TEXT NOT NULL DEFAULT '[]',
    shards     INTEGER NOT NULL DEFAULT 1,
    arch       TEXT,
    params     REAL,
    ctx_train  INTEGER,
    pulled_at  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS profiles (
    model_name     TEXT NOT NULL,
    engine_tag     TEXT NOT NULL,
    args_hash      TEXT NOT NULL,
    args_json      TEXT NOT NULL,
    benchmark_json TEXT,
    updated_at     INTEGER NOT NULL,
    PRIMARY KEY (model_name, engine_tag)
);
CREATE TABLE IF NOT EXISTS loras (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    path       TEXT NOT NULL,
    scale      REAL NOT NULL DEFAULT 1.0
);
CREATE TABLE IF NOT EXISTS key_usage (
    day      TEXT NOT NULL, -- UTC YYYY-MM-DD
    name     TEXT NOT NULL, -- [[keys]] name
    requests INTEGER NOT NULL DEFAULT 0,
    tokens   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, name)
);
CREATE TABLE IF NOT EXISTS bench_history (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    ts    INTEGER NOT NULL,          -- unix secs
    engine_tag TEXT NOT NULL,
    model TEXT NOT NULL,
    tg_tokens_per_sec REAL NOT NULL, -- llama-bench tg128 median
    pp_tokens_per_sec REAL NOT NULL DEFAULT 0,
    ctx   INTEGER NOT NULL DEFAULT 0
);
";

/// Row shapes shared across crates.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EngineRow {
    pub tag: String,
    pub asset: String,
    pub sha256: String,
    pub installed_at: i64,
    pub active: bool,
    pub manifest: String,
    /// Which upstream project this engine wraps. Rows written before
    /// the column existed deserialize as `llamacpp` (serde default).
    #[serde(default)]
    pub kind: crate::engine_kind::EngineKind,
}

impl EngineRow {
    /// Routing class of this lane, from the provenance stamped in its
    /// manifest: forks (`"source": "fork"`) are capability shims and
    /// lose same-kind ties to mainstream builds. Undecodable or legacy
    /// manifests count as mainstream — the pre-fork default.
    #[must_use]
    pub fn lane_class(&self) -> crate::engine_kind::LaneClass {
        let fork = serde_json::from_str::<serde_json::Value>(&self.manifest)
            .is_ok_and(|m| m.get("source").is_some_and(|s| s.as_str() == Some("fork")));
        if fork {
            crate::engine_kind::LaneClass::Fork
        } else {
            crate::engine_kind::LaneClass::Mainstream
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ModelRow {
    pub name: String,
    pub repo: String,
    pub quant: String,
    pub path: String,
    pub bytes: i64,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub mmproj_path: Option<String>,
    /// Diffusion component set (sdcpp lane): the `DiT` `path` above is
    /// unservable alone — the VAE and text encoder(s) complete the model.
    /// Flag-keyed because families differ in dialect: Qwen-Image pairs
    /// `--vae`/`--llm`, FLUX pairs `--vae`/`--t5xxl`/`--clip_l`. Empty on
    /// every text model (that emptiness IS the text/diffusion routing
    /// domain marker).
    #[serde(default)]
    pub components: Vec<ComponentFile>,
    #[serde(default = "default_shards")]
    pub shards: i64,
    #[serde(default)]
    pub arch: Option<String>,
    #[serde(default)]
    pub params: Option<f64>,
    #[serde(default)]
    pub ctx_train: Option<i64>,
    pub pulled_at: i64,
}

/// One diffusion component: the sd-server flag it rides and the local
/// file that satisfies it (`--vae` → VAE, `--t5xxl` → T5 encoder, ...).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ComponentFile {
    pub flag: String,
    pub path: String,
}

impl ComponentFile {
    #[must_use]
    pub fn new(flag: &str, path: &str) -> Self {
        Self {
            flag: flag.to_string(),
            path: path.to_string(),
        }
    }
}

/// Quantization-method tokens that mark a safetensors checkpoint as
/// engine-picky: AWQ/GPTQ/FP8 dirs serve on sglang's vLLM-style loader
/// but dequant broken on mistral.rs prebuilt builds (live-proven
/// v0.9.3: dtype-mismatch empty 200s, garbage tokens under f16).
const QUANTIZED_SAFETENSORS_TOKENS: [&str; 3] = ["awq", "gptq", "fp8"];

/// The quantized-safetensors routing signal, shared by every caller
/// (store rows, CLI name+path pairs): true when the checkpoint is NOT
/// a GGUF file and any name/repo/path token names a quantization
/// method. Word-boundary matched so names like "hawk" stay clean.
#[must_use]
pub fn quantized_safetensors_signal(name: &str, repo: &str, path: &str) -> bool {
    if path.to_ascii_lowercase().ends_with(".gguf") {
        return false;
    }
    [name, repo, path].iter().any(|s| {
        s.to_ascii_lowercase()
            .split(|c: char| !c.is_ascii_alphanumeric())
            .any(|tok| QUANTIZED_SAFETENSORS_TOKENS.contains(&tok))
    })
}

impl ModelRow {
    /// True for quantized safetensors checkpoints (AWQ/GPTQ/FP8). The
    /// `quant` column is UNRELIABLE for this (an AWQ dir pulls with
    /// quant `4BIT`, an FP8-dynamic dir with `BF16`) — the honest
    /// signal is the quantization-method token in the name/repo/path,
    /// which the pull convention always carries
    /// (`model-instruct-awq.d`). GGUF rows are never "quantized
    /// safetensors" — GGUF quants are a different, well-served lane.
    #[must_use]
    pub fn is_quantized_safetensors(&self) -> bool {
        quantized_safetensors_signal(&self.name, &self.repo, &self.path)
    }

    /// Local path for a component flag, if the set carries it.
    #[must_use]
    pub fn component(&self, flag: &str) -> Option<&str> {
        self.components
            .iter()
            .find(|c| c.flag == flag)
            .map(|c| c.path.as_str())
    }

    /// Diffusion-domain routing marker: a row with a `--vae` component
    /// serves on the sdcpp lane; every text row carries no set at all.
    #[must_use]
    pub fn has_component_set(&self) -> bool {
        self.component("--vae").is_some()
    }

    /// Image edits (`/v1/images/edits`) need the vision-encoder
    /// companion; families without one teach instead of re-pull-looping.
    #[must_use]
    pub fn serves_image_edits(&self) -> bool {
        self.component("--llm_vision").is_some()
    }
}

fn default_shards() -> i64 {
    1
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProfileRow {
    pub model_name: String,
    pub engine_tag: String,
    pub args_hash: String,
    pub args_json: String,
    #[serde(default)]
    pub benchmark_json: Option<String>,
    pub updated_at: i64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LoraRow {
    pub id: i64,
    pub model_name: String,
    pub path: String,
    pub scale: f64,
}

impl Store {
    /// Open (creating directories and file as needed) and migrate.
    pub fn open(dirs: &BlazarDirs) -> CoreResult<Self> {
        std::fs::create_dir_all(&dirs.data_dir)?;
        let conn = Connection::open(dirs.db_file())?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.pragma_update(None, "busy_timeout", 5000)?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    /// Consistent snapshot of the live database via `SQLite` `VACUUM INTO`.
    /// A raw file copy of a live WAL database can be torn mid-transaction
    /// (F130); `VACUUM INTO` is transactionally consistent by construction.
    /// Fails if the destination file already exists (`SQLite` contract).
    pub fn snapshot_db_to(&self, dst: &std::path::Path) -> CoreResult<()> {
        let dst = dst.to_string_lossy().into_owned();
        self.conn.execute("VACUUM INTO ?1", [dst])?;
        Ok(())
    }

    /// Column names of `table` — the source of truth for additive
    /// migrations deciding whether an ALTER is still owed.
    fn table_columns(&self, table: &str) -> CoreResult<Vec<String>> {
        let mut stmt = self.conn.prepare(&format!("PRAGMA table_info({table})"))?;
        let mut cols = stmt.query([])?;
        let mut names = Vec::new();
        while let Some(r) = cols.next()? {
            names.push(r.get(1)?);
        }
        Ok(names)
    }

    fn migrate(&self) -> CoreResult<()> {
        let version: i32 = self
            .conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if version < SCHEMA_VERSION {
            self.conn.execute_batch(SCHEMA_SQL)?;
            // v4 added engines.kind. CREATE TABLE IF NOT EXISTS covers
            // fresh databases; existing ones need the explicit ALTER.
            if !self.table_columns("engines")?.contains(&"kind".to_string()) {
                self.conn.execute(
                    "ALTER TABLE engines ADD COLUMN kind TEXT NOT NULL DEFAULT 'llamacpp'",
                    [],
                )?;
            }
            // v5→v6: the three fixed diffusion columns became one
            // flag-keyed `components` JSON column (families differ in
            // dialect: Qwen pairs --vae/--llm, FLUX pairs --t5xxl/
            // --clip_l). Fresh databases get it from SCHEMA_SQL; v5
            // databases fold their legacy columns into it and drop them.
            let model_cols = self.table_columns("models")?;
            if !model_cols.contains(&"components".to_string()) {
                self.conn.execute(
                    "ALTER TABLE models ADD COLUMN components TEXT NOT NULL DEFAULT '[]'",
                    [],
                )?;
            }
            if model_cols.iter().any(|c| c.starts_with("vae_path")) {
                // Collected up front: rewriting rows while the SELECT
                // cursor is still open on the same table is undefined.
                type LegacyComponentCols = (String, Option<String>, Option<String>, Option<String>);
                let legacy: Vec<LegacyComponentCols> = self
                    .conn
                    .prepare("SELECT name, vae_path, llm_path, llm_vision_path FROM models")?
                    .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
                    .collect::<Result<Vec<_>, _>>()?;
                for (name, vae, llm, vision) in legacy {
                    let folded: Vec<ComponentFile> =
                        [("--vae", vae), ("--llm", llm), ("--llm_vision", vision)]
                            .into_iter()
                            .filter_map(|(flag, path)| path.map(|p| ComponentFile::new(flag, &p)))
                            .collect();
                    let folded_json = serde_json::to_string(&folded)
                        .map_err(|e| CoreError::Catalog(e.to_string()))?;
                    self.conn.execute(
                        "UPDATE models SET components = ?1 WHERE name = ?2",
                        params![folded_json, name],
                    )?;
                }
                for col in ["vae_path", "llm_path", "llm_vision_path"] {
                    self.conn
                        .execute(&format!("ALTER TABLE models DROP COLUMN {col}"), [])?;
                }
            }
            self.conn
                .pragma_update(None, "user_version", SCHEMA_VERSION)?;
        }
        Ok(())
    }

    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    pub fn upsert_engine(&self, e: &EngineRow) -> CoreResult<()> {
        self.conn.execute(
            "INSERT INTO engines (tag, asset, sha256, installed_at, active, manifest, kind)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(tag) DO UPDATE SET
               asset = excluded.asset, sha256 = excluded.sha256,
               installed_at = excluded.installed_at, manifest = excluded.manifest,
               kind = excluded.kind",
            params![
                e.tag,
                e.asset,
                e.sha256,
                e.installed_at,
                i64::from(e.active),
                e.manifest,
                e.kind
            ],
        )?;
        Ok(())
    }

    /// Flip the active engine. Fails (changing nothing) when `tag` is not
    /// installed — never silently leaves the store with zero active engines.
    pub fn set_active_engine(&self, tag: &str) -> CoreResult<()> {
        // F120: both UPDATEs inside one transaction — a crash between
        // them used to leave the store with ZERO active engines.
        let tx = self.conn.unchecked_transaction()?;
        let exists: bool = tx
            .query_row(
                "SELECT COUNT(*) FROM engines WHERE tag = ?1",
                params![tag],
                |r| r.get::<_, i64>(0),
            )
            .map(|n| n > 0)?;
        if !exists {
            return Err(CoreError::Store(rusqlite::Error::QueryReturnedNoRows));
        }
        tx.execute("UPDATE engines SET active = 0", [])?;
        tx.execute("UPDATE engines SET active = 1 WHERE tag = ?1", params![tag])?;
        tx.commit()?;
        Ok(())
    }

    /// Rewrite one engine row's manifest JSON (supersede marking,
    /// lazy architecture-coverage mining). Fails when `tag` is unknown
    /// so a stale caller can never invent a row.
    pub fn update_engine_manifest(&self, tag: &str, manifest_json: &str) -> CoreResult<()> {
        let changed = self.conn.execute(
            "UPDATE engines SET manifest = ?2 WHERE tag = ?1",
            params![tag, manifest_json],
        )?;
        if changed == 0 {
            return Err(CoreError::Store(rusqlite::Error::QueryReturnedNoRows));
        }
        Ok(())
    }

    pub fn active_engine(&self) -> CoreResult<Option<EngineRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT tag, asset, sha256, installed_at, active, manifest, kind FROM engines WHERE active = 1",
        )?;
        let mut rows = stmt.query([])?;
        if let Some(r) = rows.next()? {
            return Ok(Some(EngineRow {
                tag: r.get(0)?,
                asset: r.get(1)?,
                sha256: r.get(2)?,
                installed_at: r.get(3)?,
                active: r.get::<_, i64>(4)? != 0,
                manifest: r.get(5)?,
                kind: r.get(6)?,
            }));
        }
        Ok(None)
    }

    pub fn list_engines(&self) -> CoreResult<Vec<EngineRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT tag, asset, sha256, installed_at, active, manifest, kind FROM engines ORDER BY installed_at DESC, rowid DESC",
        )?;
        let rows = stmt.query_map([], engine_from_row)?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    pub fn delete_engine(&self, tag: &str) -> CoreResult<()> {
        self.conn
            .execute("DELETE FROM engines WHERE tag = ?1", params![tag])?;
        Ok(())
    }

    pub fn upsert_model(&self, m: &ModelRow) -> CoreResult<()> {
        // `#N` is the internal replica-key separator (supervisor B1):
        // a model literally named `x#2` would collide with replica keys.
        if m.name.contains('#') {
            return Err(CoreError::Catalog(format!(
                "model name {:?} contains '#', which is reserved for replica keys \
                 (supervisor `model#N`); rename the model",
                m.name
            )));
        }
        self.conn.execute(
            "INSERT INTO models (name, repo, quant, path, bytes, sha256, mmproj_path, components,
                                 shards, arch, params, ctx_train, pulled_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)
             ON CONFLICT(name) DO UPDATE SET
               repo = excluded.repo, quant = excluded.quant, path = excluded.path,
               bytes = excluded.bytes, sha256 = excluded.sha256,
               mmproj_path = excluded.mmproj_path, components = excluded.components,
               shards = excluded.shards,
               arch = excluded.arch, params = excluded.params,
               ctx_train = excluded.ctx_train, pulled_at = excluded.pulled_at",
            params![
                m.name,
                m.repo,
                m.quant,
                m.path,
                m.bytes,
                m.sha256,
                m.mmproj_path,
                serde_json::to_string(&m.components)
                    .map_err(|e| CoreError::Catalog(e.to_string()))?,
                m.shards,
                m.arch,
                m.params,
                m.ctx_train,
                m.pulled_at
            ],
        )?;
        Ok(())
    }

    pub fn get_model(&self, name: &str) -> CoreResult<Option<ModelRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT name, repo, quant, path, bytes, sha256, mmproj_path, components, shards, arch,
                    params, ctx_train, pulled_at
             FROM models WHERE name = ?1",
        )?;
        let mut rows = stmt.query(params![name])?;
        if let Some(r) = rows.next()? {
            return Ok(Some(model_from_row(r)?));
        }
        Ok(None)
    }

    pub fn list_models(&self) -> CoreResult<Vec<ModelRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT name, repo, quant, path, bytes, sha256, mmproj_path, components, shards, arch,
                    params, ctx_train, pulled_at
             FROM models ORDER BY name",
        )?;
        let rows = stmt.query_map([], model_from_row)?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Map a user-supplied model name onto a store row for ollama
    /// migrants: exact match first, then a `model:tag` → `model-tag`
    /// `:`→`-` swap. Returns the input unchanged when nothing matches,
    /// so callers keep their own not-found error text (fail loud, no
    /// hidden rewrite). Shared rule for the CLI boundary and the
    /// gateway's `ensure` path; `pull` never uses it (its colon is the
    /// `owner/repo:QUANT` separator).
    pub fn resolve_model_name(&self, name: &str) -> String {
        if !name.contains(':') {
            return name.to_string();
        }
        let hit = |n: &str| self.get_model(n).is_ok_and(|r| r.is_some());
        if hit(name) {
            return name.to_string();
        }
        let swapped = name.replace(':', "-");
        if hit(&swapped) {
            return swapped;
        }
        name.to_string()
    }

    pub fn delete_model(&self, name: &str) -> CoreResult<bool> {
        Ok(self
            .conn
            .execute("DELETE FROM models WHERE name = ?1", params![name])?
            == 1)
    }

    pub fn upsert_profile(&self, p: &ProfileRow) -> CoreResult<()> {
        self.conn.execute(
            "INSERT INTO profiles (model_name, engine_tag, args_hash, args_json, benchmark_json, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6)
             ON CONFLICT(model_name, engine_tag) DO UPDATE SET
               args_hash = excluded.args_hash, args_json = excluded.args_json,
               benchmark_json = excluded.benchmark_json, updated_at = excluded.updated_at",
            params![p.model_name, p.engine_tag, p.args_hash, p.args_json, p.benchmark_json, p.updated_at],
        )?;
        Ok(())
    }

    pub fn get_profile(&self, model: &str, engine_tag: &str) -> CoreResult<Option<ProfileRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT model_name, engine_tag, args_hash, args_json, benchmark_json, updated_at
             FROM profiles WHERE model_name = ?1 AND engine_tag = ?2",
        )?;
        let mut rows = stmt.query(params![model, engine_tag])?;
        if let Some(r) = rows.next()? {
            return Ok(Some(ProfileRow {
                model_name: r.get(0)?,
                engine_tag: r.get(1)?,
                args_hash: r.get(2)?,
                args_json: r.get(3)?,
                benchmark_json: r.get(4)?,
                updated_at: r.get(5)?,
            }));
        }
        Ok(None)
    }

    pub fn add_lora(&self, model_name: &str, path: &str, scale: f64) -> CoreResult<i64> {
        self.conn.execute(
            "INSERT INTO loras (model_name, path, scale) VALUES (?1, ?2, ?3)",
            params![model_name, path, scale],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn delete_lora(&self, id: i64) -> CoreResult<bool> {
        Ok(self
            .conn
            .execute("DELETE FROM loras WHERE id = ?1", params![id])?
            == 1)
    }

    pub fn list_loras(&self, model_name: Option<&str>) -> CoreResult<Vec<LoraRow>> {
        let (sql, binds): (&str, Vec<&str>) = match model_name {
            Some(m) => (
                "SELECT id, model_name, path, scale FROM loras WHERE model_name = ?1 ORDER BY id",
                vec![m],
            ),
            None => (
                "SELECT id, model_name, path, scale FROM loras ORDER BY id",
                vec![],
            ),
        };
        let mut stmt = self.conn.prepare(sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(binds.iter()), |r| {
            Ok(LoraRow {
                id: r.get(0)?,
                model_name: r.get(1)?,
                path: r.get(2)?,
                scale: r.get(3)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Today's (or any day's) per-key counters, for `/api/keys` + budgets.
    pub fn key_usage(&self, day: &str) -> CoreResult<Vec<KeyUsageRow>> {
        let mut stmt = self
            .conn
            .prepare("SELECT day, name, requests, tokens FROM key_usage WHERE day = ?1")?;
        let rows = stmt.query_map(rusqlite::params![day], |r| {
            Ok(KeyUsageRow {
                day: r.get(0)?,
                name: r.get(1)?,
                requests: r.get(2)?,
                tokens: r.get(3)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Append a bench measurement (tune --search winner, engine gates).
    pub fn record_bench(
        &self,
        engine_tag: &str,
        model: &str,
        tg: f64,
        pp: f64,
        ctx: i64,
    ) -> CoreResult<()> {
        self.conn.execute(
            "INSERT INTO bench_history (ts, engine_tag, model, tg_tokens_per_sec, pp_tokens_per_sec, ctx) \
             VALUES (unixepoch(), ?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![engine_tag, model, tg, pp, ctx],
        )?;
        Ok(())
    }

    /// The single most recent bench row overall (engine gate baseline).
    pub fn latest_bench_by_time(&self) -> CoreResult<Option<(String, f64, String)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT engine_tag, tg_tokens_per_sec, model FROM bench_history ORDER BY id DESC LIMIT 1")?;
        let mut rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
        Ok(rows.next().transpose()?)
    }

    /// Most recent bench row per model: `engine_tag`, `tg` t/s, `model`.
    pub fn latest_benches(&self) -> CoreResult<Vec<(String, f64, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT engine_tag, tg_tokens_per_sec, model FROM bench_history b \
             WHERE id = (SELECT MAX(id) FROM bench_history b2 WHERE b2.model = b.model) \
             ORDER BY model",
        )?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Overwrite one day's counters for a batch of keys (absolute values,
    /// not deltas — the gateway holds the live counters and writes the
    /// whole day-state behind). One transaction per batch.
    pub fn set_key_usage_day(&self, day: &str, rows: &[(String, u64, u64)]) -> CoreResult<()> {
        let tx = self.conn.unchecked_transaction()?;
        for (name, requests, tokens) in rows {
            tx.execute(
                "DELETE FROM key_usage WHERE day = ?1 AND name = ?2",
                rusqlite::params![day, name],
            )?;
            tx.execute(
                "INSERT INTO key_usage (day, name, requests, tokens) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![
                    day,
                    name,
                    i64::try_from(*requests).unwrap_or(i64::MAX),
                    i64::try_from(*tokens).unwrap_or(i64::MAX)
                ],
            )?;
        }
        tx.commit()?;
        Ok(())
    }
}

/// Daily usage counters for one key.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct KeyUsageRow {
    pub day: String,
    pub name: String,
    pub requests: i64,
    pub tokens: i64,
}

fn engine_from_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<EngineRow> {
    Ok(EngineRow {
        tag: r.get(0)?,
        asset: r.get(1)?,
        sha256: r.get(2)?,
        installed_at: r.get(3)?,
        active: r.get::<_, i64>(4)? != 0,
        manifest: r.get(5)?,
        kind: r.get(6)?,
    })
}

fn model_from_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<ModelRow> {
    let raw_components: Option<String> = r.get(7)?;
    let components = match raw_components.as_deref() {
        // Empty/NULL predates a pull ever attaching a set; anything else
        // must decode — a silently-dropped set would re-route the row to
        // the text lanes and die at spawn with a confusing crash.
        None | Some("") => Vec::new(),
        Some(raw) => serde_json::from_str(raw).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(7, rusqlite::types::Type::Text, Box::new(e))
        })?,
    };
    Ok(ModelRow {
        name: r.get(0)?,
        repo: r.get(1)?,
        quant: r.get(2)?,
        path: r.get(3)?,
        bytes: r.get(4)?,
        sha256: r.get(5)?,
        mmproj_path: r.get(6)?,
        components,
        shards: r.get(8)?,
        arch: r.get(9)?,
        params: r.get(10)?,
        ctx_train: r.get(11)?,
        pulled_at: r.get(12)?,
    })
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::engine_kind::EngineKind;

    fn tmp_store() -> (tempfile::TempDir, Store) {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        let store = Store::open(&dirs).unwrap();
        (tmp, store)
    }

    #[test]
    fn unit__store_schema_created__tables_exist() {
        let (_t, s) = tmp_store();
        let n: i64 = s
            .conn()
            .query_row("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('engines','models','profiles','loras')", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 4);
    }

    #[test]
    fn unit__lane_class__reads_manifest_source_field() {
        use crate::engine_kind::LaneClass;
        let row = |manifest: String| EngineRow {
            tag: "t".into(),
            asset: "a".into(),
            sha256: "x".into(),
            installed_at: 0,
            active: false,
            manifest,
            kind: EngineKind::LlamaCpp,
        };
        // Fork provenance stamps Fork.
        assert_eq!(
            row(r#"{"source":"fork"}"#.into()).lane_class(),
            LaneClass::Fork
        );
        // Legacy/missing field and other sources stay mainstream.
        assert_eq!(row("{}".into()).lane_class(), LaneClass::Mainstream);
        assert_eq!(
            row(r#"{"source":"upstream"}"#.into()).lane_class(),
            LaneClass::Mainstream
        );
        // Undecodable manifest: fail toward the pre-fork default.
        assert_eq!(row("not json".into()).lane_class(), LaneClass::Mainstream);
    }

    #[test]
    fn unit__resolve_model_name__colon_swaps_only_onto_existing_rows() {
        let (_t, s) = tmp_store();
        s.upsert_model(&base_model("qwen3.5-9b")).unwrap();
        // Flat names pass through untouched (no store probe needed).
        assert_eq!(s.resolve_model_name("qwen3.5-9b"), "qwen3.5-9b");
        // ollama migrant input resolves onto the flat row.
        assert_eq!(s.resolve_model_name("qwen3.5:9b"), "qwen3.5-9b");
        // Miss on both forms returns the input verbatim (caller errors).
        assert_eq!(s.resolve_model_name("nope:9b"), "nope:9b");
    }

    fn base_model(name: &str) -> ModelRow {
        ModelRow {
            name: name.into(),
            repo: "o/x".into(),
            quant: "Q4_K_M".into(),
            path: "/tmp/x.gguf".into(),
            bytes: 1,
            sha256: None,
            mmproj_path: None,
            components: vec![],
            shards: 1,
            arch: None,
            params: None,
            ctx_train: None,
            pulled_at: 0,
        }
    }

    #[test]
    fn unit__upsert_model__rejects_hash_in_name() {
        let (_t, s) = tmp_store();
        let err = s.upsert_model(&base_model("x#2")).unwrap_err();
        assert!(
            err.to_string().contains("reserved for replica keys"),
            "teaching error, got: {err}"
        );
        // The rejected row never lands.
        assert!(s.get_model("x#2").unwrap().is_none());
        // Normal names still work.
        s.upsert_model(&base_model("x")).unwrap();
        assert!(s.get_model("x").unwrap().is_some());
    }

    #[test]
    fn unit__engine_upsert_activate_cycle() {
        let (_t, s) = tmp_store();
        for tag in ["b100", "b200"] {
            s.upsert_engine(&EngineRow {
                tag: tag.into(),
                asset: "ubuntu-vulkan-x64".into(),
                sha256: format!("{tag}deadbeef"),
                installed_at: 1,
                active: false,
                manifest: "{}".into(),
                kind: EngineKind::LlamaCpp,
            })
            .unwrap();
        }
        assert!(s.active_engine().unwrap().is_none());
        s.set_active_engine("b100").unwrap();
        assert_eq!(s.active_engine().unwrap().unwrap().tag, "b100");
        s.set_active_engine("b200").unwrap();
        assert_eq!(s.active_engine().unwrap().unwrap().tag, "b200");
        // Re-activating an unknown tag errors AND leaves state untouched.
        assert!(s.set_active_engine("b999").is_err());
        assert_eq!(
            s.active_engine().unwrap().unwrap().tag,
            "b200",
            "failed activation must not clear the active engine"
        );
        assert_eq!(s.list_engines().unwrap().len(), 2);
    }

    #[test]
    fn unit__model_roundtrip_and_delete() {
        let (_t, s) = tmp_store();
        let m = ModelRow {
            name: "qwen3-0.6b".into(),
            repo: "ggml-org/Qwen3-0.6B-GGUF".into(),
            quant: "Q4_K_M".into(),
            path: "/models/qwen3-0.6b-q4_k_m.gguf".into(),
            bytes: 500_000_000,
            sha256: Some("abc".into()),
            mmproj_path: None,
            components: vec![],
            shards: 2,
            arch: Some("qwen3".into()),
            params: Some(0.6),
            ctx_train: Some(32_768),
            pulled_at: 42,
        };
        s.upsert_model(&m).unwrap();
        let got = s.get_model("qwen3-0.6b").unwrap().unwrap();
        assert_eq!(got, m);
        assert_eq!(s.list_models().unwrap().len(), 1);
        assert!(s.delete_model("qwen3-0.6b").unwrap());
        assert!(!s.delete_model("qwen3-0.6b").unwrap());
        assert!(s.get_model("qwen3-0.6b").unwrap().is_none());
    }

    #[test]
    fn unit__profile_upsert_replaces() {
        let (_t, s) = tmp_store();
        let p = ProfileRow {
            model_name: "m".into(),
            engine_tag: "b1".into(),
            args_hash: "h1".into(),
            args_json: "[\"--jinja\"]".into(),
            benchmark_json: None,
            updated_at: 1,
        };
        s.upsert_profile(&p).unwrap();
        let mut p2 = p.clone();
        p2.args_hash = "h2".into();
        p2.updated_at = 2;
        s.upsert_profile(&p2).unwrap();
        let got = s.get_profile("m", "b1").unwrap().unwrap();
        assert_eq!((got.args_hash.as_str(), got.updated_at), ("h2", 2));
    }

    #[test]
    fn unit__lora_add_list_delete() {
        let (_t, s) = tmp_store();
        let id = s.add_lora("m", "/loras/a.bin", 0.8).unwrap();
        s.add_lora("other", "/loras/b.bin", 1.0).unwrap();
        let for_m = s.list_loras(Some("m")).unwrap();
        assert_eq!(for_m.len(), 1);
        assert!((for_m[0].scale - 0.8).abs() < f64::EPSILON);
        assert_eq!(s.list_loras(None).unwrap().len(), 2);
        assert!(s.delete_lora(id).unwrap());
        assert!(s.list_loras(Some("m")).unwrap().is_empty());
    }

    /// The quantized-safetensors routing signal: pull-convention dir
    /// names carry the quant-method token; `quant` column values never
    /// enter it (awq rows pull as `4BIT`, fp8-dynamic as `BF16`).
    #[test]
    fn unit__quantized_safetensors_signal__tokens_and_boundaries() {
        let row = |name: &str, repo: &str, path: &str| ModelRow {
            name: name.into(),
            repo: repo.into(),
            quant: "BF16".into(), // unreliable on purpose — never read
            path: path.into(),
            bytes: 1,
            sha256: None,
            mmproj_path: None,
            components: vec![],
            shards: 1,
            arch: None,
            params: None,
            ctx_train: None,
            pulled_at: 0,
        };
        // The live incident rows: awq/gptq/fp8 dirs quantize true.
        assert!(row(
            "qwen2.5-0.5b-instruct-awq",
            "qwen/qwen2.5-0.5b-instruct-awq",
            "/models/qwen2.5-0.5b-instruct-awq.d"
        )
        .is_quantized_safetensors());
        assert!(row("m-gptq", "r", "/models/m-gptq.d").is_quantized_safetensors());
        assert!(row("m-fp8-dynamic", "r", "/models/m-fp8-dynamic.d").is_quantized_safetensors());
        // Word boundary: 'hawk' embeds awq as a substring but splits
        // into its own token — stays a normal lane.
        assert!(!row("hawk", "r", "/models/hawk.d").is_quantized_safetensors());
        // Plain BF16 dir: the lane mistral.rs serves perfectly.
        assert!(!row(
            "qwen2.5-0.5b-instruct",
            "qwen/qwen2.5-0.5b-instruct",
            "/models/qwen2.5-0.5b-instruct.d"
        )
        .is_quantized_safetensors());
        // GGUF never counts, even when the filename carries a token.
        assert!(!row("m-awq", "r", "/models/m-awq-q4_k_m.gguf").is_quantized_safetensors());
        // Repo token alone (a dir renamed clean) still signals.
        assert!(quantized_safetensors_signal(
            "renamed",
            "qwen/qwen2.5-0.5b-instruct-awq",
            "/models/renamed.d"
        ));
    }

    #[test]
    fn unit__reopen_migrates_idempotently() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        {
            let s = Store::open(&dirs).unwrap();
            s.upsert_model(&ModelRow {
                name: "m".into(),
                repo: "r".into(),
                quant: "Q4_K_M".into(),
                path: "p".into(),
                bytes: 1,
                sha256: None,
                mmproj_path: None,
                components: vec![],
                shards: 1,
                arch: None,
                params: None,
                ctx_train: None,
                pulled_at: 1,
            })
            .unwrap();
        }
        // Second open must not lose or duplicate data.
        let s2 = Store::open(&dirs).unwrap();
        assert_eq!(s2.list_models().unwrap().len(), 1);
        let v: i32 = s2
            .conn()
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(v, SCHEMA_VERSION);
    }

    #[test]
    fn unit__migrate_v4_db__adds_components_col_and_keeps_rows() {
        // A database last written by a v4 daemon: models table without
        // any diffusion columns. Opening it must add the components
        // column in place, keep every existing row, and read the set as
        // empty (that emptiness is the text-model domain marker).
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        std::fs::create_dir_all(&dirs.data_dir).unwrap();
        {
            let conn = Connection::open(dirs.db_file()).unwrap();
            conn.execute_batch(
                "CREATE TABLE models (
                    name       TEXT PRIMARY KEY,
                    repo       TEXT NOT NULL,
                    quant      TEXT NOT NULL,
                    path       TEXT NOT NULL,
                    bytes      INTEGER NOT NULL,
                    sha256     TEXT,
                    mmproj_path TEXT,
                    shards     INTEGER NOT NULL DEFAULT 1,
                    arch       TEXT,
                    params     REAL,
                    ctx_train  INTEGER,
                    pulled_at  INTEGER NOT NULL
                );
                INSERT INTO models (name, repo, quant, path, bytes, shards, pulled_at)
                VALUES ('old-m', 'r', 'Q4_K_M', 'p', 7, 1, 1);
                PRAGMA user_version = 4;",
            )
            .unwrap();
        }
        let s = Store::open(&dirs).unwrap();
        let mut cols = s.conn().prepare("PRAGMA table_info(models)").unwrap();
        let mut rows = cols.query([]).unwrap();
        let mut names: Vec<String> = Vec::new();
        while let Some(r) = rows.next().unwrap() {
            names.push(r.get::<_, String>(1).unwrap());
        }
        names.sort_unstable();
        assert!(
            names.iter().any(|n| n == "components"),
            "missing components: {names:?}"
        );
        let old = s.get_model("old-m").unwrap().unwrap();
        assert_eq!(old.bytes, 7);
        assert!(old.components.is_empty());
        assert!(!old.has_component_set());
        // And the new field roundtrips through the migrated table.
        s.upsert_model(&ModelRow {
            name: "qwen-image-2.1".into(),
            repo: "abenzerps/Qwen-Image-2.1-GGUF".into(),
            quant: "Q4_K_M".into(),
            path: "p.gguf".into(),
            bytes: 4_608_000_000,
            sha256: None,
            mmproj_path: None,
            components: vec![
                ComponentFile::new("--vae", "vae.safetensors"),
                ComponentFile::new("--llm", "te.gguf"),
            ],
            shards: 1,
            arch: None,
            params: None,
            ctx_train: None,
            pulled_at: 2,
        })
        .unwrap();
        let set = s.get_model("qwen-image-2.1").unwrap().unwrap();
        assert_eq!(set.component("--vae"), Some("vae.safetensors"));
        assert_eq!(set.component("--llm"), Some("te.gguf"));
        assert!(set.has_component_set());
        assert!(!set.serves_image_edits());
        assert_eq!(s.list_models().unwrap().len(), 2);
    }

    #[test]
    fn unit__migrate_v5_db__folds_legacy_component_cols_and_drops_them() {
        // v5 wrote three fixed columns (--vae/--llm/--llm_vision paths).
        // v6 folds them into the flag-keyed components JSON and drops
        // the legacy columns — FLUX needs --t5xxl/--clip_l, which the
        // fixed shape could never carry.
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        std::fs::create_dir_all(&dirs.data_dir).unwrap();
        {
            let conn = Connection::open(dirs.db_file()).unwrap();
            conn.execute_batch(
                "CREATE TABLE models (
                    name       TEXT PRIMARY KEY,
                    repo       TEXT NOT NULL,
                    quant      TEXT NOT NULL,
                    path       TEXT NOT NULL,
                    bytes      INTEGER NOT NULL,
                    sha256     TEXT,
                    mmproj_path TEXT,
                    vae_path     TEXT,
                    llm_path     TEXT,
                    llm_vision_path TEXT,
                    shards     INTEGER NOT NULL DEFAULT 1,
                    arch       TEXT,
                    params     REAL,
                    ctx_train  INTEGER,
                    pulled_at  INTEGER NOT NULL
                );
                INSERT INTO models (name, repo, quant, path, bytes, shards, pulled_at,
                                    vae_path, llm_path, llm_vision_path)
                VALUES ('qwen', 'r', 'Q4_K_M', 'p', 7, 1, 1,
                        'vae.safetensors', 'te.gguf', 'vis.gguf'),
                       ('text-m', 'r', 'Q4_K_M', 't', 8, 1, 1, NULL, NULL, NULL);
                PRAGMA user_version = 5;",
            )
            .unwrap();
        }
        let s = Store::open(&dirs).unwrap();
        let qwen = s.get_model("qwen").unwrap().unwrap();
        assert_eq!(
            qwen.components,
            vec![
                ComponentFile::new("--vae", "vae.safetensors"),
                ComponentFile::new("--llm", "te.gguf"),
                ComponentFile::new("--llm_vision", "vis.gguf"),
            ]
        );
        assert!(qwen.serves_image_edits());
        let text = s.get_model("text-m").unwrap().unwrap();
        assert!(text.components.is_empty());
        let mut cols = s.conn().prepare("PRAGMA table_info(models)").unwrap();
        let mut rows = cols.query([]).unwrap();
        while let Some(r) = rows.next().unwrap() {
            let name: String = r.get(1).unwrap();
            assert!(
                !name.ends_with("_path") || name == "mmproj_path",
                "legacy column {name} survived the fold"
            );
        }
    }
}
