use std::str::FromStr;

use rusqlite::{Connection, params};

use crate::dirs::BlazarDirs;
use crate::engine_kind::EngineKind;
use crate::error::{CoreError, CoreResult};

/// SQLite-backed persistent state (WAL mode). One DB file at
/// `<data_dir>/blazar.db`. Schema versioned via `PRAGMA user_version`.
///
/// Tables:
/// - engines:  installed llama-server builds + capability manifest JSON
/// - models:   pulled GGUFs (plain files, no blob store)
/// - profiles: per-(model, engine) launch argv + benchmark results
/// - loras:    `LoRA` adapters per model
/// - jobs / `job_events`: durable background-job spine (v8) — every async
///   surface (audio, image, video) records its job here so records,
///   events, and terminal results survive gateway restarts
/// - responses: durable Responses-API registry (v8) — `previous_response_id`
///   chaining survives restarts
/// - `model_caps`: per-(model, engine) capability certificates (v8)
#[derive(Debug)]
pub struct Store {
    conn: Connection,
}

// v11 adds `bench_results` (plain-bench payloads persisted by
// `blazar bench`; scorecards fall back to it when no tuned profile
// carries benchmark_json). Purely additive — CREATE TABLE IF NOT
// EXISTS inside the migration batch covers both fresh and old stores.
// v12 adds `completion_cards` (slim OpenAI chat-completion metadata
// cards — the id→metadata mapping behind POST /v1/chat/completions/{id}).
// Cards exist ONLY for completions created with a `metadata` field;
// the cap prunes oldest-first so the table stays bounded.
const SCHEMA_VERSION: i32 = 12;

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
    pulled_at  INTEGER NOT NULL,
    last_used_at INTEGER NOT NULL DEFAULT 0
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
    ctx   INTEGER NOT NULL DEFAULT 0,
    kind  TEXT NOT NULL DEFAULT 'llamacpp', -- lane that produced the row
    ttft_ms REAL,                    -- HTTP lane: first-token latency
    gen_tps REAL,                    -- HTTP lane: generation t/s (usage)
    prompt_tps REAL,                 -- HTTP lane: prompt t/s (usage)
    images_per_sec REAL,             -- HTTP image lane: images/s
    detail TEXT                      -- HTTP lane: probe params JSON
);
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,      -- audio | image | video | batch
    model         TEXT,               -- serving model, when one is named
    state         TEXT NOT NULL,      -- queued | running | completed | failed | cancelled | abandoned
    request_json  TEXT NOT NULL DEFAULT '{}', -- kind-specific submit descriptor
    result_json   TEXT,               -- terminal payload (small, inline)
    error         TEXT,
    artifact_path TEXT,               -- larger result/input file, relative to data_dir
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
CREATE INDEX IF NOT EXISTS idx_jobs_updated ON jobs(updated_at);
CREATE TABLE IF NOT EXISTS job_events (
    seq       INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id    TEXT NOT NULL,
    ts        INTEGER NOT NULL,
    kind      TEXT NOT NULL,          -- created | started | completed | failed | cancelled | abandoned
    data_json TEXT                    -- optional event detail
);
CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events(job_id, seq);
CREATE TABLE IF NOT EXISTS responses (
    id            TEXT PRIMARY KEY,   -- resp_... client-chainable id
    model         TEXT NOT NULL,
    input_json    TEXT NOT NULL,
    output_json   TEXT NOT NULL,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    ts            INTEGER NOT NULL,
    conversation  TEXT NOT NULL DEFAULT '',  -- v10: grouping key for /v1/conversations
    body_json     TEXT                            -- v10: exact completed body (background mode fidelity)
);
CREATE TABLE IF NOT EXISTS model_caps (
    model      TEXT NOT NULL,
    engine_tag TEXT NOT NULL,
    tested_at  INTEGER NOT NULL,
    caps_json  TEXT NOT NULL,
    PRIMARY KEY (model, engine_tag)
);
CREATE TABLE IF NOT EXISTS bench_results (
    model        TEXT NOT NULL,
    engine_tag   TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    updated_at   INTEGER NOT NULL,
    PRIMARY KEY (model, engine_tag)
);
CREATE TABLE IF NOT EXISTS completion_cards (
    id        TEXT PRIMARY KEY,
    model     TEXT NOT NULL,
    created   INTEGER NOT NULL,
    metadata  TEXT NOT NULL,
    updated_at INTEGER NOT NULL
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

/// One HTTP-lane bench measurement (non-llamacpp engines: prompt/tg via
/// streaming + usage stats, images via time-to-image). Fields the lane
/// did not measure stay `None` — the image lane has no TTFT, the text
/// lane has no images/s.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HttpBenchRecord {
    /// Serving kind that produced the row (schema default `llamacpp`
    /// keeps legacy rows readable without a join).
    pub kind: String,
    pub engine_tag: String,
    pub model: String,
    pub ttft_ms: Option<f64>,
    pub gen_tps: Option<f64>,
    pub prompt_tps: Option<f64>,
    pub images_per_sec: Option<f64>,
    /// Probe parameters (reps, `max_tokens`, size/steps) so a stored row
    /// is reproducible without reading the code that wrote it.
    pub detail_json: Option<String>,
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

/// UTC calendar date (`YYYY-MM-DD`) for an epoch-seconds stamp.
/// Scorecards and capability certificates cite verification dates;
/// sharing one implementation keeps the CLI and gateway citing the
/// same calendar without a date dependency. Civil-calendar inverse of
/// the epoch day count (Hinnant's `civil_from_days`); pinned against
/// known dates including a leap day and a non-leap century year.
#[must_use]
pub fn epoch_to_utc_date(secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
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
    /// Last time this model went resident (spawn admission). Stays fresh
    /// while loaded by definition; ages out only when nothing re-spawns it.
    /// Backfilled from `pulled_at` at v9 migration.
    #[serde(default)]
    pub last_used_at: i64,
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

/// MLX-format signal: the pull convention carries the marker in the
/// repo (`mlx-community/...`) or name/path token. Callers gate on the
/// row being a directory (the safetensors class) exactly as they do
/// for `safetensors` — a GGUF file from an mlx repo is a GGUF row.
#[must_use]
pub fn mlx_signal(name: &str, repo: &str, path: &str) -> bool {
    [name, repo, path].iter().any(|s| {
        s.to_ascii_lowercase()
            .split(|c: char| !c.is_ascii_alphanumeric())
            .any(|tok| tok == "mlx")
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

    /// True for MLX-format community dirs. The token marker (repo
    /// `mlx-community/...`, or an `mlx` name/path token) plus the
    /// caller's dir gate is the discriminator — MLX dirs are
    /// quantized-safetensors-shaped, so routing consults this BEFORE
    /// [`Self::is_quantized_safetensors`].
    #[must_use]
    pub fn is_mlx(&self) -> bool {
        mlx_signal(&self.name, &self.repo, &self.path)
    }

    /// Local path for a component flag, if the set carries it.
    #[must_use]
    pub fn component(&self, flag: &str) -> Option<&str> {
        self.components
            .iter()
            .find(|c| c.flag == flag)
            .map(|c| c.path.as_str())
    }

    /// Diffusion-domain routing marker: a row carrying any component
    /// set serves on the sdcpp lane — component families carry
    /// `--vae`/text-encoder files, standalone checkpoints (SD 1.5,
    /// SDXL) carry a self-referencing `--model` entry. Every text row
    /// has no set at all.
    #[must_use]
    pub fn has_component_set(&self) -> bool {
        !self.components.is_empty()
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

#[derive(Debug, Clone)]
/// One row of the `bench_results` table — the durable plain-bench
/// history `blazar bench` upserts (model + lane keyed).
pub struct BenchResultRow {
    pub model: String,
    pub engine_tag: String,
    pub payload_json: String,
    pub updated_at: i64,
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

/// One durable background job (audio/image/video async work). `state` is
/// the persistence-level truth; the gateway keeps live handles alongside.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct JobRow {
    pub id: String,
    pub kind: String,
    pub model: Option<String>,
    pub state: String,
    pub request_json: String,
    pub result_json: Option<String>,
    pub error: Option<String>,
    pub artifact_path: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

/// One append-only job history event (the "story" behind a job).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct JobEventRow {
    pub seq: i64,
    pub job_id: String,
    pub ts: i64,
    pub kind: String,
    pub data_json: Option<String>,
}

/// Slim chat-completion card: the id→metadata mapping that lets
/// `POST /v1/chat/completions/{id}` update metadata after the fact.
/// Stored only for completions whose create request carried
/// `metadata` (non-stream; streams cannot be persisted after the
/// fact — bytes are already on the wire).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CompletionCardRow {
    pub id: String,
    pub model: String,
    pub created: i64,
    pub metadata_json: String,
    pub updated_at: i64,
}

/// Persisted Responses-API entry: enough to reconstruct a
/// `previous_response_id` chain after a restart.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StoredResponseRow {
    pub id: String,
    pub model: String,
    pub input_json: String,
    pub output_json: String,
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub ts: i64,
    /// v10: grouping key set when the request carried `conversation`
    /// (or inherited it through a `previous_response_id` chain).
    pub conversation: String,
    /// v10: the exact completed body JSON when the gateway kept it
    /// (background mode always; foreground when cheap). `None` on
    /// pre-v10 rows — GET reconstructs the shape instead.
    pub body_json: Option<String>,
}

#[must_use]
pub fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(0))
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
            // v10→v11: `bench_results` (plain-bench persistence) is
            // CREATE-TABLE-IF-NOT-EXISTS in SCHEMA_SQL — nothing more
            // to do here; the version bump replays the batch on
            // existing stores.
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
            // v9→v10: responses grew `conversation` (listing handle for
            // /v1/conversations) and `body_json` (the exact completed body,
            // so background-mode polls return what the foreground call would
            // have received). Both nullable-or-default — no backfill owed:
            // pre-v10 rows keep the reconstructed-shape GET contract.
            let resp_cols = self.table_columns("responses")?;
            if !resp_cols.contains(&"conversation".to_string()) {
                self.conn.execute(
                    "ALTER TABLE responses ADD COLUMN conversation TEXT NOT NULL DEFAULT ''",
                    [],
                )?;
            }
            if !resp_cols.contains(&"body_json".to_string()) {
                self.conn
                    .execute("ALTER TABLE responses ADD COLUMN body_json TEXT", [])?;
            }
            // Index lives here, not in SCHEMA_SQL: on an upgrade database
            // SCHEMA_SQL runs before these ALTERs and the index references
            // the not-yet-added column (caught by the v9→v10 upgrade pin).
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_responses_conversation ON responses(conversation, ts)",
                [],
            )?;
            // v8→v9: models gained `last_used_at` (disk intelligence:
            // `blazar prune --unused` ages models out by spawn activity).
            // Fresh databases get it from SCHEMA_SQL; v8 databases take the
            // ALTER plus a backfill from `pulled_at` so an existing model
            // is never instantly "unused" the day this ships.
            if !model_cols.contains(&"last_used_at".to_string()) {
                self.conn.execute(
                    "ALTER TABLE models ADD COLUMN last_used_at INTEGER NOT NULL DEFAULT 0",
                    [],
                )?;
                self.conn.execute(
                    "UPDATE models SET last_used_at = pulled_at WHERE last_used_at = 0",
                    [],
                )?;
            }
            // v6→v7: bench_history grew HTTP-lane columns (kind + probe
            // metrics). Fresh databases get them from SCHEMA_SQL; v6
            // databases take additive ALTERs and keep every legacy row
            // (NULL metrics mark rows the llama-bench lane wrote).
            let bench_cols = self.table_columns("bench_history")?;
            if !bench_cols.contains(&"kind".to_string()) {
                self.conn.execute(
                    "ALTER TABLE bench_history ADD COLUMN kind TEXT NOT NULL DEFAULT 'llamacpp'",
                    [],
                )?;
                for col in [
                    "ttft_ms REAL",
                    "gen_tps REAL",
                    "prompt_tps REAL",
                    "images_per_sec REAL",
                    "detail TEXT",
                ] {
                    let (name, ty) = col.split_once(' ').unwrap_or((col, "TEXT"));
                    self.conn.execute(
                        &format!("ALTER TABLE bench_history ADD COLUMN {name} {ty}"),
                        [],
                    )?;
                }
            }
            self.conn
                .pragma_update(None, "user_version", SCHEMA_VERSION)?;
        }
        Ok(())
    }

    // ---- durable jobs (v8) -------------------------------------------

    /// Insert a job row plus its `created` event in one transaction — a
    /// crash between the two must never leave an event-less job.
    pub fn insert_job(&self, job: &JobRow) -> CoreResult<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO jobs (id, kind, model, state, request_json, result_json, error, artifact_path, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                job.id,
                job.kind,
                job.model,
                job.state,
                job.request_json,
                job.result_json,
                job.error,
                job.artifact_path,
                job.created_at,
                job.updated_at
            ],
        )?;
        tx.execute(
            "INSERT INTO job_events (job_id, ts, kind) VALUES (?1, ?2, 'created')",
            params![job.id, job.created_at],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Transition a job's state (and terminal payload/error). Returns
    /// `false` when no such job row exists — callers decide whether that
    /// is a 404 or an in-memory-only straggler.
    pub fn set_job_state(
        &self,
        id: &str,
        state: &str,
        result_json: Option<&str>,
        error: Option<&str>,
    ) -> CoreResult<bool> {
        // One-way close: only a still-open row (queued/running) may flip
        // state. First terminal state wins — a cancelled row can never be
        // silently resurrected to completed by a late racer (e.g. a
        // doctor/audio task that outlived its cancel).
        let n = self.conn.execute(
            "UPDATE jobs SET state = ?2, result_json = COALESCE(?3, result_json), error = ?4, updated_at = ?5 \
             WHERE id = ?1 AND state IN ('queued','running')",
            params![id, state, result_json, error, unix_now()],
        )?;
        Ok(n == 1)
    }

    /// Attach a spilled-result artifact path to a job row (results too
    /// large for the inline column). Separate from `set_job_state` so a
    /// transition never needs to know about disk spillover.
    pub fn set_job_artifact(&self, id: &str, artifact_path: &str) -> CoreResult<bool> {
        let n = self.conn.execute(
            "UPDATE jobs SET artifact_path = ?2, updated_at = ?3 WHERE id = ?1",
            params![id, artifact_path, unix_now()],
        )?;
        Ok(n == 1)
    }

    pub fn append_job_event(
        &self,
        job_id: &str,
        kind: &str,
        data_json: Option<&str>,
    ) -> CoreResult<()> {
        self.conn.execute(
            "INSERT INTO job_events (job_id, ts, kind, data_json) VALUES (?1, ?2, ?3, ?4)",
            params![job_id, unix_now(), kind, data_json],
        )?;
        Ok(())
    }

    pub fn get_job(&self, id: &str) -> CoreResult<Option<JobRow>> {
        self.conn
            .query_row(
                "SELECT id, kind, model, state, request_json, result_json, error, artifact_path, created_at, updated_at FROM jobs WHERE id = ?1",
                params![id],
                |r| {
                    Ok(JobRow {
                        id: r.get(0)?,
                        kind: r.get(1)?,
                        model: r.get(2)?,
                        state: r.get(3)?,
                        request_json: r.get(4)?,
                        result_json: r.get(5)?,
                        error: r.get(6)?,
                        artifact_path: r.get(7)?,
                        created_at: r.get(8)?,
                        updated_at: r.get(9)?,
                    })
                },
            )
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other.into()),
            })
    }

    /// Newest-first job listing. `state`/`kind` filter, `limit` clamped
    /// to 1..=1000 (the admin plane, not a firehose).
    pub fn list_jobs(
        &self,
        state: Option<&str>,
        kind: Option<&str>,
        limit: u64,
    ) -> CoreResult<Vec<JobRow>> {
        let limit = limit.clamp(1, 1000);
        let mut sql = String::from(
            "SELECT id, kind, model, state, request_json, result_json, error, artifact_path, created_at, updated_at FROM jobs",
        );
        let mut conds: Vec<&str> = Vec::new();
        // Owned SQL values: filter strings are pattern-scoped borrows, so the
        // bind vec must own its data to outlive the if-let arms.
        let mut binds: Vec<rusqlite::types::Value> = Vec::new();
        if let Some(s) = state {
            conds.push("state = ?");
            binds.push(rusqlite::types::Value::Text(s.to_string()));
        }
        if let Some(k) = kind {
            conds.push("kind = ?");
            binds.push(rusqlite::types::Value::Text(k.to_string()));
        }
        if !conds.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&conds.join(" AND "));
        }
        sql.push_str(" ORDER BY updated_at DESC, id LIMIT ?");
        // limit is clamped to 1..=1000 by the caller; try_from keeps the
        // cast total instead of `as` truncation.
        let limit_i = i64::try_from(limit).unwrap_or(1000);
        binds.push(rusqlite::types::Value::Integer(limit_i));
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(binds), |r| {
            Ok(JobRow {
                id: r.get(0)?,
                kind: r.get(1)?,
                model: r.get(2)?,
                state: r.get(3)?,
                request_json: r.get(4)?,
                result_json: r.get(5)?,
                error: r.get(6)?,
                artifact_path: r.get(7)?,
                created_at: r.get(8)?,
                updated_at: r.get(9)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Job events, oldest-first (the story reads in order).
    pub fn job_events(&self, job_id: &str, limit: u64) -> CoreResult<Vec<JobEventRow>> {
        let limit = i64::try_from(limit.clamp(1, 1000)).unwrap_or(1000);
        let mut stmt = self.conn.prepare(
            "SELECT seq, job_id, ts, kind, data_json FROM (
                SELECT * FROM job_events WHERE job_id = ?1 ORDER BY seq DESC LIMIT ?2
            ) ORDER BY seq ASC",
        )?;
        let rows = stmt.query_map(params![job_id, limit], |r| {
            Ok(JobEventRow {
                seq: r.get(0)?,
                job_id: r.get(1)?,
                ts: r.get(2)?,
                kind: r.get(3)?,
                data_json: r.get(4)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Boot-time honesty sweep: every job still `queued`/`running` whose
    /// `updated_at` predates `cutoff_unix` belonged to a dead gateway —
    /// mark it `abandoned` with a teaching error and an event. Two callers,
    /// two cutoffs: at boot the cutoff is `now - grace` (the grace window
    /// keeps a concurrently-live second daemon from reaping the other's
    /// fresh work), and a delayed second pass cuts at this daemon's start
    /// time — catching rows whose last update fell inside the grace window
    /// (killed less than `grace` before boot) which the first pass
    /// deliberately spared. This daemon's own jobs always have
    /// `updated_at >= its start`, so live work is never reaped.
    /// Returns the swept ids (for the boot log).
    pub fn sweep_jobs_updated_before(&self, cutoff_unix: i64) -> CoreResult<Vec<String>> {
        let swept: Vec<String> = {
            let mut stmt = self.conn.prepare(
                "SELECT id FROM jobs WHERE state IN ('queued','running') AND updated_at < ?1",
            )?;
            let rows = stmt.query_map(params![cutoff_unix], |r| r.get::<_, String>(0))?;
            rows.collect::<Result<Vec<_>, _>>()?
        };
        let now = unix_now();
        for id in &swept {
            self.conn.execute(
                "UPDATE jobs SET state = 'abandoned', error = 'gateway restarted while the job was in flight — resubmit to rerun; completed artifacts were kept', updated_at = ?2 WHERE id = ?1",
                params![id, now],
            )?;
            self.conn.execute(
                "INSERT INTO job_events (job_id, ts, kind, data_json) VALUES (?1, ?2, 'abandoned', ?3)",
                params![id, now, r#"{"reason":"gateway_restart"}"#],
            )?;
        }
        Ok(swept)
    }

    /// Bounded retention: terminal jobs older than `older_than_secs`
    /// (and their events) are deleted. Called at boot — the table must
    /// not grow forever.
    pub fn prune_jobs(&self, older_than_secs: i64) -> CoreResult<u64> {
        let cutoff = unix_now() - older_than_secs.max(0);
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "DELETE FROM job_events WHERE job_id IN (
                SELECT id FROM jobs WHERE state IN ('completed','failed','cancelled','abandoned') AND updated_at < ?1
            )",
            params![cutoff],
        )?;
        let n = tx.execute(
            "DELETE FROM jobs WHERE state IN ('completed','failed','cancelled','abandoned') AND updated_at < ?1",
            params![cutoff],
        )?;
        tx.commit()?;
        Ok(u64::try_from(n).unwrap_or(0))
    }

    // ---- durable responses (v8) --------------------------------------

    pub fn put_response(&self, r: &StoredResponseRow) -> CoreResult<()> {
        self.conn.execute(
            "INSERT INTO responses (id, model, input_json, output_json, input_tokens, output_tokens, ts, conversation, body_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(id) DO UPDATE SET
               model = excluded.model, input_json = excluded.input_json,
               output_json = excluded.output_json, input_tokens = excluded.input_tokens,
               output_tokens = excluded.output_tokens, ts = excluded.ts,
               conversation = excluded.conversation, body_json = excluded.body_json",
            params![
                r.id,
                r.model,
                r.input_json,
                r.output_json,
                r.input_tokens,
                r.output_tokens,
                r.ts,
                r.conversation,
                r.body_json
            ],
        )?;
        Ok(())
    }

    pub fn get_response(&self, id: &str) -> CoreResult<Option<StoredResponseRow>> {
        self.conn
            .query_row(
                "SELECT id, model, input_json, output_json, input_tokens, output_tokens, ts, conversation, body_json FROM responses WHERE id = ?1",
                params![id],
                |r| {
                    Ok(StoredResponseRow {
                        id: r.get(0)?,
                        model: r.get(1)?,
                        input_json: r.get(2)?,
                        output_json: r.get(3)?,
                        input_tokens: r.get(4)?,
                        output_tokens: r.get(5)?,
                        ts: r.get(6)?,
                        conversation: r.get(7)?,
                        body_json: r.get(8)?,
                    })
                },
            )
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other.into()),
            })
    }

    /// Every stored response grouped under `conversation`, oldest first —
    /// the listing behind `GET /v1/conversations/{id}`.
    pub fn list_conversation(&self, conversation: &str) -> CoreResult<Vec<StoredResponseRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, model, input_json, output_json, input_tokens, output_tokens, ts, conversation, body_json
             FROM responses WHERE conversation = ?1 ORDER BY ts ASC, id ASC",
        )?;
        let rows = stmt.query_map(params![conversation], |r| {
            Ok(StoredResponseRow {
                id: r.get(0)?,
                model: r.get(1)?,
                input_json: r.get(2)?,
                output_json: r.get(3)?,
                input_tokens: r.get(4)?,
                output_tokens: r.get(5)?,
                ts: r.get(6)?,
                conversation: r.get(7)?,
                body_json: r.get(8)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Prune every response in `conversation` (DELETE endpoint). Returns
    /// the number of rows removed so the API can say what it did.
    pub fn delete_conversation(&self, conversation: &str) -> CoreResult<u64> {
        let n = self.conn.execute(
            "DELETE FROM responses WHERE conversation = ?1",
            params![conversation],
        )?;
        Ok(u64::try_from(n).unwrap_or(0))
    }

    // ---- completion metadata cards (v12) ------------------------------

    /// Upper bound on persisted cards. Completions vastly outnumber
    /// stored responses (every metadata-tagged chat passes here), so the
    /// cap prunes oldest-updated-first instead of growing forever.
    const COMPLETION_CARD_CAP: i64 = 4096;

    /// Persist (or overwrite) the slim card for a completion created
    /// with `metadata`. Prunes beyond the cap in the same transaction
    /// window — a card that ages out simply stops being updatable, which
    /// the update endpoint teaches.
    pub fn put_completion_card(&self, card: &CompletionCardRow) -> CoreResult<()> {
        self.conn.execute(
            "INSERT INTO completion_cards (id, model, created, metadata, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(id) DO UPDATE SET
               model = excluded.model, created = excluded.created,
               metadata = excluded.metadata, updated_at = excluded.updated_at",
            params![
                card.id,
                card.model,
                card.created,
                card.metadata_json,
                card.updated_at
            ],
        )?;
        self.conn.execute(
            "DELETE FROM completion_cards WHERE id IN (
                SELECT id FROM completion_cards
                 ORDER BY updated_at DESC, id DESC
                 LIMIT -1 OFFSET ?1
            )",
            params![Self::COMPLETION_CARD_CAP],
        )?;
        Ok(())
    }

    /// Update only the metadata of an existing card. `Ok(None)` = no
    /// card under this id (never created with metadata, streamed,
    /// expired, or pruned) — the caller turns that into the teaching 404.
    pub fn update_completion_card(
        &self,
        id: &str,
        metadata_json: &str,
        updated_at: i64,
    ) -> CoreResult<Option<CompletionCardRow>> {
        let n = self.conn.execute(
            "UPDATE completion_cards SET metadata = ?2, updated_at = ?3 WHERE id = ?1",
            params![id, metadata_json, updated_at],
        )?;
        if n == 0 {
            return Ok(None);
        }
        self.get_completion_card(id)
    }

    pub fn get_completion_card(&self, id: &str) -> CoreResult<Option<CompletionCardRow>> {
        self.conn
            .query_row(
                "SELECT id, model, created, metadata, updated_at FROM completion_cards WHERE id = ?1",
                params![id],
                |r| {
                    Ok(CompletionCardRow {
                        id: r.get(0)?,
                        model: r.get(1)?,
                        created: r.get(2)?,
                        metadata_json: r.get(3)?,
                        updated_at: r.get(4)?,
                    })
                },
            )
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other.into()),
            })
    }

    /// Mirror of the in-memory registry contract: entries older than
    /// `ttl_secs` die; past `cap` rows the oldest die. Keeps SQLite the
    /// same shape the RAM registry always had.
    pub fn prune_responses(&self, ttl_secs: u64, cap: usize) -> CoreResult<()> {
        let now = unix_now();
        let ttl = i64::try_from(ttl_secs).unwrap_or(i64::MAX);
        self.conn.execute(
            "DELETE FROM responses WHERE ts < ?1",
            params![now.saturating_sub(ttl)],
        )?;
        let excess: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM responses", [], |r| r.get::<_, i64>(0))?
            - i64::try_from(cap).unwrap_or(i64::MAX);
        if excess > 0 {
            self.conn.execute(
                "DELETE FROM responses WHERE id IN (
                    SELECT id FROM responses ORDER BY ts ASC LIMIT ?1
                )",
                params![excess],
            )?;
        }
        Ok(())
    }

    // ---- model capability certificates (v8) ---------------------------

    pub fn put_model_caps(&self, model: &str, engine_tag: &str, caps_json: &str) -> CoreResult<()> {
        self.conn.execute(
            "INSERT INTO model_caps (model, engine_tag, tested_at, caps_json)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(model, engine_tag) DO UPDATE SET
               tested_at = excluded.tested_at, caps_json = excluded.caps_json",
            params![model, engine_tag, unix_now(), caps_json],
        )?;
        Ok(())
    }

    pub fn get_model_caps(&self, model: &str) -> CoreResult<Option<(String, String)>> {
        // Latest verification wins: certs are keyed (model, engine_tag),
        // so a model served by two lanes carries two rows — admission
        // and scorecards must read the most recent measurement, not
        // whichever row SQLite happens to return first.
        self.conn
            .query_row(
                "SELECT engine_tag, caps_json FROM model_caps WHERE model = ?1 \
                 ORDER BY tested_at DESC LIMIT 1",
                params![model],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
            )
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other.into()),
            })
    }

    /// Same record as [`get_model_caps`] with its `tested_at` epoch stamp —
    /// scorecards cite the verification date, so it travels with the caps.
    pub fn get_model_caps_dated(&self, model: &str) -> CoreResult<Option<(String, i64, String)>> {
        self.conn
            .query_row(
                "SELECT engine_tag, tested_at, caps_json FROM model_caps WHERE model = ?1 \
                 ORDER BY tested_at DESC LIMIT 1",
                params![model],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, i64>(1)?,
                        r.get::<_, String>(2)?,
                    ))
                },
            )
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other.into()),
            })
    }

    // ---- bench result records -----------------------------------------
    // `blazar bench` upserts its measured rows here (tune-owned launch
    // profiles keep their richer payload in `profiles.benchmark_json`;
    // this table is the plain-bench history scorecards fall back to).

    pub fn put_bench_result(
        &self,
        model: &str,
        engine_tag: &str,
        payload_json: &str,
    ) -> CoreResult<()> {
        self.conn.execute(
            "INSERT INTO bench_results (model, engine_tag, payload_json, updated_at)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(model, engine_tag) DO UPDATE SET
               payload_json = excluded.payload_json, updated_at = excluded.updated_at",
            params![model, engine_tag, payload_json, unix_now()],
        )?;
        Ok(())
    }

    /// Latest plain-bench payload for a model — `engine_tag = Some`
    /// pins the lane, `None` takes the most recent across lanes.
    pub fn latest_bench_result(
        &self,
        model: &str,
        engine_tag: Option<&str>,
    ) -> CoreResult<Option<(String, String, i64)>> {
        self.conn
            .query_row(
                "SELECT engine_tag, payload_json, updated_at FROM bench_results
                 WHERE model = ?1 AND (?2 IS NULL OR engine_tag = ?2)
                 ORDER BY updated_at DESC LIMIT 1",
                params![model, engine_tag],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, i64>(2)?,
                    ))
                },
            )
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other.into()),
            })
    }

    /// Every plain-bench payload on record — one query for the console's
    /// Benchmarks view instead of a per-model lookup per row.
    pub fn list_bench_results(&self) -> CoreResult<Vec<BenchResultRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT model, engine_tag, payload_json, updated_at FROM bench_results
             ORDER BY model, engine_tag",
        )?;
        let rows = stmt
            .query_map([], |r| {
                Ok(BenchResultRow {
                    model: r.get(0)?,
                    engine_tag: r.get(1)?,
                    payload_json: r.get(2)?,
                    updated_at: r.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Every tuned launch profile on record — pairs with
    /// `list_bench_results` to show measured vs tuned per model + lane.
    pub fn list_profiles(&self) -> CoreResult<Vec<ProfileRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT model_name, engine_tag, args_hash, args_json, benchmark_json, updated_at
             FROM profiles ORDER BY model_name, engine_tag",
        )?;
        let rows = stmt
            .query_map([], |r| {
                Ok(ProfileRow {
                    model_name: r.get(0)?,
                    engine_tag: r.get(1)?,
                    args_hash: r.get(2)?,
                    args_json: r.get(3)?,
                    benchmark_json: r.get(4)?,
                    updated_at: r.get(5)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
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

    /// Boot-time self-heal for the active flag. Two states heal:
    ///
    /// 1. Lazy-active: the flagged row is an audio lane (whisper, piper)
    ///    that a pre-guard store flipped on — a text boot over that
    ///    store picks no adapter and dies. Demote it, then fall through
    ///    to the pick (an audio-only box lands at zero-active, its
    ///    pre-lane state; serving through `/v1/audio` never read the
    ///    flag and keeps working).
    /// 2. Zero-active: no row holds the flag but serving-capable
    ///    engines are installed (the class F120's transaction fixed
    ///    mid-write; an update interrupted before its activation step
    ///    lands here): activate the best one — llamacpp lanes first
    ///    (router mode's native lane), newest within the kind — so one
    ///    stale flag can never brick `serve` while good engines sit
    ///    installed.
    ///
    /// Returns the activated tag, or None when the store needs no
    /// healing. Lazy lanes (whisper, piper, sdcpp) never claim the slot:
    /// the active row is the serving adapter and must stay a text lane.
    pub fn heal_active_engine(&self) -> CoreResult<Option<String>> {
        if let Some(active) = self.active_engine()? {
            if !active.kind.is_lazy_lane() {
                return Ok(None);
            }
            self.conn.execute(
                "UPDATE engines SET active = 0 WHERE tag = ?1",
                params![active.tag],
            )?;
            tracing::info!(
                "demoted lazy {} row {} off the serving-active slot",
                active.kind.as_str(),
                active.tag
            );
        }
        let rank = |k: crate::engine_kind::EngineKind| match k {
            crate::engine_kind::EngineKind::LlamaCpp => 2,
            crate::engine_kind::EngineKind::MistralRs | crate::engine_kind::EngineKind::Sglang => 1,
            _ => 0,
        };
        let pick = self
            .list_engines()?
            .into_iter()
            .filter(|r| rank(r.kind) > 0)
            .max_by_key(|r| (rank(r.kind), r.installed_at));
        match pick {
            Some(row) => {
                self.set_active_engine(&row.tag)?;
                Ok(Some(row.tag))
            }
            None => Ok(None),
        }
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

    /// Kind of a specific engine tag, retired or active — capability
    /// certificates store only the tag, and tags are arbitrary build
    /// strings (`b11370-cuda`, `mlx-0.32.0`); the kind comparison the
    /// certificate gate needs lives here. `None` when the row is gone
    /// or its kind no longer parses (callers fail open).
    pub fn engine_kind_of_tag(&self, tag: &str) -> CoreResult<Option<EngineKind>> {
        let mut stmt = self
            .conn
            .prepare("SELECT kind FROM engines WHERE tag = ?")?;
        let mut rows = stmt.query([tag])?;
        if let Some(r) = rows.next()? {
            let raw: String = r.get(0)?;
            Ok(EngineKind::from_str(&raw).ok())
        } else {
            Ok(None)
        }
    }

    pub fn active_engine(&self) -> CoreResult<Option<EngineRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT tag, asset, sha256, installed_at, active, manifest, kind FROM engines WHERE active = 1",
        )?;
        let mut rows = stmt.query([])?;
        while let Some(r) = rows.next()? {
            match read_engine_row(r)? {
                EngineRowRead::Known(row) => return Ok(Some(row)),
                // An active row the binary cannot parse is treated as
                // absent: heal_active_engine then claims the slot for
                // the newest known serving lane instead of bricking
                // boot. The row itself is never touched — a newer
                // binary parses it again.
                EngineRowRead::UnknownKind { tag, kind } => tracing::warn!(
                    tag = %tag,
                    kind = %kind,
                    "active engine row kind unknown to this binary — treated as absent, healing picks a known lane"
                ),
            }
        }
        Ok(None)
    }

    pub fn list_engines(&self) -> CoreResult<Vec<EngineRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT tag, asset, sha256, installed_at, active, manifest, kind FROM engines ORDER BY installed_at DESC, rowid DESC",
        )?;
        let mut rows = stmt.query([])?;
        let mut engines = Vec::new();
        while let Some(r) = rows.next()? {
            match read_engine_row(r)? {
                EngineRowRead::Known(row) => engines.push(row),
                EngineRowRead::UnknownKind { tag, kind } => tracing::warn!(
                    tag = %tag,
                    kind = %kind,
                    "engine row kind unknown to this binary — quarantined until upgrade"
                ),
            }
        }
        Ok(engines)
    }

    /// Roster rows whose kind this binary cannot parse, as `(tag, kind
    /// text)` pairs, for surfaces that teach the upgrade path (doctor).
    /// The rows stay untouched in the store and parse again under a
    /// newer binary — quarantine is read-side only.
    pub fn unknown_kind_engine_rows(&self) -> CoreResult<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT tag, asset, sha256, installed_at, active, manifest, kind FROM engines ORDER BY installed_at DESC, rowid DESC",
        )?;
        let mut rows = stmt.query([])?;
        let mut quarantined = Vec::new();
        while let Some(r) = rows.next()? {
            if let EngineRowRead::UnknownKind { tag, kind } = read_engine_row(r)? {
                quarantined.push((tag, kind));
            }
        }
        Ok(quarantined)
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
        // `last_used_at = 0` means "never went resident"; a freshly
        // pulled model starts aged at its pull stamp so it is never
        // instantly "unused" (same semantics the v8→v9 migration
        // backfills for pre-existing rows).
        let last_used_at = if m.last_used_at == 0 {
            m.pulled_at
        } else {
            m.last_used_at
        };
        self.conn.execute(
            "INSERT INTO models (name, repo, quant, path, bytes, sha256, mmproj_path, components,
                                 shards, arch, params, ctx_train, pulled_at, last_used_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)
             ON CONFLICT(name) DO UPDATE SET
               repo = excluded.repo, quant = excluded.quant, path = excluded.path,
               bytes = excluded.bytes, sha256 = excluded.sha256,
               mmproj_path = excluded.mmproj_path, components = excluded.components,
               shards = excluded.shards,
               arch = excluded.arch, params = excluded.params,
               ctx_train = excluded.ctx_train, pulled_at = excluded.pulled_at,
               last_used_at = excluded.last_used_at",
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
                m.pulled_at,
                last_used_at
            ],
        )?;
        Ok(())
    }

    pub fn get_model(&self, name: &str) -> CoreResult<Option<ModelRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT name, repo, quant, path, bytes, sha256, mmproj_path, components, shards, arch,
                    params, ctx_train, pulled_at, last_used_at
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
                    params, ctx_train, pulled_at, last_used_at
             FROM models ORDER BY name",
        )?;
        let rows = stmt.query_map([], model_from_row)?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Stamp `last_used_at = now` for a model going resident (spawn
    /// admission touch). Returns whether a row was updated — false is a
    /// caller-side curiosity (name vanished mid-flight), not an error.
    pub fn touch_model_used(&self, name: &str) -> CoreResult<bool> {
        let n = self.conn.execute(
            "UPDATE models SET last_used_at = ?2 WHERE name = ?1",
            params![name, unix_now()],
        )?;
        Ok(n > 0)
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

    /// One HTTP-lane probe outcome. `tg_tokens_per_sec` mirrors
    /// `gen_tps` so the latest-bench readers stay meaningful across
    /// lanes; the kind + detail columns carry what the llama-bench row
    /// never had (dialect, probe parameters, TTFT).
    pub fn record_http_bench(&self, rec: &HttpBenchRecord) -> CoreResult<()> {
        self.conn.execute(
            "INSERT INTO bench_history \
             (ts, engine_tag, model, tg_tokens_per_sec, pp_tokens_per_sec, ctx, \
              kind, ttft_ms, gen_tps, prompt_tps, images_per_sec, detail) \
             VALUES (unixepoch(), ?1, ?2, ?3, 0, 0, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![
                rec.engine_tag,
                rec.model,
                rec.gen_tps.unwrap_or(0.0),
                rec.kind,
                rec.ttft_ms,
                rec.gen_tps,
                rec.prompt_tps,
                rec.images_per_sec,
                rec.detail_json,
            ],
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

/// One roster row read leniently: a kind string this binary does not
/// know (a lane shipped by a newer blazar) surfaces as
/// [`EngineRowRead::UnknownKind`] instead of a hard SQL error, so one
/// forward-written row cannot fail every roster read. Live incident
/// 2026-09-27: an engines row written by a newer binary bricked serve
/// boot at store-open and crash-looped the daemon 518 times.
enum EngineRowRead {
    Known(EngineRow),
    UnknownKind { tag: String, kind: String },
}

fn read_engine_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<EngineRowRead> {
    let tag: String = r.get(0)?;
    let kind_text: String = r.get(6)?;
    let Ok(kind) = EngineKind::from_str(&kind_text) else {
        return Ok(EngineRowRead::UnknownKind {
            tag,
            kind: kind_text,
        });
    };
    Ok(EngineRowRead::Known(EngineRow {
        tag,
        asset: r.get(1)?,
        sha256: r.get(2)?,
        installed_at: r.get(3)?,
        active: r.get::<_, i64>(4)? != 0,
        manifest: r.get(5)?,
        kind,
    }))
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
        // NULL and '' both mean "no projector": hand-edited stores and
        // older writers use the empty-string dialect, and treating
        // Some("") as attached would misclassify the model as multimodal
        // (cache_reuse skip, mmproj lazy/offload flags firing on text
        // models). Same normalization the components field applies above.
        mmproj_path: match r.get::<_, Option<String>>(6)? {
            Some(p) if !p.is_empty() => Some(p),
            _ => None,
        },
        components,
        shards: r.get(8)?,
        arch: r.get(9)?,
        params: r.get(10)?,
        ctx_train: r.get(11)?,
        pulled_at: r.get(12)?,
        last_used_at: r.get(13)?,
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
    fn unit__completion_card__put_update_get_roundtrip() {
        let (_t, s) = tmp_store();
        let card = CompletionCardRow {
            id: "chatcmpl-abc".into(),
            model: "qwen3:14b".into(),
            created: 1_760_000_000,
            metadata_json: r#"{"tag":"eval-42"}"#.into(),
            updated_at: 1_760_000_000,
        };
        s.put_completion_card(&card).unwrap();
        let got = s.get_completion_card("chatcmpl-abc").unwrap().unwrap();
        assert_eq!(got.metadata_json, r#"{"tag":"eval-42"}"#);
        assert_eq!(got.model, "qwen3:14b");

        let upd = s
            .update_completion_card("chatcmpl-abc", r#"{"tag":"eval-43"}"#, 1_760_000_900)
            .unwrap()
            .unwrap();
        assert_eq!(upd.metadata_json, r#"{"tag":"eval-43"}"#);
        assert_eq!(upd.updated_at, 1_760_000_900);

        // Unknown id: Ok(None) — the API layer turns this into the
        // teaching 404, not a store error.
        assert!(
            s.update_completion_card("chatcmpl-nope", "{}", 1)
                .unwrap()
                .is_none()
        );
        assert!(s.get_completion_card("chatcmpl-nope").unwrap().is_none());
    }

    #[test]
    fn integration__completion_card__v11_store_migrates_to_v12() {
        // Simulate a v11 database: completion_cards absent, user_version 11.
        // (Open a real store first, then drop only the new table — a
        // hand-rolled minimal schema would fool the older-step column
        // backfills the migration also replays.)
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        {
            let s = Store::open(&dirs).unwrap();
            s.conn()
                .execute_batch("DROP TABLE completion_cards; PRAGMA user_version = 11;")
                .unwrap();
        }
        // Reopen: migrate() recreates the table, stamps v12, and the
        // card accessors work on the upgraded store.
        let s = Store::open(&dirs).unwrap();
        let v: i32 = s
            .conn()
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(v, SCHEMA_VERSION);
        s.put_completion_card(&CompletionCardRow {
            id: "x".into(),
            model: "m".into(),
            created: 1,
            metadata_json: "{}".into(),
            updated_at: 1,
        })
        .unwrap();
        assert!(s.get_completion_card("x").unwrap().is_some());
    }

    #[test]
    fn unit__store_schema_created__tables_exist() {
        let (_t, s) = tmp_store();
        let n: i64 = s
            .conn()
            .query_row("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('engines','models','profiles','loras','jobs','job_events','responses','model_caps')", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 8);
    }

    fn job_row(id: &str, kind: &str, state: &str, updated_at: i64) -> JobRow {
        JobRow {
            id: id.into(),
            kind: kind.into(),
            model: Some("qwen3-8b".into()),
            state: state.into(),
            request_json: "{}".into(),
            result_json: None,
            error: None,
            artifact_path: None,
            created_at: updated_at,
            updated_at,
        }
    }

    #[test]
    fn unit__jobs__lifecycle_rows_and_events() {
        let (_t, s) = tmp_store();
        s.insert_job(&job_row("job_1", "audio", "queued", 1000))
            .unwrap();
        // insert_job stamps the 'created' event itself — the story starts with one row.
        let events = s.job_events("job_1", 10).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, "created");

        assert!(s.set_job_state("job_1", "running", None, None).unwrap());
        s.append_job_event("job_1", "progress", Some(r#"{"pct":50}"#))
            .unwrap();
        assert!(
            s.set_job_state("job_1", "completed", Some(r#"{"text":"hi"}"#), None)
                .unwrap()
        );
        // Unknown id: the contract returns false, not an error.
        assert!(!s.set_job_state("job_9", "completed", None, None).unwrap());

        let row = s.get_job("job_1").unwrap().unwrap();
        assert_eq!(row.state, "completed");
        assert_eq!(row.result_json.as_deref(), Some(r#"{"text":"hi"}"#));
        let events = s.job_events("job_1", 10).unwrap();
        assert_eq!(
            events.iter().map(|e| e.kind.as_str()).collect::<Vec<_>>(),
            vec!["created", "progress"]
        );

        // Listing filters: state and kind both apply, newest-updated first.
        // job_1's transitions stamped real-clock updated_at, so job_2 must
        // sit strictly in the future to be the newest deterministically.
        s.insert_job(&job_row("job_2", "image", "completed", unix_now() + 10))
            .unwrap();
        let listed = s.list_jobs(Some("completed"), None, 10).unwrap();
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].id, "job_2");
        let listed = s.list_jobs(None, Some("image"), 10).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, "job_2");
        let listed = s.list_jobs(None, None, 1).unwrap();
        assert_eq!(listed.len(), 1);
    }

    #[test]
    fn unit__jobs__boot_sweep_abandons_inflight_with_grace() {
        let (_t, s) = tmp_store();
        let now = unix_now();
        // Stale in-flight row: last touched an hour ago — must be reaped.
        s.insert_job(&job_row("old", "audio", "running", now - 3600))
            .unwrap();
        // Fresh in-flight row: last touched now — a live concurrent daemon
        // must never be reaped by a second gateway's boot sweep.
        s.insert_job(&job_row("fresh", "audio", "running", now))
            .unwrap();
        // Terminal row: sweep does not touch it even when ancient.
        s.insert_job(&job_row("done", "audio", "completed", now - 3600))
            .unwrap();

        let swept = s.sweep_jobs_updated_before(unix_now() - 5).unwrap();
        assert_eq!(swept, vec!["old".to_string()]);

        let old = s.get_job("old").unwrap().unwrap();
        assert_eq!(old.state, "abandoned");
        assert!(old.error.as_deref().unwrap().contains("resubmit"));
        let fresh = s.get_job("fresh").unwrap().unwrap();
        assert_eq!(fresh.state, "running");
        let done = s.get_job("done").unwrap().unwrap();
        assert_eq!(done.state, "completed");
        // The abandonment is part of the story.
        let events = s.job_events("old", 10).unwrap();
        assert!(events.iter().any(|e| e.kind == "abandoned"));
    }

    #[test]
    fn unit__jobs__late_sweep_reaps_grace_window_survivors_only() {
        let (_t, s) = tmp_store();
        let now = unix_now();
        // Killed moments before boot: survived the grace pass (updated_at
        // inside the window), but predates the daemon — the late pass
        // with the daemon-start cutoff must reap it.
        s.insert_job(&job_row("zombie", "doctor", "running", now - 3))
            .unwrap();
        // This daemon's own live work: updated after boot — never reaped,
        // even when a long probe keeps the row silent for minutes.
        s.insert_job(&job_row("live", "doctor", "running", now + 30))
            .unwrap();

        let swept = s.sweep_jobs_updated_before(now).unwrap();
        assert_eq!(swept, vec!["zombie".to_string()]);
        assert_eq!(s.get_job("zombie").unwrap().unwrap().state, "abandoned");
        assert_eq!(s.get_job("live").unwrap().unwrap().state, "running");
    }

    #[test]
    fn unit__jobs__prune_removes_terminal_and_events() {
        let (_t, s) = tmp_store();
        let now = unix_now();
        s.insert_job(&job_row("ancient", "video", "completed", now - 30 * 86400))
            .unwrap();
        s.insert_job(&job_row("recent", "video", "completed", now))
            .unwrap();
        s.insert_job(&job_row(
            "old_inflight",
            "video",
            "running",
            now - 30 * 86400,
        ))
        .unwrap();

        let n = s.prune_jobs(7 * 86400).unwrap();
        assert_eq!(n, 1);
        assert!(s.get_job("ancient").unwrap().is_none());
        // Events die with their job — no orphans.
        assert_eq!(s.job_events("ancient", 10).unwrap(), Vec::new());
        // Recent terminal and old-but-in-flight both survive (the sweep owns in-flight).
        assert!(s.get_job("recent").unwrap().is_some());
        assert!(s.get_job("old_inflight").unwrap().is_some());
    }

    #[test]
    fn unit__responses__put_get_prune_ttl_cap() {
        let (_t, s) = tmp_store();
        let row = |id: &str, ts: i64| StoredResponseRow {
            id: id.into(),
            model: "qwen3-8b".into(),
            input_json: "[]".into(),
            output_json: "[]".into(),
            input_tokens: Some(1),
            output_tokens: Some(2),
            ts,
            conversation: String::new(),
            body_json: None,
        };
        let now = unix_now();
        s.put_response(&row("r_old_ttl", now - 7200)).unwrap();
        s.put_response(&row("r_a", now)).unwrap();
        s.put_response(&row("r_b", now)).unwrap();
        s.put_response(&row("r_c", now)).unwrap();
        s.put_response(&row("r_d", now)).unwrap();

        let got = s.get_response("r_a").unwrap().unwrap();
        assert_eq!(got.model, "qwen3-8b");
        assert!(s.get_response("missing").unwrap().is_none());
        // Upsert on conflict: same id re-put updates, not errors.
        s.put_response(&row("r_a", now + 1)).unwrap();
        assert_eq!(s.get_response("r_a").unwrap().unwrap().ts, now + 1);

        // TTL death + cap trim in one prune: expired row goes, then the
        // oldest of the survivors until 3 rows remain.
        s.prune_responses(3600, 3).unwrap();
        assert!(s.get_response("r_old_ttl").unwrap().is_none());
        assert!(s.get_response("r_a").unwrap().is_some());
        assert!(s.get_response("r_d").unwrap().is_some());
        let n: i64 = s
            .conn()
            .query_row("SELECT COUNT(*) FROM responses", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 3);
    }

    fn engine_row(tag: &str, at: i64, active: bool, kind: EngineKind) -> EngineRow {
        EngineRow {
            tag: tag.into(),
            asset: "cpu".into(),
            sha256: "x".into(),
            installed_at: at,
            active,
            // Manifest JSON is opaque at this layer — the runtime layer
            // parses it; heal only reads kind/active/installed_at.
            manifest: "{}".into(),
            kind,
        }
    }

    #[test]
    fn unit__heal_active_engine__demotes_lazy_active_and_picks_text_lane() {
        let (_t, s) = tmp_store();
        // The pre-guard incident shape: a whisper row holds the serving
        // flag while text lanes sit installed — a text boot over this
        // store would pick no adapter and die.
        s.upsert_engine(&engine_row("b5130", 5000, true, EngineKind::Whisper))
            .unwrap();
        s.upsert_engine(&engine_row("m1", 4000, false, EngineKind::MistralRs))
            .unwrap();
        s.upsert_engine(&engine_row("b4000", 3000, false, EngineKind::LlamaCpp))
            .unwrap();

        let healed = s.heal_active_engine().unwrap();
        assert_eq!(
            healed,
            Some("b4000".to_string()),
            "llamacpp outranks a newer mistralrs row"
        );
        let active = s.active_engine().unwrap().expect("a text lane serves");
        assert_eq!(
            (active.tag.as_str(), active.kind),
            ("b4000", EngineKind::LlamaCpp)
        );
        let whisper = s
            .list_engines()
            .unwrap()
            .into_iter()
            .find(|r| r.tag == "b5130")
            .unwrap();
        assert!(
            !whisper.active,
            "the lazy row was demoted off the serving slot"
        );
    }

    #[test]
    fn unit__heal_active_engine__audio_only_box_lands_zero_active() {
        let (_t, s) = tmp_store();
        s.upsert_engine(&engine_row("2023.11.14-2", 5000, true, EngineKind::Piper))
            .unwrap();
        // Demotion with nothing to promote: the box rests at zero-active
        // (its pre-lane state); /v1/audio serving never read the flag.
        assert_eq!(s.heal_active_engine().unwrap(), None);
        assert!(s.active_engine().unwrap().is_none());
        // Idempotent: the second pass is a plain no-op.
        assert_eq!(s.heal_active_engine().unwrap(), None);
    }

    #[test]
    fn unit__heal_active_engine__healthy_active_untouched() {
        let (_t, s) = tmp_store();
        s.upsert_engine(&engine_row("b4000", 3000, true, EngineKind::LlamaCpp))
            .unwrap();
        s.upsert_engine(&engine_row("b5000", 5000, false, EngineKind::LlamaCpp))
            .unwrap();
        assert_eq!(s.heal_active_engine().unwrap(), None, "no healing to do");
        assert_eq!(
            s.active_engine().unwrap().map(|r| r.tag),
            Some("b4000".to_string()),
            "the newer inactive row did not steal the slot"
        );
    }

    #[test]
    fn unit__store_schema_upgrades__v7_to_v8() {
        // Simulate a v7 database: the four v8 tables absent, user_version 7.
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        {
            let s = Store::open(&dirs).unwrap();
            s.conn()
                .execute_batch(
                    "DROP TABLE jobs; DROP TABLE job_events; DROP TABLE responses; DROP TABLE model_caps; PRAGMA user_version = 7;",
                )
                .unwrap();
        }
        // Reopen: migrate() must recreate the v8 tables and stamp version 8.
        let s = Store::open(&dirs).unwrap();
        let v: i32 = s
            .conn()
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(v, SCHEMA_VERSION);
        // And the recreated tables are functional, not decorative.
        s.insert_job(&job_row("upgrade_probe", "audio", "queued", 1000))
            .unwrap();
        assert!(s.get_job("upgrade_probe").unwrap().is_some());
    }

    #[test]
    fn unit__model_caps_dated__round_trips_with_stamp() {
        let (_tmp, s) = tmp_store();
        assert!(s.get_model_caps_dated("m1").unwrap().is_none());
        s.put_model_caps("m1", "llamacpp-b1", "{\"caps\":{}}")
            .unwrap();
        let (tag, tested_at, caps_json) = s
            .get_model_caps_dated("m1")
            .unwrap()
            .expect("row exists after put");
        assert_eq!(tag, "llamacpp-b1");
        assert!(
            tested_at > 1_700_000_000,
            "stamp is a real epoch: {tested_at}"
        );
        assert_eq!(caps_json, "{\"caps\":{}}");
        // Undated view stays consistent with the dated one.
        assert_eq!(s.get_model_caps("m1").unwrap().unwrap().0, tag);
    }

    #[test]
    fn unit__bench_results__round_trip_pinned_then_latest_any() {
        let (_tmp, s) = tmp_store();
        assert!(s.latest_bench_result("m1", None).unwrap().is_none());
        s.put_bench_result("m1", "lane-a", "[]").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1100));
        s.put_bench_result("m1", "lane-b", "[]").unwrap();
        // Pinned lookup is exact; unpinned takes the most recent.
        let (tag, payload, _) = s
            .latest_bench_result("m1", Some("lane-a"))
            .unwrap()
            .expect("pinned row exists");
        assert_eq!((tag.as_str(), payload.as_str()), ("lane-a", "[]"));
        let (tag, _, _) = s
            .latest_bench_result("m1", None)
            .unwrap()
            .expect("any row exists");
        assert_eq!(tag, "lane-b", "latest across lanes wins: {tag}");
        // Upsert, not append: re-benching the same (model, lane) replaces.
        s.put_bench_result("m1", "lane-a", "[{\"ts\":1}]").unwrap();
        let (tag, payload, _) = s
            .latest_bench_result("m1", Some("lane-a"))
            .unwrap()
            .expect("row exists after upsert");
        assert_eq!((tag.as_str(), payload.as_str()), ("lane-a", "[{\"ts\":1}]"));
    }

    #[test]
    fn unit__list_bench_results_and_profiles__single_query_inventories() {
        let (_tmp, s) = tmp_store();
        s.put_bench_result("m2", "lane-b", "[]").unwrap();
        s.put_bench_result("m1", "lane-a", "[]").unwrap();
        let listed = s.list_bench_results().unwrap();
        assert_eq!(
            listed
                .iter()
                .map(|r| (r.model.as_str(), r.engine_tag.as_str()))
                .collect::<Vec<_>>(),
            vec![("m1", "lane-a"), ("m2", "lane-b")],
            "sorted by model then lane"
        );
        assert!(s.list_profiles().unwrap().is_empty());
        s.upsert_profile(&ProfileRow {
            model_name: "m1".into(),
            engine_tag: "lane-a".into(),
            args_hash: "h".into(),
            args_json: "{}".into(),
            benchmark_json: None,
            updated_at: 7,
        })
        .unwrap();
        let profiles = s.list_profiles().unwrap();
        assert_eq!(profiles.len(), 1);
        assert_eq!(
            (profiles[0].model_name.as_str(), profiles[0].updated_at),
            ("m1", 7)
        );
    }

    #[test]
    fn unit__store_schema_upgrades__v10_to_v11() {
        // Simulate a v10 database: no bench_results table, user_version 10.
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        {
            let s = Store::open(&dirs).unwrap();
            s.conn()
                .execute_batch("DROP TABLE bench_results; PRAGMA user_version = 10;")
                .unwrap();
        }
        // Reopen: migrate() recreates the table, stamps v11, and the
        // accessors work on the upgraded store (the exact gap the live
        // bench hit: version-gated schema replay).
        let s = Store::open(&dirs).unwrap();
        let v: i32 = s
            .conn()
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(v, SCHEMA_VERSION);
        s.put_bench_result("m1", "lane-a", "[]").unwrap();
        assert!(s.latest_bench_result("m1", None).unwrap().is_some());
    }

    #[test]
    fn unit__store_schema_upgrades__v9_to_v10() {
        // Simulate a v9 database: responses without the conversation/body_json
        // columns, user_version 9.
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        {
            let s = Store::open(&dirs).unwrap();
            s.conn()
                .execute_batch(
                    "DROP TABLE responses; CREATE TABLE responses (id TEXT PRIMARY KEY, model TEXT NOT NULL, input_json TEXT NOT NULL, output_json TEXT NOT NULL, input_tokens INTEGER, output_tokens INTEGER, ts INTEGER NOT NULL); INSERT INTO responses (id, model, input_json, output_json, input_tokens, output_tokens, ts) VALUES ('legacy', 'm', '[]', '[]', 1, 2, 100); PRAGMA user_version = 9;",
                )
                .unwrap();
        }
        // Reopen: migrate() adds the columns, keeps legacy rows, stamps v10.
        let s = Store::open(&dirs).unwrap();
        let v: i32 = s
            .conn()
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(v, SCHEMA_VERSION);
        let legacy = s.get_response("legacy").unwrap().unwrap();
        assert_eq!(legacy.model, "m");
        assert_eq!(legacy.conversation, "");
        assert!(legacy.body_json.is_none());
    }

    #[test]
    fn unit__conversations__stamp_list_delete() {
        let (_t, s) = tmp_store();
        let row = |id: &str, conv: &str, ts: i64| StoredResponseRow {
            id: id.into(),
            model: "m".into(),
            input_json: "[]".into(),
            output_json: "[]".into(),
            input_tokens: None,
            output_tokens: None,
            ts,
            conversation: conv.into(),
            body_json: None,
        };
        // Same conversation, distinct timestamps; ordering must be ts ASC.
        s.put_response(&row("r2", "conv-a", 200)).unwrap();
        s.put_response(&row("r1", "conv-a", 100)).unwrap();
        s.put_response(&row("other", "conv-b", 150)).unwrap();

        let listed = s.list_conversation("conv-a").unwrap();
        assert_eq!(
            listed.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            vec!["r1", "r2"],
            "conversation listing is chronological"
        );
        assert!(
            s.list_conversation("missing").unwrap().is_empty(),
            "missing conversation lists nothing"
        );

        let deleted = s.delete_conversation("conv-a").unwrap();
        assert_eq!(deleted, 2);
        assert!(
            s.list_conversation("conv-a").unwrap().is_empty(),
            "deleted conversation lists nothing"
        );
        // Other conversations untouched.
        assert_eq!(s.list_conversation("conv-b").unwrap().len(), 1);
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

    #[test]
    fn unit__model_from_row__empty_mmproj_dialect_reads_as_no_projector() {
        let (_t, s) = tmp_store();
        s.upsert_model(&base_model("text-only")).unwrap();
        // Simulate the empty-string dialect (hand-edited store / older
        // writer): both NULL and '' must read back as None so profile
        // never classifies the model as multimodal.
        s.conn()
            .execute(
                "UPDATE models SET mmproj_path = '' WHERE name = 'text-only'",
                [],
            )
            .unwrap();
        assert_eq!(s.get_model("text-only").unwrap().unwrap().mmproj_path, None);
        s.conn()
            .execute(
                "UPDATE models SET mmproj_path = NULL WHERE name = 'text-only'",
                [],
            )
            .unwrap();
        assert_eq!(s.get_model("text-only").unwrap().unwrap().mmproj_path, None);
        // A real projector path survives normalization.
        s.conn()
            .execute(
                "UPDATE models SET mmproj_path = '/tmp/mmproj-X.gguf' WHERE name = 'text-only'",
                [],
            )
            .unwrap();
        assert_eq!(
            s.get_model("text-only").unwrap().unwrap().mmproj_path,
            Some("/tmp/mmproj-X.gguf".into())
        );
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
            last_used_at: 0,
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

    fn engine_row_of(tag: &str, kind: EngineKind, installed_at: i64) -> EngineRow {
        EngineRow {
            tag: tag.into(),
            asset: "ubuntu-vulkan-x64".into(),
            sha256: format!("{tag}deadbeef"),
            installed_at,
            active: false,
            manifest: "{}".into(),
            kind,
        }
    }

    #[test]
    fn unit__heal_active_engine__zero_active_prefers_llamacpp_then_newest() {
        let (_t, s) = tmp_store();
        // llamacpp older than mistral.rs, whisper newest of all: kind
        // preference beats recency, and the lazy whisper lane never
        // claims the serving slot.
        s.upsert_engine(&engine_row_of("v0.9.3", EngineKind::MistralRs, 20))
            .unwrap();
        s.upsert_engine(&engine_row_of("b5130", EngineKind::Whisper, 30))
            .unwrap();
        s.upsert_engine(&engine_row_of("b100", EngineKind::LlamaCpp, 10))
            .unwrap();
        let healed = s.heal_active_engine().unwrap();
        assert_eq!(healed.as_deref(), Some("b100"));
        assert_eq!(s.active_engine().unwrap().unwrap().tag, "b100");
        // Without a llamacpp lane the newest serving engine wins; the
        // lazy whisper row still never does.
        let (_t2, s2) = tmp_store();
        s2.upsert_engine(&engine_row_of("v0.9.3", EngineKind::MistralRs, 20))
            .unwrap();
        s2.upsert_engine(&engine_row_of("sglang-0.5", EngineKind::Sglang, 40))
            .unwrap();
        s2.upsert_engine(&engine_row_of("b5130", EngineKind::Whisper, 60))
            .unwrap();
        assert_eq!(
            s2.heal_active_engine().unwrap().as_deref(),
            Some("sglang-0.5")
        );
    }

    #[test]
    fn unit__heal_active_engine__active_present_is_noop() {
        let (_t, s) = tmp_store();
        s.upsert_engine(&engine_row_of("b100", EngineKind::LlamaCpp, 10))
            .unwrap();
        s.set_active_engine("b100").unwrap();
        assert_eq!(s.heal_active_engine().unwrap(), None);
        assert_eq!(s.active_engine().unwrap().unwrap().tag, "b100");
    }

    #[test]
    fn unit__heal_active_engine__lazy_only_store_stays_unhealed() {
        // Only lazy lanes installed: the serving error path must stay
        // honest (activating whisper as the serving adapter would
        // regress the whisper-dethroning bug this store already fixed).
        let (_t, s) = tmp_store();
        s.upsert_engine(&engine_row_of("b5130", EngineKind::Whisper, 30))
            .unwrap();
        s.upsert_engine(&engine_row_of("master-890", EngineKind::SdCpp, 40))
            .unwrap();
        assert_eq!(s.heal_active_engine().unwrap(), None);
        assert!(s.active_engine().unwrap().is_none());
    }

    /// Forward-written row shape (2026-09-27 incident): a kind only a
    /// newer blazar understands must quarantine the row, not fail the
    /// roster read for every caller.
    #[test]
    fn unit__list_engines__unknown_kind_row_quarantined_not_fatal() {
        let (_t, s) = tmp_store();
        s.upsert_engine(&engine_row_of("b100", EngineKind::LlamaCpp, 10))
            .unwrap();
        s.conn()
            .execute(
                "INSERT INTO engines (tag, asset, sha256, installed_at, active, manifest, kind)
                 VALUES ('warp-9', 'overlay', 'deadbeef', 20, 0, '{}', 'warpdrive')",
                [],
            )
            .unwrap();
        let tags: Vec<String> = s
            .list_engines()
            .unwrap()
            .into_iter()
            .map(|e| e.tag)
            .collect();
        assert_eq!(tags, ["b100".to_string()]);
        assert_eq!(
            s.unknown_kind_engine_rows().unwrap(),
            [("warp-9".to_string(), "warpdrive".to_string())]
        );
    }

    /// The brick path itself: an unknown-kind row squatting on the
    /// active flag reads as absent, healing reclaims the slot for a
    /// known lane, and the forward row survives deactivated for the
    /// newer binary to reclaim.
    #[test]
    fn unit__active_engine__unknown_kind_row_treated_absent_and_heals() {
        let (_t, s) = tmp_store();
        s.conn()
            .execute(
                "INSERT INTO engines (tag, asset, sha256, installed_at, active, manifest, kind)
                 VALUES ('warp-9', 'overlay', 'deadbeef', 20, 1, '{}', 'warpdrive')",
                [],
            )
            .unwrap();
        s.upsert_engine(&engine_row_of("b100", EngineKind::LlamaCpp, 10))
            .unwrap();
        assert!(s.active_engine().unwrap().is_none());
        assert_eq!(s.heal_active_engine().unwrap().as_deref(), Some("b100"));
        assert_eq!(s.active_engine().unwrap().unwrap().tag, "b100");
        let warp_active: i64 = s
            .conn()
            .query_row("SELECT active FROM engines WHERE tag = 'warp-9'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(warp_active, 0);
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
            last_used_at: 42,
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
        assert_eq!(s.list_loras(Some("m")).unwrap().len(), 0);
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
            last_used_at: 0,
        };
        // The live incident rows: awq/gptq/fp8 dirs quantize true.
        assert!(
            row(
                "qwen2.5-0.5b-instruct-awq",
                "qwen/qwen2.5-0.5b-instruct-awq",
                "/models/qwen2.5-0.5b-instruct-awq.d"
            )
            .is_quantized_safetensors()
        );
        assert!(row("m-gptq", "r", "/models/m-gptq.d").is_quantized_safetensors());
        assert!(row("m-fp8-dynamic", "r", "/models/m-fp8-dynamic.d").is_quantized_safetensors());
        // Word boundary: 'hawk' embeds awq as a substring but splits
        // into its own token — stays a normal lane.
        assert!(!row("hawk", "r", "/models/hawk.d").is_quantized_safetensors());
        // Plain BF16 dir: the lane mistral.rs serves perfectly.
        assert!(
            !row(
                "qwen2.5-0.5b-instruct",
                "qwen/qwen2.5-0.5b-instruct",
                "/models/qwen2.5-0.5b-instruct.d"
            )
            .is_quantized_safetensors()
        );
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
                last_used_at: 1,
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
        assert_eq!(old.components.len(), 0);
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
            last_used_at: 2,
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
        assert_eq!(text.components.len(), 0);
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
