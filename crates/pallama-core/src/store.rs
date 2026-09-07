use rusqlite::{params, Connection};

use crate::dirs::PallamaDirs;
use crate::error::{CoreError, CoreResult};

/// SQLite-backed persistent state (WAL mode). One DB file at
/// `<data_dir>/pallama.db`. Schema versioned via `PRAGMA user_version`.
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

const SCHEMA_VERSION: i32 = 3;

const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS engines (
    tag          TEXT PRIMARY KEY,
    asset        TEXT NOT NULL,
    sha256       TEXT NOT NULL,
    installed_at INTEGER NOT NULL,
    active       INTEGER NOT NULL DEFAULT 0,
    manifest     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
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
    pub fn open(dirs: &PallamaDirs) -> CoreResult<Self> {
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

    fn migrate(&self) -> CoreResult<()> {
        let version: i32 = self
            .conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if version < SCHEMA_VERSION {
            self.conn.execute_batch(SCHEMA_SQL)?;
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
            "INSERT INTO engines (tag, asset, sha256, installed_at, active, manifest)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(tag) DO UPDATE SET
               asset = excluded.asset, sha256 = excluded.sha256,
               installed_at = excluded.installed_at, manifest = excluded.manifest",
            params![
                e.tag,
                e.asset,
                e.sha256,
                e.installed_at,
                i64::from(e.active),
                e.manifest
            ],
        )?;
        Ok(())
    }

    /// Flip the active engine. Fails (changing nothing) when `tag` is not
    /// installed — never silently leaves the store with zero active engines.
    pub fn set_active_engine(&self, tag: &str) -> CoreResult<()> {
        let exists: bool = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM engines WHERE tag = ?1",
                params![tag],
                |r| r.get::<_, i64>(0),
            )
            .map(|n| n > 0)?;
        if !exists {
            return Err(CoreError::Store(rusqlite::Error::QueryReturnedNoRows));
        }
        self.conn.execute("UPDATE engines SET active = 0", [])?;
        self.conn
            .execute("UPDATE engines SET active = 1 WHERE tag = ?1", params![tag])?;
        Ok(())
    }

    pub fn active_engine(&self) -> CoreResult<Option<EngineRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT tag, asset, sha256, installed_at, active, manifest FROM engines WHERE active = 1",
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
            }));
        }
        Ok(None)
    }

    pub fn list_engines(&self) -> CoreResult<Vec<EngineRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT tag, asset, sha256, installed_at, active, manifest FROM engines ORDER BY installed_at DESC, rowid DESC",
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
            "INSERT INTO models (name, repo, quant, path, bytes, sha256, mmproj_path, shards,
                                 arch, params, ctx_train, pulled_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)
             ON CONFLICT(name) DO UPDATE SET
               repo = excluded.repo, quant = excluded.quant, path = excluded.path,
               bytes = excluded.bytes, sha256 = excluded.sha256,
               mmproj_path = excluded.mmproj_path, shards = excluded.shards,
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
            "SELECT name, repo, quant, path, bytes, sha256, mmproj_path, shards, arch, params,
                    ctx_train, pulled_at
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
            "SELECT name, repo, quant, path, bytes, sha256, mmproj_path, shards, arch, params,
                    ctx_train, pulled_at
             FROM models ORDER BY name",
        )?;
        let rows = stmt.query_map([], model_from_row)?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
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

    /// Add to a key's daily counters (upsert). Bumped by the gateway's
    /// write-behind flusher, never on the hot path.
    pub fn bump_key_usage(
        &self,
        day: &str,
        name: &str,
        delta_requests: i64,
        delta_tokens: i64,
    ) -> CoreResult<()> {
        self.conn.execute(
            "INSERT INTO key_usage (day, name, requests, tokens) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(day, name) DO UPDATE SET
               requests = requests + excluded.requests,
               tokens   = tokens   + excluded.tokens",
            rusqlite::params![day, name, delta_requests, delta_tokens],
        )?;
        Ok(())
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
    })
}

fn model_from_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<ModelRow> {
    Ok(ModelRow {
        name: r.get(0)?,
        repo: r.get(1)?,
        quant: r.get(2)?,
        path: r.get(3)?,
        bytes: r.get(4)?,
        sha256: r.get(5)?,
        mmproj_path: r.get(6)?,
        shards: r.get(7)?,
        arch: r.get(8)?,
        params: r.get(9)?,
        ctx_train: r.get(10)?,
        pulled_at: r.get(11)?,
    })
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    fn tmp_store() -> (tempfile::TempDir, Store) {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = PallamaDirs {
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

    fn base_model(name: &str) -> ModelRow {
        ModelRow {
            name: name.into(),
            repo: "o/x".into(),
            quant: "Q4_K_M".into(),
            path: "/tmp/x.gguf".into(),
            bytes: 1,
            sha256: None,
            mmproj_path: None,
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

    #[test]
    fn unit__reopen_migrates_idempotently() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = PallamaDirs {
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
}
