//! The `Engine` abstraction: everything Pallama needs from a serving
//! backend. Sole implementation today: `LlamaCppEngine`. vLLM/SGLang
//! adapters later implement the same trait — gateway, supervisor and
//! lifecycle stay engine-agnostic.

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;

use pallama_core::profile::{Endpoint, Profile};

/// A spawned engine child process: how to reach it and how to stop it.
#[derive(Debug)]
pub struct ChildHandle {
    pub endpoint: Endpoint,
    child: tokio::process::Child,
}

impl ChildHandle {
    #[must_use] 
    pub fn new(endpoint: Endpoint, child: tokio::process::Child) -> Self {
        Self { endpoint, child }
    }

    pub async fn wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
        self.child.wait().await
    }

    /// Non-blocking exit check.
    pub fn try_status(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        self.child.try_wait()
    }

    /// Hard kill (teardown last resort).
    pub async fn kill(&mut self) -> std::io::Result<()> {
        self.child.kill().await
    }

    /// Reap the exited process.
    pub async fn reap(&mut self) -> std::io::Result<std::process::ExitStatus> {
        self.child.wait().await
    }

    #[must_use] 
    pub fn id(&self) -> Option<u32> {
        self.child.id()
    }
}

/// One running instance of a model, as the supervisor sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineEndpoint {
    pub host: String,
    pub port: u16,
}

#[async_trait]
pub trait Engine: Send + Sync {
    /// Capabilities of the active engine build (manifest).
    fn capabilities(&self) -> &crate::engine::manifest::Manifest;

    /// Build the argv (without the binary) for one model instance.
    fn build_argv(
        &self,
        model: &pallama_core::ModelRow,
        profile: &Profile,
        endpoint: &Endpoint,
    ) -> Vec<String>;

    /// Spawn a child for the compiled argv. Implementations own process
    /// group setup and log piping.
    async fn spawn(&self, argv: &[String], endpoint: &Endpoint) -> Result<ChildHandle>;

    /// Poll the child's /health until {"status":"ok"} or timeout.
    async fn health_check(&self, endpoint: &Endpoint, timeout: std::time::Duration) -> Result<()>;
}

/// llama.cpp llama-server engine.
pub struct LlamaCppEngine {
    pub manifest: crate::engine::manifest::Manifest,
    /// HTTP client for health polls (children are loopback).
    http: reqwest::Client,
    /// Extra env injected into children (tests: STUB_* knobs).
    pub child_env: Vec<(String, String)>,
}

impl LlamaCppEngine {
    #[must_use] 
    pub fn new(manifest: crate::engine::manifest::Manifest) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .expect("health client");
        Self { manifest, http, child_env: Vec::new() }
    }

    fn base_url(endpoint: &Endpoint) -> String {
        match endpoint {
            Endpoint::Tcp { host, port } => format!("http://{host}:{port}"),
            Endpoint::Unix { .. } => String::new(), // unix sockets: probe skipped
        }
    }
}

#[async_trait]
impl Engine for LlamaCppEngine {
    fn capabilities(&self) -> &crate::engine::manifest::Manifest {
        &self.manifest
    }

    fn build_argv(
        &self,
        _model: &pallama_core::ModelRow,
        profile: &Profile,
        endpoint: &Endpoint,
    ) -> Vec<String> {
        // The profile already carries the full argv for this endpoint
        // (it was compiled with the endpoint baked in).
        let _ = endpoint;
        profile.argv.clone()
    }

    async fn spawn(&self, argv: &[String], endpoint: &Endpoint) -> Result<ChildHandle> {
        let mut cmd = tokio::process::Command::new(&self.manifest.server_path);
        cmd.args(argv)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            // If the owning process dies without teardown, the child must
            // not linger (test leakage, daemon crash).
            .kill_on_drop(true);
        #[cfg(unix)]
        {
            // Own process group: SIGTERM to the group takes down the whole
            // engine tree, and Ctrl-C on the daemon does not hit children.
            cmd.process_group(0);
        }
        for (k, v) in &self.child_env {
            cmd.env(k, v);
        }
        let mut child = cmd
            .spawn()
            .with_context(|| format!("spawn {}", self.manifest.server_path))?;

        // Pipe child logs into tracing with the model/endpoint prefix.
        if let Some(stdout) = child.stdout.take() {
            tokio::spawn(pipe_logs(stdout, "stdout"));
        }
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(pipe_logs(stderr, "stderr"));
        }
        Ok(ChildHandle::new(endpoint.clone(), child))
    }

    async fn health_check(&self, endpoint: &Endpoint, timeout: std::time::Duration) -> Result<()> {
        let url = Self::base_url(endpoint);
        if url.is_empty() {
            return Ok(()); // unix transport: health via socket handled by caller
        }
        let deadline = tokio::time::Instant::now() + timeout;
        let started = std::time::Instant::now();
        loop {
            match self
                .http
                .get(format!("{url}/health"))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    let body: serde_json::Value = resp.json().await.unwrap_or_default();
                    if body["status"] == "ok" {
                        tracing::debug!("engine healthy at {url} after {:?}", started.elapsed());
                        return Ok(());
                    }
                }
                _ => {}
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(anyhow!(
                    "engine at {url} did not become healthy within {timeout:?} (model_load_timeout)"
                ));
            }
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        }
    }
}

async fn pipe_logs<R: tokio::io::AsyncRead + Unpin + Send + 'static>(r: R, stream: &str) {
    use tokio::io::AsyncBufReadExt;
    let mut lines = tokio::io::BufReader::new(r).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        tracing::debug!(target: "pallama::engine", stream, "{line}");
    }
}
