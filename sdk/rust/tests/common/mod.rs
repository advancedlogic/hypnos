//! Starts the mock Hypnos server (the real router with mock engines) for a test.
//!
//! The binary is `$HYPNOS_MOCK_SERVER` when set, else `backend/target/debug/examples/mock_server`,
//! built with cargo when it is missing. Every test gets its own server, so tests that inject
//! failures or flip readiness cannot disturb each other.

#![allow(dead_code)]

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use hypnos_client::{Client, RetryPolicy};
use serde_json::json;

pub const CHAT: &str = "gemma4:26b-q4";
pub const STT: &str = "gemma4:e4b-q4";
pub const EMBED: &str = "gemmaembedding2";
pub const TTS: &str = "kokoro-82m-q4";
pub const IMAGE: &str = "moss-image";
pub const DOCS: &str = "docling";
pub const NER: &str = "gliner2.5-multi";

static BUILD: Mutex<()> = Mutex::new(());

fn backend() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../backend")
}

fn mock_binary() -> PathBuf {
    if let Ok(path) = std::env::var("HYPNOS_MOCK_SERVER") {
        return PathBuf::from(path);
    }
    let exe = if cfg!(windows) {
        "mock_server.exe"
    } else {
        "mock_server"
    };
    let binary = backend().join("target/debug/examples").join(exe);
    let _guard = BUILD.lock().unwrap_or_else(|e| e.into_inner());
    if !binary.exists() {
        let mut cmd = Command::new("cargo");
        cmd.current_dir(backend()).args([
            "build",
            "-p",
            "hypnos-server",
            "--example",
            "mock_server",
            "--no-default-features",
        ]);
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            cmd.args(["--features", "metal"]);
        }
        assert!(
            cmd.status().expect("cargo").success(),
            "building the mock server failed"
        );
    }
    binary
}

/// A running mock server; killed when dropped.
pub struct MockServer {
    child: Child,
    pub url: String,
}

impl MockServer {
    pub fn start() -> Self {
        Self::with_args(&[])
    }

    /// `--api-key secret --limited-key limited:1`.
    pub fn keyed() -> Self {
        Self::with_args(&["--api-key", "secret", "--limited-key", "limited:1"])
    }

    pub fn with_args(args: &[&str]) -> Self {
        let binary = mock_binary();
        let target = binary
            .parent()
            .and_then(|p| p.parent())
            .expect("target dir");
        let sep = if cfg!(windows) { ";" } else { ":" };
        let libs = format!("{}{sep}{}", target.display(), target.join("deps").display());
        let mut cmd = Command::new(&binary);
        cmd.args(["--port", "0"])
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        for var in ["LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PATH"] {
            let old = std::env::var(var).unwrap_or_default();
            cmd.env(var, format!("{libs}{sep}{old}"));
        }
        let mut child = cmd.spawn().expect("start the mock server");
        let mut line = String::new();
        BufReader::new(child.stdout.take().expect("stdout"))
            .read_line(&mut line)
            .expect("read the mock server's first line");
        let url = line
            .trim()
            .strip_prefix("listening on ")
            .unwrap_or_else(|| panic!("mock server did not start: {line:?}"))
            .to_string();
        Self { child, url }
    }

    pub async fn control(&self, path: &str, body: serde_json::Value) {
        reqwest::Client::new()
            .post(format!("{}/mock/{path}", self.url))
            .json(&body)
            .send()
            .await
            .and_then(|r| r.error_for_status())
            .expect("mock control");
    }

    pub async fn fail(&self, model: &str, error: &str, times: u32) {
        self.control(
            "fail",
            json!({ "model": model, "error": error, "times": times }),
        )
        .await;
    }

    pub async fn set_ready(&self, ready: bool) {
        self.control("ready", json!({ "ready": ready })).await;
    }

    /// A client with retries that do not really wait.
    pub fn client(&self) -> Client {
        Client::builder()
            .base_url(&self.url)
            .no_api_key()
            .retry(fast_retry())
            .build()
            .expect("client")
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub fn fast_retry() -> RetryPolicy {
    RetryPolicy {
        busy_backoff: Duration::from_millis(10),
        memory_backoff: Duration::from_millis(10),
        backoff: Duration::from_millis(10),
        unavailable_wait: Duration::from_millis(10),
        ..RetryPolicy::default()
    }
}
