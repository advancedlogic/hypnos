//! The process's one ONNX Runtime environment. ort 2.0.0-rc.14 made the environment explicit and
//! allows a single one per process, which can never be built again once dropped, so engines must
//! not build their own: Kokoro used to, and failed to reload after its first eviction. docling-onnx
//! builds it on first use (telemetry off) and docling's sessions always use that one, so every ONNX
//! engine in Hypnos (Kokoro, docling, GLiNER) borrows it.

use hypnos_core::error::{HypnosError, Result};
use ort::environment::Environment;

pub(crate) fn environment() -> Result<&'static Environment> {
    docling_onnx::environment()
        .map_err(|e| HypnosError::BackendError(format!("ONNX Runtime init failed: {e}")))
}
