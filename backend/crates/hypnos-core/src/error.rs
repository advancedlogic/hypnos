use thiserror::Error;

#[derive(Debug, Error)]
pub enum HypnosError {
    #[error("Model not found in registry: '{0}'")]
    ModelNotFound(String),

    /// The model is configured but switched off (`enabled = false`, or `POST /v1/models/{id}/disable`).
    #[error("Model '{0}' is disabled; enable it with POST /v1/models/{0}/enable or `enabled = true` in its config")]
    ModelDisabled(String),

    #[error("Modality mismatch for model '{model}': expected {expected:?}, got {actual:?}")]
    ModalityMismatch {
        model: String,
        expected: crate::config::Modality,
        actual: crate::config::Modality,
    },

    #[error("Insufficient memory across Tier 1 (VRAM) and Tier 2 (Pinned RAM) to load model '{model}': required {required_bytes} bytes, available {available_bytes} bytes")]
    InsufficientMemory {
        model: String,
        required_bytes: u64,
        available_bytes: u64,
    },

    /// Loading the model would push the process past `hardware.max_process_memory_bytes`.
    #[error("Loading model '{model}' needs {required_bytes} bytes but only {available_bytes} bytes remain under the process memory cap of {cap_bytes} bytes (hardware.max_process_memory_bytes)")]
    ProcessMemoryCap {
        model: String,
        required_bytes: u64,
        available_bytes: u64,
        cap_bytes: u64,
    },

    /// Too many requests are already queued or running for this model.
    #[error(
        "Model '{model}' is busy: {in_flight} requests already queued (limit {limit}); retry later"
    )]
    Busy {
        model: String,
        in_flight: usize,
        limit: usize,
    },

    /// The estimated wait before this request could start exceeds the deadline it gave (PERF-10).
    #[error(
        "Model '{model}' cannot start within {deadline_ms} ms: the estimated wait is {estimated_ms} ms; retry later or with a longer deadline"
    )]
    DeadlineUnmeetable {
        model: String,
        estimated_ms: u64,
        deadline_ms: u64,
    },

    /// The client went away or the server is shutting down.
    #[error("Request cancelled")]
    Cancelled,

    #[error("Request timed out after {0} seconds")]
    Timeout(u64),

    #[error("Hardware backend error: {0}")]
    HardwareError(String),

    #[error("Backend execution error: {0}")]
    BackendError(String),

    /// The engine ran out of device memory while running a request on a model that was already
    /// placed (sd.cpp workspace, llama.cpp compute buffers): the device is fuller than the
    /// bookings say, usually because other models' runtime buffers grew.
    #[error("Model '{model}' ran out of device memory while running the request: {detail}")]
    DeviceOutOfMemory { model: String, detail: String },

    #[error("Invalid request parameter: {0}")]
    InvalidRequest(String),

    /// A resource addressed by the URL (a stored document, a log entry) does not exist.
    #[error("Not found: {0}")]
    NotFound(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Configuration error: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, HypnosError>;
