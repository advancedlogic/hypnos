use thiserror::Error;

#[derive(Debug, Error)]
pub enum HypnosError {
    #[error("Model not found in registry: '{0}'")]
    ModelNotFound(String),

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

    /// The client went away or the server is shutting down.
    #[error("Request cancelled")]
    Cancelled,

    #[error("Request timed out after {0} seconds")]
    Timeout(u64),

    #[error("Hardware backend error: {0}")]
    HardwareError(String),

    #[error("Backend execution error: {0}")]
    BackendError(String),

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
