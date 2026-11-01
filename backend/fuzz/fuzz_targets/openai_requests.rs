//! The JSON request bodies of the API, as the routes deserialize them.
#![no_main]
use hypnos_core::openai::*;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(req) = serde_json::from_slice::<ChatCompletionRequest>(data) {
        let _ = req.thinking();
    }
    let _ = serde_json::from_slice::<EmbeddingRequest>(data);
    let _ = serde_json::from_slice::<AudioSpeechRequest>(data);
    let _ = serde_json::from_slice::<ImageGenerationRequest>(data);
    let _ = serde_json::from_slice::<DocumentConversionRequest>(data);
    let _ = serde_json::from_slice::<EntityExtractionRequest>(data);
});
