//! `POST /v1/entities` (Hypnos extension): zero-shot named-entity extraction with a `gliner`
//! model. JSON in (`input`: one text or a list, `labels`, `threshold`, `flat`), one entity list
//! per text out, each span with its label, character offsets and score.

use super::{finish_telemetry, with_log, ApiError, AppState};
use crate::request_log::LogDetails;
use axum::{
    extract::State,
    response::{IntoResponse, Response},
    Json,
};
use hypnos_core::openai::{EntityExtractionRequest, EntityExtractionResponse};
use serde_json::json;
use std::time::Instant;

pub(crate) async fn extract_entities(
    State(state): State<AppState>,
    Json(req): Json<EntityExtractionRequest>,
) -> Response {
    let mut details = LogDetails {
        model: Some(req.model.clone()),
        input: state.request_log.capture_text().then(|| {
            format!(
                "labels: {}\n{}",
                req.labels.join(", "),
                req.input.clone().into_vec().join("\n")
            )
        }),
        request_payload: state
            .request_log
            .capture_payloads()
            .then(|| serde_json::to_value(&req).unwrap_or(serde_json::Value::Null)),
        ..Default::default()
    };
    match extract_entities_inner(&state, &req, &mut details).await {
        Ok(resp) => with_log(resp.into_response(), details),
        Err(e) => {
            let message = e.0.to_string();
            with_log(e.into_response(), details.with_error(message))
        }
    }
}

async fn extract_entities_inner(
    state: &AppState,
    req: &EntityExtractionRequest,
    details: &mut LogDetails,
) -> Result<Json<EntityExtractionResponse>, ApiError> {
    let ctx = state.new_request();
    let guard = ctx.cancel_on_drop();
    let lease = state
        .scheduler
        .acquire_model(&req.model, None, 1, &ctx)
        .await?;
    let infer_start = Instant::now();
    let mut resp = lease.backend.extract_entities(req, &ctx).await?;
    resp.hypnos_telemetry = Some(finish_telemetry(
        &lease,
        infer_start,
        resp.hypnos_telemetry.take(),
    ));
    guard.disarm();
    details.telemetry = resp.hypnos_telemetry.clone();
    let found: usize = resp.data.iter().map(|d| d.entities.len()).sum();
    if state.request_log.capture_text() {
        details.output = Some(
            resp.data
                .iter()
                .flat_map(|d| d.entities.iter())
                .map(|e| format!("{} [{}] {:.2}", e.text, e.label, e.score))
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }
    if state.request_log.capture_payloads() {
        details.response_payload = Some(json!({
            "texts": resp.usage.texts,
            "entities": found,
            "hypnos_telemetry": resp.hypnos_telemetry,
        }));
    }
    Ok(Json(resp))
}
