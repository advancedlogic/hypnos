//! The microphone button (DESK-9): record from the default input device with `cpal`, then
//! transcribe on the server. A cpal stream must stay on the thread that built it (CoreAudio's is
//! not `Send`), so each recording runs on its own thread; samples are mixed to mono as they come
//! and an input level goes to the UI for the meter.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Runtime, State};

use crate::attachments::{resample, wav_bytes};
use crate::hypnos::{ErrorPayload, Hypnos};

/// A recording stops by itself after this long.
const MAX_SECONDS: u64 = 300;
const LEVEL_EVERY: Duration = Duration::from_millis(100);

struct Recording {
    stop: Arc<AtomicBool>,
    thread: JoinHandle<Result<(Vec<f32>, u32), String>>,
}

#[derive(Default)]
pub struct Mic(Mutex<Option<Recording>>);

fn fail(message: impl Into<String>) -> ErrorPayload {
    ErrorPayload {
        kind: "mic".into(),
        message: message.into(),
        detail: None,
        retry_after_secs: None,
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Level {
    rms: f32,
    seconds: f32,
}

/// Starts recording; `mic-level` events carry `{ rms, seconds }` until it stops.
#[tauri::command]
pub fn mic_start<R: Runtime>(app: AppHandle<R>, mic: State<'_, Mic>) -> Result<(), ErrorPayload> {
    let mut slot = mic.0.lock().unwrap();
    if slot.is_some() {
        return Err(fail("Already recording."));
    }
    let stop = Arc::new(AtomicBool::new(false));
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();
    let thread_stop = stop.clone();
    let thread = std::thread::spawn(move || record(app, thread_stop, ready_tx));
    // Report a missing device or a refused permission right away.
    match ready_rx.recv_timeout(Duration::from_secs(5)) {
        Ok(Ok(())) => {
            *slot = Some(Recording { stop, thread });
            Ok(())
        }
        Ok(Err(e)) => {
            let _ = thread.join();
            Err(fail(e))
        }
        Err(_) => {
            stop.store(true, Ordering::SeqCst);
            Err(fail("The microphone did not start."))
        }
    }
}

fn record<R: Runtime>(
    app: AppHandle<R>,
    stop: Arc<AtomicBool>,
    ready: std::sync::mpsc::Sender<Result<(), String>>,
) -> Result<(Vec<f32>, u32), String> {
    let host = cpal::default_host();
    let Some(device) = host.default_input_device() else {
        let _ = ready.send(Err("No microphone found.".into()));
        return Err("no input device".into());
    };
    let config = match device.default_input_config() {
        Ok(c) => c,
        Err(e) => {
            let _ = ready.send(Err(format!("The microphone cannot be opened ({e}).")));
            return Err(e.to_string());
        }
    };
    let rate = config.sample_rate();
    let channels = config.channels() as usize;
    let samples = Arc::new(Mutex::new(Vec::<f32>::with_capacity(rate as usize * 30)));
    let stream = match config.sample_format() {
        SampleFormat::F32 => build::<f32>(&device, &config.config(), channels, samples.clone()),
        SampleFormat::I16 => build::<i16>(&device, &config.config(), channels, samples.clone()),
        SampleFormat::U16 => build::<u16>(&device, &config.config(), channels, samples.clone()),
        SampleFormat::I32 => build::<i32>(&device, &config.config(), channels, samples.clone()),
        other => Err(format!("unsupported sample format {other:?}")),
    };
    let stream = match stream.and_then(|s| s.play().map(|()| s).map_err(|e| e.to_string())) {
        Ok(s) => s,
        Err(e) => {
            let _ = ready.send(Err(format!(
                "The microphone cannot be opened ({e}). Check that the app may use it in the system's privacy settings."
            )));
            return Err(e);
        }
    };
    let _ = ready.send(Ok(()));

    let started = Instant::now();
    let mut seen = 0;
    while !stop.load(Ordering::SeqCst) && started.elapsed() < Duration::from_secs(MAX_SECONDS) {
        std::thread::sleep(LEVEL_EVERY);
        let buf = samples.lock().unwrap();
        let recent = &buf[seen..];
        let rms = if recent.is_empty() {
            0.0
        } else {
            (recent.iter().map(|s| s * s).sum::<f32>() / recent.len() as f32).sqrt()
        };
        seen = buf.len();
        drop(buf);
        let _ = app.emit(
            "mic-level",
            Level {
                rms,
                seconds: started.elapsed().as_secs_f32(),
            },
        );
    }
    drop(stream);
    let recorded = std::mem::take(&mut *samples.lock().unwrap());
    if started.elapsed() >= Duration::from_secs(MAX_SECONDS) {
        let _ = app.emit("mic-limit", MAX_SECONDS);
    }
    Ok((recorded, rate))
}

fn build<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    out: Arc<Mutex<Vec<f32>>>,
) -> Result<cpal::Stream, String>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    device
        .build_input_stream::<T, _, _>(
            *config,
            move |data: &[T], _| {
                let mut out = out.lock().unwrap();
                out.extend(data.chunks(channels.max(1)).map(|frame| {
                    frame.iter().map(|s| s.to_sample::<f32>()).sum::<f32>() / frame.len() as f32
                }));
            },
            |e| eprintln!("microphone stream error: {e}"),
            None,
        )
        .map_err(|e| e.to_string())
}

fn finish(mic: &Mic) -> Result<(Vec<f32>, u32), ErrorPayload> {
    let recording = mic
        .0
        .lock()
        .unwrap()
        .take()
        .ok_or_else(|| fail("Not recording."))?;
    recording.stop.store(true, Ordering::SeqCst);
    recording
        .thread
        .join()
        .map_err(|_| fail("The recording thread failed."))?
        .map_err(fail)
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Spoken {
    pub text: String,
    pub seconds: f64,
    pub model: String,
}

/// Stops recording and transcribes it with the model the router picks for speech to text.
#[tauri::command]
pub async fn mic_stop(
    mic: State<'_, Mic>,
    hypnos: State<'_, Hypnos>,
    chat_model: Option<String>,
) -> Result<Spoken, ErrorPayload> {
    let (samples, rate) = finish(&mic)?;
    let seconds = samples.len() as f64 / rate as f64;
    if seconds < 0.3 {
        return Err(fail("The recording is too short."));
    }
    let wav = tauri::async_runtime::spawn_blocking(move || {
        wav_bytes(&resample(&samples, rate, 16_000), 16_000)
    })
    .await
    .map_err(|e| fail(e.to_string()))?;
    let model = hypnos.pick("speech_to_text", chat_model.as_deref()).await?;
    let client = hypnos.client().await?;
    let result = client
        .transcribe(&hypnos_client::TranscriptionRequest::new(
            model.clone(),
            "speech.wav",
            wav,
        ))
        .await?;
    Ok(Spoken {
        text: result.text.trim().to_string(),
        seconds,
        model,
    })
}

/// Stops recording and throws it away.
#[tauri::command]
pub fn mic_cancel(mic: State<'_, Mic>) -> Result<(), ErrorPayload> {
    let _ = finish(&mic);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tauri::Manager;

    /// Records 1.5 s from the default input device: `cargo test mic -- --ignored`.
    #[test]
    #[ignore]
    fn records_from_the_default_microphone() {
        let app = tauri::test::mock_builder()
            .manage(Mic::default())
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();
        mic_start(app.handle().clone(), app.state()).unwrap();
        assert!(
            mic_start(app.handle().clone(), app.state()).is_err(),
            "a second start is refused"
        );
        std::thread::sleep(Duration::from_millis(1500));
        let (samples, rate) = finish(&app.state::<Mic>()).unwrap();
        let seconds = samples.len() as f64 / rate as f64;
        assert!((1.2..2.0).contains(&seconds), "{seconds} s at {rate} Hz");
        assert!(finish(&app.state::<Mic>()).is_err(), "nothing left to stop");
    }
}
