//! The speaker button (DESK-10): answers read aloud. The webview sends the answer's text as it
//! grows; each complete sentence is turned into speakable text, synthesized on the server's
//! text-to-speech model and queued on a player thread that owns the audio device (rodio, on the
//! same cpal as the microphone). Playback starts after the first sentence.

use std::io::Cursor;
use std::sync::mpsc as std_mpsc;
use std::sync::Mutex;
use std::time::Duration;

use hypnos_client::SpeechRequest;
use serde::Serialize;
use tauri::async_runtime::JoinHandle;
use tauri::{AppHandle, Emitter, Manager, Runtime, State};
use tokio::sync::mpsc;

use crate::hypnos::{ErrorPayload, Hypnos};

enum PlayerCmd {
    Append {
        wav: Vec<u8>,
    },
    /// No more audio for this answer: report the end once the queue has drained.
    End {
        id: String,
    },
    Clear,
}

struct Session {
    id: String,
    /// Sentences already handed to the worker.
    sent: usize,
    tx: mpsc::UnboundedSender<String>,
    worker: JoinHandle<()>,
}

#[derive(Default)]
pub struct Speaker {
    session: Mutex<Option<Session>>,
    player: Mutex<Option<std_mpsc::Sender<PlayerCmd>>>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct SpeechState {
    id: String,
    playing: bool,
    error: Option<String>,
}

fn fail(message: impl Into<String>) -> ErrorPayload {
    ErrorPayload {
        kind: "speech".into(),
        message: message.into(),
        detail: None,
        retry_after_secs: None,
    }
}

impl Speaker {
    fn player<R: Runtime>(
        &self,
        app: &AppHandle<R>,
    ) -> Result<std_mpsc::Sender<PlayerCmd>, ErrorPayload> {
        let mut slot = self.player.lock().unwrap();
        if let Some(tx) = slot.as_ref() {
            return Ok(tx.clone());
        }
        let (tx, rx) = std_mpsc::channel();
        let (ready_tx, ready_rx) = std_mpsc::channel();
        let app = app.clone();
        std::thread::spawn(move || play(app, rx, ready_tx));
        ready_rx
            .recv_timeout(Duration::from_secs(5))
            .map_err(|_| fail("The speakers did not start."))?
            .map_err(fail)?;
        *slot = Some(tx.clone());
        Ok(tx)
    }

    /// Stops the synthesis (if any) and silences the player. The session is gone once its last
    /// sentence is synthesized while the audio still plays, so the player is cleared either way;
    /// it reports the end of the answer it was finishing.
    fn stop_session(&self) -> Option<String> {
        let session = self.session.lock().unwrap().take();
        if let Some(s) = &session {
            s.worker.abort();
        }
        if let Some(tx) = self.player.lock().unwrap().as_ref() {
            let _ = tx.send(PlayerCmd::Clear);
        }
        session.map(|s| s.id)
    }
}

/// The player thread: the output device must stay on the thread that opened it.
fn play<R: Runtime>(
    app: AppHandle<R>,
    rx: std_mpsc::Receiver<PlayerCmd>,
    ready: std_mpsc::Sender<Result<(), String>>,
) {
    let sink = match rodio::DeviceSinkBuilder::open_default_sink() {
        Ok(s) => s,
        Err(e) => {
            let _ = ready.send(Err(format!("No audio output ({e}).")));
            return;
        }
    };
    let player = rodio::Player::connect_new(sink.mixer());
    let _ = ready.send(Ok(()));
    let mut ending: Option<String> = None;
    loop {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(PlayerCmd::Append { wav }) => match rodio::Decoder::try_from(Cursor::new(wav)) {
                Ok(source) => {
                    player.append(source);
                    player.play();
                }
                Err(e) => eprintln!("cannot decode speech audio: {e}"),
            },
            Ok(PlayerCmd::End { id }) => ending = Some(id),
            Ok(PlayerCmd::Clear) => {
                player.clear();
                if let Some(id) = ending.take() {
                    let _ = app.emit(
                        "speech-state",
                        SpeechState {
                            id,
                            playing: false,
                            error: None,
                        },
                    );
                }
            }
            Err(std_mpsc::RecvTimeoutError::Timeout) => {}
            Err(std_mpsc::RecvTimeoutError::Disconnected) => return,
        }
        if player.empty() {
            if let Some(id) = ending.take() {
                let _ = app.emit(
                    "speech-state",
                    SpeechState {
                        id,
                        playing: false,
                        error: None,
                    },
                );
            }
        }
    }
}

/// Reads an answer aloud. Call it with the text so far while the answer streams (`done =
/// false`) and once more at the end (`done = true`); a call for another `id` stops the previous
/// answer. `speech-state` events report `{ id, playing, error }`.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn speak<R: Runtime>(
    app: AppHandle<R>,
    speaker: State<'_, Speaker>,
    hypnos: State<'_, Hypnos>,
    id: String,
    text: String,
    done: bool,
    voice: Option<String>,
    speed: Option<f32>,
) -> Result<(), ErrorPayload> {
    let player = speaker.player(&app)?;
    let sentences = sentences(&speech_text(&text), done);
    let is_new = {
        let slot = speaker.session.lock().unwrap();
        slot.as_ref().is_none_or(|s| s.id != id)
    };
    if is_new {
        speaker.stop_session();
        let model = hypnos.pick("text_to_speech", None).await?;
        let client = hypnos.client().await?;
        let (tx, rx) = mpsc::unbounded_channel();
        let worker = tauri::async_runtime::spawn(synthesize(
            app.clone(),
            client,
            model,
            voice,
            speed,
            id.clone(),
            rx,
            player.clone(),
        ));
        let _ = app.emit(
            "speech-state",
            SpeechState {
                id: id.clone(),
                playing: true,
                error: None,
            },
        );
        *speaker.session.lock().unwrap() = Some(Session {
            id: id.clone(),
            sent: 0,
            tx,
            worker,
        });
    }
    let mut slot = speaker.session.lock().unwrap();
    let Some(session) = slot.as_mut().filter(|s| s.id == id) else {
        return Ok(()); // finished or replaced meanwhile
    };
    for sentence in sentences.iter().skip(session.sent) {
        let _ = session.tx.send(sentence.clone());
    }
    session.sent = session.sent.max(sentences.len());
    if done {
        // Closing the channel lets the worker finish the queue and report the end.
        let (closed, _) = mpsc::unbounded_channel();
        session.tx = closed;
    }
    Ok(())
}

/// Synthesizes the sentences in order and queues the audio; reports the end or a failure.
#[allow(clippy::too_many_arguments)]
async fn synthesize<R: Runtime>(
    app: AppHandle<R>,
    client: std::sync::Arc<hypnos_client::Client>,
    model: String,
    voice: Option<String>,
    speed: Option<f32>,
    id: String,
    mut rx: mpsc::UnboundedReceiver<String>,
    player: std_mpsc::Sender<PlayerCmd>,
) {
    while let Some(sentence) = rx.recv().await {
        let mut request = SpeechRequest::new(model.clone(), sentence).response_format("wav");
        if let Some(v) = voice.as_deref().filter(|v| !v.trim().is_empty()) {
            request = request.voice(v);
        }
        if let Some(s) = speed {
            request = request.speed(s);
        }
        match client.speech(&request).await {
            Ok(speech) => {
                let _ = player.send(PlayerCmd::Append {
                    wav: speech.audio.to_vec(),
                });
            }
            Err(e) => {
                let message = ErrorPayload::from(e).message;
                let _ = app.emit(
                    "speech-state",
                    SpeechState {
                        id,
                        playing: false,
                        error: Some(message),
                    },
                );
                return;
            }
        }
    }
    let _ = player.send(PlayerCmd::End { id: id.clone() });
    // The session ends with its audio; forget it so a replay of the same answer starts afresh.
    let speaker = app.state::<Speaker>();
    let mut slot = speaker.session.lock().unwrap();
    if slot.as_ref().is_some_and(|s| s.id == id) {
        *slot = None;
    }
}

/// Stops reading aloud.
#[tauri::command]
pub fn speak_stop<R: Runtime>(app: AppHandle<R>, speaker: State<'_, Speaker>) {
    if let Some(id) = speaker.stop_session() {
        let _ = app.emit(
            "speech-state",
            SpeechState {
                id,
                playing: false,
                error: None,
            },
        );
    }
}

/// Markdown as it should be heard: code blocks announced instead of read, links reduced to
/// their text, bare URLs, emphasis marks, headings, list bullets and table pipes dropped. An
/// unclosed code fence (an answer still streaming) silences the rest until it closes.
pub fn speech_text(markdown: &str) -> String {
    let mut out = String::with_capacity(markdown.len());
    let mut in_code = false;
    for line in markdown.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            if !in_code {
                out.push_str("\nCode block.\n");
            }
            in_code = !in_code;
            continue;
        }
        if in_code {
            continue;
        }
        let mut l = trimmed
            .trim_start_matches('#')
            .trim_start_matches('>')
            .trim_start();
        if l.chars()
            .all(|c| matches!(c, '-' | '*' | '_' | '=' | '|' | ':' | ' '))
        {
            out.push('\n');
            continue;
        }
        for bullet in ["- ", "* ", "+ "] {
            if let Some(rest) = l.strip_prefix(bullet) {
                l = rest;
            }
        }
        if let Some(dot) = l.find(". ") {
            if dot <= 3 && l[..dot].chars().all(|c| c.is_ascii_digit()) {
                l = &l[dot + 2..];
            }
        }
        if l.starts_with('|') {
            // A table row: its cells, separated by commas.
            let cells: Vec<String> = l
                .split('|')
                .map(|c| inline(c.trim()))
                .filter(|c| !c.is_empty())
                .collect();
            out.push_str(&cells.join(", "));
        } else {
            out.push_str(&inline(l));
        }
        out.push('\n');
    }
    out
}

fn inline(line: &str) -> String {
    let mut s = String::with_capacity(line.len());
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        // [text](url) and ![alt](url) -> text
        if c == '[' || (c == '!' && chars.get(i + 1) == Some(&'[')) {
            let start = if c == '!' { i + 2 } else { i + 1 };
            if let Some(close) = (start..chars.len()).find(|&j| chars[j] == ']') {
                if chars.get(close + 1) == Some(&'(') {
                    if let Some(end) = (close + 2..chars.len()).find(|&j| chars[j] == ')') {
                        s.extend(&chars[start..close]);
                        i = end + 1;
                        continue;
                    }
                }
            }
        }
        if line[char_offset(&chars, i)..].starts_with("http://")
            || line[char_offset(&chars, i)..].starts_with("https://")
        {
            while i < chars.len() && !chars[i].is_whitespace() {
                i += 1;
            }
            s.push_str("a link");
            continue;
        }
        match c {
            '*' | '`' | '~' => {}
            '|' => s.push_str(", "),
            '_' if chars.get(i + 1) == Some(&'_') => i += 1,
            _ => s.push(c),
        }
        i += 1;
    }
    s.trim_matches(|c: char| c == ',' || c.is_whitespace())
        .to_string()
}

fn char_offset(chars: &[char], i: usize) -> usize {
    chars[..i].iter().map(|c| c.len_utf8()).sum()
}

/// Splits speakable text into sentences. Without `done`, the last piece is kept back unless
/// it already ends a sentence (more text may still be on its way).
pub fn sentences(text: &str, done: bool) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if c == '\n' {
            push(&mut out, &mut current);
            continue;
        }
        current.push(c);
        let next = chars.get(i + 1);
        if matches!(c, '.' | '!' | '?' | '…' | '。' | '！' | '？')
            && next.is_some_and(|n| n.is_whitespace())
        {
            push(&mut out, &mut current);
        }
    }
    // Otherwise the trailing piece waits for more text.
    if done || text.ends_with(['.', '!', '?', '…', '\n']) {
        push(&mut out, &mut current);
    }
    out
}

fn push(out: &mut Vec<String>, current: &mut String) {
    let s = current.trim();
    if s.chars().any(char::is_alphanumeric) {
        out.push(s.to_string());
    }
    current.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_is_read_without_its_symbols() {
        let md = "## Title\n\nSee [the docs](https://x.org/a) or https://y.org now.\n\n- **Bold** item\n1. First `step`\n\n```rust\nfn main() {}\n```\n| a | b |\n|---|---|\n";
        let text = speech_text(md);
        assert!(text.contains("Title"));
        assert!(text.contains("See the docs or a link now."));
        assert!(text.contains("Bold item"));
        assert!(text.contains("First step"));
        assert!(text.contains("Code block."));
        assert!(!text.contains("fn main"));
        assert!(text.contains("a, b"));
        assert!(!text.contains("---"));
    }

    #[test]
    fn an_open_code_fence_silences_the_rest() {
        let text = speech_text("Here it is:\n```\nlet x = 1;");
        assert!(text.contains("Here it is:"));
        assert!(!text.contains("let x"));
    }

    #[test]
    fn only_complete_sentences_are_spoken_while_streaming() {
        assert_eq!(
            sentences("Hello there. How are", false),
            vec!["Hello there."]
        );
        assert_eq!(
            sentences("Hello there. How are you?", false),
            vec!["Hello there.", "How are you?"]
        );
        assert_eq!(
            sentences("Hello there. How are", true),
            vec!["Hello there.", "How are"]
        );
        // Version numbers do not split.
        assert_eq!(sentences("Use v1.2 today.", true), vec!["Use v1.2 today."]);
        assert_eq!(sentences("Line one\nLine two", false), vec!["Line one"]);
    }

    /// Synthesizes an answer's sentences on a running server and decodes each clip the way the
    /// player would, without opening the speakers: `HYPNOS_TEST_URL=... cargo test tts -- --ignored`.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn synthesizes_decodable_speech_for_each_sentence() {
        use crate::hypnos::hypnos_connect;
        use rodio::Source;

        let url = std::env::var("HYPNOS_TEST_URL").expect("HYPNOS_TEST_URL");
        let app = tauri::test::mock_builder()
            .manage(Hypnos::default())
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();
        hypnos_connect(app.state(), url).await.unwrap();
        let hypnos = app.state::<Hypnos>();
        let model = hypnos.pick("text_to_speech", None).await.unwrap();
        let client = hypnos.client().await.unwrap();

        let answer = "## Result\n\nThe build **passed**. See [the log](https://ci.example/1).\n\n```sh\nmake\n```";
        let spoken = sentences(&speech_text(answer), true);
        assert_eq!(
            spoken,
            vec!["Result", "The build passed.", "See the log.", "Code block."]
        );
        for sentence in spoken {
            let speech = client
                .speech(
                    &SpeechRequest::new(model.clone(), sentence.clone())
                        .response_format("wav")
                        .speed(1.1),
                )
                .await
                .unwrap();
            let decoder = rodio::Decoder::try_from(Cursor::new(speech.audio.to_vec())).unwrap();
            let rate = decoder.sample_rate().get() as f64;
            let channels = decoder.channels().get() as f64;
            let seconds = decoder.count() as f64 / rate / channels;
            assert!(seconds > 0.2, "{sentence}: {seconds} s");
        }
    }
}
