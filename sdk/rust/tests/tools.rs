//! Tool calling against the mock server (`docs/api.md` section 7.5).
//!
//! The mock calls one tool with `{"text": <last user text>}`: the named one for a function
//! `tool_choice`, the first for `required`, and for `auto` the first whose name is in the last
//! user text. After a trailing `tool` message it answers `tool said: <content>`.

mod common;

use common::*;
use hypnos_client::*;
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize, PartialEq)]
struct Args {
    text: String,
}

fn weather() -> ToolSpec {
    ToolSpec::function(
        "get_weather",
        "Current weather for a city",
        json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
    )
}

fn tools() -> Vec<ToolSpec> {
    vec![
        weather(),
        ToolSpec::function("get_time", "Current time", json!({"type": "object"})),
    ]
}

#[tokio::test]
async fn auto_calls_the_named_tool() {
    let server = MockServer::start();
    let text = "use get_weather for Paris";
    let reply = server
        .client()
        .chat(&ChatRequest::new(CHAT, vec![ChatMessage::user(text)]).tools(tools()))
        .await
        .unwrap();
    assert_eq!(reply.finish_reason(), "tool_calls");
    assert_eq!(reply.text(), "");
    let calls = reply.tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name(), "get_weather");
    assert_eq!(calls[0].kind, "function");
    assert!(!calls[0].id.is_empty());
    assert_eq!(
        calls[0].parse_arguments::<Args>().unwrap(),
        Args { text: text.into() }
    );
    assert_eq!(calls[0].arguments_json().unwrap(), json!({"text": text}));
}

#[tokio::test]
async fn auto_without_a_match_answers_in_text() {
    let server = MockServer::start();
    let reply = server
        .client()
        .chat(
            &ChatRequest::new(CHAT, vec![ChatMessage::user("hello")])
                .tools(tools())
                .tool_choice(ToolChoice::Auto),
        )
        .await
        .unwrap();
    assert_eq!(reply.text(), "echo: hello");
    assert!(reply.tool_calls().is_empty());
    assert!(reply.choices[0].message.tool_calls.is_none());
}

#[tokio::test]
async fn round_trip_with_tool_results() {
    let server = MockServer::start();
    let c = server.client();
    let mut messages = vec![ChatMessage::user("what does get_weather say?")];
    let first = c
        .chat(&ChatRequest::new(CHAT, messages.clone()).tools(tools()))
        .await
        .unwrap();
    assert_eq!(first.finish_reason(), "tool_calls");
    let history = first.to_message();
    assert_eq!(history.content, Content::Null);
    messages.push(history);
    for call in first.tool_calls() {
        messages.push(ChatMessage::tool(&call.id, r#"{"temperature": 18}"#).with_name(call.name()));
    }
    let second = c
        .chat(&ChatRequest::new(CHAT, messages).tools(tools()))
        .await
        .unwrap();
    assert_eq!(second.text(), r#"tool said: {"temperature": 18}"#);
    assert_eq!(second.finish_reason(), "stop");
}

#[tokio::test]
async fn streaming_accumulates_tool_calls() {
    let server = MockServer::start();
    let mut stream = server
        .client()
        .chat_stream(
            &ChatRequest::new(CHAT, vec![ChatMessage::user("anything")])
                .tools(tools())
                .tool_choice(ToolChoice::Required),
        )
        .await
        .unwrap();
    let completion = stream.final_completion().await.unwrap();
    assert_eq!(stream.finish_reason(), Some("tool_calls"));
    assert_eq!(stream.tool_calls().len(), 1);
    assert_eq!(stream.tool_calls()[0].name(), "get_weather");
    assert_eq!(completion.finish_reason(), "tool_calls");
    let calls = completion.tool_calls();
    assert_eq!(calls.len(), 1);
    assert!(!calls[0].id.is_empty());
    assert_eq!(
        calls[0].parse_arguments::<Args>().unwrap(),
        Args {
            text: "anything".into()
        }
    );
}

#[tokio::test]
async fn named_tool_choice() {
    let server = MockServer::start();
    let reply = server
        .client()
        .chat(
            &ChatRequest::new(CHAT, vec![ChatMessage::user("now")])
                .tools(tools())
                .tool_choice(ToolChoice::function("get_time")),
        )
        .await
        .unwrap();
    assert_eq!(reply.tool_calls()[0].name(), "get_time");
}

#[tokio::test]
async fn unknown_named_tool_choice_is_bad_request() {
    let server = MockServer::start();
    let err = server
        .client()
        .chat(
            &ChatRequest::new(CHAT, vec![ChatMessage::user("hi")])
                .tools(vec![weather()])
                .tool_choice(ToolChoice::function("no_such_tool")),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, Error::BadRequest(_)), "{err:?}");
}
