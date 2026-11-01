//! The tool-call parsers (`hypnos-backends/src/tools.rs`): they read what a model writes, which a
//! user's prompt can steer. The module is compiled here on its own (it needs only hypnos-core and
//! serde_json), so the target does not build the engines. The first byte picks the format, the
//! second how the rest is cut into the pieces a stream delivers.
#![no_main]
use libfuzzer_sys::fuzz_target;

#[allow(dead_code)]
#[path = "../../crates/hypnos-backends/src/tools.rs"]
mod tools;

use hypnos_core::openai::ToolSpec;
use tools::{CallParser, ToolFormat};

fn tool_specs() -> Vec<ToolSpec> {
    serde_json::from_value(serde_json::json!([
        { "type": "function", "function": { "name": "get_weather", "description": "Weather",
          "parameters": { "type": "object", "properties": {
            "city": { "type": "string" }, "days": { "type": "integer" } }, "required": ["city"] } } },
        { "type": "function", "function": { "name": "fetch_url",
          "parameters": { "type": "object", "properties": { "url": { "type": "string" } } } } }
    ]))
    .unwrap()
}

fuzz_target!(|data: &[u8]| {
    let [format, cut, rest @ ..] = data else { return };
    let Ok(text) = std::str::from_utf8(rest) else { return };
    let format = match format % 3 {
        0 => ToolFormat::Gemma4,
        1 => ToolFormat::QwenXml,
        _ => ToolFormat::GlmXml,
    };
    let specs = tool_specs();
    let mut parser = CallParser::new(format, "", "call", &specs);
    let step = usize::from(cut % 16) + 1;
    let mut piece = String::new();
    for (i, ch) in text.chars().enumerate() {
        piece.push(ch);
        if i % step == 0 {
            let _ = parser.feed(&piece);
            piece.clear();
        }
    }
    let _ = parser.feed(&piece);
    let _ = parser.finish();
    let _ = tools::parse_call(text);
});
