//! OpenAI tool calling on Gemma 4's native format (API-9), as its GGUF chat template defines it:
//! tools are declared in the system turn as `<|tool>declaration:NAME{...}<tool|>`, the model calls
//! one with `<|tool_call>call:NAME{key:value,...}<tool_call|>` and then emits `<|tool_response>`
//! (the engine stops there), and results come back in the same model turn as
//! `<|tool_response>response:NAME{value:...}<tool_response|>`. Values use Gemma's own notation:
//! strings between `<|"|>` tokens, bare keys inside a call.

use hypnos_core::openai::{FunctionCall, ToolCall, ToolSpec};
use serde_json::{Map, Value};

pub const QUOTE: &str = "<|\"|>";
pub const CALL_OPEN: &str = "<|tool_call>";
pub const CALL_CLOSE: &str = "<tool_call|>";
pub const RESPONSE_OPEN: &str = "<|tool_response>";
pub const RESPONSE_CLOSE: &str = "<tool_response|>";

/// The engine passes these marker tokens through as text when tools are active.
pub const PASS_THROUGH: [&str; 3] = [CALL_OPEN, CALL_CLOSE, QUOTE];

/// `format_argument` of the template: a JSON value in Gemma notation.
pub fn format_value(v: &Value, escape_keys: bool) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => format!("{QUOTE}{s}{QUOTE}"),
        Value::Array(items) => format!(
            "[{}]",
            items
                .iter()
                .map(|i| format_value(i, escape_keys))
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let body: Vec<String> = keys
                .iter()
                .map(|k| {
                    let key = if escape_keys {
                        format!("{QUOTE}{k}{QUOTE}")
                    } else {
                        (*k).clone()
                    };
                    format!("{key}:{}", format_value(&map[*k], escape_keys))
                })
                .collect();
            format!("{{{}}}", body.join(","))
        }
    }
}

fn upper_type(v: &Value) -> String {
    v.as_str().unwrap_or("string").to_ascii_uppercase()
}

fn quoted(s: &str) -> String {
    format!("{QUOTE}{s}{QUOTE}")
}

fn required_list(req: &Value) -> Option<String> {
    let items: Vec<String> = req
        .as_array()?
        .iter()
        .filter_map(|r| r.as_str())
        .map(quoted)
        .collect();
    (!items.is_empty()).then(|| format!("required:[{}]", items.join(",")))
}

/// `format_parameters`: each property as `name:{description:...,type:<|"|>STRING<|"|>}`.
fn format_properties(props: &Map<String, Value>) -> String {
    let mut keys: Vec<&String> = props.keys().collect();
    keys.sort();
    keys.iter()
        .map(|k| {
            let p = &props[*k];
            let mut fields = Vec::new();
            if let Some(d) = p
                .get("description")
                .and_then(Value::as_str)
                .filter(|d| !d.is_empty())
            {
                fields.push(format!("description:{}", quoted(d)));
            }
            let kind = upper_type(p.get("type").unwrap_or(&Value::Null));
            match kind.as_str() {
                "STRING" => {
                    if let Some(e) = p.get("enum").filter(|e| e.is_array()) {
                        fields.push(format!("enum:{}", format_value(e, true)));
                    }
                }
                "ARRAY" => {
                    if let Some(items) = p
                        .get("items")
                        .and_then(Value::as_object)
                        .filter(|i| !i.is_empty())
                    {
                        let mut item_fields = Vec::new();
                        let mut item_keys: Vec<&String> = items.keys().collect();
                        item_keys.sort();
                        for ik in item_keys {
                            let iv = &items[ik];
                            match ik.as_str() {
                                "properties" => {
                                    if let Some(obj) = iv.as_object() {
                                        item_fields.push(format!(
                                            "properties:{{{}}}",
                                            format_properties(obj)
                                        ));
                                    }
                                }
                                "required" => item_fields.extend(required_list(iv)),
                                "type" => {
                                    item_fields.push(format!("type:{}", quoted(&upper_type(iv))))
                                }
                                _ if !iv.is_null() => {
                                    item_fields.push(format!("{ik}:{}", format_value(iv, true)))
                                }
                                _ => {}
                            }
                        }
                        fields.push(format!("items:{{{}}}", item_fields.join(",")));
                    }
                }
                "OBJECT" => {
                    if let Some(obj) = p.get("properties").and_then(Value::as_object) {
                        fields.push(format!("properties:{{{}}}", format_properties(obj)));
                    }
                    if let Some(r) = p.get("required").and_then(required_list) {
                        fields.push(r);
                    }
                }
                _ => {}
            }
            if p.get("nullable").and_then(Value::as_bool) == Some(true) {
                fields.push("nullable:true".into());
            }
            fields.push(format!("type:{}", quoted(&kind)));
            format!("{k}:{{{}}}", fields.join(","))
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// `format_function_declaration`, wrapped in `<|tool>...<tool|>`.
pub fn declaration(tool: &ToolSpec) -> String {
    let f = &tool.function;
    let mut out = format!(
        "<|tool>declaration:{}{{description:{}",
        f.name,
        quoted(f.description.as_deref().unwrap_or(""))
    );
    if let Some(params) = f.parameters.as_ref().filter(|p| p.is_object()) {
        out.push_str(",parameters:{");
        if let Some(props) = params
            .get("properties")
            .and_then(Value::as_object)
            .filter(|p| !p.is_empty())
        {
            out.push_str(&format!("properties:{{{}}},", format_properties(props)));
        }
        if let Some(r) = params.get("required").and_then(required_list) {
            out.push_str(&r);
            out.push(',');
        }
        if let Some(t) = params.get("type") {
            out.push_str(&format!("type:{}}}", quoted(&upper_type(t))));
        }
    }
    out.push_str("}<tool|>");
    out
}

/// A past call as the model wrote it. `arguments` is the OpenAI JSON string.
pub fn call_text(call: &ToolCall) -> String {
    let args: Value =
        serde_json::from_str(&call.function.arguments).unwrap_or(Value::Object(Map::new()));
    let body = match &args {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            keys.iter()
                .map(|k| format!("{k}:{}", format_value(&map[*k], false)))
                .collect::<Vec<_>>()
                .join(",")
        }
        _ => String::new(),
    };
    format!(
        "{CALL_OPEN}call:{}{{{body}}}{CALL_CLOSE}",
        call.function.name
    )
}

/// A tool's result; OpenAI sends it as a string, which the template wraps as `{value:...}`.
pub fn response_text(name: &str, content: &str) -> String {
    format!(
        "{RESPONSE_OPEN}response:{name}{{value:{}}}{RESPONSE_CLOSE}",
        quoted(content)
    )
}

// --- parsing --------------------------------------------------------------------------------

/// Reads Gemma notation back into JSON: `{key:value}` with bare or quoted keys, `<|"|>` strings,
/// numbers, `true` / `false` / `null`, arrays.
struct Reader<'a> {
    s: &'a str,
    i: usize,
}

impl<'a> Reader<'a> {
    fn rest(&self) -> &'a str {
        &self.s[self.i..]
    }

    fn skip_ws(&mut self) {
        while self.rest().starts_with(char::is_whitespace) {
            self.i += self.rest().chars().next().map_or(1, char::len_utf8);
        }
    }

    fn eat(&mut self, token: &str) -> bool {
        self.skip_ws();
        if self.rest().starts_with(token) {
            self.i += token.len();
            true
        } else {
            false
        }
    }

    fn string(&mut self) -> Option<String> {
        if !self.eat(QUOTE) {
            return None;
        }
        let end = self.rest().find(QUOTE)?;
        let s = self.rest()[..end].to_string();
        self.i += end + QUOTE.len();
        Some(s)
    }

    fn key(&mut self) -> Option<String> {
        self.skip_ws();
        if self.rest().starts_with(QUOTE) {
            return self.string();
        }
        let end = self.rest().find(|c: char| c == ':' || c.is_whitespace())?;
        let k = self.rest()[..end].trim().to_string();
        self.i += end;
        (!k.is_empty()).then_some(k)
    }

    fn value(&mut self) -> Option<Value> {
        self.skip_ws();
        let r = self.rest();
        if r.starts_with(QUOTE) {
            return self.string().map(Value::String);
        }
        if r.starts_with('{') {
            return self.object().map(Value::Object);
        }
        if r.starts_with('[') {
            self.i += 1;
            let mut items = Vec::new();
            if self.eat("]") {
                return Some(Value::Array(items));
            }
            loop {
                items.push(self.value()?);
                if self.eat("]") {
                    return Some(Value::Array(items));
                }
                if !self.eat(",") {
                    return None;
                }
            }
        }
        // A scalar runs to the next separator.
        let end = r.find([',', '}', ']']).unwrap_or(r.len());
        let raw = r[..end].trim();
        self.i += end;
        match raw {
            "true" => Some(Value::Bool(true)),
            "false" => Some(Value::Bool(false)),
            "null" => Some(Value::Null),
            _ => serde_json::from_str::<serde_json::Number>(raw)
                .map(Value::Number)
                // A bare word the model failed to quote is kept as a string.
                .or_else(|_| Ok::<_, ()>(Value::String(raw.to_string())))
                .ok(),
        }
    }

    fn object(&mut self) -> Option<Map<String, Value>> {
        if !self.eat("{") {
            return None;
        }
        let mut map = Map::new();
        if self.eat("}") {
            return Some(map);
        }
        loop {
            let k = self.key()?;
            if !self.eat(":") {
                return None;
            }
            let v = self.value()?;
            map.insert(k, v);
            if self.eat("}") {
                return Some(map);
            }
            if !self.eat(",") {
                return None;
            }
        }
    }
}

/// `call:NAME{...}` (the text between the call markers) as a name and JSON arguments.
pub fn parse_call(body: &str) -> Option<(String, Value)> {
    let rest = body.trim().strip_prefix("call:")?;
    let brace = rest.find('{')?;
    let name = rest[..brace].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let mut reader = Reader {
        s: &rest[brace..],
        i: 0,
    };
    let args = reader.object()?;
    Some((name, Value::Object(args)))
}

/// What a piece of raw output turned into.
#[derive(Debug, Clone, PartialEq)]
pub enum Parsed {
    Text(String),
    Call(ToolCall),
}

/// Splits the engine's raw output (tool markers included) into text and calls as it streams.
#[derive(Debug, Default)]
pub struct CallParser {
    buf: String,
    in_call: bool,
    calls: usize,
    prefix: String,
}

impl CallParser {
    /// `prefill` is what the prompt already wrote of the answer (a forced call's opening).
    pub fn new(prefill: &str, id_prefix: &str) -> Self {
        let mut p = Self {
            prefix: id_prefix.to_string(),
            ..Self::default()
        };
        if let Some(rest) = prefill.strip_prefix(CALL_OPEN) {
            p.in_call = true;
            p.buf.push_str(rest);
        }
        p
    }

    pub fn calls(&self) -> usize {
        self.calls
    }

    pub fn feed(&mut self, piece: &str) -> Vec<Parsed> {
        self.buf.push_str(piece);
        let mut out = Vec::new();
        loop {
            if self.in_call {
                let Some(end) = self.buf.find(CALL_CLOSE) else {
                    break;
                };
                let body: String = self.buf.drain(..end + CALL_CLOSE.len()).collect();
                self.in_call = false;
                out.extend(self.call(&body[..end]));
            } else if let Some(start) = self.buf.find(CALL_OPEN) {
                let text: String = self.buf.drain(..start + CALL_OPEN.len()).collect();
                let text = &text[..start];
                if !text.is_empty() {
                    out.push(Parsed::Text(text.to_string()));
                }
                self.in_call = true;
            } else {
                // Keep a possible marker prefix at the end for the next piece.
                let keep = (1..CALL_OPEN.len())
                    .rev()
                    .find(|&n| self.buf.ends_with(&CALL_OPEN[..n]))
                    .unwrap_or(0);
                let cut = self.buf.len() - keep;
                if cut > 0 {
                    let text: String = self.buf.drain(..cut).collect();
                    out.push(Parsed::Text(text.replace(QUOTE, "\"")));
                }
                break;
            }
        }
        out
    }

    /// At the end: an unfinished call is still parsed if it can be, else returned as text.
    pub fn finish(&mut self) -> Vec<Parsed> {
        let rest = std::mem::take(&mut self.buf);
        if self.in_call {
            self.in_call = false;
            let parsed = self.call(&rest);
            if !parsed.is_empty() {
                return parsed;
            }
            return vec![Parsed::Text(rest.replace(QUOTE, "\""))];
        }
        if rest.is_empty() {
            vec![]
        } else {
            vec![Parsed::Text(rest.replace(QUOTE, "\""))]
        }
    }

    fn call(&mut self, body: &str) -> Vec<Parsed> {
        match parse_call(body) {
            Some((name, args)) => {
                self.calls += 1;
                vec![Parsed::Call(ToolCall {
                    id: format!("call_{}_{}", self.prefix, self.calls),
                    kind: "function".into(),
                    function: FunctionCall {
                        name,
                        arguments: args.to_string(),
                    },
                })]
            }
            None => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn weather_tool() -> ToolSpec {
        serde_json::from_value(json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": { "type": "string", "description": "City name" },
                        "unit": { "type": "string", "enum": ["c", "f"] },
                        "days": { "type": "integer" }
                    },
                    "required": ["city"]
                }
            }
        }))
        .unwrap()
    }

    #[test]
    fn declarations_follow_the_template() {
        assert_eq!(
            declaration(&weather_tool()),
            "<|tool>declaration:get_weather{description:<|\"|>Current weather for a city<|\"|>,parameters:{properties:{city:{description:<|\"|>City name<|\"|>,type:<|\"|>STRING<|\"|>},days:{type:<|\"|>INTEGER<|\"|>},unit:{enum:[<|\"|>c<|\"|>,<|\"|>f<|\"|>],type:<|\"|>STRING<|\"|>}},required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>"
        );
    }

    #[test]
    fn calls_and_responses_round_trip() {
        let call = ToolCall {
            id: "call_1".into(),
            kind: "function".into(),
            function: FunctionCall {
                name: "get_weather".into(),
                arguments: json!({"city": "Paris", "days": 2, "opts": {"metric": true}, "tags": ["a", "b"]}).to_string(),
            },
        };
        let text = call_text(&call);
        assert_eq!(
            text,
            "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:2,opts:{metric:true},tags:[<|\"|>a<|\"|>,<|\"|>b<|\"|>]}<tool_call|>"
        );
        let body = text
            .strip_prefix(CALL_OPEN)
            .unwrap()
            .strip_suffix(CALL_CLOSE)
            .unwrap();
        let (name, args) = parse_call(body).unwrap();
        assert_eq!(name, "get_weather");
        assert_eq!(
            args,
            serde_json::from_str::<Value>(&call.function.arguments).unwrap()
        );
        assert_eq!(
            response_text("get_weather", "18 C, sunny"),
            "<|tool_response>response:get_weather{value:<|\"|>18 C, sunny<|\"|>}<tool_response|>"
        );
    }

    #[test]
    fn the_parser_splits_text_and_calls_across_pieces() {
        let mut p = CallParser::new("", "x");
        let mut out = Vec::new();
        for piece in [
            "Let me check. ",
            "<|tool_call>",
            "call:get_weather{city:",
            "<|\"|>",
            "Paris",
            "<|\"|>",
            "}",
            "<tool_call|>",
        ] {
            out.extend(p.feed(piece));
        }
        out.extend(p.finish());
        assert_eq!(out[0], Parsed::Text("Let me check. ".into()));
        let Parsed::Call(call) = &out[1] else {
            panic!("{out:?}")
        };
        assert_eq!(call.function.name, "get_weather");
        assert_eq!(call.function.arguments, r#"{"city":"Paris"}"#);
        assert_eq!(call.id, "call_x_1");
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn a_forced_call_starts_inside_the_call() {
        let mut p = CallParser::new("<|tool_call>call:get_weather{", "y");
        let mut out = p.feed("city:<|\"|>Rome<|\"|>}<tool_call|>");
        out.extend(p.finish());
        let Parsed::Call(call) = &out[0] else {
            panic!("{out:?}")
        };
        assert_eq!(call.function.arguments, r#"{"city":"Rome"}"#);
    }

    #[test]
    fn plain_answers_pass_through_and_quotes_read_as_quotes() {
        let mut p = CallParser::new("", "z");
        let mut out = p.feed("It is <|\"|>fine<|\"|>.");
        out.extend(p.finish());
        let text: String = out
            .iter()
            .map(|o| match o {
                Parsed::Text(t) => t.as_str(),
                _ => "",
            })
            .collect();
        assert_eq!(text, "It is \"fine\".");
        assert_eq!(p.calls(), 0);
    }
}
