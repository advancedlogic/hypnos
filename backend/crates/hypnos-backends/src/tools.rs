//! OpenAI tool calling on each model family's native format: Gemma 4's (API-9) and Qwen's XML
//! (ENG-6, below [`ToolFormat::QwenXml`]).
//!
//! Gemma 4, as its GGUF chat template defines it:
//! tools are declared in the system turn as `<|tool>declaration:NAME{...}<tool|>`, the model calls
//! one with `<|tool_call>call:NAME{key:value,...}<tool_call|>` and then emits `<|tool_response>`
//! (the engine stops there), and results come back in the same model turn as
//! `<|tool_response>response:NAME{value:...}<tool_response|>`. Values use Gemma's own notation:
//! strings between `<|"|>` tokens, bare keys inside a call.

use std::collections::BTreeSet;

use hypnos_core::openai::{FunctionCall, ToolCall, ToolChoice, ToolSpec};
use serde_json::{Map, Value};

pub const QUOTE: &str = "<|\"|>";
pub const CALL_OPEN: &str = "<|tool_call>";
pub const CALL_CLOSE: &str = "<tool_call|>";
pub const RESPONSE_OPEN: &str = "<|tool_response>";
pub const RESPONSE_CLOSE: &str = "<tool_response|>";

pub const QWEN_CALL_OPEN: &str = "<tool_call>";
pub const QWEN_CALL_CLOSE: &str = "</tool_call>";

/// The engine passes these marker tokens through as text when tools are active (Qwen's are
/// ordinary tokens already; listed in case a GGUF marks them special).
pub const PASS_THROUGH: [&str; 5] = [
    CALL_OPEN,
    CALL_CLOSE,
    QUOTE,
    QWEN_CALL_OPEN,
    QWEN_CALL_CLOSE,
];

/// A model family's tool-call notation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolFormat {
    /// `<|tool_call>call:NAME{key:value}<tool_call|>` (Gemma 4).
    Gemma4,
    /// `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>`
    /// (Qwen 3.5 and later, Qwen3-Coder, Nemotron 3.5, LFM3).
    QwenXml,
    /// `<tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value></tool_call>` (GLM 4.x).
    GlmXml,
}

impl ToolFormat {
    fn open(self) -> &'static str {
        match self {
            ToolFormat::Gemma4 => CALL_OPEN,
            ToolFormat::QwenXml | ToolFormat::GlmXml => QWEN_CALL_OPEN,
        }
    }

    fn close(self) -> &'static str {
        match self {
            ToolFormat::Gemma4 => CALL_CLOSE,
            ToolFormat::QwenXml | ToolFormat::GlmXml => QWEN_CALL_CLOSE,
        }
    }

    /// What the prompt writes after the generation prompt to force a call (no grammar: a
    /// required or named call is forced by writing its opening).
    pub fn prefill(self, choice: &ToolChoice) -> String {
        match (self, choice) {
            (_, ToolChoice::Auto | ToolChoice::None) => String::new(),
            (ToolFormat::Gemma4, ToolChoice::Required) => format!("{CALL_OPEN}call:"),
            (ToolFormat::Gemma4, ToolChoice::Function(name)) => {
                format!("{CALL_OPEN}call:{name}{{")
            }
            (ToolFormat::QwenXml, ToolChoice::Required) => format!("{QWEN_CALL_OPEN}\n<function="),
            (ToolFormat::QwenXml, ToolChoice::Function(name)) => {
                format!("{QWEN_CALL_OPEN}\n<function={name}>\n")
            }
            (ToolFormat::GlmXml, ToolChoice::Required) => QWEN_CALL_OPEN.to_string(),
            (ToolFormat::GlmXml, ToolChoice::Function(name)) => format!("{QWEN_CALL_OPEN}{name}"),
        }
    }
}

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

// --- Qwen ---------------------------------------------------------------------------------------

/// The system-turn block that declares `tools`, as Qwen 3.5+ templates write it.
pub fn qwen_tools_block(tools: &[ToolSpec]) -> String {
    let mut out = String::from("# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for tool in tools {
        out.push('\n');
        out.push_str(&py_json(&tool_json(tool)));
    }
    out.push_str("\n</tools>");
    out.push_str(QWEN_TOOL_INSTRUCTIONS);
    out
}

/// A tool as templates see it: `{"type": "function", "function": {name, description, parameters}}`
/// in that order, without absent fields.
pub fn tool_json(tool: &ToolSpec) -> Value {
    let mut outer = Map::new();
    outer.insert("type".into(), Value::String(tool.kind.clone()));
    outer.insert("function".into(), function_json(tool));
    Value::Object(outer)
}

/// The `function` object of a tool (LFM3 declares these alone).
pub fn function_json(tool: &ToolSpec) -> Value {
    let f = &tool.function;
    let mut m = Map::new();
    m.insert("name".into(), Value::String(f.name.clone()));
    if let Some(d) = &f.description {
        m.insert("description".into(), Value::String(d.clone()));
    }
    if let Some(p) = &f.parameters {
        m.insert("parameters".into(), p.clone());
    }
    Value::Object(m)
}

/// Qwen3-Coder's (and Nemotron 3.5's) declaration block: each tool as XML, after `# Tools`.
pub fn coder_tools_block(tools: &[ToolSpec]) -> String {
    let extra = |map: &Map<String, Value>, handled: &[&str]| {
        let mut out = String::new();
        for (k, v) in map.iter().filter(|(k, _)| !handled.contains(&k.as_str())) {
            let v = match v {
                Value::Object(_) | Value::Array(_) => py_json(v),
                other => py_str(other),
            };
            out.push_str(&format!("\n<{k}>{v}</{k}>"));
        }
        out
    };
    let mut out =
        String::from("\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for tool in tools {
        let f = &tool.function;
        out.push_str(&format!("\n<function>\n<name>{}</name>", f.name));
        if let Some(d) = &f.description {
            out.push_str(&format!("\n<description>{}</description>", d.trim()));
        }
        out.push_str("\n<parameters>");
        let params = f.parameters.as_ref().and_then(Value::as_object);
        if let Some(props) = params
            .and_then(|p| p.get("properties"))
            .and_then(Value::as_object)
        {
            for (name, fields) in props {
                out.push_str(&format!("\n<parameter>\n<name>{name}</name>"));
                if let Some(fields) = fields.as_object() {
                    if let Some(t) = fields.get("type") {
                        out.push_str(&format!("\n<type>{}</type>", py_str(t)));
                    }
                    if let Some(d) = fields.get("description").and_then(Value::as_str) {
                        out.push_str(&format!("\n<description>{}</description>", d.trim()));
                    }
                    out.push_str(&extra(fields, &["name", "type", "description"]));
                }
                out.push_str("\n</parameter>");
            }
        }
        if let Some(p) = params {
            out.push_str(&extra(p, &["type", "properties"]));
        }
        out.push_str("\n</parameters>\n</function>");
    }
    out.push_str("\n</tools>");
    out.push_str(QWEN_TOOL_INSTRUCTIONS);
    out
}

/// Jinja's `| string` of a JSON value, as Python prints it (`True`, `None`, numbers, strings).
fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Null => "None".into(),
        Value::Object(_) | Value::Array(_) => py_json(v),
        Value::Number(n) => n.to_string(),
    }
}

/// How a family writes a past call's argument values: strings as they are; mappings and lists
/// as JSON; scalars as `| string` (Coder, Nemotron) or as JSON (Qwen 3.5+, LFM3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgStyle {
    Json,
    PyString,
}

/// A past call in the Qwen XML notation (no separator before it).
pub fn xml_call_text(call: &ToolCall, style: ArgStyle) -> String {
    let args: Value =
        serde_json::from_str(&call.function.arguments).unwrap_or(Value::Object(Map::new()));
    let mut out = format!("{QWEN_CALL_OPEN}\n<function={}>\n", call.function.name);
    if let Value::Object(map) = &args {
        for (k, v) in map {
            let v = match (v, style) {
                (Value::String(s), _) => s.clone(),
                (Value::Object(_) | Value::Array(_), _) | (_, ArgStyle::Json) => py_json(v),
                (other, ArgStyle::PyString) => py_str(other),
            };
            out.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n"));
        }
    }
    out.push_str("</function>\n");
    out.push_str(QWEN_CALL_CLOSE);
    out
}

/// GLM 4.x: the declaration block opening the prompt's first system turn.
pub fn glm_tools_block(tools: &[ToolSpec]) -> String {
    let mut out = String::from("<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n");
    for tool in tools {
        out.push_str(&py_json(&tool_json(tool)));
        out.push('\n');
    }
    out.push_str("</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>");
    out
}

/// A past call in GLM's notation.
pub fn glm_call_text(call: &ToolCall) -> String {
    let args: Value =
        serde_json::from_str(&call.function.arguments).unwrap_or(Value::Object(Map::new()));
    let mut out = format!("{QWEN_CALL_OPEN}{}", call.function.name);
    if let Value::Object(map) = &args {
        for (k, v) in map {
            let v = match v {
                Value::String(s) => s.clone(),
                other => py_json(other),
            };
            out.push_str(&format!("<arg_key>{k}</arg_key><arg_value>{v}</arg_value>"));
        }
    }
    out.push_str(QWEN_CALL_CLOSE);
    out
}

/// `NAME<arg_key>K</arg_key><arg_value>V</arg_value>...` (GLM) as a name and JSON arguments.
pub fn parse_glm_call(body: &str, string_params: &BTreeSet<String>) -> Option<(String, Value)> {
    let body = body.trim();
    let name_end = body.find("<arg_key>").unwrap_or(body.len());
    let name = body[..name_end].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let mut map = Map::new();
    let mut s = &body[name_end..];
    while let Some(k0) = s.find("<arg_key>") {
        let rest = &s[k0 + "<arg_key>".len()..];
        let k1 = rest.find("</arg_key>")?;
        let key = rest[..k1].trim().to_string();
        let rest = &rest[k1 + "</arg_key>".len()..];
        let v0 = rest.find("<arg_value>")?;
        let rest = &rest[v0 + "<arg_value>".len()..];
        let v1 = rest.find("</arg_value>").unwrap_or(rest.len());
        let raw = &rest[..v1];
        let value = if string_params.contains(&format!("{name}.{key}")) {
            Value::String(raw.to_string())
        } else {
            serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
        };
        map.insert(key, value);
        s = rest.get(v1 + "</arg_value>".len()..).unwrap_or("");
    }
    Some((name, Value::Object(map)))
}

const QWEN_TOOL_INSTRUCTIONS: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// A past call as a Qwen template writes it in the assistant turn (without the separator before
/// it, which depends on its place).
pub fn qwen_call_text(call: &ToolCall) -> String {
    xml_call_text(call, ArgStyle::Json)
}

/// One tool result inside the user turn that carries them.
pub fn qwen_response_text(content: &str) -> String {
    format!("\n<tool_response>\n{content}\n</tool_response>")
}

/// Jinja's `tojson` (Python `json.dumps`: `", "` and `": "` separators, non-ASCII kept).
pub fn py_json(v: &Value) -> String {
    match v {
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(py_json).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(k, v)| format!("{}: {}", Value::String(k.clone()), py_json(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => other.to_string(),
    }
}

/// `<function=NAME>` and its `<parameter=KEY>VALUE</parameter>` blocks (the text between the
/// call markers) as a name and JSON arguments. A value is a string where the tool's schema says
/// so, else JSON when it parses as JSON, else a string.
pub fn parse_qwen_call(body: &str, string_params: &BTreeSet<String>) -> Option<(String, Value)> {
    let rest = body.trim_start().strip_prefix("<function=")?;
    let gt = rest.find('>')?;
    let name = rest[..gt].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let mut s = &rest[gt + 1..];
    let mut map = Map::new();
    while let Some(i) = s.find("<parameter=") {
        let after = &s[i + "<parameter=".len()..];
        let gt = after.find('>')?;
        let key = after[..gt].trim().to_string();
        let tail = &after[gt + 1..];
        let end = tail
            .find("</parameter>")
            .or_else(|| tail.find("</function>"))
            .unwrap_or(tail.len());
        let raw = tail[..end].strip_prefix('\n').unwrap_or(&tail[..end]);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        let value = if string_params.contains(&format!("{name}.{key}")) {
            Value::String(raw.to_string())
        } else {
            serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
        };
        map.insert(key, value);
        s = tail[end..]
            .strip_prefix("</parameter>")
            .unwrap_or(&tail[end..]);
    }
    Some((name, Value::Object(map)))
}

/// `function.param` for every string-typed parameter of `tools`.
fn string_params(tools: &[ToolSpec]) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for t in tools {
        let props = t
            .function
            .parameters
            .as_ref()
            .and_then(|p| p.get("properties"))
            .and_then(Value::as_object);
        for (k, v) in props.into_iter().flatten() {
            if v.get("type").and_then(Value::as_str) == Some("string") {
                out.insert(format!("{}.{k}", t.function.name));
            }
        }
    }
    out
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
#[derive(Debug)]
pub struct CallParser {
    format: ToolFormat,
    buf: String,
    in_call: bool,
    calls: usize,
    prefix: String,
    /// `function.param` of the string-typed parameters (Qwen values carry no quotes).
    string_params: BTreeSet<String>,
}

impl CallParser {
    /// `prefill` is what the prompt already wrote of the answer (a forced call's opening).
    pub fn new(format: ToolFormat, prefill: &str, id_prefix: &str, tools: &[ToolSpec]) -> Self {
        let mut p = Self {
            format,
            buf: String::new(),
            in_call: false,
            calls: 0,
            prefix: id_prefix.to_string(),
            string_params: string_params(tools),
        };
        if let Some(rest) = prefill.strip_prefix(format.open()) {
            p.in_call = true;
            p.buf.push_str(rest);
        }
        p
    }

    /// Text outside calls; Gemma's quote token reads as `"`.
    fn text(&self, t: &str) -> String {
        match self.format {
            ToolFormat::Gemma4 => t.replace(QUOTE, "\""),
            ToolFormat::QwenXml | ToolFormat::GlmXml => t.to_string(),
        }
    }

    #[cfg(test)]
    pub fn calls(&self) -> usize {
        self.calls
    }

    pub fn feed(&mut self, piece: &str) -> Vec<Parsed> {
        self.buf.push_str(piece);
        let (open, close) = (self.format.open(), self.format.close());
        let mut out = Vec::new();
        loop {
            if self.in_call {
                let Some(end) = self.buf.find(close) else {
                    break;
                };
                let body: String = self.buf.drain(..end + close.len()).collect();
                self.in_call = false;
                out.extend(self.call(&body[..end]));
            } else if let Some(start) = self.buf.find(open) {
                let text: String = self.buf.drain(..start + open.len()).collect();
                let text = &text[..start];
                if !text.is_empty() {
                    out.push(Parsed::Text(text.to_string()));
                }
                self.in_call = true;
            } else {
                // Keep a possible marker prefix at the end for the next piece.
                let keep = (1..open.len())
                    .rev()
                    .find(|&n| self.buf.ends_with(&open[..n]))
                    .unwrap_or(0);
                let cut = self.buf.len() - keep;
                if cut > 0 {
                    let text: String = self.buf.drain(..cut).collect();
                    out.push(Parsed::Text(self.text(&text)));
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
            return vec![Parsed::Text(self.text(&rest))];
        }
        if rest.is_empty() {
            vec![]
        } else {
            vec![Parsed::Text(self.text(&rest))]
        }
    }

    fn call(&mut self, body: &str) -> Vec<Parsed> {
        let parsed = match self.format {
            ToolFormat::Gemma4 => parse_call(body),
            ToolFormat::QwenXml => parse_qwen_call(body, &self.string_params),
            ToolFormat::GlmXml => parse_glm_call(body, &self.string_params),
        };
        match parsed {
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
        let mut p = CallParser::new(ToolFormat::Gemma4, "", "x", &[]);
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
        let mut p = CallParser::new(
            ToolFormat::Gemma4,
            "<|tool_call>call:get_weather{",
            "y",
            &[],
        );
        let mut out = p.feed("city:<|\"|>Rome<|\"|>}<tool_call|>");
        out.extend(p.finish());
        let Parsed::Call(call) = &out[0] else {
            panic!("{out:?}")
        };
        assert_eq!(call.function.arguments, r#"{"city":"Rome"}"#);
    }

    #[test]
    fn plain_answers_pass_through_and_quotes_read_as_quotes() {
        let mut p = CallParser::new(ToolFormat::Gemma4, "", "z", &[]);
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

    #[test]
    fn qwen_calls_render_and_parse_like_the_template() {
        let call = ToolCall {
            id: "c1".into(),
            kind: "function".into(),
            function: FunctionCall {
                name: "get_weather".into(),
                arguments: r#"{"city":"Paris","days":3,"opts":{"unit":"c"}}"#.into(),
            },
        };
        let text = qwen_call_text(&call);
        assert_eq!(
            text,
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n\
             <parameter=days>\n3\n</parameter>\n<parameter=opts>\n{\"unit\": \"c\"}\n</parameter>\n\
             </function>\n</tool_call>"
        );
        // Back again: the schema keeps a numeric-looking string a string.
        let strings: BTreeSet<String> = ["get_weather.city".to_string()].into();
        let body = &text[QWEN_CALL_OPEN.len()..text.len() - QWEN_CALL_CLOSE.len()];
        let (name, args) = parse_qwen_call(body, &strings).unwrap();
        assert_eq!(name, "get_weather");
        assert_eq!(
            args,
            json!({"city": "Paris", "days": 3, "opts": {"unit": "c"}})
        );
        let zip = "<function=f>\n<parameter=zip>\n02134\n</parameter>\n</function>";
        let (_, a) = parse_qwen_call(zip, &["f.zip".to_string()].into()).unwrap();
        assert_eq!(a, json!({"zip": "02134"}));
        // The declarations block lists each tool as Python-style JSON.
        let block = qwen_tools_block(&[weather_tool()]);
        assert!(block.starts_with("# Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\""), "{block}");
        assert!(block.contains("</tools>\n\nIf you choose to call a function ONLY reply"));
    }

    #[test]
    fn the_qwen_parser_streams_text_and_calls() {
        let tools = [weather_tool()];
        let mut p = CallParser::new(ToolFormat::QwenXml, "", "q", &tools);
        let mut parsed = Vec::new();
        for piece in [
            "Let me check.\n\n<tool",
            "_call>\n<function=get_weather>\n<parameter=city>\nPar",
            "is\n</parameter>\n</function>\n</tool_call>",
        ] {
            parsed.extend(p.feed(piece));
        }
        parsed.extend(p.finish());
        let text: String = parsed
            .iter()
            .filter_map(|x| match x {
                Parsed::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "Let me check.\n\n");
        let calls: Vec<&ToolCall> = parsed
            .iter()
            .filter_map(|x| match x {
                Parsed::Call(c) => Some(c),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, r#"{"city":"Paris"}"#);
        // A forced call: the prompt wrote the opening, the model continues inside it.
        let prefill = ToolFormat::QwenXml.prefill(&ToolChoice::Function("get_weather".into()));
        let mut p = CallParser::new(ToolFormat::QwenXml, &prefill, "r", &tools);
        let mut parsed = p.feed("<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>");
        parsed.extend(p.finish());
        assert!(
            matches!(&parsed[..], [Parsed::Call(c)] if c.function.arguments == r#"{"city":"Oslo"}"#),
            "{parsed:?}"
        );
    }

    #[test]
    fn glm_calls_round_trip() {
        let call = ToolCall {
            id: "c1".into(),
            kind: "function".into(),
            function: FunctionCall {
                name: "get_weather".into(),
                arguments: r#"{"city":"Paris","days":3}"#.into(),
            },
        };
        let text = glm_call_text(&call);
        assert_eq!(
            text,
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>\
             <arg_key>days</arg_key><arg_value>3</arg_value></tool_call>"
        );
        let tools = [weather_tool()];
        let mut p = CallParser::new(ToolFormat::GlmXml, "", "g", &tools);
        let mut parsed = p.feed("Checking.");
        parsed.extend(p.feed(&text));
        parsed.extend(p.finish());
        let call = parsed
            .iter()
            .find_map(|x| match x {
                Parsed::Call(c) => Some(c),
                _ => None,
            })
            .unwrap();
        assert_eq!(call.function.name, "get_weather");
        assert_eq!(call.function.arguments, r#"{"city":"Paris","days":3}"#);
        // A forced named call starts inside the call, after the name.
        let prefill = ToolFormat::GlmXml.prefill(&ToolChoice::Function("get_weather".into()));
        let mut p = CallParser::new(ToolFormat::GlmXml, &prefill, "h", &tools);
        let mut parsed = p.feed("<arg_key>city</arg_key><arg_value>Rome</arg_value></tool_call>");
        parsed.extend(p.finish());
        assert!(
            matches!(&parsed[..], [Parsed::Call(c)] if c.function.arguments == r#"{"city":"Rome"}"#),
            "{parsed:?}"
        );
    }
}
