/**
 * Helpers for tool calling (docs/api.md section 7.5): building the history messages of a
 * round trip and reading a call's arguments.
 */

import type { AssistantMessage, ChatCompletion, CompletionMessage, ToolCall, ToolMessage } from "./types.js";

/** The `tool` message carrying the result of call `toolCallId`; send JSON as a string. */
export function toolMessage(toolCallId: string, content: string): ToolMessage {
  return { role: "tool", tool_call_id: toolCallId, content };
}

/**
 * The assistant message of a completion (or the message itself), ready to append to the
 * history: with its `tool_calls`, and `content` `null` when it only called tools.
 */
export function assistantMessage(source: ChatCompletion | CompletionMessage): AssistantMessage {
  const message = "choices" in source ? source.choices[0]?.message : source;
  if (!message) throw new TypeError("the completion has no choices");
  const calls = message.tool_calls ?? [];
  const out: AssistantMessage = {
    role: "assistant",
    content: calls.length && !message.content ? null : (message.content ?? ""),
  };
  if (calls.length) out.tool_calls = calls;
  return out;
}

/** `JSON.parse` of a call's `function.arguments` (`{}` when empty); throws `SyntaxError` on bad JSON. */
export function parseArguments<T = Record<string, unknown>>(call: ToolCall): T {
  const raw = call.function.arguments;
  return (raw && raw.trim() ? JSON.parse(raw) : {}) as T;
}
