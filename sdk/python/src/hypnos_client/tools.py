"""Helpers for tool calling (docs/api.md section 7.5).

    reply = hypnos.chat(model, messages, tools=[weather])
    message = reply["choices"][0]["message"]
    if reply["choices"][0]["finish_reason"] == "tool_calls":
        messages.append(assistant_message(reply))
        for call in message["tool_calls"]:
            result = run(call["function"]["name"], parse_arguments(call))
            messages.append(tool_message(call["id"], json.dumps(result)))
        reply = hypnos.chat(model, messages, tools=[weather])
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional, Union, cast

from .types import ChatCompletion, ChatMessage, ResponseMessage, ToolCall, ToolSpec


def function_tool(
    name: str,
    description: Optional[str] = None,
    parameters: Optional[Mapping[str, Any]] = None,
) -> ToolSpec:
    """A ``tools`` entry: ``{"type": "function", "function": {name, description?, parameters?}}``."""
    function: Dict[str, Any] = {"name": name}
    if description is not None:
        function["description"] = description
    if parameters is not None:
        function["parameters"] = dict(parameters)
    return cast(ToolSpec, {"type": "function", "function": function})


def tool_message(tool_call_id: str, content: str, name: Optional[str] = None) -> ChatMessage:
    """The ``tool`` message that sends a call's result back. Send JSON as a string."""
    message: Dict[str, Any] = {"role": "tool", "tool_call_id": tool_call_id, "content": content}
    if name is not None:
        message["name"] = name
    return cast(ChatMessage, message)


def assistant_message(reply: Union[ChatCompletion, ResponseMessage, Mapping[str, Any]]) -> ChatMessage:
    """The assistant turn of a completion (or of its ``choices[0].message``) as a history message.

    It keeps the ``tool_calls``; ``content`` becomes ``None`` when the model only called tools.
    """
    if "choices" in reply:
        reply = cast(ChatCompletion, reply)["choices"][0]["message"]
    content = reply.get("content")
    calls = reply.get("tool_calls")
    message: Dict[str, Any] = {"role": "assistant", "content": (content or None) if calls else (content or "")}
    if calls:
        message["tool_calls"] = [
            {
                "id": c["id"],
                "type": c.get("type", "function"),
                "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]},
            }
            for c in calls
        ]
    return cast(ChatMessage, message)


def parse_arguments(call: Union[ToolCall, Mapping[str, Any]]) -> Any:
    """A call's ``function.arguments`` decoded from JSON (``{}`` when empty).

    Raises :class:`ValueError` when the model wrote invalid JSON: validate the result before
    running the tool, the model follows the schema only as well as it manages.
    """
    raw = call["function"].get("arguments") or ""
    if not raw.strip():
        return {}
    return json.loads(raw)
