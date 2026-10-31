"""Tool calling (docs/api.md section 7.5) against the mock server's deterministic tool model."""

from __future__ import annotations

import json

import pytest

from hypnos_client import (
    AsyncClient,
    BadRequestError,
    Client,
    assistant_message,
    function_tool,
    parse_arguments,
    tool_message,
)

CHAT = "gemma4:26b-q4"

WEATHER = function_tool(
    "get_weather",
    "Current weather for a city",
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)
TIME = {"type": "function", "function": {"name": "get_time"}}


def test_auto_calls_the_named_tool(client: Client) -> None:
    text = "please get_weather for Paris"
    reply = client.chat(CHAT, [{"role": "user", "content": text}], tools=[TIME, WEATHER])
    choice = reply["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == ""
    [call] = choice["message"]["tool_calls"]
    assert call["type"] == "function"
    assert call["id"]
    assert call["function"]["name"] == "get_weather"
    assert parse_arguments(call) == {"text": text}


def test_auto_without_a_match_answers_in_text(client: Client) -> None:
    reply = client.chat(CHAT, [{"role": "user", "content": "hello"}], tools=[WEATHER])
    message = reply["choices"][0]["message"]
    assert message["content"] == "echo: hello"
    assert "tool_calls" not in message
    assert reply["choices"][0]["finish_reason"] == "stop"


def test_round_trip(client: Client) -> None:
    messages = [{"role": "user", "content": "what is the weather?"}]
    reply = client.chat(
        CHAT, messages, tools=[WEATHER],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )  # fmt: skip
    assert reply["choices"][0]["finish_reason"] == "tool_calls"
    history = assistant_message(reply)
    assert history["role"] == "assistant"
    assert history["content"] is None
    [call] = history["tool_calls"]
    messages.append(history)
    messages.append(tool_message(call["id"], json.dumps({"temperature": 18})))
    assert messages[-1] == {"role": "tool", "tool_call_id": call["id"], "content": '{"temperature": 18}'}

    final = client.chat(CHAT, messages, tools=[WEATHER])
    assert final["choices"][0]["message"]["content"] == 'tool said: {"temperature": 18}'
    assert final["choices"][0]["finish_reason"] == "stop"
    assert assistant_message(final) == {"role": "assistant", "content": final["choices"][0]["message"]["content"]}


def test_stream_accumulates_tool_calls(client: Client) -> None:
    text = "anything"
    with client.chat_stream(
        CHAT, [{"role": "user", "content": text}], tools=[WEATHER, TIME], tool_choice="required"
    ) as stream:
        chunks = list(stream)
    assert any(c["choices"] and c["choices"][0]["delta"].get("tool_calls") for c in chunks)
    assert stream.finish_reason == "tool_calls"
    [call] = stream.tool_calls
    assert call["function"]["name"] == "get_weather"
    assert call["id"]
    assert parse_arguments(call) == {"text": text}
    completion = stream.final()
    assert completion["choices"][0]["finish_reason"] == "tool_calls"
    assert completion["choices"][0]["message"]["tool_calls"] == [call]


def test_tool_choice_none_hides_the_tools(client: Client) -> None:
    reply = client.chat(CHAT, [{"role": "user", "content": "get_weather"}], tools=[WEATHER], tool_choice="none")
    assert reply["choices"][0]["message"]["content"] == "echo: get_weather"


def test_unknown_named_tool_is_a_bad_request(client: Client) -> None:
    with pytest.raises(BadRequestError) as err:
        client.chat(
            CHAT, [{"role": "user", "content": "hi"}], tools=[WEATHER],
            tool_choice={"type": "function", "function": {"name": "nope"}},
        )  # fmt: skip
    assert err.value.status == 400
    assert err.value.type == "invalid_request_error"


def test_stream_merges_fragments_by_index() -> None:
    # OpenAI servers split a call's arguments over chunks; the accumulator joins them by index.
    from hypnos_client.streams import _ChatState

    state = _ChatState(None)
    frames = [
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c0", "type": "function",
                                                              "function": {"name": "f", "arguments": '{"a"'}}]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 1, "id": "c1",
                                                              "function": {"name": "g", "arguments": "{}"}}]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": ": 1}"}}]}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]  # fmt: skip
    for frame in frames:
        state.add(json.dumps(frame))
    calls = state.completion()["choices"][0]["message"]["tool_calls"]
    assert [c["id"] for c in calls] == ["c0", "c1"]
    assert parse_arguments(calls[0]) == {"a": 1}
    assert calls[1]["function"]["name"] == "g"
    assert state.finish_reason == "tool_calls"


@pytest.mark.anyio
async def test_async_tool_stream(server) -> None:
    async with AsyncClient(server.url) as c:
        stream = await c.chat_stream(
            CHAT, [{"role": "user", "content": "x"}], tools=[WEATHER],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )  # fmt: skip
        completion = await stream.final()
    assert completion["choices"][0]["finish_reason"] == "tool_calls"
    assert stream.tool_calls[0]["function"]["name"] == "get_weather"
