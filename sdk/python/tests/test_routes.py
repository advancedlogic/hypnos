"""Every route against the mock server: request shapes, response shapes, streaming."""

from __future__ import annotations

import base64
import time

import pytest

from hypnos_client import APIError, BadRequestError, Client, NotFoundError, StreamError

CHAT = "gemma4:26b-q4"
STT = "gemma4:e4b-q4"
EMBED = "gemmaembedding2"
TTS = "kokoro-82m-q4"
IMAGE = "moss-image"
DOCS = "docling"
NER = "gliner2.5-multi"

USER = [{"role": "user", "content": "hello there"}]


def test_health_ready_models(client: Client) -> None:
    assert client.health()["status"] == "ok"
    assert client.is_ready()
    client.wait_until_ready(timeout=1)
    ids = {m["id"] for m in client.models()["data"]}
    assert {CHAT, STT, EMBED, TTS, IMAGE, DOCS, NER} <= ids
    assert "hypnos_http_requests_total" in client.metrics()


def test_models_catalog(client: Client) -> None:
    catalog = client.models_catalog()
    assert catalog["object"] == "list"
    assert len(catalog["data"]) >= 10
    entries = {e["id"]: e for e in catalog["data"]}
    assert entries[CHAT]["configured"] is True
    lfm3 = entries["lfm3:1.7b"]
    assert lfm3["configured"] is False and "enabled" not in lfm3
    quants = {v["quant"]: v for v in lfm3["variants"]}
    assert {"Q8_0", "Q4_K_M"} <= set(quants)
    assert quants["Q8_0"]["default"] is True
    assert 'weights_path = "LFM3-1.7B-Q8_0.gguf"' in quants["Q8_0"]["config_block"]
    assert not any(v["installed"] for e in catalog["data"] for v in e["variants"])
    vision = {e["id"] for e in client.models_catalog(modality="vision")["data"]}
    assert {"gemma4:26b-q4", "gemma4:e4b-q4", "lfm3-vl:2b"} <= vision
    assert "lfm3:1.7b" not in vision


GIB = 1 << 30


def test_advise(client: Client) -> None:
    advice = client.advise(["chat", "voice"], vram_bytes=24 * GIB, ram_bytes=64 * GIB)
    assert advice["machine"] == {
        "vram_bytes": 24 * GIB,
        "ram_bytes": 64 * GIB,
        "unified_memory": False,
    }
    assert advice["interactive_coresident"] is True
    assert advice["specs"]
    assert "[[models]]" in advice["config"]
    assert TTS in {m["id"] for m in advice["models"]}
    with pytest.raises(BadRequestError):
        client.advise(["images"], vram_bytes=2 * GIB, ram_bytes=8 * GIB)
    with pytest.raises(BadRequestError):
        client.advise([])


def test_chat(client: Client) -> None:
    reply = client.chat(CHAT, USER, max_tokens=20, temperature=0.2, seed=1, num_ctx=4096)
    assert reply["choices"][0]["message"]["content"] == "echo: hello there"
    assert reply["choices"][0]["finish_reason"] == "stop"
    assert reply["usage"]["completion_tokens"] == 3
    telemetry = reply["hypnos_telemetry"]
    assert telemetry["placement"]["primary_tier"] == "tier1_gpu_vram"
    assert telemetry["previous_tier"] in ("tier1_gpu_vram", "tier2_pinned_ram", "tier3_nvme_ssd")


def test_chat_with_media_parts(client: Client) -> None:
    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}},
            ],
        }
    ]
    reply = client.chat(CHAT, messages)
    assert reply["choices"][0]["message"]["content"].startswith("echo: what is this")


def test_chat_stream(client: Client) -> None:
    with client.chat_stream(CHAT, [{"role": "user", "content": "count"}], max_tokens=5) as stream:
        pieces = list(stream.text())
    assert "".join(pieces) == "1 2 3 4 5"
    assert stream.finish_reason == "length"
    assert stream.usage is not None and stream.usage["completion_tokens"] == 5
    assert stream.telemetry is not None
    assert stream.request_id and stream.request_id.startswith("req-")
    assert stream.final()["choices"][0]["message"]["content"] == "1 2 3 4 5"


def test_chat_reasoning(client: Client) -> None:
    hi = [{"role": "user", "content": "hi"}]
    reply = client.chat(CHAT, hi, reasoning_effort="medium")
    assert reply["choices"][0]["message"]["reasoning_content"] == "Mock thoughts (medium)."
    with client.chat_stream(CHAT, hi, reasoning_effort="medium") as stream:
        completion = stream.final()
    assert stream.reasoning_so_far == "Mock thoughts (medium)."
    assert completion["choices"][0]["message"]["reasoning_content"] == "Mock thoughts (medium)."
    assert completion["choices"][0]["message"]["content"] == "echo: hi"
    off = client.chat(CHAT, hi, reasoning_effort="none")
    assert "reasoning_content" not in off["choices"][0]["message"]


def test_chat_stream_final(client: Client) -> None:
    stream = client.chat_stream(CHAT, USER)
    completion = stream.final()
    assert completion["choices"][0]["message"]["content"] == "echo: hello there"
    assert completion["object"] == "chat.completion"
    assert "hypnos_telemetry" in completion


def test_closing_a_stream_cancels_it(client: Client) -> None:
    started = time.monotonic()
    with client.chat_stream(CHAT, [{"role": "user", "content": "count"}], max_tokens=2000) as s:
        for i, _ in enumerate(s.text()):
            if i == 3:
                break
    # 2000 tokens at 10 ms each would take 20 s.
    assert time.monotonic() - started < 5
    # The model is free again.
    assert client.chat(CHAT, USER)["choices"][0]["message"]["content"]


def test_a_stream_that_fails_raises(client: Client, server) -> None:
    server.fail(CHAT, "stream_error")
    stream = client.chat_stream(CHAT, [{"role": "user", "content": "count"}], max_tokens=10)
    with pytest.raises(StreamError) as err:
        for _ in stream:
            pass
    assert err.value.status == 200
    assert "stream_error" in err.value.message
    assert stream.text_so_far == "1 2"


def test_embeddings(client: Client) -> None:
    one = client.embeddings(EMBED, "a")
    assert len(one["data"]) == 1
    many = client.embeddings(EMBED, ["a", "b", "c"])
    assert [d["index"] for d in many["data"]] == [0, 1, 2]
    assert many["data"][0]["embedding"] != many["data"][1]["embedding"]
    assert len(many["data"][0]["embedding"]) == 8


def test_transcribe(client: Client, tmp_path) -> None:
    clip = b"RIFF\x00\x00\x00\x00WAVEdata"
    reply = client.transcribe(STT, clip, filename="clip.wav", language="it", prompt="names")
    assert reply["text"] == f"mock transcript of {len(clip)} bytes"
    assert reply["language"] == "it"
    path = tmp_path / "clip.wav"
    path.write_bytes(clip)
    assert client.transcribe(STT, path)["text"] == reply["text"]


def test_speech(client: Client) -> None:
    wav = client.speech(TTS, "one two three", voice="alloy")
    assert wav.content_type == "audio/wav"
    assert wav.audio[:4] == b"RIFF"
    assert wav.telemetry.previous_tier in ("tier1_gpu_vram", "tier2_pinned_ram", "tier3_nvme_ssd")
    assert wav.telemetry.gpu_layers is not None
    pcm = client.speech(TTS, "one two three", response_format="pcm")
    assert pcm.content_type == "audio/pcm"
    # 50 ms per word at 24 kHz, 16-bit: 3 * 1200 samples * 2 bytes.
    assert len(pcm.audio) == 3 * 1200 * 2


def test_speech_stream(client: Client) -> None:
    with client.speech_stream(TTS, "one two three four", response_format="pcm") as stream:
        chunks = list(stream)
    assert sum(len(c) for c in chunks) == 4 * 1200 * 2
    assert stream.content_type == "audio/pcm"
    assert stream.telemetry.previous_tier is not None


def test_images(client: Client) -> None:
    reply = client.images(IMAGE, "a lighthouse", n=2, size="512x512", steps=4, seed=7)
    assert len(reply["data"]) == 2
    assert base64.b64decode(reply["data"][0]["b64_json"])[:8] == b"\x89PNG\r\n\x1a\n"
    assert "hypnos_telemetry" in reply


def test_entities(client: Client) -> None:
    text = "Mario Rossi works in Milan."
    reply = client.entities(NER, [text, "nothing here"], ["person", "city"], threshold=0.4)
    found = reply["data"][0]["entities"]
    assert [e["text"] for e in found] == ["Mario", "Rossi", "Milan"]
    assert all(text[e["start"] : e["end"]] == e["text"] for e in found)
    assert reply["data"][1]["entities"] == []
    assert reply["usage"]["texts"] == 2


def test_documents_and_pictures(client: Client) -> None:
    doc = client.convert_document(
        DOCS, b"%PDF-1.4 fake", filename="report.pdf", pages="2-3", return_pictures=True
    )
    assert doc["page_count"] == 3
    assert [p["page"] for p in doc["pages"]] == [2, 3]
    assert "## Page 2" in doc["markdown"]
    doc_id = doc["id"]
    assert doc["pages"][0]["picture_urls"] == [f"/v1/documents/{doc_id}/pictures/1"]

    listed = client.document_pictures(doc_id)
    assert [p["index"] for p in listed["data"]] == [1, 2]
    assert "image" not in listed["data"][0]
    picture = client.document_picture(doc_id, 1)
    assert picture["caption"] == "Figure 2"
    assert base64.b64decode(picture["image"])[:4] == b"\x89PNG"
    raw, mime = client.document_picture_bytes(doc_id, 2)
    assert mime == "image/png" and raw[:4] == b"\x89PNG"

    assert client.delete_document(doc_id)["deleted"] is True
    with pytest.raises(NotFoundError):
        client.document_pictures(doc_id)


def test_document_stream(client: Client) -> None:
    with client.convert_document_stream(DOCS, b"%PDF fake", filename="a.pdf", output_format="json") as s:
        kinds = [event["object"] for event in s]
    assert kinds == ["document.started", "document.page", "document.page", "document.page", "document"]
    assert s.started is not None and s.started["pages"] == [1, 2, 3]

    stream = client.convert_document_stream(DOCS, b"%PDF fake", filename="a.pdf")
    doc = stream.final()
    # The server leaves `pages` empty in its last event; the stream puts them back.
    assert [p["page"] for p in doc["pages"]] == [1, 2, 3]
    assert doc["markdown"].count("## Page") == 3

    # Iterating first and asking for the result afterwards works too.
    stream = client.convert_document_stream(DOCS, b"%PDF fake", filename="a.pdf")
    events = list(stream)
    assert len(events) == 5 and len(stream.final()["pages"]) == 3

    pages = list(client.convert_document_stream(DOCS, b"%PDF fake", pages="2").pages())
    assert [p["page"] for p in pages] == [2]


def test_document_needs_file_or_url(client: Client) -> None:
    with pytest.raises(ValueError):
        client.convert_document(DOCS)


def test_memory_and_placement(client: Client) -> None:
    client.chat(CHAT, USER)
    status = client.memory_status()
    model = next(m for m in status["models"] if m["id"] == CHAT)
    assert model["current_tier"] == "tier1_gpu_vram"
    moved = client.set_tier(CHAT, "tier2_pinned_ram")
    assert moved["placement"]["primary_tier"] == "tier2_pinned_ram"
    back = client.set_tier(CHAT, "tier1_gpu_vram", context_len=4096)
    assert back["placement"]["primary_tier"] == "tier1_gpu_vram"


def test_kv_park(client: Client) -> None:
    status = client.kv_park()
    assert status["enabled"] is False
    assert status["used_bytes"] == 0
    try:
        on = client.set_kv_park(enabled=True, max_bytes=1073741824)
        assert on["enabled"] is True
        assert on["max_bytes"] == 1073741824
        assert on["models"] == []
        assert client.memory_status()["kv_park"]["enabled"] is True
        with pytest.raises(ValueError):
            client.set_kv_park()
    finally:
        off = client.set_kv_park(enabled=False)
    assert off["enabled"] is False


def test_enable_and_disable(client: Client) -> None:
    off = client.disable_model(IMAGE)
    assert off["enabled"] is False
    assert IMAGE not in {m["id"] for m in client.models()["data"]}
    with pytest.raises(NotFoundError) as err:
        client.images(IMAGE, "x")
    assert err.value.type == "model_disabled"
    assert client.enable_model(IMAGE)["enabled"] is True
    assert client.images(IMAGE, "x")["data"]


def test_model_pool(client: Client) -> None:
    pooled = "lfm3:1.7b"
    try:
        added = client.add_model(catalog=pooled)
        assert added["model"] == pooled and added["action"] == "added"
        assert isinstance(added["persisted"], bool) and "placement" in added
        assert pooled in {m["id"] for m in client.models()["data"]}
        reply = client.chat(pooled, USER, max_tokens=10)
        assert reply["choices"][0]["message"]["content"] == "echo: hello there"
        with pytest.raises(BadRequestError):
            client.add_model(catalog=pooled)
        replaced = client.replace_model(pooled, catalog="lfm3:1.7b:Q4_K_M", drain_secs=5)
        assert replaced["model"] == pooled and replaced["action"] == "replaced"
        removed = client.remove_model(pooled, drain_secs=5)
        assert removed == {**removed, "model": pooled, "action": "removed"}
        assert "placement" not in removed
        with pytest.raises(NotFoundError):
            client.chat(pooled, USER)
        with pytest.raises(NotFoundError):
            client.remove_model(pooled)
        with pytest.raises(NotFoundError):
            client.replace_model(pooled, catalog=pooled)
    finally:
        try:
            client.remove_model(pooled)
        except NotFoundError:
            pass
    with pytest.raises(ValueError):
        client.add_model()
    with pytest.raises(ValueError):
        client.add_model(catalog=pooled, config="[[models]]")


def test_moe_offload(client: Client) -> None:
    try:
        switch = client.set_moe_offload(CHAT, "on")
        assert switch["model"] == CHAT
        assert switch["moe_offload"] == "on"
        assert switch["applies"] in ("now", "next_load")
        assert switch["placement"].get("cpu_expert_layers", 0) == 0
        status = {m["id"]: m for m in client.memory_status()["models"]}
        assert status[CHAT]["moe_offload"] == "on"
        assert status[CHAT]["expert_bytes"] >= 0
        with pytest.raises(NotFoundError):
            client.set_moe_offload("no-such-model", "on")
        with pytest.raises(APIError) as err:
            client.set_moe_offload(CHAT, "sometimes")  # type: ignore[arg-type]
        assert err.value.status == 422
    finally:
        assert client.set_moe_offload(CHAT, "off")["moe_offload"] == "off"


def test_logs(client: Client) -> None:
    reply = client.chat(CHAT, USER)
    assert reply["id"]
    logs = client.logs(route="/v1/chat/completions", limit=5)
    assert logs["records"], logs
    record = client.log(logs["records"][0]["id"])
    assert record["route"] == "/v1/chat/completions"
    assert client.set_log_level("debug")["level"] == "debug"
    client.set_log_level("info")
    assert client.clear_logs()["cleared"] is True
