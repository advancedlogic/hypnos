"""Streams a chat reply from gemma4:26b-q4, then prints what the request cost.

    python examples/chat.py "Explain sliding-window attention in two sentences."
"""

import sys

from hypnos_client import Client

prompt = " ".join(sys.argv[1:]) or "Explain sliding-window attention in two sentences."
with Client() as hypnos:
    hypnos.wait_until_ready()
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": prompt},
    ]
    with hypnos.chat_stream("gemma4:26b-q4", messages, max_tokens=300) as stream:
        for piece in stream.text():
            print(piece, end="", flush=True)
    print()
    t = stream.telemetry or {}
    print(
        f"\n{stream.usage['completion_tokens'] if stream.usage else '?'} tokens, "
        f"model was in {t.get('previous_tier')}, load {t.get('tier_switch_latency_ms', 0):.0f} ms, "
        f"decode {t.get('engine', {}).get('decode_tokens_per_s', 0):.0f} tok/s"
    )
