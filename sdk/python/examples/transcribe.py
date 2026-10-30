"""Transcribes a recording with gemma4:e4b-q4 (WAV, 16-bit PCM is the reliable format).

    python examples/transcribe.py clip.wav [language]
"""

import sys

from hypnos_client import Client

with Client() as hypnos:
    result = hypnos.transcribe(
        "gemma4:e4b-q4", sys.argv[1], language=sys.argv[2] if len(sys.argv) > 2 else None
    )
    print(result["text"])
    engine = result.get("hypnos_telemetry", {}).get("engine", {})
    if "rtf" in engine:
        print(f"\n{engine.get('audio_secs', 0):.1f} s of audio, real-time factor {engine['rtf']:.2f}")
