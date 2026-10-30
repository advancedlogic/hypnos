"""Synthesizes speech with kokoro-82m-q4: the whole file, then the same text streamed.

    python examples/speech.py "Hello from Hypnos." out.wav
"""

import sys

from hypnos_client import Client

text = sys.argv[1] if len(sys.argv) > 1 else "Hello from Hypnos."
out = sys.argv[2] if len(sys.argv) > 2 else "speech.wav"

with Client() as hypnos:
    speech = hypnos.speech("kokoro-82m-q4", text, voice="nova")
    with open(out, "wb") as f:
        f.write(speech.audio)
    print(f"wrote {out} ({speech.content_type}, real-time factor {speech.telemetry.rtf})")

    # Streaming: raw 16-bit PCM at 24 kHz, playable as it arrives (e.g. piped to `aplay`).
    with hypnos.speech_stream("kokoro-82m-q4", text, response_format="pcm") as stream:
        received = sum(len(chunk) for chunk in stream)
    print(f"streamed {received / 48000:.2f} s of audio")
