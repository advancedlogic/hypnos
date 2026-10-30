"""Generates an image with moss-image and saves it as PNG.

    python examples/images.py "a lighthouse at dawn, oil painting" lighthouse.png
"""

import base64
import sys

from hypnos_client import Client

prompt = sys.argv[1] if len(sys.argv) > 1 else "a lighthouse at dawn, oil painting"
out = sys.argv[2] if len(sys.argv) > 2 else "image.png"

with Client() as hypnos:
    result = hypnos.images("moss-image", prompt, size="1024x1024", steps=20, seed=7)
    with open(out, "wb") as f:
        f.write(base64.b64decode(result["data"][0]["b64_json"]))
    print(f"wrote {out} in {result.get('hypnos_telemetry', {}).get('inference_latency_ms', 0) / 1000:.1f} s")
