"""Asks gemma4:26b-q4 about an image.

    python examples/vision.py photo.jpg "What is in this picture?"
"""

import base64
import mimetypes
import sys

from hypnos_client import Client

path = sys.argv[1]
question = " ".join(sys.argv[2:]) or "Describe this picture."
mime = mimetypes.guess_type(path)[0] or "image/png"
with open(path, "rb") as f:
    data_url = f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"

with Client() as hypnos:
    reply = hypnos.chat(
        "gemma4:26b-q4",
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        max_tokens=300,
    )
    print(reply["choices"][0]["message"]["content"])
