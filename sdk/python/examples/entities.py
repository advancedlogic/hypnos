"""Finds people, places and companies with gliner2.5-multi (zero-shot: any labels work).

    python examples/entities.py "Mario Rossi works in Milan for Ferrari."
"""

import sys

from hypnos_client import Client

text = " ".join(sys.argv[1:]) or "Mario Rossi works in Milan for Ferrari."

with Client() as hypnos:
    result = hypnos.entities("gliner2.5-multi", text, ["person", "city", "company"])
    for e in result["data"][0]["entities"]:
        # start/end are character offsets: Python string indexes.
        print(f"{e['label']:>8}  {text[e['start']:e['end']]!r}  {e['score']:.2f}")
