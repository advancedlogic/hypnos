"""Embeds a few sentences with gemmaembedding2 and ranks them against a query.

    python examples/embeddings.py "how do I reset my password"
"""

import sys

from hypnos_client import Client

docs = [
    "To change your password, open Settings and choose Security.",
    "Our office is closed on public holidays.",
    "Invoices are sent on the first day of each month.",
]
query = " ".join(sys.argv[1:]) or "how do I reset my password"

with Client() as hypnos:
    vectors = [d["embedding"] for d in hypnos.embeddings("gemmaembedding2", [query, *docs])["data"]]
    q, rest = vectors[0], vectors[1:]
    # The vectors are L2-normalised: the dot product is the cosine similarity.
    scores = [sum(a * b for a, b in zip(q, v)) for v in rest]
    for score, doc in sorted(zip(scores, docs), reverse=True):
        print(f"{score:.3f}  {doc}")
