"""Converts a document to Markdown with docling, page by page, and saves its pictures.

    python examples/documents.py report.pdf [pages]
"""

import sys

from hypnos_client import Client

path = sys.argv[1]
pages = sys.argv[2] if len(sys.argv) > 2 else None

with Client() as hypnos:
    stream = hypnos.convert_document_stream("docling", path, pages=pages, return_pictures=True)
    for event in stream:
        if event["object"] == "document.started":
            print(f"{event['page_count']} pages, converting {event['pages']}")
        elif event["object"] == "document.page":
            print(f"page {event['page']}: {event['tables']} tables, {event['pictures']} pictures")
    doc = stream.final()
    with open(path + ".md", "w") as f:
        f.write(doc["markdown"])
    print(f"wrote {path}.md")

    if "id" in doc:
        for picture in hypnos.document_pictures(doc["id"])["data"]:
            data, mime = hypnos.document_picture_bytes(doc["id"], picture["index"])
            name = f"{path}.picture{picture['index']}.{mime.split('/')[-1]}"
            with open(name, "wb") as f:
                f.write(data)
            print(f"wrote {name} ({picture.get('caption', 'no caption')})")
        hypnos.delete_document(doc["id"])
