"""Writes a two-page PDF with a paragraph, an embedded RGB picture and a caption on each page."""
import sys

out = sys.argv[1]
W, H = 96, 64


def picture(seed):
    data = bytearray()
    for y in range(H):
        for x in range(W):
            data += bytes(((x * 255 // W + seed) % 256, (y * 255 // H) % 256, (seed * 3) % 256))
    return bytes(data)


objects = []  # (number, bytes)


def add(body: bytes) -> int:
    objects.append(body)
    return len(objects)


font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
pages_placeholder = add(b"")  # filled later
page_ids = []
for n, seed in ((1, 0), (2, 90)):
    img = picture(seed)
    image_id = add(
        b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB "
        b"/BitsPerComponent 8 /Length %d >>\nstream\n" % (W, H, len(img)) + img + b"\nendstream"
    )
    text = (
        f"BT /F1 18 Tf 72 740 Td (Hypnos picture store test, page {n}) Tj ET\n"
        f"BT /F1 11 Tf 72 710 Td (This page has one embedded picture below this paragraph. "
        f"The picture store should keep it.) Tj ET\n"
        f"q 288 0 0 192 72 480 cm /Im1 Do Q\n"
        f"BT /F1 11 Tf 72 460 Td (Figure {n}: a colour gradient rendered as a picture) Tj ET\n"
    ).encode()
    content_id = add(b"<< /Length %d >>\nstream\n" % len(text) + text + b"\nendstream")
    page_id = add(
        b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] /Contents %d 0 R "
        b"/Resources << /Font << /F1 %d 0 R >> /XObject << /Im1 %d 0 R >> >> >>"
        % (pages_placeholder, content_id, font, image_id)
    )
    page_ids.append(page_id)

objects[pages_placeholder - 1] = (
    b"<< /Type /Pages /Kids [" + b" ".join(b"%d 0 R" % p for p in page_ids) + b"] /Count %d >>" % len(page_ids)
)
catalog = add(b"<< /Type /Catalog /Pages %d 0 R >>" % pages_placeholder)

buf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
offsets = []
for i, body in enumerate(objects, start=1):
    offsets.append(len(buf))
    buf += b"%d 0 obj\n" % i + body + b"\nendobj\n"
xref = len(buf)
buf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
for off in offsets:
    buf += b"%010d 00000 n \n" % off
buf += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, catalog, xref)
open(out, "wb").write(buf)
print(out, len(buf), "bytes")
