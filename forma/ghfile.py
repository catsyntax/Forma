"""Reader for Grasshopper definition files (.gh binary and .ghx XML).

Both formats store the same GH_IO archive tree: chunks that hold named items and child chunks.
This module only reads the tree; ``forma.ghimport`` turns it into a Forma node graph.
"""
from __future__ import annotations

import base64
import struct
import uuid
import zlib
import xml.etree.ElementTree as ET


class Chunk:
    def __init__(self, name, index=-1):
        self.name, self.index = name, index
        self.items: list[tuple] = []          # (name, index, value, type code)
        self.chunks: list[Chunk] = []

    # ---- lookups
    def item(self, name, index=-1, default=None):
        for n, i, v, *_ in self.items:
            if n == name and (index == -1 or i == index):
                return v
        return default

    def items_named(self, name):
        return [(i, v) for n, i, v, *_ in self.items if n == name]

    def set_item(self, name, value, type_code=None, index=-1):
        """Replace an item's value (keeping its type) or add it."""
        for k, (n, i, v, *t) in enumerate(self.items):
            if n == name and (index == -1 or i == index):
                self.items[k] = (n, i, value, t[0] if t else (type_code or _guess_type(value)))
                return
        self.items.append((name, index, value, type_code or _guess_type(value)))

    def chunk(self, name, index=-1):
        for c in self.chunks:
            if c.name == name and (index == -1 or c.index == index):
                return c
        return None

    def chunks_named(self, name):
        return [c for c in self.chunks if c.name == name]

    def walk(self):
        yield self
        for c in self.chunks:
            yield from c.walk()

    def __repr__(self):
        return f"<Chunk {self.name}[{self.index}] items={len(self.items)} chunks={len(self.chunks)}>"


# ------------------------------------------------------------------ binary (.gh)
class _Reader:
    def __init__(self, data):
        self.d, self.i = data, 0

    def take(self, n):
        b = self.d[self.i:self.i + n]
        if len(b) < n:
            raise ValueError("unexpected end of Grasshopper file")
        self.i += n
        return b

    def i32(self):
        return struct.unpack("<i", self.take(4))[0]

    def string(self):
        n = s = 0
        while True:
            b = self.take(1)[0]
            n |= (b & 0x7F) << s
            s += 7
            if b < 0x80:
                break
        return self.take(n).decode("utf-8", "replace")

    def value(self, t):
        f = lambda fmt: struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))
        if t == 1:
            return bool(self.take(1)[0])
        if t == 2:
            return self.take(1)[0]
        if t == 3:
            return f("i")[0]
        if t == 4:
            return f("q")[0]
        if t == 5:
            return f("f")[0]
        if t == 6:
            return f("d")[0]
        if t == 7:
            return self.take(16)
        if t == 8:
            return f("q")[0]
        if t == 9:
            return str(uuid.UUID(bytes_le=self.take(16)))
        if t == 10:
            return self.string()
        if t in (20, 37):
            return self.take(self.i32())
        if t == 21:
            n = self.i32()
            return list(f("%dd" % n)) if n else []
        if t in (30, 32):
            return f("ii")
        if t in (31, 33):
            return f("ff")
        if t == 34:
            return f("iiii")
        if t == 35:
            return f("ffff")
        if t == 36:
            return f("I")[0]
        if t == 50:
            return f("dd")
        if t == 51:
            return f("ddd")
        if t == 52:
            return f("dddd")
        if t == 60:
            return f("dd")
        if t == 61:
            return f("dddd")
        if t in (70, 71):
            return f("dddddd")
        if t == 72:
            return f("ddddddddd")
        if t == 80:
            return f("iii")
        raise ValueError(f"unknown Grasshopper item type {t}")

    def chunk(self):
        c = Chunk(self.string(), self.i32())
        nitems, nchunks = self.i32(), self.i32()
        for _ in range(nitems):
            name, idx, t = self.string(), self.i32(), self.i32()
            c.items.append((name, idx, self.value(t), t))
        for _ in range(nchunks):
            c.chunks.append(self.chunk())
        return c


def _read_binary(raw: bytes) -> Chunk:
    for wbits in (-15, 15, 31):
        try:
            data = zlib.decompress(raw, wbits)
            break
        except zlib.error:
            continue
    else:
        data = raw
    return _Reader(data).chunk()


# ------------------------------------------------------------------ XML (.ghx)
_XML_TYPES = {"gh_bool": 1, "gh_int32": 3, "gh_int64": 4, "gh_single": 5, "gh_double": 6, "gh_guid": 9,
              "gh_string": 10, "gh_bytearray": 20, "gh_drawing_bitmap": 37}


def _xml_chunk(el) -> Chunk:
    c = Chunk(el.get("name", ""), int(el.get("index", -1)))
    items = el.find("items")
    if items is not None:
        for it in items.findall("item"):
            t = it.get("type_name", "gh_string")
            name, idx = it.get("name", ""), int(it.get("index", -1))
            txt = it.text or ""
            if t == "gh_bool":
                v = txt.strip().lower() == "true"
            elif t in ("gh_int32", "gh_int64", "gh_byte"):
                v = int(txt or 0)
            elif t in ("gh_single", "gh_double"):
                v = float(txt or 0)
            elif t in ("gh_bytearray", "gh_drawing_bitmap"):
                v = base64.b64decode(txt) if txt else b""
            elif t == "gh_drawing_rectanglef":
                v = tuple(float(it.findtext(k, "0")) for k in ("X", "Y", "W", "H"))
            elif t == "gh_drawing_pointf":
                v = tuple(float(it.findtext(k, "0")) for k in ("X", "Y"))
            elif t == "gh_version":
                v = tuple(int(it.findtext(k, "0")) for k in ("Major", "Minor", "Revision"))
            else:
                v = txt
            code = {"gh_drawing_rectanglef": 35, "gh_drawing_pointf": 31, "gh_version": 80}.get(t, _XML_TYPES.get(t, 10))
            c.items.append((name, idx, v, code))
    chunks = el.find("chunks")
    if chunks is not None:
        for ch in chunks.findall("chunk"):
            c.chunks.append(_xml_chunk(ch))
    return c


def _guess_type(v):
    if isinstance(v, bool):
        return 1
    if isinstance(v, int):
        return 3
    if isinstance(v, float):
        return 6
    if isinstance(v, (bytes, bytearray)):
        return 20
    return 10


# ------------------------------------------------------------------ writing (.gh)
class _Writer:
    def __init__(self):
        self.b = bytearray()

    def i32(self, v):
        self.b += struct.pack("<i", v)

    def string(self, s):
        raw = s.encode("utf-8")
        n = len(raw)
        while True:
            byte = n & 0x7F
            n >>= 7
            self.b.append(byte | (0x80 if n else 0))
            if not n:
                break
        self.b += raw

    def value(self, t, v):
        p = lambda fmt, *a: self.b.extend(struct.pack("<" + fmt, *a))
        if t == 1:
            self.b.append(1 if v else 0)
        elif t == 2:
            self.b.append(int(v) & 0xFF)
        elif t == 3:
            p("i", int(v))
        elif t in (4, 8):
            p("q", int(v))
        elif t == 5:
            p("f", float(v))
        elif t == 6:
            p("d", float(v))
        elif t == 7:
            self.b += bytes(v)
        elif t == 9:
            self.b += uuid.UUID(str(v)).bytes_le
        elif t == 10:
            self.string(str(v))
        elif t in (20, 37):
            self.i32(len(v))
            self.b += bytes(v)
        elif t == 21:
            self.i32(len(v))
            if v:
                p("%dd" % len(v), *v)
        elif t in (30, 32):
            p("ii", *v)
        elif t in (31, 33):
            p("ff", *v)
        elif t == 34:
            p("iiii", *v)
        elif t == 35:
            p("ffff", *v)
        elif t == 36:
            p("I", int(v))
        elif t in (50, 60):
            p("dd", *v)
        elif t == 51:
            p("ddd", *v)
        elif t in (52, 61):
            p("dddd", *v)
        elif t in (70, 71):
            p("dddddd", *v)
        elif t == 72:
            p("ddddddddd", *v)
        elif t == 80:
            p("iii", *v)
        else:
            raise ValueError(f"cannot write Grasshopper item type {t}")

    def chunk(self, c):
        self.string(c.name)
        self.i32(c.index)
        self.i32(len(c.items))
        self.i32(len(c.chunks))
        for name, idx, v, *t in c.items:
            self.string(name)
            self.i32(idx)
            code = t[0] if t else _guess_type(v)
            self.i32(code)
            self.value(code, v)
        for ch in c.chunks:
            self.chunk(ch)


def to_bytes(root: Chunk) -> bytes:
    """Uncompressed GH_IO archive bytes."""
    w = _Writer()
    w.chunk(root)
    return bytes(w.b)


def write(root: Chunk, path: str):
    """Write a binary .gh file (deflate-compressed GH_IO archive) that Grasshopper opens."""
    co = zlib.compressobj(9, zlib.DEFLATED, -15)
    data = co.compress(to_bytes(root)) + co.flush()
    with open(path, "wb") as f:
        f.write(data)


def read_bytes(raw: bytes) -> Chunk:
    if raw.lstrip()[:5] in (b"<?xml", b"<Arch"):
        root = ET.fromstring(raw)
        top = root if root.tag == "Archive" else root.find("Archive")
        return _xml_chunk(top)
    return _read_binary(raw)


def read(path: str) -> Chunk:
    raw = open(path, "rb").read()
    if raw.lstrip()[:5] in (b"<?xml", b"<Arch"):
        root = ET.fromstring(raw)
        top = root if root.tag == "Archive" else root.find("Archive")
        return _xml_chunk(top)
    return _read_binary(raw)
