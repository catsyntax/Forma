"""Minimal binary FBX 7.4 writer and reader (meshes, object names, layer colours, transforms).

Written from the public description of the binary FBX node format. Covers what 3ds Max, Maya,
Blender, Unity and Unreal need to bring in static geometry. No animation, skinning or textures.
"""
from __future__ import annotations

import struct
import zlib
import time

import numpy as np

_HEAD = b"Kaydara FBX Binary  \x00\x1a\x00"
_SENT = b"\x00" * 13
_FOOT_ID = b"\xfa\xbc\xab\x09\xd0\xc8\xd4\x66\xb1\x76\xfb\x83\x1c\xf7\x26\x7e"
_FOOT_MAGIC = b"\xf8\x5a\x8c\x6a\xde\xf5\xd9\x7e\xec\xe9\x0c\xe3\x75\x8f\x29\x0b"
VERSION = 7400


# ======================================================================= writing
class Node:
    def __init__(self, name, *props):
        self.name = name.encode() if isinstance(name, str) else name
        self.props = list(props)
        self.children = []

    def add(self, name, *props):
        n = Node(name, *props)
        self.children.append(n)
        return n

    def P(self, *props):  # Properties70 entry
        return self.add("P", *props)


class Raw(bytes):
    pass


class L(int):
    """Force a 64-bit integer property (object ids)."""


def _prop(v):
    if isinstance(v, bool):
        return b"C" + struct.pack("<?", v)
    if isinstance(v, L):
        return b"L" + struct.pack("<q", int(v))
    if isinstance(v, np.ndarray):
        if v.dtype.kind == "f":
            code, arr = b"d", v.astype("<f8")
        elif v.dtype == np.int64:
            code, arr = b"l", v.astype("<i8")
        else:
            code, arr = b"i", v.astype("<i4")
        data = arr.tobytes()
        comp = zlib.compress(data) if len(data) > 128 else None
        if comp is not None and len(comp) < len(data):
            return code + struct.pack("<III", len(arr), 1, len(comp)) + comp
        return code + struct.pack("<III", len(arr), 0, len(data)) + data
    if isinstance(v, Raw):
        return b"R" + struct.pack("<I", len(v)) + bytes(v)
    if isinstance(v, (bytes, str)):
        b = v.encode() if isinstance(v, str) else v
        return b"S" + struct.pack("<I", len(b)) + b
    if isinstance(v, int):
        if -2 ** 31 <= v < 2 ** 31:
            return b"I" + struct.pack("<i", v)
        return b"L" + struct.pack("<q", v)
    if isinstance(v, float):
        return b"D" + struct.pack("<d", v)
    raise TypeError(type(v))


def _write_node(buf, n, offset):
    props = b"".join(_prop(p) for p in n.props)
    head_len = 4 + 4 + 4 + 1 + len(n.name)
    start = offset
    body = bytearray()
    pos = start + head_len + len(props)
    for c in n.children:
        cb = _write_node(None, c, pos)
        body += cb
        pos += len(cb)
    if n.children or not n.props:
        body += _SENT
        pos += len(_SENT)
    rec = struct.pack("<IIIB", pos, len(n.props), len(props), len(n.name)) + n.name + props + bytes(body)
    if buf is not None:
        buf += rec
    return rec


def _p70(parent, entries):
    p = parent.add("Properties70")
    for e in entries:
        p.P(*e)
    return p


def write(path, objects, unit_scale_cm=0.1, up="Z"):
    """objects: list of dicts {name, V (N,3), F (list of index lists), color (r,g,b 0..1)}.
    unit_scale_cm: size of one model unit in centimetres (mm = 0.1)."""
    nodes = []
    hdr = Node("FBXHeaderExtension")
    hdr.add("FBXHeaderVersion", 1003)
    hdr.add("FBXVersion", VERSION)
    hdr.add("EncryptionType", 0)
    t = time.localtime()
    ts = hdr.add("CreationTimeStamp")
    ts.add("Version", 1000)
    for k, v in (("Year", t.tm_year), ("Month", t.tm_mon), ("Day", t.tm_mday), ("Hour", t.tm_hour),
                 ("Minute", t.tm_min), ("Second", t.tm_sec), ("Millisecond", 0)):
        ts.add(k, v)
    hdr.add("Creator", "Forma FBX exporter")
    nodes.append(hdr)
    nodes.append(Node("FileId", Raw(b"\x28\xb3\x2a\xeb\xb6\x24\xcc\xc2\xbf\xc8\xb0\x2a\xa9\x2b\xfc\xf1")))
    nodes.append(Node("CreationTime", "1970-01-01 10:00:00:000"))
    nodes.append(Node("Creator", "Forma"))

    gs = Node("GlobalSettings")
    gs.add("Version", 1000)
    if up == "Z":
        axes = [("UpAxis", "int", "Integer", "", 2), ("UpAxisSign", "int", "Integer", "", 1),
                ("FrontAxis", "int", "Integer", "", 1), ("FrontAxisSign", "int", "Integer", "", -1),
                ("CoordAxis", "int", "Integer", "", 0), ("CoordAxisSign", "int", "Integer", "", 1)]
    else:
        axes = [("UpAxis", "int", "Integer", "", 1), ("UpAxisSign", "int", "Integer", "", 1),
                ("FrontAxis", "int", "Integer", "", 2), ("FrontAxisSign", "int", "Integer", "", 1),
                ("CoordAxis", "int", "Integer", "", 0), ("CoordAxisSign", "int", "Integer", "", 1)]
    _p70(gs, axes + [("OriginalUpAxis", "int", "Integer", "", 2 if up == "Z" else 1),
                     ("OriginalUpAxisSign", "int", "Integer", "", 1),
                     ("UnitScaleFactor", "double", "Number", "", float(unit_scale_cm)),
                     ("OriginalUnitScaleFactor", "double", "Number", "", float(unit_scale_cm))])
    nodes.append(gs)

    docs = Node("Documents")
    docs.add("Count", 1)
    d = docs.add("Document", L(1000000000), "", "Scene")
    _p70(d, [("SourceObject", "object", "", ""), ("ActiveAnimStackName", "KString", "", "", "")])
    d.add("RootNode", L(0))
    nodes.append(docs)
    nodes.append(Node("References"))

    n_obj = len(objects)
    defs = Node("Definitions")
    defs.add("Version", 100)
    defs.add("Count", 1 + 3 * n_obj)
    ot = defs.add("ObjectType", "GlobalSettings")
    ot.add("Count", 1)
    for typ in ("Model", "Geometry", "Material"):
        ot = defs.add("ObjectType", typ)
        ot.add("Count", n_obj)
    nodes.append(defs)

    objs = Node("Objects")
    conns = Node("Connections")
    uid = 2000000000
    for o in objects:
        V = np.asarray(o["V"], float)
        name = o.get("name") or "Object"
        gid, mid, matid = uid, uid + 1, uid + 2
        uid += 3
        g = objs.add("Geometry", L(gid), f"{name}\x00\x01Geometry".encode(), "Mesh")
        g.add("Vertices", V.ravel().astype(np.float64))
        idx = []
        for f in o["F"]:
            f = [int(i) for i in f]
            idx += f[:-1] + [-(f[-1]) - 1]
        g.add("PolygonVertexIndex", np.array(idx, dtype=np.int32))
        g.add("GeometryVersion", 124)
        nrm = _normals(V, o["F"])
        ln = g.add("LayerElementNormal", 0)
        ln.add("Version", 101)
        ln.add("Name", "")
        ln.add("MappingInformationType", "ByPolygonVertex")
        ln.add("ReferenceInformationType", "Direct")
        ln.add("Normals", nrm.ravel())
        lm = g.add("LayerElementMaterial", 0)
        lm.add("Version", 101)
        lm.add("Name", "")
        lm.add("MappingInformationType", "AllSame")
        lm.add("ReferenceInformationType", "IndexToDirect")
        lm.add("Materials", np.array([0], dtype=np.int32))
        layer = g.add("Layer", 0)
        layer.add("Version", 100)
        for typ in ("LayerElementNormal", "LayerElementMaterial"):
            le = layer.add("LayerElement")
            le.add("Type", typ)
            le.add("TypedIndex", 0)

        m = objs.add("Model", L(mid), f"{name}\x00\x01Model".encode(), "Mesh")
        m.add("Version", 232)
        _p70(m, [("Lcl Translation", "Lcl Translation", "", "A", 0.0, 0.0, 0.0),
                 ("Lcl Rotation", "Lcl Rotation", "", "A", 0.0, 0.0, 0.0),
                 ("Lcl Scaling", "Lcl Scaling", "", "A", 1.0, 1.0, 1.0),
                 ("DefaultAttributeIndex", "int", "Integer", "", 0)])
        m.add("Shading", True)
        m.add("Culling", "CullingOff")

        r, gg, b = o.get("color", (0.8, 0.8, 0.8))
        mat = objs.add("Material", L(matid), f"{o.get('material', name)}\x00\x01Material".encode(), "")
        mat.add("Version", 102)
        mat.add("ShadingModel", "phong")
        mat.add("MultiLayer", 0)
        _p70(mat, [("DiffuseColor", "Color", "", "A", float(r), float(gg), float(b)),
                   ("Diffuse", "Vector3D", "Vector", "", float(r), float(gg), float(b)),
                   ("DiffuseFactor", "Number", "", "A", 1.0)])

        conns.add("C", "OO", L(mid), L(0))
        conns.add("C", "OO", L(gid), L(mid))
        conns.add("C", "OO", L(matid), L(mid))
    nodes.append(objs)
    nodes.append(conns)
    tk = Node("Takes")
    tk.add("Current", "")
    nodes.append(tk)

    out = bytearray(_HEAD + struct.pack("<I", VERSION))
    for n in nodes:
        out += _write_node(None, n, len(out))
    out += _SENT
    out += _FOOT_ID + b"\x00\x00\x00\x00"
    pad = ((len(out) + 15) & ~15) - len(out)
    out += b"\x00" * (pad or 16)
    out += struct.pack("<I", VERSION) + b"\x00" * 120 + _FOOT_MAGIC
    with open(path, "wb") as f:
        f.write(out)


def _normals(V, F):
    out = []
    for f in F:
        P = V[list(f)]
        n = np.zeros(3)
        for a, b in zip(P, np.roll(P, -1, axis=0)):
            n += np.cross(a, b)
        n /= np.linalg.norm(n) or 1
        out += [n] * len(f)
    return np.array(out, float).reshape(-1, 3)


# ======================================================================= reading
class RNode:
    __slots__ = ("name", "props", "children")

    def __init__(self, name, props, children):
        self.name, self.props, self.children = name, props, children

    def find(self, name):
        for c in self.children:
            if c.name == name:
                return c
        return None

    def findall(self, name):
        return [c for c in self.children if c.name == name]


def _read_props(data, pos, n):
    props = []
    for _ in range(n):
        t = data[pos:pos + 1]
        pos += 1
        if t == b"C":
            props.append(bool(data[pos]))
            pos += 1
        elif t == b"Y":
            props.append(struct.unpack_from("<h", data, pos)[0])
            pos += 2
        elif t == b"I":
            props.append(struct.unpack_from("<i", data, pos)[0])
            pos += 4
        elif t == b"L":
            props.append(struct.unpack_from("<q", data, pos)[0])
            pos += 8
        elif t == b"F":
            props.append(struct.unpack_from("<f", data, pos)[0])
            pos += 4
        elif t == b"D":
            props.append(struct.unpack_from("<d", data, pos)[0])
            pos += 8
        elif t in (b"S", b"R"):
            ln = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            props.append(bytes(data[pos:pos + ln]))
            pos += ln
        elif t in (b"f", b"d", b"l", b"i", b"b", b"c"):
            alen, enc, clen = struct.unpack_from("<III", data, pos)
            pos += 12
            raw = bytes(data[pos:pos + clen])
            pos += clen
            if enc == 1:
                raw = zlib.decompress(raw)
            dt = {b"f": "<f4", b"d": "<f8", b"l": "<i8", b"i": "<i4", b"b": "u1", b"c": "u1"}[t]
            props.append(np.frombuffer(raw, dtype=dt, count=alen))
        else:
            raise ValueError(f"Unknown FBX property type {t!r}")
    return props, pos


def _read_node(data, pos, ver):
    if ver >= 7500:
        end, nprops, plen = struct.unpack_from("<QQQ", data, pos)
        pos += 24
    else:
        end, nprops, plen = struct.unpack_from("<III", data, pos)
        pos += 12
    nlen = data[pos]
    pos += 1
    if end == 0:
        return None, pos
    name = bytes(data[pos:pos + nlen]).decode("latin-1")
    pos += nlen
    props, pos = _read_props(data, pos, nprops)
    children = []
    sent = 25 if ver >= 7500 else 13
    while pos < end:
        if end - pos == sent and data[pos:end] == b"\x00" * sent:
            pos = end
            break
        c, pos = _read_node(data, pos, ver)
        if c is None:
            break
        children.append(c)
    return RNode(name, props, children), end


def read(path):
    """Return list of dicts {name, V, F, M (4x4 world transform)} plus (unit_cm, up_axis)."""
    data = memoryview(open(path, "rb").read())
    if bytes(data[:len(_HEAD)]) != _HEAD:
        raise ValueError("Only binary FBX files are supported (save as binary FBX from your app).")
    ver = struct.unpack_from("<I", data, 23)[0]
    pos = 27
    top = []
    while pos < len(data) - 13:
        n, pos = _read_node(data, pos, ver)
        if n is None:
            break
        top.append(n)
    root = RNode("root", [], top)
    unit_cm, up = 1.0, 1
    gs = root.find("GlobalSettings")
    if gs is not None and gs.find("Properties70") is not None:
        for p in gs.find("Properties70").children:
            key = p.props[0].decode("latin-1") if isinstance(p.props[0], bytes) else p.props[0]
            if key == "UnitScaleFactor":
                unit_cm = float(p.props[4])
            if key == "UpAxis":
                up = int(p.props[4])
    objs = root.find("Objects")
    conns = root.find("Connections")
    if objs is None:
        return [], unit_cm, up
    geoms, models = {}, {}
    for o in objs.children:
        if o.name == "Geometry" and len(o.props) >= 3 and o.props[2] == b"Mesh":
            V = o.find("Vertices")
            I = o.find("PolygonVertexIndex")
            if V is None or I is None:
                continue
            Vv = np.asarray(V.props[0], float).reshape(-1, 3)
            F, cur = [], []
            for i in np.asarray(I.props[0], np.int64):
                if i < 0:
                    cur.append(int(-i - 1))
                    F.append(cur)
                    cur = []
                else:
                    cur.append(int(i))
            geoms[o.props[0]] = (Vv, F)
        elif o.name == "Model":
            nm = o.props[1].split(b"\x00\x01")[0].decode("utf-8", "replace") if isinstance(o.props[1], bytes) else str(o.props[1])
            T, R, S = np.zeros(3), np.zeros(3), np.ones(3)
            p70 = o.find("Properties70")
            if p70 is not None:
                for p in p70.children:
                    key = p.props[0].decode("latin-1") if isinstance(p.props[0], bytes) else p.props[0]
                    if key == "Lcl Translation":
                        T = np.array(p.props[4:7], float)
                    elif key == "Lcl Rotation":
                        R = np.array(p.props[4:7], float)
                    elif key == "Lcl Scaling":
                        S = np.array(p.props[4:7], float)
            models[o.props[0]] = {"name": nm, "M": _trs(T, R, S)}
    parent, geom_of = {}, {}
    if conns is not None:
        for c in conns.children:
            if len(c.props) >= 3 and c.props[0] == b"OO":
                a, b = c.props[1], c.props[2]
                if a in geoms and b in models:
                    geom_of.setdefault(b, []).append(a)
                elif a in models:
                    parent[a] = b

    def world(mid, depth=0):
        M = models[mid]["M"]
        p = parent.get(mid)
        if p in models and depth < 64:
            return world(p, depth + 1) @ M
        return M
    out = []
    for mid, gids in geom_of.items():
        W = world(mid)
        for gid in gids:
            V, F = geoms[gid]
            out.append({"name": models[mid]["name"], "V": V, "F": F, "M": W})
    return out, unit_cm, up


def _trs(T, R, S):
    rx, ry, rz = np.radians(R)
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    M = np.eye(4)
    M[:3, :3] = Rz @ Ry @ Rx @ np.diag(S)  # FBX default rotation order XYZ (applied X first)
    M[:3, 3] = T
    return M
