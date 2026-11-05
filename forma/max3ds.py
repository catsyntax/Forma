""".3ds (Autodesk 3D Studio) mesh writer and reader. 3ds Max, SketchUp, Blender and most tools read it."""
from __future__ import annotations

import struct

import numpy as np

MAIN, VERSION, MDATA, MESH_VERSION, MASTER_SCALE = 0x4D4D, 0x0002, 0x3D3D, 0x3D3E, 0x0100
NAMED_OBJECT, TRI_OBJECT, POINT_ARRAY, FACE_ARRAY, MSH_MAT_GROUP, MESH_MATRIX = 0x4000, 0x4100, 0x4110, 0x4120, 0x4130, 0x4160
MAT_ENTRY, MAT_NAME, MAT_DIFFUSE, COLOR_24 = 0xAFFF, 0xA000, 0xA020, 0x0011
LIMIT = 65000


def _chunk(cid, data=b"", children=b""):
    return struct.pack("<HI", cid, 6 + len(data) + len(children)) + data + children


def _cstr(s, maxlen=None):
    b = s.encode("ascii", "replace")
    if maxlen:
        b = b[:maxlen]
    return b + b"\x00"


def write(path, objects, master_scale=1.0):
    """objects: list of {name, V, F (index lists), color}."""
    mats, objs = b"", b""
    used = set()
    for k, o in enumerate(objects):
        V = np.asarray(o["V"], np.float32)
        tris = []
        for f in o["F"]:
            for j in range(1, len(f) - 1):
                tris.append((f[0], f[j], f[j + 1]))
        tris = np.array(tris, int).reshape(-1, 3)
        base = "".join(ch for ch in (o.get("name") or "Object") if ch.isalnum() or ch in "_-")[:7] or "Obj"
        mname = f"M{k}_{base}"[:16]
        r, g, b = (int(max(0, min(1, c)) * 255) for c in o.get("color", (0.8, 0.8, 0.8)))
        mats += _chunk(MAT_ENTRY, b"", _chunk(MAT_NAME, _cstr(mname)) + _chunk(MAT_DIFFUSE, b"", _chunk(COLOR_24, bytes([r, g, b]))))
        # split into pieces under the 16-bit limits
        start = 0
        part = 0
        while start < len(tris) or (start == 0 and len(tris) == 0):
            chunk_tris = tris[start:start + LIMIT]
            used_v = np.unique(chunk_tris.ravel()) if len(chunk_tris) else np.arange(min(len(V), LIMIT))
            if len(used_v) > LIMIT:
                chunk_tris = chunk_tris[: LIMIT // 3]
                used_v = np.unique(chunk_tris.ravel())
            remap = {int(v): i for i, v in enumerate(used_v)}
            pts = V[used_v]
            faces = np.array([[remap[int(a)], remap[int(b)], remap[int(c)]] for a, b, c in chunk_tris], np.uint16).reshape(-1, 3)
            name = (base if part == 0 else f"{base[:6]}_{part}")[:10]
            while name in used:
                name = (name[:8] + f"{len(used) % 100:02d}")[:10]
            used.add(name)
            pa = struct.pack("<H", len(pts)) + pts.astype("<f4").tobytes()
            fa_data = struct.pack("<H", len(faces)) + b"".join(struct.pack("<HHHH", a, b, c, 7) for a, b, c in faces)
            grp = _cstr(mname) + struct.pack("<H", len(faces)) + np.arange(len(faces), dtype="<u2").tobytes()
            fa = _chunk(FACE_ARRAY, fa_data, _chunk(MSH_MAT_GROUP, grp))
            mm = _chunk(MESH_MATRIX, np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], "<f4").tobytes())
            tri = _chunk(TRI_OBJECT, b"", _chunk(POINT_ARRAY, pa) + fa + mm)
            objs += _chunk(NAMED_OBJECT, _cstr(name), tri)
            part += 1
            start += max(len(chunk_tris), 1)
            if len(tris) == 0:
                break
    mdata = _chunk(MDATA, b"", _chunk(MESH_VERSION, struct.pack("<I", 3)) +
                   _chunk(MASTER_SCALE, struct.pack("<f", master_scale)) + mats + objs)
    data = _chunk(MAIN, b"", _chunk(VERSION, struct.pack("<I", 3)) + mdata)
    with open(path, "wb") as f:
        f.write(data)


def read(path):
    data = open(path, "rb").read()
    out = []
    mat_colors = {}

    def walk(pos, end, ctx):
        while pos + 6 <= end:
            cid, ln = struct.unpack_from("<HI", data, pos)
            if ln < 6:
                break
            body, cend = pos + 6, min(pos + ln, end)
            if cid in (MAIN, MDATA, TRI_OBJECT, MAT_ENTRY, MAT_DIFFUSE):
                if cid == TRI_OBJECT:
                    ctx = dict(ctx, mesh={"V": None, "F": [], "mat": None})
                    walk(body, cend, ctx)
                    m = ctx["mesh"]
                    if m["V"] is not None:
                        out.append({"name": ctx.get("name", "Object"), "V": m["V"], "F": m["F"],
                                    "material": m["mat"]})
                elif cid == MAT_ENTRY:
                    mctx = {"mat": {}}
                    walk(body, cend, mctx)
                    mm = mctx["mat"]
                    if "name" in mm:
                        mat_colors[mm["name"]] = mm.get("color")
                else:
                    walk(body, cend, ctx)
            elif cid == NAMED_OBJECT:
                z = data.index(b"\x00", body)
                name = data[body:z].decode("latin-1")
                walk(z + 1, cend, dict(ctx, name=name))
            elif cid == POINT_ARRAY and "mesh" in ctx:
                n = struct.unpack_from("<H", data, body)[0]
                ctx["mesh"]["V"] = np.frombuffer(data, "<f4", n * 3, body + 2).reshape(-1, 3).astype(float)
            elif cid == FACE_ARRAY and "mesh" in ctx:
                n = struct.unpack_from("<H", data, body)[0]
                arr = np.frombuffer(data, "<u2", n * 4, body + 2).reshape(-1, 4)
                ctx["mesh"]["F"] = [list(map(int, r[:3])) for r in arr]
                sub = body + 2 + n * 8
                walk(sub, cend, ctx)
            elif cid == MSH_MAT_GROUP and "mesh" in ctx:
                z = data.index(b"\x00", body)
                ctx["mesh"]["mat"] = data[body:z].decode("latin-1")
            elif cid == MAT_NAME and "mat" in ctx:
                z = data.index(b"\x00", body)
                ctx["mat"]["name"] = data[body:z].decode("latin-1")
            elif cid == COLOR_24 and "mat" in ctx:
                ctx["mat"]["color"] = tuple(c / 255 for c in data[body:body + 3])
            pos += ln

    walk(0, len(data), {})
    for o in out:
        o["color"] = mat_colors.get(o.get("material"))
    return out
