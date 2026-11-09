"""Import / export hub for every supported file format.

Exact NURBS:   .3dm (Rhino), .step/.stp, .igs/.iges
CAD drawing:   .dxf, .dwg (DWG through the free ODA File Converter, if installed)
Vector:        .svg
Mesh / scene:  .fbx, .obj, .3ds, .dae (SketchUp / Collada), .glb/.gltf, .stl, .ply, .off, .3mf
"""
from __future__ import annotations

import math
import os
import re

import numpy as np

from . import geometry as G
from .geometry import Geom
from .document import Layer

UNIT_M = {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254, "ft": 0.3048}

FORMATS = [
    # ext, label, import, export
    ("forma", "Forma project", True, True),
    ("3dm", "Rhino 3D", True, True),
    ("step", "STEP (exact NURBS solids)", True, True),
    ("stp", "STEP", True, True),
    ("igs", "IGES", True, True),
    ("iges", "IGES", True, True),
    ("fbx", "FBX (3ds Max, Maya, Blender, Unity)", True, True),
    ("obj", "Wavefront OBJ", True, True),
    ("3ds", "3D Studio (3ds Max)", True, True),
    ("dae", "Collada (SketchUp)", True, True),
    ("dxf", "AutoCAD DXF", True, True),
    ("dwg", "AutoCAD DWG (needs ODA File Converter)", True, True),
    ("svg", "SVG vector drawing", True, True),
    ("glb", "glTF binary", True, True),
    ("gltf", "glTF", True, True),
    ("stl", "STL (3D printing)", True, True),
    ("ply", "PLY", True, True),
    ("off", "OFF", True, True),
    ("3mf", "3MF (3D printing)", True, True),
    ("gh", "Grasshopper definition (opens in Nodes)", True, False),
    ("ghx", "Grasshopper definition (opens in Nodes)", True, False),
]


def file_filter(import_=True):
    exts = [e for e, _, i, x in FORMATS if (i if import_ else x)]
    parts = ["All supported (" + " ".join(f"*.{e}" for e in exts) + ")"]
    seen = set()
    for e, label, i, x in FORMATS:
        if (i if import_ else x) and label not in seen:
            same = [f"*.{e2}" for e2, l2, i2, x2 in FORMATS if l2 == label]
            parts.append(f"{label} ({' '.join(same)})")
            seen.add(label)
    return ";;".join(parts)


def _ext(path):
    return os.path.splitext(path)[1].lower().lstrip(".")


def _hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _rgb_hex(c):
    return "#%02x%02x%02x" % tuple(int(max(0, min(1, x)) * 255) for x in c[:3])


# ======================================================================= gather
def gather(doc, objects=None):
    """Flatten objects (components expanded) into (geom, layer, color_hex, name) tuples."""
    src = list(doc.objects.values()) if objects is None else objects
    out = []
    for g in src:
        lay = doc.layer(g.layer)
        if objects is None and lay is not None and not lay.visible:
            continue
        col = doc.layer_color(g.layer)
        parts = doc.expand(g) if g.kind == "instance" else [g]
        base = g.name or (doc.definitions[g.def_id].name if g.kind == "instance" and g.def_id in doc.definitions else "")
        for k, p in enumerate(parts):
            nm = base or p.data.get("source", p.kind)
            if len(parts) > 1:
                nm = f"{nm}_{k + 1}"
            out.append((p, g.layer, col, nm))
    return out


def mesh_of(g):
    """(V, faces) for surfaces/solids/meshes, else None."""
    if g.kind == "brep":
        from . import brep as B
        V, N, T, _ = B.tessellate(g)
        return np.asarray(V, float), [list(t) for t in T]
    if g.kind == "mesh":
        return np.asarray(g.V, float), g.mesh_faces()
    return None


def _zup_to_yup(V):
    V = np.asarray(V, float)
    return np.c_[V[:, 0], V[:, 2], -V[:, 1]]


def _yup_to_zup(V):
    V = np.asarray(V, float)
    return np.c_[V[:, 0], -V[:, 2], V[:, 1]]


# ======================================================================= export
def export_file(doc, path, objects=None):
    """Write `path` in the format of its extension. Returns (count, warnings)."""
    ext = _ext(path)
    units = getattr(doc, "units", "mm")
    if ext == "forma":
        doc.save(path)
        return len(doc.objects), []
    if ext == "3dm":
        from .io3dm import export_3dm
        items = [p for p, *_ in gather(doc, objects)]
        n = export_3dm(doc, path, items)
        fb = getattr(export_3dm, "last_fallback_faces", 0)
        return n, ([f"{fb} trimmed curved faces were written as meshes (use STEP for fully exact solids)."] if fb else [])
    if ext in ("step", "stp", "igs", "iges"):
        from . import brep as B
        items = [p for p, *_ in gather(doc, objects) if p.kind == "brep"]
        if not items:
            raise ValueError("No NURBS surfaces or solids to export.")
        (B.export_step if ext in ("step", "stp") else B.export_iges)(items, path)
        return len(items), []
    items = gather(doc, objects)
    if ext == "fbx":
        return _export_fbx(items, path, units)
    if ext == "3ds":
        return _export_3ds(items, path)
    if ext == "obj":
        return _export_obj(items, path)
    if ext in ("dxf", "dwg"):
        return _export_dxf(items, path, doc, units)
    if ext == "svg":
        return _export_svg(items, path, units)
    if ext in ("glb", "gltf", "stl", "ply", "off", "3mf", "dae"):
        return _export_trimesh(items, path, ext, units)
    raise ValueError(f"Unsupported export format: .{ext}")


def _mesh_items(items):
    out, skipped = [], 0
    for g, layer, col, name in items:
        m = mesh_of(g)
        if m is None:
            skipped += 1
            continue
        V, F = m
        if len(F):
            out.append({"name": name, "V": V, "F": F, "color": _hex_rgb(col), "material": layer})
    return out, skipped


def _skip_note(skipped, fmt):
    return [f"{skipped} curves/points skipped ({fmt} stores meshes only)."] if skipped else []


def _export_fbx(items, path, units):
    from . import fbx
    objs, skipped = _mesh_items(items)
    fbx.write(path, objs, unit_scale_cm=UNIT_M.get(units, 0.001) * 100, up="Z")
    return len(objs), _skip_note(skipped, "FBX")


def _export_3ds(items, path):
    from . import max3ds
    objs, skipped = _mesh_items(items)
    max3ds.write(path, objs)
    return len(objs), _skip_note(skipped, "3DS")


def _export_obj(items, path):
    """OBJ + MTL, Y-up (the convention 3ds Max, Blender, Maya and SketchUp expect)."""
    base = os.path.splitext(path)[0]
    mtl = os.path.basename(base) + ".mtl"
    lines = [f"# Forma OBJ export", f"mtllib {mtl}"]
    mats = {}
    vi = 1
    n = 0
    for g, layer, col, name in items:
        mats.setdefault(layer, col)
        safe = re.sub(r"\s+", "_", name)
        m = mesh_of(g)
        if m is not None:
            V, F = m
            if not len(F):
                continue
            lines.append(f"o {safe}")
            lines.append(f"usemtl {re.sub(chr(32), '_', layer)}")
            for v in _zup_to_yup(V):
                lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
            for f in F:
                lines.append("f " + " ".join(str(vi + int(i)) for i in f))
            vi += len(V)
            n += 1
        elif g.is_curve or g.kind == "point":
            P = g.sample() if g.is_curve else g.p.reshape(1, 3)
            lines.append(f"o {safe}")
            for v in _zup_to_yup(P):
                lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
            if len(P) > 1:
                lines.append("l " + " ".join(str(vi + i) for i in range(len(P))))
            vi += len(P)
            n += 1
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(os.path.dirname(path), mtl), "w") as f:
        for lay, col in mats.items():
            r, g_, b = _hex_rgb(col)
            f.write(f"newmtl {re.sub(chr(32), '_', lay)}\nKd {r:.4f} {g_:.4f} {b:.4f}\nKa 0 0 0\nd 1\nillum 2\n\n")
    return n, []


def _export_trimesh(items, path, ext, units):
    import trimesh
    objs, skipped = _mesh_items(items)
    if not objs:
        raise ValueError("Nothing to export (this format stores surfaces/meshes only).")
    scene = trimesh.Scene()
    um = UNIT_M.get(units, 0.001)
    for o in objs:
        V = o["V"]
        if ext in ("glb", "gltf"):
            V = _zup_to_yup(V) * um  # glTF is Y-up, metres
        tris = []
        for f in o["F"]:
            for j in range(1, len(f) - 1):
                tris.append((f[0], f[j], f[j + 1]))
        m = trimesh.Trimesh(vertices=V, faces=np.array(tris, int), process=False)
        rgba = [int(c * 255) for c in o["color"]] + [255]
        m.visual.face_colors = np.tile(rgba, (len(tris), 1))
        scene.add_geometry(m, node_name=o["name"], geom_name=o["name"])
    if ext == "dae":
        from trimesh.exchange.dae import export_collada
        meshes = [scene.geometry[k] for k in scene.geometry]
        data = export_collada(meshes)
        txt = data.decode("utf-8") if isinstance(data, bytes) else data
        unit_name = {"mm": "millimeter", "cm": "centimeter", "m": "meter", "in": "inch", "ft": "foot"}.get(units, "millimeter")
        txt = re.sub(r"\s*<up_axis>.*?</up_axis>", "", txt)
        txt = re.sub(r"\s*<unit[^>]*/>", "", txt)
        txt = txt.replace("</asset>", f'<unit meter="{um}" name="{unit_name}"/><up_axis>Z_UP</up_axis></asset>', 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)
    elif ext == "off":
        trimesh.util.concatenate([scene.geometry[k] for k in scene.geometry]).export(path, file_type="off")
    else:
        scene.export(path, file_type=ext)
    return len(objs), _skip_note(skipped, ext.upper())


def _export_dxf(items, path, doc, units):
    import ezdxf
    from ezdxf import units as U
    from ezdxf.math import Vec3
    dxf = ezdxf.new("R2018", setup=True)
    dxf.units = {"mm": U.MM, "cm": U.CM, "m": U.M, "in": U.IN, "ft": U.FT}.get(units, U.MM)
    msp = dxf.modelspace()
    for l in doc.layers:
        name = re.sub(r'[<>/\\":;?*|=`]', "_", l.name)
        if name not in dxf.layers:
            r, g, b = (int(c * 255) for c in _hex_rgb(l.color))
            dxf.layers.add(name, true_color=ezdxf.colors.rgb2int((r, g, b)))
    n, warn = 0, []
    for g, layer, col, name in items:
        att = {"layer": re.sub(r'[<>/\\":;?*|=`]', "_", layer)}
        k = g.kind
        try:
            if k == "point":
                msp.add_point(tuple(g.p), dxfattribs=att)
            elif k == "polyline":
                pts = g.pts
                if np.allclose(pts[:, 2], pts[0, 2]):
                    msp.add_lwpolyline([tuple(p[:2]) for p in pts], close=g.closed,
                                       dxfattribs=dict(att, elevation=float(pts[0, 2])))
                else:
                    msp.add_polyline3d([tuple(p) for p in pts], close=g.closed, dxfattribs=att)
            elif k == "ellipse":
                ru, rv = np.linalg.norm(g.u), np.linalg.norm(g.v)
                nrm = np.cross(g.u, g.v)
                nrm = nrm / (np.linalg.norm(nrm) or 1)
                circular = abs(ru - rv) < 1e-9 * max(ru, 1) and abs(np.dot(g.u, g.v)) < 1e-9 * max(ru * rv, 1)
                if circular and nrm[2] > 1 - 1e-9:
                    rot = math.degrees(math.atan2(g.u[1], g.u[0]))
                    if g.closed:
                        msp.add_circle(tuple(g.c), ru, dxfattribs=dict(att))
                    else:
                        msp.add_arc(tuple(g.c[:2]) + (float(g.c[2]),), ru, math.degrees(g.a0) + rot, math.degrees(g.a1) + rot,
                                    dxfattribs=att)
                else:
                    if ru >= rv:
                        major, ratio, s0, s1 = g.u, rv / ru, g.a0, g.a1
                    else:
                        major, ratio, s0, s1 = g.v, ru / rv, g.a0 - math.pi / 2, g.a1 - math.pi / 2
                    if g.closed:
                        s0, s1 = 0.0, 2 * math.pi
                    msp.add_ellipse(tuple(g.c), major_axis=tuple(major), ratio=max(min(ratio, 1.0), 1e-6),
                                    start_param=s0, end_param=s1, dxfattribs=dict(att, extrusion=tuple(nrm)))
            elif k == "nurbs":
                if np.allclose(g.w, 1):
                    msp.add_open_spline([tuple(p) for p in g.cps], degree=int(g.deg), knots=list(map(float, g.knots)), dxfattribs=att)
                else:
                    msp.add_rational_spline([tuple(p) for p in g.cps], list(map(float, g.w)), degree=int(g.deg),
                                            knots=list(map(float, g.knots)), dxfattribs=att)
            else:
                m = mesh_of(g)
                if m is None:
                    continue
                V, F = m
                mesh = msp.add_mesh(dxfattribs=att)
                with mesh.edit_data() as md:
                    md.vertices = [Vec3(*v) for v in V]
                    md.faces = [list(map(int, f)) for f in F]
            n += 1
        except Exception as e:
            warn.append(f"{name}: {e}")
    if _ext(path) == "dwg":
        from ezdxf.addons import odafc
        if not odafc.is_installed():
            raise RuntimeError("DWG needs the free ODA File Converter (opendesign.com/guestfiles/oda_file_converter). "
                               "Install it in Applications, or export DXF, which AutoCAD opens directly.")
        odafc.export_dwg(dxf, path, version="R2018", replace=True)
    else:
        dxf.saveas(path)
    return n, warn[:5]


def _export_svg(items, path, units):
    mm = UNIT_M.get(units, 0.001) / 0.001  # model units -> mm
    shapes, pts_all = [], []
    for g, layer, col, name in items:
        if g.kind == "brep":
            from . import brep as B
            plines = B.tessellate(g)[3]
            if plines:
                d = " ".join("M" + " L".join(f"{p[0]:.4f},{-p[1]:.4f}" for p in pl) for pl in plines)
                shapes.append((col, f'<path d="{d}"/>'))
                pts_all += [p[:2] for pl in plines for p in pl]
        elif g.kind == "mesh":
            segs = G.feature_edges(g)
            if segs:
                d = " ".join(f"M{a[0]:.4f},{-a[1]:.4f} L{b[0]:.4f},{-b[1]:.4f}" for a, b in segs)
                shapes.append((col, f'<path d="{d}"/>'))
                pts_all += [a[:2] for a, b in segs] + [b[:2] for a, b in segs]
        elif g.kind == "ellipse" and abs(np.cross(g.u, g.v)[2]) > 1e-12 and np.allclose([g.u[2], g.v[2]], 0) and g.closed:
            ru, rv = np.linalg.norm(g.u), np.linalg.norm(g.v)
            ang = -math.degrees(math.atan2(g.u[1], g.u[0]))
            shapes.append((col, f'<ellipse cx="{g.c[0]:.4f}" cy="{-g.c[1]:.4f}" rx="{ru:.4f}" ry="{rv:.4f}" '
                                f'transform="rotate({ang:.4f} {g.c[0]:.4f} {-g.c[1]:.4f})"/>'))
            pts_all += [g.c[:2] - max(ru, rv), g.c[:2] + max(ru, rv)]
        elif g.is_curve:
            S = g.sample()
            tag = "polygon" if g.closed else "polyline"
            P = S[:-1] if g.closed else S
            shapes.append((col, f'<{tag} points="' + " ".join(f"{p[0]:.4f},{-p[1]:.4f}" for p in P) + '"/>'))
            pts_all += list(P[:, :2])
        elif g.kind == "point":
            shapes.append((col, f'<circle cx="{g.p[0]:.4f}" cy="{-g.p[1]:.4f}" r="0.5"/>'))
            pts_all.append(g.p[:2])
    if not pts_all:
        raise ValueError("Nothing to export.")
    A = np.array(pts_all)
    lo, hi = A.min(0), A.max(0)
    pad = max((hi - lo).max() * 0.03, 1e-3)
    x0, y0 = lo[0] - pad, -(hi[1] + pad)
    w, h = (hi[0] - lo[0]) + 2 * pad, (hi[1] - lo[1]) + 2 * pad
    groups = {}
    for col, s in shapes:
        groups.setdefault(col, []).append(s)
    body = "\n".join(f'<g stroke="{col if col.lower() not in ("#d0d0d0",) else "#1b1b1d"}" fill="none" '
                     f'stroke-width="{0.35 / mm:.6g}" vector-effect="non-scaling-stroke">\n' + "\n".join(v) + "\n</g>"
                     for col, v in groups.items())
    svg = (f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" '
           f'width="{w * mm:.3f}mm" height="{h * mm:.3f}mm" viewBox="{x0:.4f} {y0:.4f} {w:.4f} {h:.4f}">\n{body}\n</svg>\n')
    with open(path, "w") as f:
        f.write(svg)
    return len(shapes), []


# ======================================================================= import
def import_file(path, units="mm"):
    """Return (layers, geoms, warnings)."""
    ext = _ext(path)
    stem = os.path.basename(path)
    if ext == "3dm":
        from .io3dm import import_3dm
        return import_3dm(path)
    if ext in ("step", "stp", "igs", "iges"):
        from . import brep as B
        return [], B.import_step_iges(path), []
    if ext == "fbx":
        return _import_fbx(path, units, stem)
    if ext == "3ds":
        return _import_3ds(path, stem)
    if ext in ("dxf", "dwg"):
        return _import_dxf(path, units)
    if ext == "svg":
        return _import_svg(path, units, stem)
    if ext == "obj":
        return _import_obj(path, stem)
    if ext in ("dae", "glb", "gltf", "stl", "ply", "off", "3mf"):
        return _import_trimesh(path, ext, units, stem)
    raise ValueError(f"Unsupported import format: .{ext}")


def _new_layer(stem):
    return Layer(f"Import {stem}"[:60], "#d0d0d0")


def _import_fbx(path, units, stem):
    from . import fbx
    objs, unit_cm, up = fbx.read(path)
    s = (unit_cm / 100.0) / UNIT_M.get(units, 0.001)
    lay = _new_layer(stem)
    out = []
    for o in objs:
        V = G.xform_pts(o["M"], o["V"])
        if up == 1:
            V = _yup_to_zup(V)
        out.append(G.mesh(V * s, o["F"], source="Imported mesh", name=o["name"], layer=lay.name))
    return [lay], out, ([] if out else ["No meshes found in the FBX file."])


def _import_3ds(path, stem):
    from . import max3ds
    lay = _new_layer(stem)
    out = [G.mesh(o["V"], o["F"], source="Imported mesh", name=o["name"], layer=lay.name) for o in max3ds.read(path)]
    return [lay], out, []


def _import_obj(path, stem):
    """OBJ reader keeping objects/groups and polylines ('l'). OBJ is treated as Y-up."""
    V = []
    objs = []
    cur = {"name": stem, "F": [], "L": []}

    def push():
        if cur["F"] or cur["L"]:
            objs.append(dict(cur))
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if not line.strip() or line[0] == "#":
                continue
            parts = line.split()
            tag = parts[0]
            if tag == "v" and len(parts) >= 4:
                V.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif tag in ("o", "g") and len(parts) > 1:
                push()
                cur = {"name": " ".join(parts[1:]), "F": [], "L": []}
            elif tag == "f":
                idx = []
                for tok in parts[1:]:
                    i = int(tok.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(V) + i)
                if len(idx) >= 3:
                    cur["F"].append(idx)
            elif tag == "l":
                idx = []
                for tok in parts[1:]:
                    i = int(tok.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(V) + i)
                cur["L"].append(idx)
    push()
    Vz = _yup_to_zup(np.array(V, float)) if V else np.zeros((0, 3))
    lay = _new_layer(stem)
    out = []
    for o in objs:
        if o["F"]:
            used = sorted({i for f in o["F"] for i in f})
            remap = {v: k for k, v in enumerate(used)}
            out.append(G.mesh(Vz[used], [[remap[i] for i in f] for f in o["F"]], source="Imported mesh",
                              name=o["name"], layer=lay.name))
        for l in o["L"]:
            P = Vz[l]
            if len(P) >= 2:
                closed = len(P) > 2 and np.allclose(P[0], P[-1])
                out.append(G.polyline(P[:-1] if closed else P, closed=closed, name=o["name"], layer=lay.name))
            elif len(P) == 1:
                out.append(G.point(P[0], layer=lay.name))
    return [lay], out, []


def _import_trimesh(path, ext, units, stem):
    import trimesh
    um = UNIT_M.get(units, 0.001)
    scale, yup = 1.0, False
    if ext in ("glb", "gltf"):
        scale, yup = 1.0 / um, True
    elif ext == "obj":
        yup = True
    elif ext == "dae":
        txt = open(path, "r", errors="ignore").read(20000)
        m = re.search(r'<unit[^>]*meter="([0-9.eE+-]+)"', txt)
        if m:
            scale = float(m.group(1)) / um
        yup = "Y_UP" in txt
    loaded = trimesh.load(path, force="scene", process=False)
    lay = _new_layer(stem)
    out, warn = [], []
    for node in loaded.graph.nodes_geometry:
        T, gname = loaded.graph[node]
        geom = loaded.geometry[gname]
        if isinstance(geom, trimesh.Trimesh):
            V = trimesh.transform_points(np.asarray(geom.vertices, float), T)
            if yup:
                V = _yup_to_zup(V)
            out.append(G.mesh(V * scale, np.asarray(geom.faces, int), source="Imported mesh", name=str(node), layer=lay.name))
        elif hasattr(geom, "discrete"):
            for P in geom.discrete:
                P = np.c_[P, np.zeros(len(P))] if P.shape[1] == 2 else P
                P = trimesh.transform_points(P, T)
                if yup:
                    P = _yup_to_zup(P)
                if len(P) >= 2:
                    out.append(G.polyline(P * scale, layer=lay.name))
    return [lay], out, warn


def _import_dxf(path, units):
    import ezdxf
    from ezdxf import units as U
    if _ext(path) == "dwg":
        from ezdxf.addons import odafc
        if not odafc.is_installed():
            raise RuntimeError("Opening DWG needs the free ODA File Converter (opendesign.com/guestfiles/oda_file_converter). "
                               "Install it in Applications, or save the drawing as DXF in your CAD app.")
        dxf = odafc.readfile(path)
    else:
        dxf = ezdxf.readfile(path)
    target = {"mm": U.MM, "cm": U.CM, "m": U.M, "in": U.IN, "ft": U.FT}.get(units, U.MM)
    try:
        s = U.conversion_factor(dxf.units, target) if dxf.units else 1.0
    except Exception:
        s = 1.0
    layers = {}
    for l in dxf.layers:
        try:
            rgb = l.rgb or ezdxf.colors.aci2rgb(abs(l.color) if l.color else 7)
            col = "#%02x%02x%02x" % tuple(rgb)
        except Exception:
            col = "#d0d0d0"
        if col.lower() in ("#ffffff", "#000000"):
            col = "#d0d0d0"
        layers[l.dxf.name] = Layer(l.dxf.name, col)
    out, faces, skipped = [], {}, [0]
    S = np.diag([s, s, s, 1.0])

    def add(g, lay):
        g.layer = lay
        out.append(g.transformed(S) if s != 1.0 else g)

    def handle(e, depth=0):
        t = e.dxftype()
        lay = e.dxf.get("layer", "0")
        try:
            if t == "LINE":
                add(G.line(np.array(e.dxf.start), np.array(e.dxf.end)), lay)
            elif t == "POINT":
                add(G.point(np.array(e.dxf.location)), lay)
            elif t in ("LWPOLYLINE", "POLYLINE") and not (t == "POLYLINE" and (e.is_poly_face_mesh or e.is_polygon_mesh)):
                from ezdxf import path as epath
                closed = bool(e.closed if t == "LWPOLYLINE" else e.is_closed)
                has_bulge = (t == "LWPOLYLINE" and e.has_arc) or (t == "POLYLINE" and any(v.dxf.bulge for v in e.vertices))
                if has_bulge:
                    p = epath.make_path(e)
                    P = np.array([tuple(v) for v in p.flattening(0.01 / max(s, 1e-9))])
                    if closed and len(P) > 2 and np.allclose(P[0], P[-1]):
                        P = P[:-1]
                else:
                    P = np.array([tuple(v) for v in (e.vertices_in_wcs() if t == "LWPOLYLINE" else e.points())])
                if len(P) >= 2:
                    add(G.polyline(P, closed=closed), lay)
            elif t in ("CIRCLE", "ARC"):
                ocs = e.ocs()
                c = np.array(ocs.to_wcs(e.dxf.center))
                r = float(e.dxf.radius)
                ux, uy = np.array(ocs.ux), np.array(ocs.uy)
                if t == "CIRCLE":
                    a0, a1 = 0.0, 2 * math.pi
                else:
                    a0, a1 = math.radians(e.dxf.start_angle), math.radians(e.dxf.end_angle)
                    while a1 <= a0:
                        a1 += 2 * math.pi
                add(Geom("ellipse", c=c, u=ux * r, v=uy * r, a0=a0, a1=a1), lay)
            elif t == "ELLIPSE":
                c = np.array(e.dxf.center)
                major = np.array(e.dxf.major_axis)
                n = np.array(e.dxf.extrusion)
                minor = np.cross(n, major)
                minor = minor / (np.linalg.norm(minor) or 1) * np.linalg.norm(major) * e.dxf.ratio
                a0, a1 = float(e.dxf.start_param), float(e.dxf.end_param)
                while a1 <= a0:
                    a1 += 2 * math.pi
                add(Geom("ellipse", c=c, u=major, v=minor, a0=a0, a1=a1), lay)
            elif t == "SPLINE":
                cps = np.array([tuple(p) for p in e.control_points])
                if len(cps) >= 2:
                    k = np.array(list(e.knots), float)
                    k = (k - k[0]) / ((k[-1] - k[0]) or 1)
                    w = np.array(list(e.weights), float) if len(e.weights) else np.ones(len(cps))
                    add(Geom("nurbs", cps=cps, w=w, knots=k, deg=int(e.dxf.degree)), lay)
                elif len(e.fit_points) >= 2:
                    add(G.interp_curve([tuple(p) for p in e.fit_points]), lay)
            elif t in ("3DFACE", "SOLID", "TRACE"):
                pts = [np.array(e.dxf.get(f"vtx{i}")) for i in range(4) if e.dxf.hasattr(f"vtx{i}")]
                if t != "3DFACE" and len(pts) == 4:
                    pts = [pts[0], pts[1], pts[3], pts[2]]
                if len(pts) == 4 and np.allclose(pts[2], pts[3]):
                    pts = pts[:3]
                faces.setdefault(lay, []).append(pts)
            elif t == "MESH":
                V = np.array([tuple(v) for v in e.vertices])
                F = [list(f) for f in e.faces]
                if len(F):
                    add(G.mesh(V, F, source="Imported mesh"), lay)
            elif t == "POLYLINE":
                for ve in e.virtual_entities():
                    handle(ve, depth + 1)
            elif t == "INSERT" and depth < 8:
                for ve in e.virtual_entities():
                    if ve.dxf.get("layer", "0") == "0":
                        ve.dxf.layer = lay
                    handle(ve, depth + 1)
            else:
                skipped[0] += 1
        except Exception:
            skipped[0] += 1

    for e in dxf.modelspace():
        handle(e)
    for lay, fl in faces.items():
        V, F = [], []
        for pts in fl:
            F.append(list(range(len(V), len(V) + len(pts))))
            V += [p for p in pts]
        add(G.mesh(np.array(V), F, source="Imported 3D faces"), lay)
    for g in out:
        flat = g.kind != "mesh" and np.allclose(g.bbox()[0][2], 0) and np.allclose(g.bbox()[1][2], 0)
        g.space = "2d" if flat else "3d"
    used = {g.layer for g in out}
    warn = [f"{skipped[0]} entities skipped (text, hatches, dimensions...)."] if skipped[0] else []
    return [l for n_, l in layers.items() if n_ in used], out, warn


def _import_svg(path, units, stem):
    from svgelements import SVG, Shape, Path, Circle, Ellipse, Move, Line, Close, CubicBezier, QuadraticBezier, Arc
    to_units = (25.4 / 96) / (UNIT_M.get(units, 0.001) / 0.001)
    svg = SVG.parse(path, reify=True, ppi=96)
    lay = _new_layer(stem)
    out = []
    P2 = lambda pt: np.array([pt[0] * to_units, -pt[1] * to_units, 0.0])

    for el in svg.elements():
        if not isinstance(el, Shape):
            continue
        if isinstance(el, (Circle, Ellipse)) and el.rotation == 0 and not getattr(el, "skew", 0):
            try:
                c = P2((el.cx, el.cy))
                out.append(Geom("ellipse", c=c, u=np.array([el.rx * to_units, 0, 0]), v=np.array([0, el.ry * to_units, 0]),
                                a0=0.0, a1=2 * math.pi, layer=lay.name, space="2d"))
                continue
            except Exception:
                pass
        path = Path(el)
        sub = []

        def flush(closed=False):
            if len(sub) == 0:
                return
            pts0 = [s for s in sub]
            if all(kind == "L" for kind, *_ in pts0):
                pts = [pts0[0][1]] + [s[2] for s in pts0]
                if closed and len(pts) > 2 and np.allclose(pts[0], pts[-1]):
                    pts = pts[:-1]
                out.append(G.polyline(pts, closed=closed, layer=lay.name, space="2d"))
            else:
                cps = [pts0[0][1]]
                for kind, a, b, c1, c2 in pts0:
                    if kind == "L":
                        cps += [a + (b - a) / 3, a + 2 * (b - a) / 3, b]
                    else:
                        cps += [c1, c2, b]
                n = len(pts0)
                knots = [0.0] * 4 + [k for i in range(1, n) for k in (i, i, i)] + [float(n)] * 4
                out.append(Geom("nurbs", cps=np.array(cps), w=np.ones(len(cps)), knots=np.array(knots) / n, deg=3,
                                layer=lay.name, space="2d"))
            sub.clear()

        for seg in path:
            if isinstance(seg, Move):
                flush(False)
            elif isinstance(seg, Close):
                if seg.start is not None and seg.end is not None and seg.start != seg.end:
                    sub.append(("L", P2(seg.start), P2(seg.end), None, None))
                flush(True)
            elif isinstance(seg, Line):
                sub.append(("L", P2(seg.start), P2(seg.end), None, None))
            elif isinstance(seg, CubicBezier):
                sub.append(("C", P2(seg.start), P2(seg.end), P2(seg.control1), P2(seg.control2)))
            elif isinstance(seg, QuadraticBezier):
                a, q, b = P2(seg.start), P2(seg.control), P2(seg.end)
                sub.append(("C", a, b, a + 2 / 3 * (q - a), b + 2 / 3 * (q - b)))
            elif isinstance(seg, Arc):
                for cb in seg.as_cubic_curves():
                    sub.append(("C", P2(cb.start), P2(cb.end), P2(cb.control1), P2(cb.control2)))
        flush(False)
    return [lay], out, []
