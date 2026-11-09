"""Nodes ported from Grasshopper scripts: 3D voxel space syntax and the modular plan generator.

The heavy engines live in ``vga3d`` and ``planlib`` (pure NumPy / Python, unchanged from the
original scripts); this module only translates Forma geometry in and out.
"""
from __future__ import annotations

import re

import numpy as np

from .. import geometry as G
from ..geometry import Geom
from .registry import node, Port, flatten, num
from .analysis import triangles_of, colored_mesh
from . import vga3d
from . import planlib

UNIT_TO_M = {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254, "ft": 0.3048}
_VGA_CACHE: dict = {}


def _flag(v, default=False):
    if v is None or v == "":
        return default
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(v, list):
        return _flag(v[0], default) if v else default
    return bool(v)


def _tris(geoms):
    V, T = triangles_of(geoms)
    if not len(T):
        return np.zeros((0, 3, 3))
    return V[T]


def _band_mesh(band, vals, origin, s, k_level, label):
    """One coloured square per observer cell at the band's height (same as the Rhino version)."""
    band = np.asarray(band)
    z = origin[2] + k_level * s
    x0 = origin[0] + band[:, 0] * s
    y0 = origin[1] + band[:, 1] * s
    n = len(band)
    V = np.zeros((n * 4, 3))
    V[0::4] = np.column_stack([x0, y0, np.full(n, z)])
    V[1::4] = np.column_stack([x0 + s, y0, np.full(n, z)])
    V[2::4] = np.column_stack([x0 + s, y0 + s, np.full(n, z)])
    V[3::4] = np.column_stack([x0, y0 + s, np.full(n, z)])
    b = np.arange(n) * 4
    T = np.column_stack([np.stack([b, b + 1, b + 2], 1), np.stack([b, b + 2, b + 3], 1)]).reshape(-1, 3)
    return colored_mesh(V, T, np.repeat(np.asarray(vals, float), 2), None, None, label)


@node("Space Syntax 3D (voxel VGA)", "Space Syntax",
      [Port("Space", None, "list", hint="one closed solid: the AIR volume of the interior"),
       Port("Furniture", None, "list", hint="solids that block standing and sight"),
       Port("Voxel Size", 200.0, hint="cell size in model units"),
       Port("Eye Band", 7, hint="eye level = floor band + N cells"),
       Port("Vis Radius", 5000.0, hint="sightline length limit; 0 = unlimited (slow on fine grids)"),
       Port("Metric", 1, hint="0..14, see Metric Names"),
       Port("Choice", False, hint="also compute Choice on the walk band (slowest part)"),
       Port("Lines", False, hint="output sightlines (can be millions)"),
       Port("Vis 3D", False, hint="eye band to ALL free 3D cells (isovist volume)"),
       Port("Run", True)],
      outputs=("Walk Heatmap", "Vis Heatmap", "Walk Values", "Vis Values", "Walk Points", "Vis Points",
               "Points", "Count", "Walk Lines", "Vis Lines", "Walk All", "Vis All", "Metric Names", "Progress"),
      desc="Voxel visibility graph analysis of a 3D interior (walkability on the floor band, visibility at eye "
           "level). Ported from the Grasshopper script 'Space Syntax Master v03'. Results are cached, so changing "
           "only Metric is instant.")
def n_vga3d(Space=None, Furniture=None, _progress=None, **kw):
    names = vga3d.METRIC_NAMES
    metric_names = [f"{i}: {n}" for i, n in enumerate(names)]
    run = _flag(kw.get("Run"), True)
    s = num(kw.get("Voxel Size"), 200.0) or 200.0
    eb = int(num(kw.get("Eye Band"), 7))
    rad = num(kw.get("Vis Radius"), 0.0)
    R2 = rad * rad if rad > 0 else None
    mi = max(0, min(int(num(kw.get("Metric"), 1)), len(names) - 1))
    want_choice = _flag(kw.get("Choice"))
    want_lines = _flag(kw.get("Lines"))
    vis3d = _flag(kw.get("Vis 3D"))
    spaces = [g for g in flatten(Space) if isinstance(g, Geom)]
    if not spaces:
        raise ValueError("Connect the AIR volume of the room (one closed solid) to Space")
    space_tris = _tris(spaces)
    furn_tris = _tris([g for g in flatten(Furniture or []) if isinstance(g, Geom)])
    if not len(space_tris):
        raise ValueError("Space has no surfaces to analyse")
    sig = (round(s, 6), eb, round(rad, 6), want_choice, vis3d, space_tris.shape, furn_tris.shape,
           tuple(np.round(space_tris.reshape(-1, 3).min(0), 4)), tuple(np.round(space_tris.reshape(-1, 3).max(0), 4)),
           float(np.round(space_tris.sum(), 3)), float(np.round(furn_tris.sum(), 3)) if len(furn_tris) else 0.0)
    log_lines = []

    def log(t):
        log_lines.append(t)
        if _progress:
            _progress(min(0.95, 0.05 + 0.07 * len(log_lines)))

    core = _VGA_CACHE.get(sig)
    if core is None:
        if not run:
            return {"Metric Names": metric_names, "Progress": "Idle (Run is off and nothing is cached)."}
        core = vga3d._compute_core(space_tris, furn_tris, s, eb, R2, want_choice, log, vis3d)
        _VGA_CACHE.clear()
        _VGA_CACHE[sig] = core
    else:
        log("Using cached result (only the displayed metric changed).")
    origin, s_, kmin, ke = core["origin"], core["s"], core["kmin"], core["ke"]
    Wc, Hc = core["W"], core["H"]
    wmet, vmet = core["w_metrics"], core["v_metrics"]
    mname = names[mi]
    if mname not in wmet:
        raise ValueError(f"{mname} needs Choice = True")
    walk_vals = [float(x) for x in wmet[mname]]
    if mi == 0:
        denom = float(max(1, (len(core["V"]) if core.get("vis3d") else len(Hc)) - 1))
        vis_vals = [100.0 * float(x) / denom for x in vmet["Connectivity"]]
    else:
        vis_vals = [float(x) for x in vmet[mname]]
    cpts = origin + (core["V"] + 0.5) * s_
    out = {
        "Walk Heatmap": _band_mesh(Wc, walk_vals, origin, s_, kmin, "Walk " + mname),
        "Vis Heatmap": _band_mesh(Hc, vis_vals, origin, s_, ke, "Vis " + mname),
        "Walk Values": walk_vals, "Vis Values": vis_vals,
        # coordinates rather than point objects: thousands of points would bury the heat maps in the preview
        "Walk Points": [tuple(map(float, p)) for p in core["Wpts"]],
        "Vis Points": [tuple(map(float, p)) for p in core["Hpts"]],
        "Points": None, "Count": len(cpts),
        "Walk Lines": [], "Vis Lines": [],
        "Walk All": {nm: [float(x) for x in wmet[nm]] for nm in names if nm in wmet},
        "Vis All": {nm: [float(x) for x in vmet[nm]] for nm in names if nm in vmet},
        "Metric Names": metric_names,
    }
    out["Points"] = [tuple(map(float, p)) for p in cpts]
    if want_lines:
        wp = core["Wpts"]
        vp = cpts if core.get("vis3d") else core["Hpts"]
        cap = 200000
        out["Walk Lines"] = [G.line(wp[a], wp[b]) for a, b in core["w_edges"][:cap]]
        out["Vis Lines"] = [G.line(vp[a], vp[b]) for a, b in core["v_edges"][:cap]]
    out["Progress"] = "\n".join(log_lines + [
        f"DONE | cells {len(cpts)}, walk band {len(Wc)} ({len(core['w_edges'])} edges), "
        f"vis band {len(Hc)} ({len(core['v_edges'])} edges) [{mname} displayed]"])
    return out


# ===================================================================== modular plans
DEFAULT_BRIEF = """# code  type      modules  attached-to
L1     living    3
L1_KI  kitchen   1.5      L1
B1     bedroom   2
B1_BA  bathroom  1        B1
B2     bedroom   2
C1     corridor  2
S1     storage   0.5
"""

_TYPES = {"l": "living", "living": "living", "lounge": "living", "b": "bedroom", "bed": "bedroom", "bedroom": "bedroom",
          "k": "kitchen", "kitchen": "kitchen", "ba": "bathroom", "bath": "bathroom", "bathroom": "bathroom",
          "s": "storage", "st": "storage", "storage": "storage", "c": "corridor", "corridor": "corridor"}


def parse_brief(text):
    """Lines of 'code type modules [attached-to]'. Returns payload nodes and default edges."""
    nodes, errs = [], []
    for ln_no, raw in enumerate(str(text or "").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = re.split(r"[\s,;]+", line)
        if len(parts) < 3:
            errs.append(f"line {ln_no}: need 'code type modules'")
            continue
        code, typ = parts[0], _TYPES.get(parts[1].lower())
        if typ is None:
            errs.append(f"line {ln_no}: unknown type '{parts[1]}' (living, bedroom, kitchen, bathroom, storage, corridor)")
            continue
        try:
            mods = float(parts[2])
        except ValueError:
            errs.append(f"line {ln_no}: modules must be a number")
            continue
        nodes.append({"id": code, "code": code, "type": typ, "modules": max(0.5, mods),
                      "parent": parts[3] if len(parts) > 3 else None})
    if errs:
        raise ValueError("Brief: " + "; ".join(errs))
    if not nodes:
        raise ValueError("The brief is empty: one space per line, e.g. 'L1 living 3'")
    return nodes


def default_edges(nodes):
    """Same defaults as the original interface: attached spaces link to their parent, every other
    space links to the first living room."""
    ids = {n["id"] for n in nodes}
    root = next((n["id"] for n in nodes if n["type"] == "living"), nodes[0]["id"])
    edges = []
    for n in nodes:
        if n["parent"] in ids:
            edges.append([n["parent"], n["id"]])
        elif n["id"] != root:
            edges.append([root, n["id"]])
    return edges


def parse_links(text, nodes):
    ids = {n["id"] for n in nodes}
    edges = []
    for a, b in re.findall(r"([A-Za-z0-9_]+)\s*[-–>]+\s*([A-Za-z0-9_]+)", str(text or "")):
        for x in (a, b):
            if x not in ids:
                raise ValueError(f"Links: '{x}' is not in the brief")
        edges.append([a, b])
    edges += [[n["parent"], n["id"]] for n in nodes if n["parent"] in ids and [n["parent"], n["id"]] not in edges]
    return edges


def code_check(nodes, short, long_, height):
    out = []
    for n in nodes:
        st = planlib.STANDARDS.get(n["type"])
        if not st:
            continue
        w, d = planlib.footprint(n["type"], n["modules"], short, long_)
        msgs = []
        if w * d + 1e-9 < st["min_area"]:
            msgs.append(f"area {w * d:.2f} m2 < {st['min_area']:.2f} m2")
        if min(w, d) + 1e-9 < st["min_dim"]:
            msgs.append(f"least dimension {min(w, d):.2f} m < {st['min_dim']:.2f} m")
        if height + 1e-9 < st["min_height"]:
            msgs.append(f"height {height:.2f} m < {st['min_height']:.2f} m")
        if msgs:
            out.append(f"{n['code']} ({st['label']}): " + ", ".join(msgs))
    return out


@node("Modular Plan Generator", "Space Planning",
      [Port("Module A", 2400.0, hint="module side, model units"), Port("Module B", 3600.0),
       Port("Module Height", 2700.0),
       Port("Brief", DEFAULT_BRIEF, hint="one space per line: code type modules [attached-to]"),
       Port("Links", "", hint="adjacencies like 'L1-B1, L1-C1'; empty = link everything to the living room"),
       Port("Plans", 5), Port("Gap", 4000.0, hint="spacing between plan options"),
       Port("Extrude", False, hint="show rooms as module-height volumes")],
      outputs=("Rooms", "Outlines", "Envelopes", "Room Codes", "Report", "Code Check"),
      desc="Packs a brief of modular rooms into several plan options, keeping linked rooms touching. Rooms grow "
           "along the module's short side, corridors along the long side. Ported from the Grasshopper script "
           "'Modular Plan Generator' (its indicative Australian room standards are used for Code Check).")
def n_plans(Brief=None, Links=None, _progress=None, _units="mm", **kw):
    u = UNIT_TO_M.get(_units, 0.001)
    a = num(kw.get("Module A"), 2400) * u
    b = num(kw.get("Module B"), 3600) * u
    h = num(kw.get("Module Height"), 2700) * u
    if min(a, b) <= 0:
        raise ValueError("Module sides must be positive")
    nodes = parse_brief(Brief if Brief not in (None, "") else DEFAULT_BRIEF)
    edges = parse_links(Links, nodes) if str(Links or "").strip() else default_edges(nodes)
    payload = {"module": {"a": a, "b": b, "h": h}, "plans": max(1, min(12, int(num(kw.get("Plans"), 5)))),
               "nodes": nodes, "edges": edges}
    if _progress:
        _progress(0.1)
    plans, adj = planlib.generate_plans(payload)
    gap = num(kw.get("Gap"), 4000) * u
    extrude = _flag(kw.get("Extrude"))
    k = 1.0 / u
    rooms, outlines, envs, codes, lines = [], [], [], [], []
    cursor = 0.0
    for pi, placed in enumerate(plans):
        x0, y0, x1, y1 = planlib.bbox_of(placed)
        bw, bh = x1 - x0, y1 - y0
        used = sum(r.area for r in placed)
        _, miss = planlib.plan_score(placed, adj)
        pname = "PLAN_%02d" % (pi + 1)
        plan_rooms, plan_out, plan_codes = [], [], []
        for r in placed:
            pts = [((r.x + cursor) * k, r.y * k, 0), ((r.x + r.w + cursor) * k, r.y * k, 0),
                   ((r.x + r.w + cursor) * k, (r.y + r.d) * k, 0), ((r.x + cursor) * k, (r.y + r.d) * k, 0)]
            crv = G.polyline(pts, closed=True)
            crv.name = f"{pname}_{r.code}"
            crv.data["role"] = r.type
            if extrude:
                body = G.extrude(crv, (0, 0, h * k * 0.98))
            else:
                body = G.planar_surface(crv)
            body.name = crv.name
            body.data["role"] = r.type
            plan_rooms.append(body)
            plan_out.append(crv)
            plan_codes.append(f"{pname} {r.code} {r.w:.2f}x{r.d:.2f} m = {r.area:.1f} m2")
        env = G.polyline([(cursor * k, 0, 0), ((cursor + bw) * k, 0, 0), ((cursor + bw) * k, bh * k, 0),
                          (cursor * k, bh * k, 0)], closed=True)
        env.name = pname + "_ENVELOPE"
        env.data["role"] = "envelope"
        rooms.append(plan_rooms)
        outlines.append(plan_out)
        envs.append(env)
        codes.append(plan_codes)
        lines.append("%s  %6.2f x %6.2f m   footprint %8.2f m2   rooms %8.2f m2   efficiency %5.1f%%   unmet links %d"
                     % (pname, bw, bh, bw * bh, used, 100.0 * used / max(1e-9, bw * bh), miss))
        cursor += bw + gap
    header = ("Module %.2f x %.2f m, height %.2f m  |  %d spaces  |  %d plans"
              % (min(a, b), max(a, b), h, len(nodes), len(plans)))
    checks = code_check(nodes, min(a, b), max(a, b), h)
    return {"Rooms": rooms, "Outlines": outlines, "Envelopes": envs, "Room Codes": codes,
            "Report": header + "\n" + "\n".join(lines),
            "Code Check": "\n".join(checks) if checks else "All spaces meet the indicative minimums."}
