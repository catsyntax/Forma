"""Components: reusable definitions, instances, connectors and parametric ("special") components.

A ComponentDef holds geometry in local coordinates, or Python ``code`` that builds geometry
from parameters. Instances are Geoms of kind "instance" referencing a def by id with a 4x4
transform and parameter overrides. Editing a definition updates every instance.
No Qt dependency.
"""
from __future__ import annotations

import json
import math
import os
import textwrap
import types

import numpy as np

from . import geometry as G
from .geometry import Geom, new_id

USER_LIBRARY = os.path.join(os.path.expanduser("~"), ".forma", "components.json")


class ComponentDef:
    def __init__(self, name, geoms=None, code="", params=None, connectors=None,
                 category="My Components", description="", id=None, version=0, base_point=(0, 0, 0)):
        self.id = id or new_id()
        self.name = name
        self.geoms: list[Geom] = geoms or []
        self.code = code
        self.params: dict = dict(params or {})
        self.connectors: list[dict] = connectors or []
        self.category = category
        self.description = description
        self.version = version
        self.base_point = list(base_point)
        self._cache = {}

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_cache"] = {}
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    @property
    def parametric(self):
        return bool(self.code.strip())

    def clone(self, **kw):
        d = ComponentDef(self.name, [g.copy() for g in self.geoms], self.code, dict(self.params),
                         [dict(c) for c in self.connectors], self.category, self.description,
                         self.id, self.version + 1, self.base_point)
        for k, v in kw.items():
            setattr(d, k, v)
        return d

    # -------------------------------------------------------------- build
    def _module(self):
        key = ("__mod__", self.version, hash(self.code))
        if key not in self._cache:
            ns = make_namespace()
            exec(compile(self.code, f"<component {self.name}>", "exec"), ns)
            self._cache[key] = ns
        return self._cache[key]

    def merged_params(self, overrides=None):
        p = dict(self.params)
        for k, v in (overrides or {}).items():
            if k in p:
                p[k] = type(p[k])(v) if not isinstance(p[k], bool) else bool(v)
        return p

    def build(self, overrides=None):
        """Local-space geometry for given parameter overrides (cached)."""
        if not self.parametric:
            return self.geoms
        p = self.merged_params(overrides)
        key = ("geo", self.version, json.dumps(p, sort_keys=True, default=str))
        if key not in self._cache:
            ns = self._module()
            res = ns["build"](Params(p))
            self._cache[key] = G.as_geom_list(res)
            if len(self._cache) > 64:
                for k in list(self._cache)[:16]:
                    if k[0] == "geo":
                        self._cache.pop(k, None)
        return self._cache[key]

    def get_connectors(self, overrides=None):
        if self.parametric:
            ns = self._module()
            if "connectors" in ns:
                return [normalise_connector(c) for c in ns["connectors"](Params(self.merged_params(overrides)))]
        return [normalise_connector(c) for c in self.connectors]

    # -------------------------------------------------------------- io
    def to_dict(self):
        return {"id": self.id, "name": self.name, "geoms": [g.to_dict() for g in self.geoms],
                "code": self.code, "params": self.params, "connectors": self.connectors,
                "category": self.category, "description": self.description,
                "version": self.version, "base_point": self.base_point}

    @staticmethod
    def from_dict(d):
        return ComponentDef(d["name"], [Geom.from_dict(g) for g in d.get("geoms", [])], d.get("code", ""),
                            d.get("params", {}), d.get("connectors", []), d.get("category", "My Components"),
                            d.get("description", ""), d.get("id"), d.get("version", 0), d.get("base_point", (0, 0, 0)))


class Params(dict):
    """Dict that also allows attribute access: p.width"""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


def make_namespace():
    ns = {"__builtins__": __builtins__, "np": np, "math": math, "G": G, "geo": G}
    for name in ("P", "point", "line", "polyline", "rectangle", "polygon", "circle", "ellipse", "arc", "arc_3pt",
                 "nurbs_curve", "interp_curve", "translation", "rotation", "scaling", "mirror", "mesh"):
        ns[name] = getattr(G, name)
    try:  # exact NURBS solids and surfaces
        from . import brep as B
        for name in ("box", "sphere", "cylinder", "cone", "torus", "plane_surface", "planar_surface", "extrude",
                     "loft", "revolve", "pipe", "sweep1", "fillet", "chamfer", "boolean", "intersect"):
            ns[name] = getattr(B, name)
        ns["B"] = B
        ns["union"] = lambda a, b: B.boolean(G.as_geom_list(a), G.as_geom_list(b), "union")
        ns["difference"] = lambda a, b: B.boolean(G.as_geom_list(a), G.as_geom_list(b), "difference")
    except ImportError:  # OpenCascade missing: fall back to mesh primitives
        for name in ("box", "sphere", "cylinder", "cone", "torus", "plane_surface", "planar_surface", "extrude",
                     "loft", "revolve", "pipe"):
            ns[name] = getattr(G, name)
    ns["connector"] = connector
    return ns


def connector(name, p, n, up=(0, 0, 1)):
    return {"name": name, "p": list(map(float, G.P(p))), "n": list(map(float, G.P(n))), "up": list(map(float, G.P(up)))}


def normalise_connector(c):
    n = G.P(c.get("n", (1, 0, 0)))
    n = n / (np.linalg.norm(n) or 1)
    up = G.P(c.get("up", (0, 0, 1)))
    up = up - np.dot(up, n) * n
    if np.linalg.norm(up) < 1e-9:
        up = G.plane_axes(n)[0]
    up = up / np.linalg.norm(up)
    return {"name": c.get("name", "C"), "p": G.P(c["p"]), "n": n, "up": up}


# ===================================================================== instances
def instance(def_id, M=None, params=None, **kw):
    return Geom("instance", def_id=def_id, M=np.eye(4) if M is None else np.asarray(M, float),
                params=dict(params or {}), **kw)


def expand(inst: Geom, defs: dict, depth=0) -> list[Geom]:
    """World-space geometry of an instance (nested instances are expanded)."""
    d = defs.get(inst.def_id)
    if d is None or depth > 8:
        return []
    out = []
    try:
        local = d.build(inst.params)
    except Exception as e:  # broken parametric code: show a marker
        local = [G.point((0, 0, 0))]
        inst.data["_error"] = str(e)
    for g in local:
        if g.kind == "instance":
            out += [x.transformed(inst.M, keep_id=False) for x in expand(g, defs, depth + 1)]
        else:
            w = g.transformed(inst.M, keep_id=False)
            w.layer = inst.layer if g.layer in ("Default", "") else g.layer
            w.space = inst.space
            out.append(w)
    return out


def world_connectors(inst: Geom, defs: dict):
    d = defs.get(inst.def_id)
    if d is None:
        return []
    M = inst.M
    R = M[:3, :3]
    out = []
    for c in d.get_connectors(inst.params):
        p = G.xform_pts(M, c["p"].reshape(1, 3))[0]
        n = R @ c["n"]
        up = R @ c["up"]
        out.append({"name": c["name"], "p": p, "n": n / (np.linalg.norm(n) or 1), "up": up / (np.linalg.norm(up) or 1)})
    return out


def frame_matrix(p, n, up):
    x = np.asarray(n, float)
    z = np.asarray(up, float)
    z = z - np.dot(z, x) * x
    z /= np.linalg.norm(z) or 1
    y = np.cross(z, x)
    F = np.eye(4)
    F[:3, 0], F[:3, 1], F[:3, 2], F[:3, 3] = x, y, z, p
    return F


def connect_matrix(moving_conn, target_conn, spin_deg=0.0):
    """Transform that puts moving connector onto target connector, facing it."""
    A = frame_matrix(moving_conn["p"], moving_conn["n"], moving_conn["up"])
    B = frame_matrix(target_conn["p"], -target_conn["n"], target_conn["up"])
    S = G.rotation(spin_deg, (1, 0, 0))
    return B @ S @ np.linalg.inv(A)


def bbox_connectors(geoms):
    """Six face-centre connectors around the bounding box (default for plain components)."""
    pts = np.vstack([g.control_points() for g in geoms if len(g.control_points())]) if geoms else np.zeros((1, 3))
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    out = []
    for axis, nm in ((0, "X"), (1, "Y"), (2, "Z")):
        for sign, lo_hi in ((-1, lo), (1, hi)):
            p = c.copy()
            p[axis] = lo_hi[axis]
            n = np.zeros(3)
            n[axis] = sign
            up = (0, 0, 1) if axis != 2 else (1, 0, 0)
            out.append(connector(("+" if sign > 0 else "-") + nm, p, n, up))
    return out


# ===================================================================== library
BUILTIN = []


def _builtin(name, category, description, params, code):
    BUILTIN.append(ComponentDef(name, code=textwrap.dedent(code).strip() + "\n", params=params, category=category,
                                description=description, id="builtin-" + name.lower().replace(" ", "-")))


_builtin("Frame Module", "Modular", "Cubic frame cell with a connector on every face. Chain them to build structures fast.",
         {"size_x": 1000.0, "size_y": 1000.0, "size_z": 1000.0, "member": 50.0},
         """
         def build(p):
             sx, sy, sz, m = p.size_x, p.size_y, p.size_z, p.member
             out = []
             for x in (0, sx - m):
                 for y in (0, sy - m):
                     out.append(box((x, y, 0), m, m, sz))          # posts
             for z in (0, sz - m):
                 for y in (0, sy - m):
                     out.append(box((m, y, z), sx - 2*m, m, m))   # x rails
                 for x in (0, sx - m):
                     out.append(box((x, m, z), m, sy - 2*m, m))   # y rails
             return out

         def connectors(p):
             sx, sy, sz = p.size_x, p.size_y, p.size_z
             return [connector('+X', (sx, sy/2, sz/2), (1, 0, 0)), connector('-X', (0, sy/2, sz/2), (-1, 0, 0)),
                     connector('+Y', (sx/2, sy, sz/2), (0, 1, 0)), connector('-Y', (sx/2, 0, sz/2), (0, -1, 0)),
                     connector('+Z', (sx/2, sy/2, sz), (0, 0, 1), up=(1, 0, 0)),
                     connector('-Z', (sx/2, sy/2, 0), (0, 0, -1), up=(1, 0, 0))]
         """)

_builtin("Shelf Unit", "Furniture", "Shelving unit with adjustable size and number of shelves.",
         {"width": 800.0, "depth": 300.0, "height": 1800.0, "shelves": 5, "thickness": 18.0},
         """
         def build(p):
             w, d, h, t, n = p.width, p.depth, p.height, p.thickness, max(2, int(p.shelves))
             out = [box((0, 0, 0), t, d, h), box((w - t, 0, 0), t, d, h), box((t, d - t/2, 0), w - 2*t, t/2, h)]
             for i in range(n):
                 z = i * (h - t) / (n - 1)
                 out.append(box((t, 0, z), w - 2*t, d - t/2, t))
             return out

         def connectors(p):
             return [connector('Left', (0, p.depth/2, 0), (-1, 0, 0)),
                     connector('Right', (p.width, p.depth/2, 0), (1, 0, 0))]
         """)

_builtin("Wall Panel", "Modular", "Straight wall panel with connectors at both ends.",
         {"length": 2400.0, "height": 2700.0, "thickness": 120.0},
         """
         def build(p):
             return [box((0, 0, 0), p.length, p.thickness, p.height)]

         def connectors(p):
             t = p.thickness
             return [connector('Start', (0, t/2, 0), (-1, 0, 0)), connector('End', (p.length, t/2, 0), (1, 0, 0))]
         """)

_builtin("Column", "Structure", "Round or square column.",
         {"size": 300.0, "height": 3000.0, "round": True},
         """
         def build(p):
             if p['round']:
                 return [cylinder((0, 0, 0), p.size/2, p.height)]
             s = p.size
             return [box((-s/2, -s/2, 0), s, s, p.height)]

         def connectors(p):
             return [connector('Base', (0, 0, 0), (0, 0, -1), up=(1, 0, 0)),
                     connector('Top', (0, 0, p.height), (0, 0, 1), up=(1, 0, 0))]
         """)

_builtin("Beam", "Structure", "Rectangular beam along X.",
         {"length": 4000.0, "width": 200.0, "depth": 400.0},
         """
         def build(p):
             return [box((0, -p.width/2, -p.depth), p.length, p.width, p.depth)]

         def connectors(p):
             return [connector('Start', (0, 0, -p.depth/2), (-1, 0, 0)),
                     connector('End', (p.length, 0, -p.depth/2), (1, 0, 0))]
         """)

_builtin("Pipe Segment", "MEP", "Straight pipe with end connectors.",
         {"length": 1000.0, "radius": 50.0},
         """
         def build(p):
             return [pipe(line((0, 0, 0), (p.length, 0, 0)), p.radius, sides=24)]

         def connectors(p):
             return [connector('A', (0, 0, 0), (-1, 0, 0)), connector('B', (p.length, 0, 0), (1, 0, 0))]
         """)

_builtin("Pipe Elbow", "MEP", "90 degree pipe bend.",
         {"bend_radius": 150.0, "radius": 50.0},
         """
         def build(p):
             r = p.bend_radius
             c = arc((0, r, 0), r, -90, 0)
             return [pipe(c, p.radius, sides=24)]

         def connectors(p):
             r = p.bend_radius
             return [connector('A', (0, 0, 0), (-1, 0, 0)), connector('B', (r, r, 0), (0, 1, 0))]
         """)

_builtin("Panel Grid", "Facade", "Grid of panels with gaps.",
         {"width": 3000.0, "height": 3000.0, "cols": 4, "rows": 3, "gap": 20.0, "thickness": 20.0},
         """
         def build(p):
             cols, rows = max(1, int(p.cols)), max(1, int(p.rows))
             pw = (p.width - (cols - 1) * p.gap) / cols
             ph = (p.height - (rows - 1) * p.gap) / rows
             out = []
             for i in range(cols):
                 for j in range(rows):
                     out.append(box((i * (pw + p.gap), 0, j * (ph + p.gap)), pw, p.thickness, ph))
             return out

         def connectors(p):
             return [connector('Left', (0, 0, 0), (-1, 0, 0)), connector('Right', (p.width, 0, 0), (1, 0, 0)),
                     connector('Top', (0, 0, p.height), (0, 0, 1), up=(1, 0, 0))]
         """)

_builtin("Stair", "Structure", "Straight stair; total rise = steps x rise.",
         {"steps": 12, "rise": 175.0, "run": 280.0, "width": 1000.0},
         """
         def build(p):
             out = []
             for i in range(max(1, int(p.steps))):
                 out.append(box((i * p.run, 0, 0), p.run, p.width, (i + 1) * p.rise))
             return out

         def connectors(p):
             n = max(1, int(p.steps))
             return [connector('Bottom', (0, p.width/2, 0), (-1, 0, 0)),
                     connector('Top', (n * p.run, p.width/2, n * p.rise), (1, 0, 0))]
         """)

_builtin("Table", "Furniture", "Table with four legs.",
         {"length": 1600.0, "width": 800.0, "height": 740.0, "top": 30.0, "leg": 60.0, "inset": 50.0},
         """
         def build(p):
             L, W, H, t, g, i = p.length, p.width, p.height, p.top, p.leg, p.inset
             out = [box((0, 0, H - t), L, W, t)]
             for x in (i, L - i - g):
                 for y in (i, W - i - g):
                     out.append(box((x, y, 0), g, g, H - t))
             return out

         def connectors(p):
             return [connector('Left', (0, p.width/2, 0), (-1, 0, 0)), connector('Right', (p.length, p.width/2, 0), (1, 0, 0))]
         """)

_builtin("Window", "Facade", "Window frame with glass pane and mullions.",
         {"width": 1200.0, "height": 1500.0, "frame": 60.0, "depth": 80.0, "mullions": 1},
         """
         def build(p):
             w, h, f, d = p.width, p.height, p.frame, p.depth
             out = [box((0, 0, 0), w, d, f), box((0, 0, h - f), w, d, f),
                    box((0, 0, f), f, d, h - 2*f), box((w - f, 0, f), f, d, h - 2*f)]
             m = max(0, int(p.mullions))
             for k in range(m):
                 x = (k + 1) * w / (m + 1) - f/4
                 out.append(box((x, d/4, f), f/2, d/2, h - 2*f))
             out.append(box((f, d/2 - 3, f), w - 2*f, 6, h - 2*f))
             return out

         def connectors(p):
             return [connector('Left', (0, p.depth/2, 0), (-1, 0, 0)), connector('Right', (p.width, p.depth/2, 0), (1, 0, 0))]
         """)

_builtin("Hex Tile", "Modular", "Hexagonal tile with six edge connectors for tessellations.",
         {"radius": 300.0, "thickness": 20.0},
         """
         def build(p):
             return [extrude(polygon((0, 0, 0), p.radius, 6, 30), (0, 0, p.thickness))]

         def connectors(p):
             out = []
             a = p.radius * math.cos(math.radians(30))
             for k in range(6):
                 t = math.radians(60 * k)
                 out.append(connector('E%d' % k, (a * math.cos(t), a * math.sin(t), 0), (math.cos(t), math.sin(t), 0)))
             return out
         """)

NEW_COMPONENT_TEMPLATE = '''# Parametric component.
# Parameters are edited in the Properties panel and passed in as `p` (p.width or p['width']).
# Available: box, sphere, cylinder, cone, torus, line, polyline, rectangle, circle, arc,
# extrude, loft, revolve, pipe, translation, rotation, np, math, connector(...)

def build(p):
    out = []
    for i in range(int(p.count)):
        out.append(box((i * (p.size + p.gap), 0, 0), p.size, p.size, p.size))
    return out

def connectors(p):
    total = p.count * p.size + (p.count - 1) * p.gap
    return [connector('Start', (0, p.size / 2, 0), (-1, 0, 0)),
            connector('End', (total, p.size / 2, 0), (1, 0, 0))]
'''
NEW_COMPONENT_PARAMS = {"count": 3, "size": 500.0, "gap": 100.0}


def load_user_library():
    try:
        with open(USER_LIBRARY) as f:
            return [ComponentDef.from_dict(d) for d in json.load(f)]
    except Exception:
        return []


def save_user_library(defs):
    os.makedirs(os.path.dirname(USER_LIBRARY), exist_ok=True)
    with open(USER_LIBRARY, "w") as f:
        json.dump([d.to_dict() for d in defs], f)


def library():
    return list(BUILTIN) + load_user_library()
