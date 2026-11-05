"""Node type library. Pure Python (no Qt) so it runs inside the compute worker process.

Each node function receives its inputs as keyword arguments. Inputs marked access="item" are
automatically mapped over lists (longest-list matching), like Grasshopper.
"""
from __future__ import annotations

import io
import math
import random
import time
import contextlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .. import geometry as G
from .. import components as C
from .. import brep as B
from ..geometry import Geom


@dataclass
class Port:
    name: str
    default: object = None
    access: str = "item"   # "item" (auto-map over lists) or "list"
    hint: str = ""


@dataclass
class NodeType:
    name: str
    category: str
    inputs: list
    outputs: list
    fn: Callable
    desc: str = ""
    widget: str | None = None   # slider, number, integer, toggle, text, panel, ref, code, component
    dynamic: bool = False       # script node with user-defined ports


NODE_TYPES: dict[str, NodeType] = {}


def node(name, category, inputs=(), outputs=("Result",), desc="", widget=None, dynamic=False):
    def deco(fn):
        NODE_TYPES[name] = NodeType(name, category, list(inputs), list(outputs), fn, desc or (fn.__doc__ or "").strip(),
                                    widget, dynamic)
        return fn
    return deco


# ===================================================================== helpers
def vec(x):
    if isinstance(x, Geom):
        if x.kind == "point":
            return x.p.copy()
        return x.centroid()
    if x is None:
        return np.zeros(3)
    if isinstance(x, (int, float, np.floating, np.integer)):
        return np.array([0.0, 0.0, float(x)])
    a = np.asarray(x, float).ravel()
    out = np.zeros(3)
    out[: min(3, len(a))] = a[:3]
    return out


def num(x, default=0.0):
    if x is None:
        return default
    if isinstance(x, Geom):
        return x.length() if x.is_curve else default
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return default
    return float(x)


def flatten(x):
    if isinstance(x, list):
        out = []
        for y in x:
            out += flatten(y)
        return out
    return [x]


def is_list(x):
    return isinstance(x, list)


def pt(p):
    return G.point(vec(p))


# ===================================================================== params
@node("Number Slider", "Params", outputs=("Value",), widget="slider", desc="Drag to change a number.")
def n_slider(_props=None, **kw):
    v = float(_props.get("value", 1.0))
    dec = int(_props.get("decimals", 2))
    return round(v, dec) if dec > 0 else int(round(v))


@node("Number", "Params", outputs=("Value",), widget="number", desc="A typed number.")
def n_number(_props=None, **kw):
    return float(_props.get("value", 0.0))


@node("Toggle", "Params", outputs=("Value",), widget="toggle", desc="True / False switch.")
def n_toggle(_props=None, **kw):
    return bool(_props.get("value", False))


@node("Text", "Params", outputs=("Text",), widget="text", desc="A text value.")
def n_text(_props=None, **kw):
    return str(_props.get("value", ""))


@node("Panel", "Params", [Port("Data", None, "list")], outputs=("Data",), widget="panel",
      desc="Shows whatever is connected to it and passes it through.")
def n_panel(Data=None, **kw):
    return Data


@node("Geometry", "Params", outputs=("Geometry",), widget="ref",
      desc="Reference objects from the Draft or Model workspace. Select objects, then click 'Set from selection'.")
def n_geometry(_refs=None, **kw):
    return list(_refs or [])


@node("Component", "Components", [Port("Location", (0, 0, 0)), Port("Rotation", 0.0), Port("Params", "", "item",
      "key=value, key=value")], outputs=("Instance",), widget="component",
      desc="Place a component (from the document or library) at points. Params like: width=500, shelves=4")
def n_component(Location=None, Rotation=0.0, Params="", _props=None, _defs=None, **kw):
    did = (_props or {}).get("def_id")
    if not did or did not in (_defs or {}):
        raise ValueError("Choose a component on the node")
    p = {}
    for part in str(Params or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip()
            v = v.strip()
            try:
                p[k] = float(v) if v.lower() not in ("true", "false") else v.lower() == "true"
            except ValueError:
                p[k] = v
    M = G.translation(vec(Location)) @ G.rotation(num(Rotation))
    return C.instance(did, M, params=p)


# ===================================================================== math
@node("Add", "Math", [Port("A", 0.0), Port("B", 0.0)])
def n_add(A=0, B=0, **kw):
    if isinstance(A, Geom) or isinstance(B, Geom) or isinstance(A, np.ndarray) or isinstance(B, np.ndarray):
        return pt(vec(A) + vec(B))
    return num(A) + num(B)


@node("Subtract", "Math", [Port("A", 0.0), Port("B", 0.0)])
def n_sub(A=0, B=0, **kw):
    if isinstance(A, (Geom, np.ndarray)) or isinstance(B, (Geom, np.ndarray)):
        return vec(A) - vec(B)
    return num(A) - num(B)


@node("Multiply", "Math", [Port("A", 1.0), Port("B", 1.0)])
def n_mul(A=1, B=1, **kw):
    if isinstance(A, np.ndarray):
        return A * num(B)
    return num(A) * num(B)


@node("Divide", "Math", [Port("A", 1.0), Port("B", 1.0)])
def n_div(A=1, B=1, **kw):
    b = num(B)
    if b == 0:
        raise ZeroDivisionError("Divide by zero")
    return num(A) / b


@node("Power", "Math", [Port("A", 2.0), Port("B", 2.0)])
def n_pow(A=2, B=2, **kw):
    return num(A) ** num(B)


@node("Expression", "Math", [Port("x", 0.0), Port("y", 0.0), Port("z", 0.0)], widget="expr",
      desc="Evaluate a math expression using x, y, z (e.g. sin(x) * y + 2).")
def n_expr(x=0, y=0, z=0, _props=None, **kw):
    ns = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    ns.update(x=num(x), y=num(y), z=num(z), abs=abs, min=min, max=max, round=round)
    return eval(str((_props or {}).get("expr", "x")), {"__builtins__": {}}, ns)


@node("Range", "Math", [Port("Start", 0.0), Port("End", 1.0), Port("Steps", 10)], outputs=("Numbers",),
      desc="Steps+1 evenly spaced numbers from Start to End.")
def n_range(Start=0, End=1, Steps=10, **kw):
    return [float(v) for v in np.linspace(num(Start), num(End), int(num(Steps, 10)) + 1)]


@node("Series", "Math", [Port("Start", 0.0), Port("Step", 1.0), Port("Count", 10)], outputs=("Numbers",))
def n_series(Start=0, Step=1, Count=10, **kw):
    return [num(Start) + i * num(Step) for i in range(int(num(Count, 10)))]


@node("Random", "Math", [Port("Count", 10), Port("Min", 0.0), Port("Max", 1.0), Port("Seed", 1)], outputs=("Numbers",))
def n_random(Count=10, Min=0, Max=1, Seed=1, **kw):
    r = random.Random(int(num(Seed)))
    return [r.uniform(num(Min), num(Max)) for _ in range(int(num(Count)))]


@node("Sin", "Math", [Port("Angle", 0.0, hint="degrees")])
def n_sin(Angle=0, **kw):
    return math.sin(math.radians(num(Angle)))


@node("Cos", "Math", [Port("Angle", 0.0, hint="degrees")])
def n_cos(Angle=0, **kw):
    return math.cos(math.radians(num(Angle)))


@node("Round", "Math", [Port("Value", 0.0), Port("Decimals", 0)])
def n_round(Value=0, Decimals=0, **kw):
    return round(num(Value), int(num(Decimals)))


@node("Sum", "Math", [Port("Numbers", None, "list")])
def n_sum(Numbers=None, **kw):
    return float(sum(num(x) for x in flatten(Numbers or [])))


@node("Average", "Math", [Port("Numbers", None, "list")])
def n_avg(Numbers=None, **kw):
    f = flatten(Numbers or [])
    return float(sum(num(x) for x in f) / len(f)) if f else 0.0


@node("Min Max", "Math", [Port("Numbers", None, "list")], outputs=("Min", "Max"))
def n_minmax(Numbers=None, **kw):
    f = [num(x) for x in flatten(Numbers or [])]
    return {"Min": min(f) if f else 0.0, "Max": max(f) if f else 0.0}


# ===================================================================== lists
@node("List Item", "Lists", [Port("List", None, "list"), Port("Index", 0)], outputs=("Item",))
def n_item(List=None, Index=0, **kw):
    L = List or []
    return L[int(num(Index)) % len(L)] if L else None


@node("List Length", "Lists", [Port("List", None, "list")], outputs=("Length",))
def n_len(List=None, **kw):
    return len(List or [])


@node("Reverse", "Lists", [Port("List", None, "list")], outputs=("List",))
def n_reverse(List=None, **kw):
    return list(reversed(List or []))


@node("Flatten", "Lists", [Port("List", None, "list")], outputs=("List",))
def n_flatten(List=None, **kw):
    return flatten(List or [])


@node("Merge", "Lists", [Port("A", None, "list"), Port("B", None, "list")], outputs=("List",))
def n_merge(A=None, B=None, **kw):
    return flatten(A if A is not None else []) + flatten(B if B is not None else [])


@node("Sort", "Lists", [Port("Numbers", None, "list")], outputs=("Sorted",))
def n_sort(Numbers=None, **kw):
    return sorted(flatten(Numbers or []), key=num)


@node("Cull Nth", "Lists", [Port("List", None, "list"), Port("N", 2)], outputs=("List",),
      desc="Remove every Nth item.")
def n_cull(List=None, N=2, **kw):
    n = max(1, int(num(N)))
    return [x for i, x in enumerate(List or []) if (i + 1) % n != 0]


# ===================================================================== points & vectors
@node("Point", "Vector", [Port("X", 0.0), Port("Y", 0.0), Port("Z", 0.0)], outputs=("Point",), desc="Construct a point.")
def n_point(X=0, Y=0, Z=0, **kw):
    return G.point((num(X), num(Y), num(Z)))


@node("Deconstruct", "Vector", [Port("Point")], outputs=("X", "Y", "Z"))
def n_decon(Point=None, **kw):
    v = vec(Point)
    return {"X": float(v[0]), "Y": float(v[1]), "Z": float(v[2])}


@node("Vector", "Vector", [Port("X", 0.0), Port("Y", 0.0), Port("Z", 1.0)], outputs=("Vector",))
def n_vector(X=0, Y=0, Z=1, **kw):
    return np.array([num(X), num(Y), num(Z)])


@node("Distance", "Vector", [Port("A"), Port("B")], outputs=("Distance",))
def n_distance(A=None, B=None, **kw):
    return float(np.linalg.norm(vec(A) - vec(B)))


@node("Grid Points", "Vector", [Port("Origin", (0, 0, 0)), Port("Count X", 5), Port("Count Y", 5),
      Port("Spacing X", 10.0), Port("Spacing Y", 10.0)], outputs=("Points",))
def n_grid(Origin=None, **kw):
    o = vec(Origin)
    nx, ny = int(num(kw.get("Count X"), 5)), int(num(kw.get("Count Y"), 5))
    sx, sy = num(kw.get("Spacing X"), 10), num(kw.get("Spacing Y"), 10)
    return [G.point(o + (i * sx, j * sy, 0)) for j in range(ny) for i in range(nx)]


# ===================================================================== curves
@node("Line", "Curve", [Port("Start", (0, 0, 0)), Port("End", (10, 0, 0))], outputs=("Line",))
def n_line(Start=None, End=None, **kw):
    return G.line(vec(Start), vec(End))


@node("Line SDL", "Curve", [Port("Start", (0, 0, 0)), Port("Direction", (1, 0, 0)), Port("Length", 10.0)],
      outputs=("Line",), desc="Line from start, direction and length.")
def n_line_sdl(Start=None, Direction=None, Length=10, **kw):
    d = vec(Direction) if Direction is not None else np.array([1.0, 0, 0])
    d = d / (np.linalg.norm(d) or 1)
    return G.line(vec(Start), vec(Start) + d * num(Length))


@node("Circle", "Curve", [Port("Center", (0, 0, 0)), Port("Radius", 5.0)], outputs=("Circle",))
def n_circle(Center=None, Radius=5, **kw):
    return G.circle(vec(Center), abs(num(Radius, 5)) or 1e-6)


@node("Arc 3Pt", "Curve", [Port("A"), Port("B"), Port("C")], outputs=("Arc",))
def n_arc3(A=None, B=None, C=None, **kw):
    return G.arc_3pt(vec(A), vec(B), vec(C))


@node("Rectangle", "Curve", [Port("Center", (0, 0, 0)), Port("Width", 10.0), Port("Height", 5.0)], outputs=("Rectangle",))
def n_rect(Center=None, Width=10, Height=5, **kw):
    c = vec(Center)
    w, h = num(Width) / 2, num(Height) / 2
    return G.rectangle(c - (w, h, 0), c + (w, h, 0))


@node("Polygon", "Curve", [Port("Center", (0, 0, 0)), Port("Radius", 5.0), Port("Sides", 6)], outputs=("Polygon",))
def n_polygon(Center=None, Radius=5, Sides=6, **kw):
    return G.polygon(vec(Center), num(Radius), max(3, int(num(Sides))))


@node("Polyline", "Curve", [Port("Points", None, "list"), Port("Closed", False)], outputs=("Polyline",))
def n_polyline(Points=None, Closed=False, **kw):
    pts = [vec(p) for p in flatten(Points or [])]
    if len(pts) < 2:
        raise ValueError("Need at least 2 points")
    return G.polyline(pts, closed=bool(Closed))


@node("Interpolate", "Curve", [Port("Points", None, "list"), Port("Degree", 3)], outputs=("Curve",),
      desc="Smooth curve through points.")
def n_interp(Points=None, Degree=3, **kw):
    pts = [vec(p) for p in flatten(Points or [])]
    return G.interp_curve(pts, int(num(Degree, 3)))


@node("Divide Curve", "Curve", [Port("Curve"), Port("Count", 10)], outputs=("Points",))
def n_divide(Curve=None, Count=10, **kw):
    if not isinstance(Curve, Geom) or not Curve.is_curve:
        raise ValueError("Input must be a curve")
    return [G.point(p) for p in Curve.divide(int(num(Count, 10)))]


@node("Curve Length", "Curve", [Port("Curve")], outputs=("Length",))
def n_clen(Curve=None, **kw):
    return Curve.length()


@node("Evaluate Curve", "Curve", [Port("Curve"), Port("t", 0.5, hint="0..1")], outputs=("Point",))
def n_eval(Curve=None, t=0.5, **kw):
    return G.point(Curve.point_at(num(t)))


@node("End Points", "Curve", [Port("Curve")], outputs=("Start", "End"))
def n_ends(Curve=None, **kw):
    s = Curve.sample()
    return {"Start": G.point(s[0]), "End": G.point(s[-1])}


@node("Offset", "Curve", [Port("Curve"), Port("Distance", 1.0)], outputs=("Curve",))
def n_offset(Curve=None, Distance=1, **kw):
    return G.offset_polyline_2d(Curve, num(Distance))


# ===================================================================== surfaces / solids
@node("Box", "Solid", [Port("Corner", (0, 0, 0)), Port("X", 10.0), Port("Y", 10.0), Port("Z", 10.0)], outputs=("Box",))
def n_box(Corner=None, X=10, Y=10, Z=10, **kw):
    return B.box(vec(Corner), num(X), num(Y), num(Z))


@node("Center Box", "Solid", [Port("Center", (0, 0, 0)), Port("X", 10.0), Port("Y", 10.0), Port("Z", 10.0)], outputs=("Box",))
def n_cbox(Center=None, X=10, Y=10, Z=10, **kw):
    c = vec(Center)
    return B.box(c - np.array([num(X), num(Y), num(Z)]) / 2, num(X), num(Y), num(Z))


@node("Sphere", "Solid", [Port("Center", (0, 0, 0)), Port("Radius", 5.0)], outputs=("Sphere",))
def n_sphere(Center=None, Radius=5, **kw):
    return B.sphere(vec(Center), abs(num(Radius)) or 1e-6)


@node("Cylinder", "Solid", [Port("Base", (0, 0, 0)), Port("Radius", 5.0), Port("Height", 10.0)], outputs=("Cylinder",))
def n_cyl(Base=None, Radius=5, Height=10, **kw):
    return B.cylinder(vec(Base), abs(num(Radius)) or 1e-6, num(Height))


@node("Extrude", "Surface", [Port("Curve"), Port("Height", 10.0, hint="number or vector"), Port("Cap", True)], outputs=("Extrusion",))
def n_extrude(Curve=None, Height=10, Cap=True, **kw):
    v = Height if isinstance(Height, np.ndarray) else (0, 0, num(Height))
    return B.extrude(Curve, v, cap=bool(Cap))


@node("Loft", "Surface", [Port("Curves", None, "list")], outputs=("Surface",))
def n_loft(Curves=None, **kw):
    return B.loft([c for c in flatten(Curves or []) if isinstance(c, Geom)])


@node("Revolve", "Surface", [Port("Curve"), Port("Axis Start", (0, 0, 0)), Port("Axis End", (0, 0, 1)), Port("Angle", 360.0)], outputs=("Surface",))
def n_revolve(Curve=None, Angle=360, **kw):
    a, b = vec(kw.get("Axis Start")), vec(kw.get("Axis End"))
    return B.revolve(Curve, a, b - a, num(Angle))


@node("Pipe", "Surface", [Port("Curve"), Port("Radius", 1.0)], outputs=("Pipe",))
def n_pipe(Curve=None, Radius=1, **kw):
    return B.pipe(Curve, abs(num(Radius)) or 1e-6)


@node("Planar Surface", "Surface", [Port("Curve")], outputs=("Surface",))
def n_planar(Curve=None, **kw):
    return B.planar_surface([Curve])


@node("Area", "Analysis", [Port("Geometry")], outputs=("Area", "Centroid"))
def n_area(Geometry=None, **kw):
    return {"Area": Geometry.area(), "Centroid": G.point(Geometry.centroid())}


@node("Volume", "Analysis", [Port("Geometry")], outputs=("Volume",))
def n_volume(Geometry=None, **kw):
    return Geometry.volume()


@node("Bounding Box", "Analysis", [Port("Geometry", None, "list")], outputs=("Box", "Min", "Max"))
def n_bbox(Geometry=None, **kw):
    pts = []
    for g in flatten(Geometry or []):
        if isinstance(g, Geom):
            lo, hi = g.bbox()
            pts += [lo, hi]
    a = np.array(pts) if pts else np.zeros((1, 3))
    lo, hi = a.min(0), a.max(0)
    d = np.maximum(hi - lo, 1e-9)
    return {"Box": B.box(lo, *d), "Min": G.point(lo), "Max": G.point(hi)}


# ===================================================================== NURBS solids
def _breps(x):
    return [g for g in flatten(x if x is not None else []) if isinstance(g, Geom) and g.kind == "brep"]


@node("Union", "Solid", [Port("A", None, "list"), Port("B", None, "list")], outputs=("Result",), desc="Exact boolean union.")
def n_union(A=None, B_=None, **kw):
    a = _breps(A) + _breps(kw.get("B"))
    if not a:
        raise ValueError("Needs NURBS solids")
    return B.boolean(a[:1], a[1:], "union")


@node("Difference", "Solid", [Port("A", None, "list"), Port("B", None, "list")], outputs=("Result",),
      desc="Exact boolean difference: A minus all of B.")
def n_difference(A=None, **kw):
    a, b = _breps(A), _breps(kw.get("B"))
    if not a:
        raise ValueError("A needs NURBS solids")
    return B.boolean(a, b, "difference") if b else (a[0] if len(a) == 1 else B.boolean(a[:1], a[1:], "union"))


@node("Intersection", "Solid", [Port("A", None, "list"), Port("B", None, "list")], outputs=("Result",),
      desc="Exact boolean intersection.")
def n_intersection(A=None, **kw):
    a, b = _breps(A), _breps(kw.get("B"))
    if not a or not b:
        raise ValueError("Needs NURBS solids on A and B")
    return B.boolean(a, b, "intersection")


@node("Fillet", "Solid", [Port("Solid"), Port("Radius", 1.0)], outputs=("Solid",), desc="Round all edges.")
def n_fillet(Solid=None, Radius=1, **kw):
    return B.fillet(Solid, abs(num(Radius)))


@node("Sweep", "Surface", [Port("Rail"), Port("Profile")], outputs=("Surface",), desc="Sweep a profile along a rail.")
def n_sweep(Rail=None, Profile=None, **kw):
    return B.sweep1(Rail, Profile)


@node("Intersect", "Curve", [Port("A"), Port("B")], outputs=("Curves",), desc="Exact intersection curves.")
def n_intersect(A=None, **kw):
    return B.intersect(A, kw.get("B"))


@node("Brep Edges", "Analysis", [Port("Brep")], outputs=("Edges",), desc="Exact edge curves of a surface or solid.")
def n_edges(Brep=None, **kw):
    return B.edge_curves(Brep)


@node("Brep Faces", "Analysis", [Port("Brep")], outputs=("Faces",), desc="Split a polysurface into its faces.")
def n_faces(Brep=None, **kw):
    return B.explode(Brep)


@node("Mesh", "Surface", [Port("Brep")], outputs=("Mesh",), desc="Mesh from a NURBS surface or solid.")
def n_mesh(Brep=None, **kw):
    return B.to_mesh(Brep)


# ===================================================================== transforms
def _xf(geo, M):
    if isinstance(geo, Geom):
        return geo.transformed(M, keep_id=False)
    if isinstance(geo, np.ndarray):
        return G.xform_pts(M, geo.reshape(1, 3))[0]
    raise ValueError("Input is not geometry")


@node("Move", "Transform", [Port("Geometry"), Port("Vector", (0, 0, 10))], outputs=("Geometry",))
def n_move(Geometry=None, Vector=None, **kw):
    return _xf(Geometry, G.translation(vec(Vector)))


@node("Rotate", "Transform", [Port("Geometry"), Port("Angle", 45.0, hint="degrees"), Port("Center", (0, 0, 0)),
      Port("Axis", (0, 0, 1))], outputs=("Geometry",))
def n_rotate(Geometry=None, Angle=45, Center=None, Axis=None, **kw):
    return _xf(Geometry, G.rotation(num(Angle), vec(Axis) if Axis is not None else (0, 0, 1), vec(Center)))


@node("Scale", "Transform", [Port("Geometry"), Port("Factor", 2.0), Port("Center", (0, 0, 0))], outputs=("Geometry",))
def n_scale(Geometry=None, Factor=2, Center=None, **kw):
    return _xf(Geometry, G.scaling(num(Factor), vec(Center)))


@node("Mirror", "Transform", [Port("Geometry"), Port("Origin", (0, 0, 0)), Port("Normal", (1, 0, 0))], outputs=("Geometry",))
def n_mirror(Geometry=None, Origin=None, Normal=None, **kw):
    return _xf(Geometry, G.mirror(vec(Origin), vec(Normal) if Normal is not None else (1, 0, 0)))


@node("Linear Array", "Transform", [Port("Geometry"), Port("Vector", (20, 0, 0)), Port("Count", 5)], outputs=("Geometry",))
def n_linarray(Geometry=None, Vector=None, Count=5, **kw):
    v = vec(Vector)
    return [_xf(Geometry, G.translation(v * i)) for i in range(int(num(Count)))]


@node("Polar Array", "Transform", [Port("Geometry"), Port("Center", (0, 0, 0)), Port("Count", 6), Port("Angle", 360.0)],
      outputs=("Geometry",))
def n_polararray(Geometry=None, Center=None, Count=6, Angle=360, **kw):
    n = max(1, int(num(Count)))
    step = num(Angle) / n if abs(num(Angle)) >= 360 else num(Angle) / max(n - 1, 1)
    return [_xf(Geometry, G.rotation(step * i, (0, 0, 1), vec(Center))) for i in range(n)]


# ===================================================================== util
@node("Delay", "Util", [Port("Data", None, "list"), Port("Seconds", 3.0)], outputs=("Data",),
      desc="Waits, then passes data through. Use it to try the Cancel dialog.")
def n_delay(Data=None, Seconds=3.0, _progress=None, **kw):
    s = num(Seconds)
    steps = max(1, int(s * 10))
    for i in range(steps):
        time.sleep(s / steps)
        if _progress:
            _progress((i + 1) / steps)
    return Data


@node("Python", "Script", [Port("x"), Port("y")], outputs=("a", "out"), widget="code", dynamic=True,
      desc="Write Python. Inputs are variables; assign outputs (e.g. a = x * 2). print() goes to 'out'.")
def n_python(_props=None, _progress=None, _defs=None, **inputs):
    code = (_props or {}).get("code", "a = x")
    ns = C.make_namespace()
    ns.update({"G": G, "progress": _progress or (lambda f: None), "Geom": Geom})
    ns.update(inputs)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, "<python node>", "exec"), ns)
    outs = (_props or {}).get("outputs", ["a"])
    res = {o: ns.get(o) for o in outs}
    res["out"] = buf.getvalue()
    return res


DEFAULT_PY_CODE = """# Inputs are available as variables (x, y). Assign to outputs (a).
# Everything from the geometry library is available: box, circle, extrude, loft, ...
# Call progress(0..1) in long loops so the compute dialog shows progress.
pts = []
n = int(x or 20)
for i in range(n):
    t = i / n * 6.283
    pts.append(point((math.cos(t) * 10, math.sin(t) * 10, i * (y or 1))))
    progress(i / n)
a = interp_curve(pts)
print("built", n, "points")
"""


def categories():
    cats = {}
    for t in NODE_TYPES.values():
        cats.setdefault(t.category, []).append(t.name)
    return cats
from . import analysis  # noqa: E402,F401  (environment and space syntax nodes)
