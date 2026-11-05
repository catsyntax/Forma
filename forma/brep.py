"""Exact NURBS boundary-representation kernel for Forma, built on OpenCascade (via OCP).

Solids and surfaces are stored as OpenCascade shapes inside Geoms of kind "brep". Curves stay
Forma NURBS and are converted exactly when they meet a brep operation. Shapes are tessellated
only for display (cached), so modeling stays exact: booleans, fillets, trims, STEP/IGES.
"""
from __future__ import annotations

import base64
import io
import math

import numpy as np

from . import geometry as G
from .geometry import Geom, P

from OCP.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Ax1, gp_Ax2, gp_Pln, gp_Circ, gp_Trsf, gp_GTrsf, gp_Mat, gp_XYZ
from OCP.TopoDS import TopoDS, TopoDS_Shape, TopoDS_Compound
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_SOLID, TopAbs_SHELL, TopAbs_REVERSED, TopAbs_WIRE
from OCP.TopExp import TopExp_Explorer, TopExp
from OCP.TopLoc import TopLoc_Location
from OCP.BRep import BRep_Tool, BRep_Builder
from OCP.BRepTools import BRepTools
from OCP.BinTools import BinTools
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Line, GeomAbs_Circle, GeomAbs_Plane
from OCP.BRepPrimAPI import (BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeCylinder,
                             BRepPrimAPI_MakeCone, BRepPrimAPI_MakeTorus, BRepPrimAPI_MakePrism, BRepPrimAPI_MakeRevol)
from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace,
                                BRepBuilderAPI_MakePolygon, BRepBuilderAPI_Transform, BRepBuilderAPI_GTransform,
                                BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid, BRepBuilderAPI_NurbsConvert)
from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections, BRepOffsetAPI_MakePipe, BRepOffsetAPI_MakePipeShell
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut, BRepAlgoAPI_Common, BRepAlgoAPI_Section
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet, BRepFilletAPI_MakeChamfer
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.Geom import Geom_BSplineCurve, Geom_TrimmedCurve
from OCP.GeomConvert import GeomConvert
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.ShapeFix import ShapeFix_Shape
from OCP.BRepLib import BRepLib_ToolTriangulatedShape
from OCP.BRepCheck import BRepCheck_Analyzer

try:  # OCP >= 7.9 / 8.x
    from OCP.collections import (Array1_gp_Pnt as _APnt, Array1_double as _ADbl, Array1_int as _AInt)
except ImportError:  # older OCP
    from OCP.TColgp import TColgp_Array1OfPnt as _APnt
    from OCP.TColStd import TColStd_Array1OfReal as _ADbl, TColStd_Array1OfInteger as _AInt


def _caster(name):
    return getattr(TopoDS, name + "_s", None) or getattr(TopoDS, name)


_face, _edge, _wire, _shell, _vertex = (_caster(n) for n in ("Face", "Edge", "Wire", "Shell", "Vertex"))


# ======================================================================= storage
def make(shape, source="Polysurface", **kw) -> Geom:
    return Geom("brep", _shape=shape, source=source, **kw)


def shape_of(g: Geom):
    s = g.data.get("_shape")
    if s is None:
        s = deserialize(g.data["brep"])
        g.data["_shape"] = s
    return s


def serialize(shape) -> str:
    bio = io.BytesIO()
    BinTools.Write_s(shape, bio)
    return base64.b64encode(bio.getvalue()).decode("ascii")


def deserialize(text: str):
    s = TopoDS_Shape()
    BinTools.Read_s(s, io.BytesIO(base64.b64decode(text)))
    return s


def freeze(g: Geom):
    """Make sure the brep has its serialised form (for saving / pickling)."""
    if "brep" not in g.data:
        g.data["brep"] = serialize(shape_of(g))


def _pnt(p):
    p = P(p)
    return gp_Pnt(float(p[0]), float(p[1]), float(p[2]))


def _dir(v):
    v = P(v)
    return gp_Dir(float(v[0]), float(v[1]), float(v[2]))


def _vec(v):
    v = P(v)
    return gp_Vec(float(v[0]), float(v[1]), float(v[2]))


def _xyz(p):
    return np.array([p.X(), p.Y(), p.Z()])


# ======================================================================= topology helpers
def _iter(shape, kind):
    ex = TopExp_Explorer(shape, kind)
    while ex.More():
        yield ex.Current()
        ex.Next()


def faces(shape):
    return [_face(f) for f in _iter(shape, TopAbs_FACE)]


def _unique(shape, kind):
    seen = {}
    for s in _iter(shape, kind):
        seen.setdefault(hash(s), s)
    return list(seen.values())


def unique_edges(shape):
    return [_edge(e) for e in _unique(shape, TopAbs_EDGE)]


def unique_vertices(shape):
    return [_vertex(v) for v in _unique(shape, TopAbs_VERTEX)]


def n_faces(shape):
    return sum(1 for _ in _iter(shape, TopAbs_FACE))


def is_solid(shape):
    return any(True for _ in _iter(shape, TopAbs_SOLID))


def _compound(shapes):
    b = BRep_Builder()
    c = TopoDS_Compound()
    b.MakeCompound(c)
    for s in shapes:
        b.Add(c, s)
    return c


def _unify(shape):
    try:
        u = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
        u.Build()
        return u.Shape()
    except Exception:
        return shape


# ======================================================================= curves
def exact_nurbs(g: Geom) -> Geom:
    """Any Forma curve as an exact NURBS Geom (circles/ellipses become rational quadratics)."""
    if g.kind == "nurbs":
        return g
    if g.kind == "polyline":
        pts = list(g.pts) + ([g.pts[0]] if g.closed else [])
        return G.nurbs_curve(pts, degree=1)
    if g.kind == "ellipse":
        span = g.a1 - g.a0
        nseg = max(1, int(math.ceil(abs(span) / (math.pi / 2) - 1e-9)))
        d = span / nseg
        w_mid = math.cos(d / 2)
        cps, ws = [], []
        for k in range(nseg):
            a = g.a0 + k * d
            b = a + d
            m = (a + b) / 2
            p0 = g.c + math.cos(a) * g.u + math.sin(a) * g.v
            p1 = g.c + (math.cos(m) * g.u + math.sin(m) * g.v) / w_mid
            if k == 0:
                cps.append(p0)
                ws.append(1.0)
            cps.append(p1)
            ws.append(w_mid)
            cps.append(g.c + math.cos(b) * g.u + math.sin(b) * g.v)
            ws.append(1.0)
        knots = [0, 0, 0]
        for k in range(1, nseg):
            knots += [k / nseg, k / nseg]
        knots += [1, 1, 1]
        return Geom("nurbs", cps=np.array(cps), w=np.array(ws), knots=np.array(knots, float), deg=2,
                    layer=g.layer, space=g.space)
    raise ValueError(f"Not a curve: {g.kind}")


def occ_bspline(n: Geom):
    U = np.asarray(n.knots, float)
    uniq, mult = [], []
    for u in U:
        if uniq and abs(u - uniq[-1]) < 1e-12:
            mult[-1] += 1
        else:
            uniq.append(u)
            mult.append(1)
    poles = _APnt(1, len(n.cps))
    w = _ADbl(1, len(n.cps))
    for i, (p, wi) in enumerate(zip(n.cps, n.w)):
        poles.SetValue(i + 1, _pnt(p))
        w.SetValue(i + 1, float(wi))
    kn = _ADbl(1, len(uniq))
    mu = _AInt(1, len(uniq))
    for i, (u, m) in enumerate(zip(uniq, mult)):
        kn.SetValue(i + 1, float(u))
        mu.SetValue(i + 1, int(m))
    return Geom_BSplineCurve(poles, w, kn, mu, int(n.deg))


def curve_edges(g: Geom):
    """OCC edges for a Forma curve (polylines keep one edge per segment for crisp corners)."""
    if g.kind == "polyline":
        pts = list(g.pts) + ([g.pts[0]] if g.closed else [])
        return [BRepBuilderAPI_MakeEdge(_pnt(a), _pnt(b)).Edge() for a, b in zip(pts[:-1], pts[1:])
                if np.linalg.norm(np.asarray(a) - np.asarray(b)) > 1e-12]
    if g.kind == "ellipse":
        ru, rv = np.linalg.norm(g.u), np.linalg.norm(g.v)
        if abs(ru - rv) < 1e-9 * max(ru, 1) and abs(np.dot(g.u, g.v)) < 1e-9 * max(ru * rv, 1):
            n = np.cross(g.u, g.v)
            ax = gp_Ax2(_pnt(g.c), _dir(n), _dir(g.u))
            circ = gp_Circ(ax, float(ru))
            if g.closed:
                return [BRepBuilderAPI_MakeEdge(circ).Edge()]
            return [BRepBuilderAPI_MakeEdge(circ, float(g.a0), float(g.a1)).Edge()]
    n = exact_nurbs(g)
    return [BRepBuilderAPI_MakeEdge(occ_bspline(n)).Edge()]


def wire(g: Geom):
    if g.kind == "brep":
        ws = [_wire(w) for w in _iter(shape_of(g), TopAbs_WIRE)]
        if ws:
            return ws[0]
    mw = BRepBuilderAPI_MakeWire()
    for e in curve_edges(g):
        mw.Add(e)
    if not mw.IsDone():
        raise ValueError("Could not make a wire from that curve")
    return mw.Wire()


def edge_to_nurbs(edge, **kw) -> Geom:
    """Exact Forma curve from an OCC edge (lines and circles stay simple, the rest become NURBS)."""
    c = BRepAdaptor_Curve(edge)
    t0, t1 = c.FirstParameter(), c.LastParameter()
    typ = c.GetType()
    if typ == GeomAbs_Line:
        return G.line(_xyz(c.Value(t0)), _xyz(c.Value(t1)), **kw)
    if typ == GeomAbs_Circle:
        circ = c.Circle()
        ax = circ.Position()
        X, Y = _xyz(ax.XDirection()), _xyz(ax.YDirection())
        r = circ.Radius()
        return Geom("ellipse", c=_xyz(ax.Location()), u=X * r, v=Y * r, a0=float(t0), a1=float(t1), **kw)
    gc = c.Curve().Curve()
    bs = GeomConvert.CurveToBSplineCurve_s(Geom_TrimmedCurve(gc, t0, t1))
    bs.Transform(c.Trsf())
    n = bs.NbPoles()
    cps = np.array([_xyz(bs.Pole(i)) for i in range(1, n + 1)])
    w = np.array([bs.Weight(i) for i in range(1, n + 1)])
    U = []
    for i in range(1, bs.NbKnots() + 1):
        U += [bs.Knot(i)] * bs.Multiplicity(i)
    U = np.array(U)
    U = (U - U[0]) / ((U[-1] - U[0]) or 1)
    return Geom("nurbs", cps=cps, w=w, knots=U, deg=bs.Degree(), **kw)


def edge_curves(g: Geom, **kw):
    sh = shape_of(g)
    seams = _seam_edges(sh)
    out = []
    for e in unique_edges(sh):
        if BRep_Tool.Degenerated_s(e) or _key(e) in seams:
            continue
        try:
            out.append(edge_to_nurbs(e, **kw))
        except Exception:
            pass
    return out


def _key(shape):
    return hash(shape)


def _seam_edges(sh):
    seams = set()
    for f in faces(sh):
        for e in _iter(f, TopAbs_EDGE):
            e = _edge(e)
            if BRep_Tool.IsClosed_s(e, f):
                seams.add(_key(e))
    return seams


# ======================================================================= primitives
def box(corner, dx, dy, dz, **kw):
    c = P(corner)
    lo = np.minimum(c, c + (dx, dy, dz))
    hi = np.maximum(c, c + (dx, dy, dz))
    hi = np.where(hi - lo < 1e-9, lo + 1e-6, hi)
    return make(BRepPrimAPI_MakeBox(_pnt(lo), _pnt(hi)).Shape(), "Box", **kw)


def sphere(center, radius, nu=None, nv=None, **kw):
    return make(BRepPrimAPI_MakeSphere(_pnt(center), float(abs(radius)) or 1e-6).Shape(), "Sphere", **kw)


def cylinder(base, radius, height, n=None, top_radius=None, axis=(0, 0, 1), **kw):
    if top_radius is not None and abs(top_radius - radius) > 1e-12:
        return cone(base, radius, height, top_radius=top_radius, axis=axis, **kw)
    h = float(height)
    d = P(axis) * (1 if h >= 0 else -1)
    ax = gp_Ax2(_pnt(base), _dir(d))
    return make(BRepPrimAPI_MakeCylinder(ax, float(abs(radius)) or 1e-6, abs(h) or 1e-6).Shape(), "Cylinder", **kw)


def cone(base, radius, height, top_radius=0.0, axis=(0, 0, 1), n=None, **kw):
    h = float(height)
    d = P(axis) * (1 if h >= 0 else -1)
    ax = gp_Ax2(_pnt(base), _dir(d))
    return make(BRepPrimAPI_MakeCone(ax, float(abs(radius)) or 1e-6, float(top_radius), abs(h) or 1e-6).Shape(), "Cone", **kw)


def torus(center, R, r, nu=None, nv=None, **kw):
    ax = gp_Ax2(_pnt(center), _dir((0, 0, 1)))
    return make(BRepPrimAPI_MakeTorus(ax, float(R), float(r)).Shape(), "Torus", **kw)


def plane_surface(a, b, **kw):
    a, b = P(a), P(b)
    z = a[2]
    pts = [(a[0], a[1], z), (b[0], a[1], z), (b[0], b[1], z), (a[0], b[1], z)]
    return planar_surface([G.polyline(pts, closed=True)], **kw)


def planar_surface(curves, **kw):
    """Planar face(s) from closed curves. Curves inside others become holes."""
    if isinstance(curves, Geom):
        curves = [curves]
    curves = [c for c in curves if c.is_curve and c.closed]
    if not curves:
        raise ValueError("Need closed planar curves")
    curves = sorted(curves, key=lambda c: -c.area())
    outer = curves[0]
    mf = BRepBuilderAPI_MakeFace(wire(outer), True)
    if not mf.IsDone():
        raise ValueError("Curve is not planar")
    for c in curves[1:]:
        w = wire(c)
        w.Reverse()
        mf.Add(_wire(w))
    fix = ShapeFix_Shape(mf.Face())
    fix.Perform()
    return make(fix.Shape(), "Planar surface", **kw)


# ======================================================================= surface ops
def extrude(obj: Geom, vector, cap=True, **kw):
    v = _vec(vector)
    if obj.kind == "brep":
        return make(BRepPrimAPI_MakePrism(shape_of(obj), v).Shape(), "Extrusion", **kw)
    if obj.kind == "mesh":
        raise ValueError("Extruding meshes is not supported; convert to a surface first")
    w = wire(obj)
    if cap and obj.closed:
        try:
            f = BRepBuilderAPI_MakeFace(w, True)
            if f.IsDone():
                return make(BRepPrimAPI_MakePrism(f.Face(), v).Shape(), "Extrusion", **kw)
        except Exception:
            pass
    return make(BRepPrimAPI_MakePrism(w, v).Shape(), "Extrusion", **kw)


def loft(curves, solid=None, ruled=False, samples=None, closed=None, **kw):
    curves = [c for c in curves if c.is_curve]
    if len(curves) < 2:
        raise ValueError("Loft needs at least 2 curves")
    closed = all(c.closed for c in curves)
    ts = BRepOffsetAPI_ThruSections(bool(closed if solid is None else solid), bool(ruled), 1e-6)
    for c in curves:
        ts.AddWire(wire(c))
    ts.CheckCompatibility(True)
    ts.Build()
    if not ts.IsDone():
        raise ValueError("Loft failed")
    return make(ts.Shape(), "Loft", **kw)


def revolve(obj: Geom, axis_pt=(0, 0, 0), axis_dir=(0, 0, 1), angle_deg=360.0, steps=None, **kw):
    sh = shape_of(obj) if obj.kind == "brep" else wire(obj)
    ax = gp_Ax1(_pnt(axis_pt), _dir(axis_dir))
    ang = math.radians(angle_deg)
    if abs(abs(angle_deg) - 360) < 1e-9:
        mk = BRepPrimAPI_MakeRevol(sh, ax, True)
    else:
        mk = BRepPrimAPI_MakeRevol(sh, ax, ang, True)
    return make(mk.Shape(), "Revolve", **kw)


def pipe(curve: Geom, radius, cap=True, sides=None, **kw):
    spine = wire(curve)
    s = curve.sample()
    t = s[1] - s[0]
    t /= np.linalg.norm(t) or 1
    circ = gp_Circ(gp_Ax2(_pnt(s[0]), _dir(t)), float(radius))
    prof = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(circ).Edge()).Wire()
    ps = BRepOffsetAPI_MakePipeShell(spine)
    ps.SetMode(False)  # corrected Frenet
    ps.Add(prof)
    ps.Build()
    if cap:
        ps.MakeSolid()
    return make(ps.Shape(), "Pipe", **kw)


def sweep1(rail: Geom, profile: Geom, solid=True, **kw):
    ps = BRepOffsetAPI_MakePipeShell(wire(rail))
    ps.SetMode(False)
    ps.Add(wire(profile), True, True)
    ps.Build()
    if not ps.IsDone():
        raise ValueError("Sweep failed")
    if solid and profile.closed:
        ps.MakeSolid()
    return make(ps.Shape(), "Sweep", **kw)


def boolean(a_list, b_list, op, **kw):
    A = [shape_of(g) for g in a_list]
    B = [shape_of(g) for g in b_list]
    cur = A[0]
    for s in A[1:]:
        f = BRepAlgoAPI_Fuse(cur, s)
        f.Build()
        cur = f.Shape()
    cls = {"union": BRepAlgoAPI_Fuse, "difference": BRepAlgoAPI_Cut, "intersection": BRepAlgoAPI_Common}[op]
    for s in B:
        o = cls(cur, s)
        o.Build()
        if not o.IsDone():
            raise ValueError(f"Boolean {op} failed")
        cur = o.Shape()
    return make(_unify(cur), f"Boolean {op}", **kw)


def _round_edges(g, size, maker, label, **kw):
    """Fillet/chamfer every edge. If OCC can't do all at once, keep every edge that works."""
    sh = shape_of(g)
    edges = [e for e in unique_edges(sh) if not BRep_Tool.Degenerated_s(e)]

    def attempt(es):
        mk = maker(sh)
        for e in es:
            mk.Add(float(size), e)
        mk.Build()
        if not mk.IsDone():
            return None
        out = mk.Shape()
        return out if BRepCheck_Analyzer(out).IsValid() else None

    res = attempt(edges)
    skipped = 0
    if res is None:
        ok = []
        for e in edges:
            if attempt(ok + [e]) is not None:
                ok.append(e)
        skipped = len(edges) - len(ok)
        res = attempt(ok) if ok else None
        if res is None:
            raise ValueError(f"{label} failed (size too large for these edges?)")
    out = make(res, f"{label}ed " + g.data.get("source", "solid").lower(), **kw)
    out.data["skipped_edges"] = skipped
    return out


def fillet(g: Geom, radius, **kw):
    return _round_edges(g, radius, BRepFilletAPI_MakeFillet, "Fillet", **kw)


def chamfer(g: Geom, dist, **kw):
    return _round_edges(g, dist, BRepFilletAPI_MakeChamfer, "Chamfer", **kw)


def intersect(a: Geom, b: Geom):
    def shp(g):
        return shape_of(g) if g.kind == "brep" else wire(g)
    s = BRepAlgoAPI_Section(shp(a), shp(b))
    s.Build()
    return [edge_to_nurbs(_edge(e)) for e in _iter(s.Shape(), TopAbs_EDGE)]


def join(gs, tol=1e-4, source="Polysurface", **kw):
    sew = BRepBuilderAPI_Sewing(tol)
    for g in gs:
        sew.Add(shape_of(g))
    sew.Perform()
    sh = sew.SewedShape()
    try:
        shells = [_shell(s) for s in _iter(sh, TopAbs_SHELL)]
        if len(shells) == 1 and BRepCheck_Analyzer(shells[0]).IsValid() and BRep_Tool.IsClosed_s(shells[0]):
            ms = BRepBuilderAPI_MakeSolid(shells[0])
            sol = ms.Solid()
            sh = sol
    except Exception:
        pass
    return make(sh, source, **kw)


def explode(g: Geom):
    return [make(f, "Surface", layer=g.layer, space=g.space) for f in faces(shape_of(g))]


def cap(g: Geom, **kw):
    """Close planar openings of an open polysurface and make a solid."""
    from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
    sh = shape_of(g)
    fb = ShapeAnalysis_FreeBounds(sh, 1e-4)
    closed = fb.GetClosedWires()
    caps = []
    for w in _iter(closed, TopAbs_WIRE):
        mf = BRepBuilderAPI_MakeFace(_wire(w), True)
        if mf.IsDone():
            caps.append(make(mf.Face(), "Cap"))
    if not caps:
        raise ValueError("No planar openings found")
    return join([g] + caps, **kw)


def transform(g: Geom, M: np.ndarray, keep_id=True) -> Geom:
    """Transform a brep. Rigid moves share geometry (OCC locations) and reuse the cached
    tessellation, so moving/copying/instancing NURBS solids costs almost nothing."""
    R = M[:3, :3]
    if not np.isfinite(M).all():
        raise ValueError("Invalid transformation")
    sv = np.linalg.svd(R, compute_uv=False)
    if sv[-1] < 1e-4 * max(sv[0], 1e-12) or sv[-1] < 1e-9 or sv[0] > 1e6:
        # flattening to (almost) zero thickness or absurd stretching crashes the geometry kernel
        raise ValueError("Scale factor too close to zero (or too large)")
    sh = shape_of(g)
    RRt = R @ R.T
    s2 = RRt[0, 0]
    similar = np.allclose(RRt, np.eye(3) * s2, atol=1e-9 * max(1.0, s2))
    det = np.linalg.det(R)
    out = g.copy(keep_id=keep_id)
    for k in ("brep", "_tess", "_props", "_grid", "_poly"):
        out.data.pop(k, None)
    if similar:
        t = gp_Trsf()
        t.SetValues(*[float(x) for x in np.hstack([R, M[:3, 3:4]]).ravel()])
        rigid = abs(s2 - 1.0) < 1e-9 and det > 0
        if rigid:
            new = sh.Moved(TopLoc_Location(t))
        else:
            new = BRepBuilderAPI_Transform(sh, t, True).Shape()
        tess = g.data.get("_tess")
        if tess is not None:
            V, N, T, E = tess
            tr = M[:3, 3]
            V2 = V @ R.T + tr
            N2 = N @ R.T
            L = np.linalg.norm(N2, axis=1, keepdims=True)
            N2 = N2 / np.where(L > 0, L, 1)
            T2 = T[:, ::-1] if det < 0 else T
            out.data["_tess"] = (V2, N2, T2, [e @ R.T + tr for e in E])
    else:
        # Bake any placement (OCC location) into the geometry first. GTransform does not carry the
        # location through to the face triangulation, which made scaled objects draw shifted.
        if not sh.Location().IsIdentity():
            trsf = sh.Location().Transformation()
            sh = BRepBuilderAPI_Transform(sh.Located(TopLoc_Location()), trsf, True).Shape()
        new = None
        # Flat-faced solids: move their corners and rebuild (exact, and avoids the kernel's general
        # non-uniform transform, which can crash on some faces).
        try:
            from . import editpoints as EP
            poly = EP.polyhedron(g)
            if poly is not None:
                Vn = G.xform_pts(M, poly["V"])
                new = EP._polyhedron_shape(poly, Vn)
        except Exception:
            new = None
        if new is None:
            from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
            from OCP.BRepLib import BRepLib
            sh = BRepBuilderAPI_Copy(sh, True, False).Shape()
            # NURBS conversion inside GTransform needs every edge to have its 2D curve on each face;
            # flat faces don't always store one, which crashed on macOS.
            for f in faces(sh):
                if BRepAdaptor_Surface(f).GetType() == GeomAbs_Plane:
                    for e in _iter(f, TopAbs_EDGE):
                        try:
                            BRepLib.BuildPCurveForEdgeOnPlane_s(_edge(e), f)
                        except Exception:
                            pass
            try:
                fix = ShapeFix_Shape(sh)
                fix.Perform()
                sh = fix.Shape()
            except Exception:
                pass
            gt = gp_GTrsf()
            gt.SetVectorialPart(gp_Mat(*[float(x) for x in R.ravel()]))
            gt.SetTranslationPart(gp_XYZ(*[float(x) for x in M[:3, 3]]))
            new = BRepBuilderAPI_GTransform(sh, gt, True).Shape()
    out.data["_shape"] = new
    return out


# ======================================================================= analysis
def bbox(g: Geom):
    b = Bnd_Box()
    try:
        BRepBndLib.AddOptimal_s(shape_of(g), b, False, False)
    except Exception:
        BRepBndLib.Add_s(shape_of(g), b, False)
    if b.IsVoid():
        return np.zeros(3), np.zeros(3)
    return _xyz(b.CornerMin()), _xyz(b.CornerMax())


def _props(g):
    cache = g.data.get("_props")
    if cache is None:
        sh = shape_of(g)
        sp = GProp_GProps()
        try:
            BRepGProp.SurfaceProperties_s(sh, sp, 1e-6)
        except Exception:
            BRepGProp.SurfaceProperties_s(sh, sp)
        vp = GProp_GProps()
        vol = 0.0
        cen = _xyz(sp.CentreOfMass()) if sp.Mass() > 0 else np.zeros(3)
        if is_solid(sh):
            try:
                BRepGProp.VolumeProperties_s(sh, vp, 1e-6)
            except Exception:
                BRepGProp.VolumeProperties_s(sh, vp)
            vol = abs(vp.Mass())
            if vol > 0:
                cen = _xyz(vp.CentreOfMass())
        cache = (sp.Mass(), vol, cen)
        g.data["_props"] = cache
    return cache


def area(g):
    return _props(g)[0]


def volume(g):
    return _props(g)[1]


def centroid(g):
    return _props(g)[2]


def describe(g: Geom):
    sh = shape_of(g)
    nf = n_faces(sh)
    kind = "Closed solid" if is_solid(sh) else ("Surface" if nf == 1 else "Polysurface")
    src = g.data.get("source", "")
    s = f"{kind} ({src}), {nf} face{'s' if nf != 1 else ''}, area {area(g):.4g}"
    if is_solid(sh):
        s += f", volume {volume(g):.4g}"
    return s


def snap_points(g: Geom):
    sh = shape_of(g)
    out = [(_xyz(BRep_Tool.Pnt_s(v)), "End") for v in unique_vertices(sh)][:400]
    for e in unique_edges(sh)[:400]:
        if BRep_Tool.Degenerated_s(e):
            continue
        c = BRepAdaptor_Curve(e)
        t0, t1 = c.FirstParameter(), c.LastParameter()
        out.append((_xyz(c.Value((t0 + t1) / 2)), "Mid"))
        if c.GetType() == GeomAbs_Circle:
            out.append((_xyz(c.Circle().Location()), "Cen"))
    return out


# ======================================================================= tessellation
def tessellate(g: Geom):
    """(V, normals, triangles, edge polylines) for display. Cached on the geom."""
    t = g.data.get("_tess")
    if t is not None:
        return t
    sh = shape_of(g)
    lo, hi = bbox(g)
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    BRepMesh_IncrementalMesh(sh, max(diag * 0.0012, 1e-7), False, 0.22, True)
    Vs, Ns, Ts = [], [], []
    off = 0
    for f in faces(sh):
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(f, loc)
        if tri is None:
            continue
        trsf = loc.Transformation()
        nn = tri.NbNodes()
        try:
            if not tri.HasNormals():
                BRepLib_ToolTriangulatedShape.ComputeNormals_s(f, tri)
            has_n = tri.HasNormals()
        except Exception:
            has_n = False
        V = np.empty((nn, 3))
        N = np.zeros((nn, 3))
        for i in range(1, nn + 1):
            p = tri.Node(i).Transformed(trsf)
            V[i - 1] = (p.X(), p.Y(), p.Z())
            if has_n:
                d = tri.Normal(i)
                d = d.Transformed(trsf) if hasattr(d, "Transformed") else d
                N[i - 1] = (d.X(), d.Y(), d.Z())
        reversed_ = f.Orientation() == TopAbs_REVERSED
        T = np.empty((tri.NbTriangles(), 3), int)
        for i in range(1, tri.NbTriangles() + 1):
            a, b, c = tri.Triangle(i).Get()
            T[i - 1] = (a - 1, c - 1, b - 1) if reversed_ else (a - 1, b - 1, c - 1)
        if reversed_:
            N = -N
        if not has_n or not np.isfinite(N).all() or np.abs(N).sum() == 0:
            N = _vertex_normals(V, T)
        Vs.append(V)
        Ns.append(N)
        Ts.append(T + off)
        off += nn
    V = np.vstack(Vs) if Vs else np.zeros((0, 3))
    N = np.vstack(Ns) if Ns else np.zeros((0, 3))
    T = np.vstack(Ts) if Ts else np.zeros((0, 3), int)
    edges = []
    seams = _seam_edges(sh)
    for e in unique_edges(sh):
        if BRep_Tool.Degenerated_s(e) or _key(e) in seams:
            continue
        c = BRepAdaptor_Curve(e)
        t0, t1 = c.FirstParameter(), c.LastParameter()
        n = 2 if c.GetType() == GeomAbs_Line else 64
        edges.append(np.array([_xyz(c.Value(t)) for t in np.linspace(t0, t1, n)]))
    t = (V, N, T, edges)
    # A stale triangulation (left over from before a non-uniform scale) can sit in the wrong place.
    # If the mesh does not match the exact bounding box, throw it away and mesh again once.
    if len(V) and not g.data.get("_remeshed"):
        if np.abs(V.min(0) - lo).max() > diag * 0.02 or np.abs(V.max(0) - hi).max() > diag * 0.02:
            BRepTools.Clean_s(sh)
            g.data["_remeshed"] = True
            return tessellate(g)
    g.data.pop("_remeshed", None)
    g.data["_tess"] = t
    return t


def _vertex_normals(V, T):
    N = np.zeros_like(V)
    if len(T):
        fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
        for k in range(3):
            np.add.at(N, T[:, k], fn)
    L = np.linalg.norm(N, axis=1, keepdims=True)
    return N / np.where(L > 0, L, 1)


def to_mesh(g: Geom, **kw) -> Geom:
    V, N, T, _ = tessellate(g)
    return G.mesh(V, T, source="Mesh from " + g.data.get("source", "surface").lower(),
                  layer=kw.get("layer", g.layer), space=kw.get("space", g.space))


# ======================================================================= exchange
def export_step(geoms, path):
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    w = STEPControl_Writer()
    for g in geoms:
        w.Transfer(shape_of(g), STEPControl_AsIs)
    w.Write(path)


def export_iges(geoms, path):
    from OCP.IGESControl import IGESControl_Writer
    w = IGESControl_Writer()
    for g in geoms:
        w.AddShape(shape_of(g))
    w.ComputeModel()
    w.Write(path)


def import_step_iges(path):
    if path.lower().endswith((".igs", ".iges")):
        from OCP.IGESControl import IGESControl_Reader as R
    else:
        from OCP.STEPControl import STEPControl_Reader as R
    r = R()
    if r.ReadFile(path) != 1:
        raise IOError("Could not read " + path)
    r.TransferRoots()
    sh = r.OneShape()
    out = []
    solids = list(_iter(sh, TopAbs_SOLID))
    if solids:
        out = [make(s, "Imported solid") for s in solids]
    else:
        shells = list(_iter(sh, TopAbs_SHELL))
        out = [make(s, "Imported polysurface") for s in shells] or [make(sh, "Imported")]
    return out


def bspline_surface_of_face(face):
    """NURBS surface of a face, trimmed to the face's parameter box, plus a natural face built on it.

    (BRepBuilderAPI_NurbsConvert can crash on faces that already carry a display mesh, so the
    underlying surface is converted directly instead.)
    """
    from OCP.Geom import Geom_RectangularTrimmedSurface
    u0, u1, v0, v1 = BRepTools.UVBounds_s(face)
    srf = BRep_Tool.Surface_s(face)
    bs = GeomConvert.SurfaceToBSplineSurface_s(Geom_RectangularTrimmedSurface(srf, u0, u1, v0, v1))
    f2 = BRepBuilderAPI_MakeFace(bs, 1e-7).Face()
    return bs, f2


def _face_area(f):
    sp = GProp_GProps()
    try:
        BRepGProp.SurfaceProperties_s(f, sp, 1e-6)
    except Exception:
        BRepGProp.SurfaceProperties_s(f, sp)
    return sp.Mass()


def rhino_objects(g: Geom, r3):
    """rhino3dm objects for a brep: one exact Brep per face where possible
    (untrimmed patches, and planar faces with a single boundary), mesh fallback otherwise."""
    out, fallback = [], []
    for f in faces(shape_of(g)):
        done = False
        try:
            ad = BRepAdaptor_Surface(f, True)
            wires = list(_iter(f, TopAbs_WIRE))
            if ad.GetType() == GeomAbs_Plane and len(wires) == 1:
                pl = ad.Plane()
                o, n, x = _xyz(pl.Location()), _xyz(pl.Axis().Direction()), _xyz(pl.XAxis().Direction())
                rp = r3.Plane(r3.Point3d(*o), r3.Vector3d(*x), r3.Vector3d(*np.cross(n, x)))
                outline = _face_outline(f, r3)
                if outline is not None:
                    b = r3.Brep.CreateTrimmedPlane(rp, outline)
                    if b is not None and b.IsValid:
                        out.append(b)
                        done = True
            if not done and len(wires) == 1:
                bs, f2 = bspline_surface_of_face(f)
                u0, u1, v0, v1 = BRepTools.UVBounds_s(f2)
                patch = BRepBuilderAPI_MakeFace(bs, u0, u1, v0, v1, 1e-7).Face()
                fa = _face_area(f)
                if abs(_face_area(patch) - fa) <= 2e-5 * max(1.0, fa):
                    bs.Segment(u0, u1, v0, v1)
                    b = r3.Brep.CreateFromSurface(_rhino_surface(bs, r3))
                    if b is not None:
                        out.append(b)
                        done = True
        except Exception:
            done = False
        if not done:
            fallback.append(f)
    if fallback:
        sub = make(_compound(fallback), "Faces")
        V, N, T, _ = tessellate(sub)
        m = r3.Mesh()
        for v in V:
            m.Vertices.Add(*map(float, v))
        for t in T:
            m.Faces.AddFace(int(t[0]), int(t[1]), int(t[2]))
        m.Normals.ComputeNormals()
        m.Compact()
        out.append(m)
    return out, len(fallback)


def _reverse_nurbs(n: Geom) -> Geom:
    U = 1.0 - np.asarray(n.knots)[::-1]
    return Geom("nurbs", cps=n.cps[::-1].copy(), w=n.w[::-1].copy(), knots=U, deg=n.deg)


def _face_outline(face, r3):
    """Ordered, oriented outer boundary of a face as one rhino3dm curve (exact)."""
    from .io3dm import to_rhino
    from OCP.BRepTools import BRepTools_WireExplorer
    ex = BRepTools_WireExplorer(BRepTools.OuterWire_s(face))
    pc = r3.PolyCurve()
    n = 0
    while ex.More():
        e = ex.Current()
        c = exact_nurbs(edge_to_nurbs(e)) if True else None
        if e.Orientation() == TopAbs_REVERSED:
            c = _reverse_nurbs(c)
        rc = to_rhino(c)
        if rc is None or not pc.Append(rc):
            return None
        n += 1
        ex.Next()
    return pc if n else None


def _rhino_surface(bs, r3):
    nu, nv = bs.NbUPoles(), bs.NbVPoles()
    rational = bs.IsURational() or bs.IsVRational()
    s = r3.NurbsSurface.Create(3, rational, bs.UDegree() + 1, bs.VDegree() + 1, nu, nv)
    for i in range(nu):
        for j in range(nv):
            p = bs.Pole(i + 1, j + 1)
            w = bs.Weight(i + 1, j + 1)
            s.Points[i, j] = r3.Point4d(p.X() * w, p.Y() * w, p.Z() * w, w)

    def flat(nk, knot, mult):
        U = []
        for k in range(1, nk + 1):
            U += [knot(k)] * mult(k)
        return U[1:-1]
    KU = flat(bs.NbUKnots(), bs.UKnot, bs.UMultiplicity)
    KV = flat(bs.NbVKnots(), bs.VKnot, bs.VMultiplicity)
    for i, k in enumerate(KU[: len(s.KnotsU)]):
        s.KnotsU[i] = float(k)
    for i, k in enumerate(KV[: len(s.KnotsV)]):
        s.KnotsV[i] = float(k)
    return s


def from_rhino_surface(srf) -> Geom:
    """Untrimmed rhino3dm surface -> exact OCC face."""
    from OCP.Geom import Geom_BSplineSurface
    try:
        from OCP.collections import Array2_gp_Pnt as _A2P, Array2_double as _A2D
    except ImportError:
        from OCP.TColgp import TColgp_Array2OfPnt as _A2P
        from OCP.TColStd import TColStd_Array2OfReal as _A2D
    ns = srf.ToNurbsSurface()
    nu, nv = ns.Points.CountU, ns.Points.CountV
    poles = _A2P(1, nu, 1, nv)
    W = _A2D(1, nu, 1, nv)
    for i in range(nu):
        for j in range(nv):
            p = ns.Points.GetControlPoint(i, j)
            w = p.W if ns.IsRational else 1.0
            poles.SetValue(i + 1, j + 1, gp_Pnt(p.X / w, p.Y / w, p.Z / w))
            W.SetValue(i + 1, j + 1, w)

    def km(knots, order):
        U = [knots[0]] + [knots[i] for i in range(len(knots))] + [knots[len(knots) - 1]]
        uniq, mult = [], []
        for u in U:
            if uniq and abs(u - uniq[-1]) < 1e-12:
                mult[-1] += 1
            else:
                uniq.append(u)
                mult.append(1)
        a, b = _ADbl(1, len(uniq)), _AInt(1, len(uniq))
        for i, (u, m) in enumerate(zip(uniq, mult)):
            a.SetValue(i + 1, float(u))
            b.SetValue(i + 1, int(m))
        return a, b
    ku, mu = km(ns.KnotsU, ns.OrderU)
    kv, mv = km(ns.KnotsV, ns.OrderV)
    bs = Geom_BSplineSurface(poles, W, ku, kv, mu, mv, ns.OrderU - 1, ns.OrderV - 1)
    f = BRepBuilderAPI_MakeFace(bs, 1e-6).Face()
    return make(f, "Imported surface")


# ===================================================================== sub-object: faces
def face_index_at(g: Geom, p):
    """Index (in faces() order) of the face of g closest to point p."""
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    v = BRepBuilderAPI_MakeVertex(_pnt(p)).Vertex()
    best, bd = None, float("inf")
    for i, f in enumerate(faces(shape_of(g))):
        d = BRepExtrema_DistShapeShape(v, f)
        if d.IsDone() and d.Value() < bd:
            best, bd = i, d.Value()
    return best


def face_info(g: Geom, i):
    """(is_planar, outward normal, centre) of face i."""
    f = faces(shape_of(g))[i]
    ad = BRepAdaptor_Surface(f)
    u = (ad.FirstUParameter() + ad.LastUParameter()) / 2
    v = (ad.FirstVParameter() + ad.LastVParameter()) / 2
    pnt, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
    ad.D1(u, v, pnt, du, dv)
    n = np.array(_xyz(du.Crossed(dv)))
    n /= np.linalg.norm(n) or 1
    if f.Orientation() == TopAbs_REVERSED:
        n = -n
    sp = GProp_GProps()
    BRepGProp.SurfaceProperties_s(f, sp)
    try:
        from .editpoints import _flat_face
        flat = _flat_face(f)
    except Exception:
        flat = ad.GetType() == GeomAbs_Plane
    return flat, n, np.array(_xyz(sp.CentreOfMass()))


def face_geom(g: Geom, i, **kw):
    """One face of a polysurface/solid as its own surface object."""
    return make(faces(shape_of(g))[i], "Surface", **kw)


def push_pull(g: Geom, i, vector):
    """Extrude face i of a solid along vector: outward adds material, inward cuts it (like push/pull)."""
    f = faces(shape_of(g))[i]
    v = np.asarray(vector, float)
    _, n, _ = face_info(g, i)
    prism = BRepPrimAPI_MakePrism(f, _vec(v)).Shape()
    sh = shape_of(g)
    op = BRepAlgoAPI_Fuse(sh, prism) if np.dot(n, v) >= 0 else BRepAlgoAPI_Cut(sh, prism)
    op.Build()
    if not op.IsDone():
        raise ValueError("Push/pull failed")
    res = op.Shape()
    try:
        u = ShapeUpgrade_UnifySameDomain(res, True, True, True)
        u.Build()
        res = u.Shape()
    except Exception:
        pass
    return make(res, g.data.get("source", "Polysurface"), layer=g.layer, space=g.space, name=g.name, id=g.id)


# ===================================================================== trim / split
def _as_single_edge(g: Geom):
    """A curve as ONE exact edge (polylines become a degree-1 B-spline, so only real cuts split it)."""
    n = exact_nurbs(g)
    return BRepBuilderAPI_MakeEdge(occ_bspline(n)).Edge()


def _cutter_shapes(cutters, target_is_curve, big):
    """OCC tool shapes. A curve that has to cut a surface or solid is swept straight up and down
    (like trimming in Top view), unless it lies flat in a tilted plane, then along that plane's normal."""
    tools = []
    for c in cutters:
        if c.kind == "brep":
            tools.append(shape_of(c))
        elif c.kind == "mesh":
            continue
        elif c.is_curve:
            if target_is_curve:
                tools.append(_as_single_edge(c))
            else:
                pts = c.sample()
                n = np.array([0, 0, 1.0])
                if len(pts) >= 3:
                    q = pts - pts.mean(0)
                    _, sv, vt = np.linalg.svd(q, full_matrices=False)
                    if sv[-1] < 1e-6 * max(sv[0], 1e-9) and abs(vt[-1][2]) < 0.99:
                        n = vt[-1]
                e = _as_single_edge(c)
                moved = BRepBuilderAPI_Transform(e, _trsf_translate(-n * big), True).Shape()
                tools.append(BRepPrimAPI_MakePrism(moved, _vec(n * 2 * big)).Shape())
    return tools


def _trsf_translate(v):
    t = gp_Trsf()
    t.SetTranslation(_vec(v))
    return t


def _extend_face_tangent(f, L, nu=24, nv=12):
    """A curved face continued straight (tangentially) past all four of its edges by length L,
    rebuilt as a B-spline through a grid of points. Used only as a cutting tool."""
    from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace as MF
    try:
        from OCP.collections import Array2_gp_Pnt as A2P
    except ImportError:
        from OCP.TColgp import TColgp_Array2OfPnt as A2P
    ad = BRepAdaptor_Surface(f)
    u0, u1, v0, v1 = BRepTools.UVBounds_s(f)
    U, V = np.linspace(u0, u1, nu), np.linspace(v0, v1, nv)
    P = np.empty((nu, nv, 3))
    Du = np.empty((nu, nv, 3))
    Dv = np.empty((nu, nv, 3))
    for i, u in enumerate(U):
        for j, v in enumerate(V):
            p, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
            ad.D1(u, v, p, du, dv)
            P[i, j], Du[i, j], Dv[i, j] = _xyz(p), _xyz(du), _xyz(dv)

    def unit(d):
        n = np.linalg.norm(d, axis=-1, keepdims=True)
        return d / np.where(n > 1e-12, n, 1)
    # exact tangent directions at the edges
    # several evenly spaced rows in each extension keep the fitted surface straight there
    k = 8
    fr = np.linspace(1, k, k)[:, None, None] / k * L
    before = P[:1] - unit(Du[:1]) * fr[::-1]
    after = P[-1:] + unit(Du[-1:]) * fr
    P = np.concatenate([before, P, after], axis=0)
    Dv = np.concatenate([np.repeat(Dv[:1], k, 0), Dv, np.repeat(Dv[-1:], k, 0)], axis=0)
    frv = np.linspace(1, k, k)[None, :, None] / k * L
    P = np.concatenate([P[:, :1] - unit(Dv[:, :1]) * frv[:, ::-1], P, P[:, -1:] + unit(Dv[:, -1:]) * frv], axis=1)
    arr = A2P(1, P.shape[0], 1, P.shape[1])
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            arr.SetValue(i + 1, j + 1, _pnt(P[i, j]))
    bs = GeomAPI_PointsToBSplineSurface(arr, 1, 3).Surface()
    return MF(bs, 1e-6).Face()


def _extended_cutters(cutters, big):
    """Cutting surfaces made large enough to cut right through: flat faces become big planes,
    curved faces are extended along their own geometry."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace as MF
    from OCP.Geom import Geom_RectangularTrimmedSurface
    out = []
    for c in cutters:
        if c.kind != "brep":
            out.append(c)
            continue
        fs = faces(shape_of(c))
        if len(fs) != 1:
            out.append(c)
            continue
        f = fs[0]
        srf = BRep_Tool.Surface_s(f)
        ad = BRepAdaptor_Surface(f)
        try:
            if ad.GetType() == GeomAbs_Plane:
                face = MF(ad.Plane(), -big, big, -big, big).Face()
            else:
                face = _extend_face_tangent(f, big / 3.0)
            out.append(make(face, "Cutter"))
        except Exception:
            out.append(c)
    return out


def _split_extended(target, cutters):
    """Try again with the cutting surfaces extended (a few lengths, since very long curved
    extensions can bend away)."""
    lo, hi = bbox(target) if target.kind == "brep" else (target.sample().min(0), target.sample().max(0))
    diag = float(np.linalg.norm(np.asarray(hi) - np.asarray(lo))) or 1.0
    for k in (3.0, 1.5, 4.5, 0.9):
        try:
            pieces = split_pieces(target, _extended_cutters(cutters, diag * k))
        except Exception:
            continue
        if len(pieces) >= 2:
            return pieces
    return []


def split_pieces(target: Geom, cutters):
    """Split a curve, surface or solid by other curves/surfaces/solids. Returns a list of OCC shapes."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Splitter
    try:
        from OCP.TopTools import TopTools_ListOfShape
    except ImportError:
        from OCP.collections import List_TopoDS_Shape as TopTools_ListOfShape
    is_curve = target.is_curve
    lo, hi = bbox(target) if target.kind == "brep" else (target.sample().min(0), target.sample().max(0))
    big = float(np.linalg.norm(np.asarray(hi) - np.asarray(lo))) * 4 + 1000.0
    for c in cutters:
        try:
            l2, h2 = bbox(c) if c.kind == "brep" else (c.sample().min(0), c.sample().max(0))
            big = max(big, float(np.linalg.norm(np.asarray(h2) - np.asarray(l2))) * 4)
        except Exception:
            pass
    arg = _as_single_edge(target) if is_curve else shape_of(target)
    args, tools = TopTools_ListOfShape(), TopTools_ListOfShape()
    args.Append(arg)
    for t in _cutter_shapes(cutters, is_curve, big):
        tools.Append(t)
    sp = BRepAlgoAPI_Splitter()
    sp.SetArguments(args)
    sp.SetTools(tools)
    sp.SetFuzzyValue(max(big * 1e-9, 1e-7))
    sp.Build()
    if not sp.IsDone():
        raise ValueError("Could not split")
    res = sp.Shape()
    if is_curve:
        return [_edge(e) for e in _iter(res, TopAbs_EDGE)]
    solids = list(_iter(res, TopAbs_SOLID))
    if solids:
        return [_caster("Solid")(s) for s in solids]
    return [_face(f) for f in _iter(res, TopAbs_FACE)]


def _dist(shape, p):
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    d = BRepExtrema_DistShapeShape(BRepBuilderAPI_MakeVertex(_pnt(p)).Vertex(), shape)
    return d.Value() if d.IsDone() else float("inf")


def distance_to(g: Geom, p):
    """Exact distance from a point to a curve or brep."""
    if g.kind == "brep":
        return _dist(shape_of(g), p)
    if g.is_curve:
        try:
            return _dist(_as_single_edge(g), p)
        except Exception:
            s = g.sample()
            return float(np.min(np.linalg.norm(s - np.asarray(p), axis=1)))
    if g.kind == "point":
        return float(np.linalg.norm(g.p - np.asarray(p)))
    return float("inf")


def _curve_from_edges(edges, like: Geom):
    """Forma curves from split edges, keeping polylines as polylines."""
    out = []
    for e in edges:
        c = edge_to_nurbs(e, layer=like.layer, space=like.space)
        if like.kind == "polyline" and c.kind == "nurbs" and c.deg == 1:
            pts = [c.cps[0]]
            for q in c.cps[1:]:
                if np.linalg.norm(q - pts[-1]) > 1e-9:
                    pts.append(q)
            c = G.polyline(pts, layer=like.layer, space=like.space)
        out.append(c)
    return out


def trim_pieces(target: Geom, cutters):
    pieces = split_pieces(target, cutters)
    if len(pieces) < 2 and any(c.kind == "brep" for c in cutters):
        pieces = _split_extended(target, cutters)
    return pieces


def piece_points(sh):
    """Sample points on a split piece (for window selection of trim parts)."""
    if sh.ShapeType() == TopAbs_EDGE:
        c = BRepAdaptor_Curve(_edge(sh))
        return np.array([_xyz(c.Value(t)) for t in np.linspace(c.FirstParameter(), c.LastParameter(), 24)])
    g = make(sh, "piece")
    V, _, T, E = tessellate(g)[:4]
    pts = [np.asarray(V, float)] + [np.asarray(e, float) for e in E]
    return np.vstack(pts) if pts else np.zeros((0, 3))


def assemble(target: Geom, rest):
    """Forma geoms from the pieces that remain after trimming."""
    if target.is_curve:
        return _curve_from_edges(rest, target)
    kw = dict(layer=target.layer, space=target.space)
    if all(s.ShapeType() == TopAbs_FACE for s in rest):
        if len(rest) == 1:
            return [make(rest[0], target.data.get("source", "Surface"), **kw)]
        return [join([make(s, "Surface") for s in rest], source="Trimmed surface", **kw)]
    return [make(s, target.data.get("source", "Polysurface"), **kw) for s in rest]


def trim(target: Geom, cutters, pick_point, keep=False):
    """Rhino-style trim: split target by cutters and delete the piece nearest pick_point
    (keep=True keeps only that piece instead). Returns a list of Forma geoms."""
    pieces = split_pieces(target, cutters)
    if len(pieces) < 2 and any(c.kind == "brep" for c in cutters):
        # like Rhino, let cutting surfaces reach across the object when they stop short
        pieces = _split_extended(target, cutters)
    if len(pieces) < 2:
        raise ValueError("The cutting objects do not cross this object")
    d = [_dist(s, pick_point) for s in pieces]
    k = int(np.argmin(d))
    rest = [pieces[k]] if keep else [s for i, s in enumerate(pieces) if i != k]
    if target.is_curve:
        return _curve_from_edges(rest, target)
    kw = dict(layer=target.layer, space=target.space)
    if all(s.ShapeType() == TopAbs_FACE for s in rest):
        if len(rest) == 1:
            return [make(rest[0], target.data.get("source", "Surface"), **kw)]
        return [join([make(s, "Surface") for s in rest], source="Trimmed surface", **kw)]
    return [make(s, target.data.get("source", "Polysurface"), **kw) for s in rest]


def split(target: Geom, cutters):
    pieces = split_pieces(target, cutters)
    if len(pieces) < 2 and any(c.kind == "brep" for c in cutters):
        pieces = _split_extended(target, cutters)
    if target.is_curve:
        return _curve_from_edges(pieces, target)
    kw = dict(layer=target.layer, space=target.space)
    return [make(s, "Split", **kw) for s in pieces]



def remove_faces(g: Geom, idxs):
    """Delete some faces of a polysurface/solid. The rest is re-joined; pieces that are no longer
    connected become separate objects. Returns a list of Forma geoms (possibly empty)."""
    idxs = set(int(i) for i in idxs)
    keep = [f for i, f in enumerate(faces(shape_of(g))) if i not in idxs]
    if not keep:
        return []
    kw = dict(layer=g.layer, space=g.space)
    sew = BRepBuilderAPI_Sewing(1e-4)
    for f in keep:
        sew.Add(f)
    sew.Perform()
    sh = sew.SewedShape()
    out = []
    used = set()
    for shell in _iter(sh, TopAbs_SHELL):
        shell = _shell(shell)
        fs = faces(shell)
        for f in fs:
            used.add(_key(f))
        if len(fs) == 1:
            out.append(make(fs[0], "Surface", **kw))
        else:
            out.append(make(shell, g.data.get("source", "Polysurface"), **kw))
    for f in _iter(sh, TopAbs_FACE):
        if _key(f) not in used:
            out.append(make(_face(f), "Surface", **kw))
    return out
