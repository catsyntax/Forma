"""Control-point ("grip") editing: which points of an object can be moved, and moving them.

Supported: points, polylines, NURBS curves, arcs/circles/ellipses (converted to exact NURBS on edit),
meshes (vertices), single-face untrimmed NURBS surfaces (their surface control points), and
flat-faced solids and polysurfaces such as boxes, extrusions of polylines and their booleans (their corners;
faces that stop being flat are split into flat triangles so the result stays an exact solid).
"""
from __future__ import annotations

import numpy as np

from . import geometry as G
from .geometry import Geom


def _B():
    from . import brep
    return brep


def grips(g: Geom):
    """(N,3) editable points, or None if this object has no editable control points."""
    k = g.kind
    if k == "point":
        return g.p.reshape(1, 3)
    if k == "polyline":
        return g.pts
    if k == "nurbs":
        cps = g.cps
        if g.closed:  # last control point duplicates the first
            return cps[:-1]
        return cps
    if k == "ellipse":
        n = _B().exact_nurbs(g)
        return n.cps[:-1] if g.closed else n.cps
    if k == "mesh":
        return g.V if len(g.V) <= 20000 else None
    if k == "brep":
        info = surface_grid(g)
        if info is not None:
            return info["poles"].reshape(-1, 3)
        poly = polyhedron(g)
        return None if poly is None else poly["V"]
    return None


def polyhedron(g: Geom):
    """Corners and face loops of a brep whose faces are all flat with straight edges (cached), else None."""
    cache = g.data.get("_poly")
    if cache is not None:
        return cache if cache else None
    B = _B()
    try:
        from OCP.BRepTools import BRepTools_WireExplorer
        from OCP.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
        from OCP.GeomAbs import GeomAbs_Plane, GeomAbs_Line
        from OCP.TopAbs import TopAbs_WIRE, TopAbs_REVERSED
        from OCP.TopExp import TopExp_Explorer
        sh = B.shape_of(g)
        fs = B.faces(sh)
        if not fs or len(fs) > 2000:
            raise ValueError
        lo, hi = B.bbox(g)
        tol = max(float(np.linalg.norm(np.asarray(hi) - np.asarray(lo))) * 1e-7, 1e-9)
        V, index, faces, fmap = [], {}, [], []

        def vid(p):
            key = tuple(np.round(np.asarray(p) / tol).astype(np.int64))
            i = index.get(key)
            if i is None:
                i = len(V)
                V.append(np.asarray(p, float))
                index[key] = i
            return i
        outer_wire = getattr(B.BRepTools, "OuterWire_s", None)
        for f in fs:
            if not _flat_face(f):
                raise ValueError
            loops = []
            ow = outer_wire(f) if outer_wire else None
            ex = TopExp_Explorer(f, TopAbs_WIRE)
            wires = []
            while ex.More():
                wires.append(B._caster("Wire")(ex.Current()) if hasattr(B, "_caster") else ex.Current())
                ex.Next()
            if ow is not None:
                wires.sort(key=lambda w: 0 if w.IsSame(ow) else 1)
            for w in wires:
                we = BRepTools_WireExplorer(w, f)
                loop = []
                while we.More():
                    e = we.Current()
                    if not _straight_edge(e):
                        raise ValueError
                    loop.append(vid(B._xyz(B.BRep_Tool.Pnt_s(we.CurrentVertex()))))
                    we.Next()
                if len(loop) >= 3:
                    loops.append(loop)
            if loops:
                faces.append(loops)
                fmap.append(fs.index(f))
        if not faces:
            raise ValueError
        info = {"V": np.array(V), "faces": faces, "solid": _is_solid(sh), "fmap": fmap}
        g.data["_poly"] = info
        return info
    except Exception:
        g.data["_poly"] = False
        return None


def _flat_face(f):
    """Plane, or a NURBS face that is flat / a simple 4-corner (bilinear) patch."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Plane, GeomAbs_BSplineSurface, GeomAbs_BezierSurface
    ad = BRepAdaptor_Surface(f)
    t = ad.GetType()
    if t == GeomAbs_Plane:
        return True
    if t in (GeomAbs_BSplineSurface, GeomAbs_BezierSurface):
        try:
            bs = ad.BSpline() if t == GeomAbs_BSplineSurface else ad.Bezier()
            if bs.UDegree() == 1 and bs.VDegree() == 1 and bs.NbUPoles() == 2 and bs.NbVPoles() == 2:
                return True
        except Exception:
            pass
        try:
            from OCP.GeomLib import GeomLib_IsPlanarSurface
            from OCP.BRep import BRep_Tool
            return GeomLib_IsPlanarSurface(BRep_Tool.Surface_s(f), 1e-6).IsPlanar()
        except Exception:
            return False
    return False


def _straight_edge(e):
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_Line, GeomAbs_BSplineCurve, GeomAbs_BezierCurve
    c = BRepAdaptor_Curve(e)
    t = c.GetType()
    if t == GeomAbs_Line:
        return True
    if t in (GeomAbs_BSplineCurve, GeomAbs_BezierCurve):
        bs = c.BSpline() if t == GeomAbs_BSplineCurve else c.Bezier()
        P = np.array([[bs.Pole(i).X(), bs.Pole(i).Y(), bs.Pole(i).Z()] for i in range(1, bs.NbPoles() + 1)])
        if len(P) == 2:
            return True
        d = P[-1] - P[0]
        L = np.linalg.norm(d)
        if L < 1e-12:
            return False
        off = np.linalg.norm(np.cross(P - P[0], d / L), axis=1)
        return bool(off.max() < 1e-6 * max(L, 1.0))
    return False


def _is_solid(sh):
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    return TopExp_Explorer(sh, TopAbs_SOLID).More()


def _planar(P, tol):
    c = P.mean(0)
    u, sv, vt = np.linalg.svd(P - c)
    return sv[-1] / max(len(P) ** 0.5, 1) <= tol


def _polyhedron_shape(info, V):
    """Rebuild an exact brep from moved corners: flat faces stay flat, bent ones become flat triangles."""
    B = _B()
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    span = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
    tol = span * 1e-6

    def wire(idx):
        mp = BRepBuilderAPI_MakePolygon()
        for i in idx:
            mp.Add(B._pnt(V[i]))
        mp.Close()
        return mp.Wire()
    sew = BRepBuilderAPI_Sewing(tol * 10)
    for loops in info["faces"]:
        outer = loops[0]
        pts = V[outer]
        allp = V[[i for l in loops for i in l]]
        if _planar(allp, tol):
            mf = BRepBuilderAPI_MakeFace(wire(outer), True)
            for h in loops[1:]:
                w = wire(h)
                w.Reverse()
                mf.Add(w)
            if mf.IsDone():
                sew.Add(mf.Face())
                continue
        # bent face: fan of flat triangles (holes are dropped in this case)
        for j in range(1, len(outer) - 1):
            tri = [outer[0], outer[j], outer[j + 1]]
            if np.linalg.norm(np.cross(V[tri[1]] - V[tri[0]], V[tri[2]] - V[tri[0]])) < tol * tol:
                continue
            mf = BRepBuilderAPI_MakeFace(wire(tri), True)
            if mf.IsDone():
                sew.Add(mf.Face())
    sew.Perform()
    sh = sew.SewedShape()
    if info.get("solid"):
        ex = TopExp_Explorer(sh, TopAbs_SHELL)
        if ex.More():
            try:
                shell = B._caster("Shell")(ex.Current()) if hasattr(B, "_caster") else ex.Current()
                sol = BRepBuilderAPI_MakeSolid(shell).Solid()
                from OCP.BRepLib import BRepLib
                BRepLib.OrientClosedSolid_s(sol)
                sh = sol
            except Exception:
                pass
    return sh


def surface_grid(g: Geom):
    """Control net of a single-face, untrimmed NURBS surface (cached), else None."""
    cache = g.data.get("_grid")
    if cache is not None:
        return cache if cache else None
    B = _B()
    try:
        fs = B.faces(B.shape_of(g))
        if len(fs) != 1:
            g.data["_grid"] = False
            return None
        f = fs[0]
        bs, f2 = B.bspline_surface_of_face(f)
        from OCP.BRepTools import BRepTools
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        u0, u1, v0, v1 = BRepTools.UVBounds_s(f2)
        patch = BRepBuilderAPI_MakeFace(bs, u0, u1, v0, v1, 1e-7).Face()
        fa = B._face_area(f)
        if abs(B._face_area(patch) - fa) > 2e-4 * max(1.0, fa):
            g.data["_grid"] = False
            return None
        bs.Segment(u0, u1, v0, v1)
        nu, nv = bs.NbUPoles(), bs.NbVPoles()
        poles = np.array([[B._xyz(bs.Pole(i + 1, j + 1)) for j in range(nv)] for i in range(nu)])
        weights = np.array([[bs.Weight(i + 1, j + 1) for j in range(nv)] for i in range(nu)])
        uk = [(bs.UKnot(i), bs.UMultiplicity(i)) for i in range(1, bs.NbUKnots() + 1)]
        vk = [(bs.VKnot(i), bs.VMultiplicity(i)) for i in range(1, bs.NbVKnots() + 1)]
        info = {"poles": poles, "weights": weights, "uk": uk, "vk": vk, "ud": bs.UDegree(), "vd": bs.VDegree(),
                "uper": bs.IsUPeriodic(), "vper": bs.IsVPeriodic()}
        g.data["_grid"] = info
        return info
    except Exception:
        g.data["_grid"] = False
        return None


def _surface_from_grid(info, poles):
    from OCP.Geom import Geom_BSplineSurface
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    B = _B()
    try:
        from OCP.collections import Array2_gp_Pnt as A2P, Array2_double as A2D
    except ImportError:
        from OCP.TColgp import TColgp_Array2OfPnt as A2P
        from OCP.TColStd import TColStd_Array2OfReal as A2D
    nu, nv = poles.shape[:2]
    P = A2P(1, nu, 1, nv)
    W = A2D(1, nu, 1, nv)
    for i in range(nu):
        for j in range(nv):
            P.SetValue(i + 1, j + 1, B._pnt(poles[i, j]))
            W.SetValue(i + 1, j + 1, float(info["weights"][i, j]))

    def km(lst):
        a, m = B._ADbl(1, len(lst)), B._AInt(1, len(lst))
        for i, (u, mu) in enumerate(lst):
            a.SetValue(i + 1, float(u))
            m.SetValue(i + 1, int(mu))
        return a, m
    uk, um = km(info["uk"])
    vk, vm = km(info["vk"])
    bs = Geom_BSplineSurface(P, W, uk, vk, um, vm, info["ud"], info["vd"], info["uper"], info["vper"])
    return BRepBuilderAPI_MakeFace(bs, 1e-6).Face()


def apply(g: Geom, indices, M: np.ndarray) -> Geom:
    """New geom with the given grips transformed by M (same id)."""
    idx = sorted(set(int(i) for i in indices))
    k = g.kind
    if k == "point":
        return g.transformed(M)
    if k == "polyline":
        pts = g.pts.copy()
        pts[idx] = G.xform_pts(M, pts[idx])
        return g.copy(pts=pts)
    if k in ("nurbs", "ellipse"):
        n = g if k == "nurbs" else _B().exact_nurbs(g)
        cps = n.cps.copy()
        closed = g.closed
        cps[idx] = G.xform_pts(M, cps[idx])
        if closed:
            cps[-1] = cps[0]
        out = Geom("nurbs", g.space, g.layer, g.name, g.id, cps=cps, w=n.w.copy(), knots=n.knots.copy(), deg=n.deg)
        return out
    if k == "mesh":
        V = g.V.copy()
        V[idx] = G.xform_pts(M, V[idx])
        return g.copy(V=V)
    if k == "brep":
        info = surface_grid(g)
        if info is None:
            poly = polyhedron(g)
            if poly is None:
                raise ValueError("This object has curved trimmed faces; Explode it to edit each surface's points")
            V = poly["V"].copy()
            V[idx] = G.xform_pts(M, V[idx])
            sh = _polyhedron_shape(poly, V)
            return _B().make(sh, g.data.get("source", "Polysurface"), layer=g.layer, space=g.space, name=g.name, id=g.id)
        poles = info["poles"].copy()
        flat = poles.reshape(-1, 3)
        flat[idx] = G.xform_pts(M, flat[idx])
        face = _surface_from_grid(info, flat.reshape(poles.shape))
        out = _B().make(face, g.data.get("source", "Surface"), layer=g.layer, space=g.space, name=g.name, id=g.id)
        return out
    return g


def grid_lines(g: Geom):
    """Control polygon / net for display: list of (N,3) polylines."""
    k = g.kind
    if k in ("polyline",):
        return []
    if k in ("nurbs", "ellipse"):
        n = g if k == "nurbs" else _B().exact_nurbs(g)
        return [n.cps]
    if k == "brep":
        info = surface_grid(g)
        if info is None:
            return []
        P = info["poles"]
        return [P[i] for i in range(P.shape[0])] + [P[:, j] for j in range(P.shape[1])]
    return []


def face_grips(g: Geom, face_index):
    """Grip indices of the corners of one face of a flat-faced solid, or None."""
    poly = polyhedron(g)
    if poly is None or face_index not in poly.get("fmap", []):
        return None
    loops = poly["faces"][poly["fmap"].index(face_index)]
    return sorted({i for l in loops for i in l})


def edge_grips(g: Geom, a, b, tol=None):
    """Grip indices of the two corners at the ends of an edge (a, b points) of a flat-faced object."""
    poly = polyhedron(g)
    if poly is None:
        return None
    V = poly["V"]
    span = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
    tol = tol or span * 1e-5
    out = []
    for p in (a, b):
        d = np.linalg.norm(V - np.asarray(p, float), axis=1)
        i = int(np.argmin(d))
        if d[i] <= tol:
            out.append(i)
    return sorted(set(out)) if len(out) == 2 else None


def faces_with_vertices(g: Geom, verts, need_all=False):
    """Face indices (in faces() order) of a flat-faced object that use the given corner(s).
    need_all=True: faces that contain every given corner as neighbours along an edge."""
    poly = polyhedron(g)
    if poly is None:
        return []
    verts = list(verts)
    out = []
    for k, loops in enumerate(poly["faces"]):
        if need_all:
            ok = False
            for l in loops:
                n = len(l)
                pairs = {frozenset((l[i], l[(i + 1) % n])) for i in range(n)}
                if frozenset(verts) in pairs:
                    ok = True
            if ok:
                out.append(poly["fmap"][k])
        elif any(v in l for l in loops for v in verts):
            out.append(poly["fmap"][k])
    return out


def delete_points(g: Geom, idx):
    """Remove control points. Returns a list of resulting geoms (may split a curve, or be empty)."""
    idx = set(int(i) for i in idx)
    k = g.kind
    if k == "point":
        return []
    if k == "polyline":
        pts = [p for i, p in enumerate(g.pts) if i not in idx]
        if len(pts) < 2:
            return []
        return [G.polyline(pts, closed=g.closed and len(pts) > 2, layer=g.layer, space=g.space)]
    if k in ("nurbs", "ellipse"):
        n = g if k == "nurbs" else _B().exact_nurbs(g)
        cps = n.cps[:-1] if g.closed else n.cps
        pts = [p for i, p in enumerate(cps) if i not in idx]
        if len(pts) < 2:
            return []
        if g.closed:
            pts = pts + [pts[0]]
        deg = min(n.deg, len(pts) - 1)
        return [G.nurbs_curve(pts, degree=deg, layer=g.layer, space=g.space)]
    if k == "mesh":
        keep = np.array([i not in idx for i in range(len(g.V))])
        remap = -np.ones(len(g.V), int)
        remap[keep] = np.arange(keep.sum())
        F = [[int(remap[i]) for i in f] for f in g.mesh_faces() if all(keep[i] for i in f)]
        if not F:
            return []
        return [G.mesh(g.V[keep], F, layer=g.layer, space=g.space)]
    if k == "brep":
        fs = faces_with_vertices(g, idx)
        if not fs:
            return None
        return _B().remove_faces(g, fs)
    return None
