"""Geometry kernel for Forma.

Every object is a ``Geom``: a kind tag plus a dict of numpy data. Geoms are treated
as immutable by the document (operations return new Geoms), which makes undo cheap.
This module has no Qt dependency so it can be used from node worker processes.
"""
from __future__ import annotations

import math
import uuid
import numpy as np

TAU = 2 * math.pi
CURVE_KINDS = {"polyline", "ellipse", "nurbs"}


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def P(*xyz) -> np.ndarray:
    """Make a 3D point from 2 or 3 numbers (or a sequence)."""
    if len(xyz) == 1:
        x = xyz[0]
        if isinstance(x, Geom):
            return x.p.copy() if x.kind == "point" else x.centroid()
        xyz = tuple(x)
    a = np.zeros(3)
    a[: len(xyz)] = xyz[:3]
    return a


class Geom:
    def __init__(self, kind: str, space: str = "3d", layer: str = "Default",
                 name: str = "", id: str | None = None, **data):
        self.kind = kind
        self.space = space
        self.layer = layer
        self.name = name
        self.id = id or new_id()
        self.data = data

    # ------------------------------------------------------------------ basics
    def __getattr__(self, item):
        d = self.__dict__.get("data")
        if d is not None and item in d:
            return d[item]
        raise AttributeError(item)

    def __getstate__(self):
        if self.kind == "brep":
            _brep().freeze(self)
            d = dict(self.__dict__)
            d["data"] = {k: v for k, v in self.data.items() if k not in ("_shape", "_tess", "_props", "_grid", "_poly")}
            return d
        return self.__dict__

    def __setstate__(self, s):
        self.__dict__.update(s)

    def __repr__(self):
        return f"<Geom {self.kind} {self.id} layer={self.layer}>"

    def copy(self, keep_id=True, **override) -> "Geom":
        data = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.data.items()}
        g = Geom(self.kind, self.space, self.layer, self.name,
                 self.id if keep_id else None, **data)
        for k, v in override.items():
            if k in ("space", "layer", "name", "kind"):
                setattr(g, k, v)
            else:
                g.data[k] = v
        return g

    @property
    def is_curve(self):
        return self.kind in CURVE_KINDS

    @property
    def is_mesh(self):
        return self.kind == "mesh"

    @property
    def is_brep(self):
        return self.kind == "brep"

    @property
    def closed(self) -> bool:
        if self.kind == "polyline":
            return bool(self.data.get("closed", False))
        if self.kind == "ellipse":
            return abs(self.a1 - self.a0) >= TAU - 1e-9
        if self.kind == "nurbs":
            c = self.cps
            return bool(np.linalg.norm(c[0] - c[-1]) < 1e-9)
        if self.kind == "brep":
            return _brep().is_solid(_brep().shape_of(self))
        return False

    # -------------------------------------------------------------- sampling
    def sample(self, density: int = 1) -> np.ndarray:
        """Polyline approximation of a curve (Nx3). Closed curves repeat the start point."""
        k = self.kind
        if k == "point":
            return self.p.reshape(1, 3)
        if k == "polyline":
            pts = self.pts
            if self.closed and len(pts) > 1:
                pts = np.vstack([pts, pts[:1]])
            return pts
        if k == "ellipse":
            span = self.a1 - self.a0
            n = max(12, int(abs(span) / TAU * 128 * density))
            t = np.linspace(self.a0, self.a1, n + 1)
            return self.c + np.outer(np.cos(t), self.u) + np.outer(np.sin(t), self.v)
        if k == "nurbs":
            n = max(64, min(4000, 24 * len(self.cps) * density))
            t = np.linspace(self.knots[self.deg], self.knots[-self.deg - 1], n + 1)
            return nurbs_eval(self.cps, self.w, self.knots, self.deg, t)
        if k == "mesh":
            return self.V
        if k == "brep":
            return _brep().tessellate(self)[0]
        return np.zeros((0, 3))

    def length(self) -> float:
        s = self.sample(4)
        return float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum()) if len(s) > 1 else 0.0

    def point_at(self, s: float) -> np.ndarray:
        """Point at normalised arc-length parameter s in [0, 1]."""
        return divide_by_count_params(self.sample(4), [s])[0]

    def divide(self, count: int) -> np.ndarray:
        count = max(1, int(count))
        ss = np.linspace(0, 1, count + 1)
        if self.closed:
            ss = ss[:-1]
        return divide_by_count_params(self.sample(4), ss)

    def resample(self, n: int) -> np.ndarray:
        return divide_by_count_params(self.sample(4), np.linspace(0, 1, n))

    def start(self):
        return self.sample()[0]

    def end(self):
        return self.sample()[-1]

    # -------------------------------------------------------------- meshes
    def mesh_faces(self) -> list[list[int]]:
        if self.kind == "brep":
            return [list(map(int, t)) for t in _brep().tessellate(self)[2]]
        f = self.F
        out, i = [], 0
        while i < len(f):
            n = int(f[i])
            out.append([int(x) for x in f[i + 1: i + 1 + n]])
            i += n + 1
        return out

    def triangles(self) -> np.ndarray:
        if self.kind == "brep":
            return _brep().tessellate(self)[2]
        tris = []
        for face in self.mesh_faces():
            for j in range(1, len(face) - 1):
                tris.append((face[0], face[j], face[j + 1]))
        return np.array(tris, dtype=int).reshape(-1, 3)

    def area(self) -> float:
        if self.kind == "brep":
            return _brep().area(self)
        if self.kind == "mesh":
            V, T = self.V, self.triangles()
            if not len(T):
                return 0.0
            cr = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
            return float(np.linalg.norm(cr, axis=1).sum() / 2)
        if self.is_curve and self.closed:
            pts = self.sample(4)[:-1]
            n = polygon_normal(pts)
            return float(np.linalg.norm(n))  # polygon_normal is area weighted
        return 0.0

    def volume(self) -> float:
        if self.kind == "brep":
            return _brep().volume(self)
        if self.kind != "mesh":
            return 0.0
        V, T = self.V, self.triangles()
        if not len(T):
            return 0.0
        return float(abs(np.einsum("ij,ij->i", V[T[:, 0]], np.cross(V[T[:, 1]], V[T[:, 2]])).sum()) / 6)

    def centroid(self) -> np.ndarray:
        if self.kind == "brep":
            return _brep().centroid(self)
        pts = self.sample(2)
        if self.is_curve and self.closed:
            pts = pts[:-1]
        return pts.mean(axis=0) if len(pts) else np.zeros(3)

    def bbox(self):
        if self.kind == "brep":
            return _brep().bbox(self)
        pts = self.control_points() if self.kind == "mesh" else self.sample()
        if not len(pts):
            return np.zeros(3), np.zeros(3)
        return pts.min(axis=0), pts.max(axis=0)

    def control_points(self) -> np.ndarray:
        k = self.kind
        if k == "point":
            return self.p.reshape(1, 3)
        if k == "polyline":
            return self.pts
        if k == "nurbs":
            return self.cps
        if k == "mesh":
            return self.V
        if k == "ellipse":
            return np.array([self.c, self.c + self.u, self.c + self.v, self.c - self.u, self.c - self.v])
        if k == "instance":
            return self.M[:3, 3].reshape(1, 3)
        if k == "brep":
            lo, hi = _brep().bbox(self)
            return np.array([lo, hi])
        return np.zeros((0, 3))

    # -------------------------------------------------------------- snaps
    def snap_points(self):
        """List of (point, label) usable for object snaps."""
        k = self.kind
        out = []
        if k == "point":
            out.append((self.p, "Point"))
        elif k == "polyline":
            pts = self.pts
            segs = list(zip(pts[:-1], pts[1:]))
            if self.closed:
                segs.append((pts[-1], pts[0]))
            for p in pts:
                out.append((p, "End"))
            for a, b in segs:
                out.append(((a + b) / 2, "Mid"))
            if self.closed:
                out.append((pts.mean(axis=0), "Cen"))
        elif k == "ellipse":
            out.append((self.c, "Cen"))
            if self.closed:
                for t in (0, TAU / 4, TAU / 2, 3 * TAU / 4):
                    out.append((self.c + math.cos(t) * self.u + math.sin(t) * self.v, "Quad"))
            else:
                s = self.sample()
                out += [(s[0], "End"), (s[-1], "End"), (self.point_at(0.5), "Mid")]
        elif k == "nurbs":
            s = self.sample()
            out += [(s[0], "End"), (s[-1], "End"), (self.point_at(0.5), "Mid")]
        elif k == "mesh":
            V = self.V
            if len(V) <= 400:
                out += [(v, "Vertex") for v in V]
        elif k == "brep":
            out += _brep().snap_points(self)
        return out

    # ------------------------------------------------------------ transforms
    def transformed(self, M: np.ndarray, keep_id=True) -> "Geom":
        if self.kind == "brep":
            return _brep().transform(self, M, keep_id=keep_id)
        g = self.copy(keep_id=keep_id)
        d = g.data
        k = self.kind
        if k == "point":
            d["p"] = xform_pts(M, d["p"].reshape(1, 3))[0]
        elif k == "polyline":
            d["pts"] = xform_pts(M, d["pts"])
        elif k == "nurbs":
            d["cps"] = xform_pts(M, d["cps"])
        elif k == "mesh":
            d["V"] = xform_pts(M, d["V"])
            if np.linalg.det(M[:3, :3]) < 0:  # mirrored: flip winding
                d["F"] = flip_faces(d["F"])
        elif k == "ellipse":
            d["c"] = xform_pts(M, d["c"].reshape(1, 3))[0]
            d["u"] = M[:3, :3] @ d["u"]
            d["v"] = M[:3, :3] @ d["v"]
        elif k == "instance":
            d["M"] = M @ d["M"]
        return g

    # ------------------------------------------------------------ serialise
    def to_dict(self):
        if self.kind == "brep":
            _brep().freeze(self)
        return {"kind": self.kind, "space": self.space, "layer": self.layer, "name": self.name,
                "id": self.id,
                "data": {k: ({"__nd__": v.tolist()} if isinstance(v, np.ndarray) else v)
                         for k, v in self.data.items() if k not in ("_shape", "_tess", "_props", "_edit", "_grid", "_poly")}}

    @staticmethod
    def from_dict(d):
        data = {k: (np.array(v["__nd__"]) if isinstance(v, dict) and "__nd__" in v else v)
                for k, v in d["data"].items()}
        if "F" in data:
            data["F"] = np.asarray(data["F"], dtype=int)
        return Geom(d["kind"], d.get("space", "3d"), d.get("layer", "Default"),
                    d.get("name", ""), d.get("id"), **data)

    def describe(self) -> str:
        k = self.kind
        if k == "brep":
            return _brep().describe(self)
        if k == "point":
            return "Point " + fmt_pt(self.p)
        if k == "polyline":
            n = len(self.pts)
            return ("Line" if n == 2 else f"{'Closed ' if self.closed else ''}Polyline ({n} pts)") + \
                f", length {self.length():.4g}"
        if k == "ellipse":
            ru, rv = np.linalg.norm(self.u), np.linalg.norm(self.v)
            t = "Circle" if abs(ru - rv) < 1e-9 else "Ellipse"
            if not self.closed:
                t = "Arc"
            return f"{t}, r={ru:.4g}" + ("" if abs(ru - rv) < 1e-9 else f"/{rv:.4g}") + f", center {fmt_pt(self.c)}"
        if k == "nurbs":
            return f"NURBS curve deg {self.deg}, {len(self.cps)} cps, length {self.length():.4g}"
        if k == "mesh":
            src = self.data.get("source", "Mesh")
            return f"{src}: {len(self.V)} vertices, {len(self.mesh_faces())} faces, area {self.area():.4g}"
        return k


def _brep():
    from . import brep
    return brep


def fmt_pt(p):
    return "(" + ", ".join(f"{x:.4g}" for x in p) + ")"


# ===================================================================== helpers
def xform_pts(M, pts):
    pts = np.asarray(pts, float).reshape(-1, 3)
    h = np.hstack([pts, np.ones((len(pts), 1))]) @ M.T
    return h[:, :3] / h[:, 3:4]


def flip_faces(F):
    F = np.asarray(F, dtype=int).copy()
    i = 0
    while i < len(F):
        n = F[i]
        F[i + 1: i + 1 + n] = F[i + 1: i + 1 + n][::-1]
        i += n + 1
    return F


def translation(v):
    M = np.eye(4)
    M[:3, 3] = P(v)
    return M


def rotation(angle_deg, axis=(0, 0, 1), center=(0, 0, 0)):
    a = math.radians(angle_deg)
    k = P(axis)
    k = k / (np.linalg.norm(k) or 1)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K
    M = np.eye(4)
    M[:3, :3] = R
    c = P(center)
    M[:3, 3] = c - R @ c
    return M


def scaling(s, center=(0, 0, 0)):
    s = np.broadcast_to(np.asarray(s, float), (3,)) if np.ndim(s) else np.array([s, s, s], float)
    M = np.diag([*s, 1.0])
    c = P(center)
    M[:3, 3] = c - s * c
    return M


def mirror(plane_pt, plane_normal):
    n = P(plane_normal)
    n = n / (np.linalg.norm(n) or 1)
    R = np.eye(3) - 2 * np.outer(n, n)
    M = np.eye(4)
    M[:3, :3] = R
    c = P(plane_pt)
    M[:3, 3] = c - R @ c
    return M


def polygon_normal(pts):
    """Newell's method; length equals polygon area."""
    pts = np.asarray(pts)
    n = np.zeros(3)
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        n += np.cross(a, b)
    return n / 2


def divide_by_count_params(samples, params):
    samples = np.asarray(samples, float)
    if len(samples) == 1:
        return np.repeat(samples, len(params), axis=0)
    seg = np.linalg.norm(np.diff(samples, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1] or 1.0
    out = []
    for s in params:
        d = min(max(s, 0.0), 1.0) * total
        i = int(np.searchsorted(cum, d, side="right") - 1)
        i = min(max(i, 0), len(seg) - 1)
        f = (d - cum[i]) / (seg[i] or 1)
        out.append(samples[i] + f * (samples[i + 1] - samples[i]))
    return np.array(out)


# ================================================================ NURBS math
def clamped_knots(n_cps, deg):
    inner = n_cps - deg - 1
    return np.concatenate([np.zeros(deg + 1), np.arange(1, inner + 1) / (inner + 1), np.ones(deg + 1)])


def find_span(n, deg, t, U):
    if t >= U[n + 1]:
        return n
    lo, hi = deg, n + 1
    mid = (lo + hi) // 2
    while t < U[mid] or t >= U[mid + 1]:
        if t < U[mid]:
            hi = mid
        else:
            lo = mid
        mid = (lo + hi) // 2
    return mid


def basis_funs(i, t, deg, U):
    N = np.zeros(deg + 1)
    N[0] = 1.0
    left, right = np.zeros(deg + 1), np.zeros(deg + 1)
    for j in range(1, deg + 1):
        left[j] = t - U[i + 1 - j]
        right[j] = U[i + j] - t
        saved = 0.0
        for r in range(j):
            den = right[r + 1] + left[j - r]
            tmp = N[r] / den if den else 0.0
            N[r] = saved + right[r + 1] * tmp
            saved = left[j - r] * tmp
        N[j] = saved
    return N


def nurbs_eval(cps, w, U, deg, ts):
    """Vectorised rational NURBS evaluation (Cox-de Boor on all parameters at once)."""
    cps = np.asarray(cps, float)
    w = np.asarray(w, float)
    U = np.asarray(U, float)
    ts = np.clip(np.asarray(ts, float), U[deg], U[-deg - 1])
    T = ts[:, None]
    N = ((U[:-1] <= T) & (T < U[1:])).astype(float)
    spans = np.where(U[:-1] < U[1:])[0]
    at_end = ts >= U[-deg - 1]
    if at_end.any() and len(spans):
        N[at_end] = 0.0
        N[at_end, spans[-1]] = 1.0
    for p in range(1, deg + 1):
        d1 = U[p:-1] - U[:-p - 1]
        d2 = U[p + 1:] - U[1:-p]
        a = np.where(d1 > 0, (T - U[:-p - 1]) / np.where(d1 > 0, d1, 1.0), 0.0)
        b = np.where(d2 > 0, (U[p + 1:] - T) / np.where(d2 > 0, d2, 1.0), 0.0)
        N = a * N[:, :-1] + b * N[:, 1:]
    Wn = N * w
    return (Wn @ cps) / Wn.sum(axis=1, keepdims=True)


def basis_matrix(params, n_cps, deg, U):
    A = np.zeros((len(params), n_cps))
    for r, t in enumerate(params):
        span = find_span(n_cps - 1, deg, t, U)
        A[r, span - deg: span + 1] = basis_funs(span, t, deg, U)
    return A


# ============================================================ constructors
def point(p, **kw):
    return Geom("point", p=P(p), **kw)


def line(a, b, **kw):
    return Geom("polyline", pts=np.array([P(a), P(b)]), closed=False, **kw)


def polyline(pts, closed=False, **kw):
    pts = np.array([P(p) for p in pts])
    if closed and len(pts) > 2 and np.linalg.norm(pts[0] - pts[-1]) < 1e-12:
        pts = pts[:-1]
    return Geom("polyline", pts=pts, closed=bool(closed), **kw)


def rectangle(a, b, **kw):
    a, b = P(a), P(b)
    z = a[2]
    return polyline([(a[0], a[1], z), (b[0], a[1], z), (b[0], b[1], z), (a[0], b[1], z)], closed=True, **kw)


def polygon(center, radius, sides=6, rotation_deg=0.0, **kw):
    c = P(center)
    t = np.radians(rotation_deg) + np.arange(sides) * TAU / sides
    pts = [c + radius * np.array([math.cos(a), math.sin(a), 0]) for a in t]
    return polyline(pts, closed=True, **kw)


def circle(center, radius, normal=(0, 0, 1), **kw):
    u, v = plane_axes(normal)
    return Geom("ellipse", c=P(center), u=u * radius, v=v * radius, a0=0.0, a1=TAU, **kw)


def ellipse(center, rx, ry, normal=(0, 0, 1), **kw):
    u, v = plane_axes(normal)
    return Geom("ellipse", c=P(center), u=u * rx, v=v * ry, a0=0.0, a1=TAU, **kw)


def arc(center, radius, a0_deg, a1_deg, normal=(0, 0, 1), **kw):
    u, v = plane_axes(normal)
    a0, a1 = math.radians(a0_deg), math.radians(a1_deg)
    while a1 <= a0:
        a1 += TAU
    return Geom("ellipse", c=P(center), u=u * radius, v=v * radius, a0=a0, a1=a1, **kw)


def arc_3pt(p1, p2, p3, **kw):
    """Arc (or full circle if closed) through three points."""
    c, r, n = circle_from_3pts(p1, p2, p3)
    if c is None:
        return line(p1, p3, **kw)
    u = (P(p1) - c)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    ang = lambda p: math.atan2(np.dot(P(p) - c, v), np.dot(P(p) - c, u)) % TAU
    a2, a3 = ang(p2), ang(p3)
    if a2 > a3:  # p2 must lie between start and end going ccw; otherwise flip normal
        v = -v
        a3 = TAU - a3
    return Geom("ellipse", c=c, u=u * r, v=v * r, a0=0.0, a1=a3 if a3 > 1e-9 else TAU, **kw)


def circle_from_3pts(p1, p2, p3):
    a, b, c = P(p1), P(p2), P(p3)
    ab, ac = b - a, c - a
    n = np.cross(ab, ac)
    nn = np.dot(n, n)
    if nn < 1e-18:
        return None, 0, None
    center = a + (np.cross(n, ab) * np.dot(ac, ac) + np.cross(ac, n) * np.dot(ab, ab)) / (2 * nn)
    return center, float(np.linalg.norm(center - a)), n / math.sqrt(nn)


def plane_axes(normal):
    n = P(normal)
    n = n / (np.linalg.norm(n) or 1)
    ref = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    if np.allclose(n, [0, 0, 1]):
        return np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    u = np.cross(ref, n)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def nurbs_curve(control_points, degree=3, weights=None, knots=None, **kw):
    cps = np.array([P(p) for p in control_points])
    deg = int(min(degree, len(cps) - 1))
    if deg < 1:
        raise ValueError("Need at least 2 control points")
    w = np.ones(len(cps)) if weights is None else np.asarray(weights, float)
    U = clamped_knots(len(cps), deg) if knots is None else np.asarray(knots, float)
    return Geom("nurbs", cps=cps, w=w, knots=U, deg=deg, **kw)


def interp_curve(points, degree=3, **kw):
    """Global interpolation: a NURBS curve that passes through every point."""
    Q = np.array([P(p) for p in points])
    n = len(Q)
    if n < 2:
        raise ValueError("Need at least 2 points")
    deg = min(degree, n - 1)
    d = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    d = np.sqrt(d)  # centripetal parametrisation
    total = d.sum() or 1
    params = np.concatenate([[0], np.cumsum(d) / total])
    U = np.zeros(n + deg + 1)
    U[-deg - 1:] = 1
    for j in range(1, n - deg):
        U[j + deg] = params[j: j + deg].mean()
    A = basis_matrix(params, n, deg, U)
    cps = np.linalg.solve(A, Q)
    return Geom("nurbs", cps=cps, w=np.ones(n), knots=U, deg=deg, **kw)


def to_nurbs_cps(g: Geom) -> Geom:
    """Convert any curve to an editable NURBS approximation through its samples."""
    if g.kind == "nurbs":
        return g.copy()
    s = g.resample(max(6, min(40, len(g.sample()))))
    return interp_curve(s, layer=g.layer, space=g.space, id=g.id)


# ================================================================== meshes
def mesh(V, faces, source="Mesh", **kw):
    flat = []
    for f in faces:
        flat.append(len(f))
        flat.extend(int(i) for i in f)
    return Geom("mesh", V=np.asarray(V, float).reshape(-1, 3), F=np.array(flat, dtype=int), source=source, **kw)


def box(corner, dx, dy, dz, **kw):
    x, y, z = P(corner)
    V = [(x, y, z), (x + dx, y, z), (x + dx, y + dy, z), (x, y + dy, z),
         (x, y, z + dz), (x + dx, y, z + dz), (x + dx, y + dy, z + dz), (x, y + dy, z + dz)]
    F = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    if (dx * dy * dz) < 0:
        F = [f[::-1] for f in F]
    return mesh(V, F, source="Box", **kw)


def grid_mesh(G, closed_u=False, closed_v=False, source="Surface", **kw):
    """Quad mesh from a (nu, nv, 3) grid of points."""
    nu, nv = G.shape[:2]
    V = G.reshape(-1, 3)
    F = []
    iu = range(nu if closed_u else nu - 1)
    iv = range(nv if closed_v else nv - 1)
    for i in iu:
        for j in iv:
            a = i * nv + j
            b = ((i + 1) % nu) * nv + j
            c = ((i + 1) % nu) * nv + (j + 1) % nv
            d = i * nv + (j + 1) % nv
            F.append((a, b, c, d))
    return mesh(V, F, source=source, **kw)


def sphere(center, radius, nu=48, nv=24, **kw):
    c = P(center)
    th = np.linspace(0, TAU, nu, endpoint=False)
    ph = np.linspace(-math.pi / 2, math.pi / 2, nv + 1)
    V = [c + radius * np.array([0, 0, -1.0])]
    for p in ph[1:-1]:
        for t in th:
            V.append(c + radius * np.array([math.cos(p) * math.cos(t), math.cos(p) * math.sin(t), math.sin(p)]))
    V.append(c + radius * np.array([0, 0, 1.0]))
    F = []
    rings = nv - 1
    idx = lambda r, k: 1 + r * nu + (k % nu)
    for k in range(nu):
        F.append((0, idx(0, k + 1), idx(0, k)))
    for r in range(rings - 1):
        for k in range(nu):
            F.append((idx(r, k), idx(r, k + 1), idx(r + 1, k + 1), idx(r + 1, k)))
    top = len(V) - 1
    for k in range(nu):
        F.append((idx(rings - 1, k), idx(rings - 1, k + 1), top))
    return mesh(V, F, source="Sphere", **kw)


def cylinder(base, radius, height, n=48, top_radius=None, **kw):
    b = P(base)
    tr = radius if top_radius is None else top_radius
    th = np.linspace(0, TAU, n, endpoint=False)
    V = [b + np.array([radius * math.cos(t), radius * math.sin(t), 0]) for t in th]
    if tr > 1e-12:
        V += [b + np.array([tr * math.cos(t), tr * math.sin(t), height]) for t in th]
    else:
        V.append(b + np.array([0, 0, height]))
    F = [tuple(range(n - 1, -1, -1))]
    if tr > 1e-12:
        for k in range(n):
            F.append((k, (k + 1) % n, n + (k + 1) % n, n + k))
        F.append(tuple(range(n, 2 * n)))
    else:
        apex = n
        for k in range(n):
            F.append((k, (k + 1) % n, apex))
    return mesh(V, F, source="Cone" if tr != radius else "Cylinder", **kw)


def cone(base, radius, height, n=48, **kw):
    return cylinder(base, radius, height, n=n, top_radius=0.0, **kw)


def torus(center, R, r, nu=48, nv=24, **kw):
    c = P(center)
    G = np.zeros((nu, nv, 3))
    for i, t in enumerate(np.linspace(0, TAU, nu, endpoint=False)):
        for j, p in enumerate(np.linspace(0, TAU, nv, endpoint=False)):
            G[i, j] = c + np.array([(R + r * math.cos(p)) * math.cos(t), (R + r * math.cos(p)) * math.sin(t), r * math.sin(p)])
    return grid_mesh(G, True, True, source="Torus", **kw)


def plane_surface(a, b, **kw):
    a, b = P(a), P(b)
    z = a[2]
    return mesh([(a[0], a[1], z), (b[0], a[1], z), (b[0], b[1], z), (a[0], b[1], z)], [(0, 1, 2, 3)], source="Plane", **kw)


def triangulate_polygon(pts):
    """Ear clipping on a planar (possibly non-convex) polygon. Returns list of index triples."""
    pts = np.asarray(pts, float)
    n = len(pts)
    if n < 3:
        return []
    nrm = polygon_normal(pts)
    if np.linalg.norm(nrm) < 1e-15:
        return []
    u, v = plane_axes(nrm)
    p2 = np.c_[pts @ u, pts @ v]
    idx = list(range(n))
    area = 0.5 * sum(p2[i, 0] * p2[(i + 1) % n, 1] - p2[(i + 1) % n, 0] * p2[i, 1] for i in range(n))
    if area < 0:
        idx.reverse()

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    tris, guard = [], 0
    while len(idx) > 3 and guard < 10 * n:
        guard += 1
        m = len(idx)
        for k in range(m):
            i0, i1, i2 = idx[(k - 1) % m], idx[k], idx[(k + 1) % m]
            a, b, c = p2[i0], p2[i1], p2[i2]
            if cross(a, b, c) <= 1e-14:
                continue
            ok = True
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                p = p2[j]
                if cross(a, b, p) >= 0 and cross(b, c, p) >= 0 and cross(c, a, p) >= 0:
                    ok = False
                    break
            if ok:
                tris.append((i0, i1, i2))
                idx.pop(k)
                break
        else:
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    if area < 0:
        tris = [t[::-1] for t in tris]
    return tris


def planar_surface(curve: Geom, **kw):
    pts = curve.sample(2)[:-1] if curve.closed else curve.sample(2)
    tris = triangulate_polygon(pts)
    return mesh(pts, tris, source="Planar surface", **kw)


def extrude(curve: Geom, vector, cap=True, **kw):
    vec = P(vector)
    if curve.kind == "mesh":
        return curve.transformed(translation(vec), keep_id=False)
    s = curve.sample(2)
    closed = curve.closed
    if closed:
        s = s[:-1]
    G = np.stack([s, s + vec], axis=1)  # (n, 2, 3)
    m = grid_mesh(G, closed_u=closed, source="Extrusion", **kw)
    if cap and closed and len(s) >= 3:
        tris = triangulate_polygon(s)
        nrm = polygon_normal(s)
        V = m.V
        F = list(m.mesh_faces())
        n = len(s)
        flip = np.dot(nrm, vec) > 0
        for t in tris:
            bottom = [2 * i for i in t]
            top = [2 * i + 1 for i in t]
            F.append(bottom[::-1] if flip else bottom)
            F.append(top if flip else top[::-1])
        m = mesh(V, F, source="Extrusion", **kw)
        # orient outward
        if signed_volume(m) < 0:
            m.data["F"] = flip_faces(m.F)
    return m


def signed_volume(m: Geom):
    V, T = m.V, m.triangles()
    if not len(T):
        return 0.0
    return float(np.einsum("ij,ij->i", V[T[:, 0]], np.cross(V[T[:, 1]], V[T[:, 2]])).sum() / 6)


def loft(curves, samples=64, closed=False, **kw):
    curves = [c for c in curves if c.is_curve]
    if len(curves) < 2:
        raise ValueError("Loft needs at least 2 curves")
    all_closed = all(c.closed for c in curves)
    rows = []
    for c in curves:
        r = c.resample(samples + (1 if all_closed else 0))
        if all_closed:
            r = r[:-1]
        rows.append(r)
    # align seams of closed curves to minimise twist
    if all_closed:
        for i in range(1, len(rows)):
            prev = rows[i - 1]
            best, bshift = None, 0
            for sh in range(len(rows[i])):
                d = np.linalg.norm(np.roll(rows[i], -sh, axis=0) - prev, axis=1).sum()
                if best is None or d < best:
                    best, bshift = d, sh
            rows[i] = np.roll(rows[i], -bshift, axis=0)
    G = np.stack(rows, axis=0)  # (ncurves, samples, 3)
    return grid_mesh(G, closed_u=closed, closed_v=all_closed, source="Loft", **kw)


def revolve(curve: Geom, axis_pt=(0, 0, 0), axis_dir=(0, 0, 1), angle_deg=360.0, steps=48, **kw):
    s = curve.sample(2)
    full = abs(angle_deg) >= 360 - 1e-9
    n = steps if full else max(2, int(steps * abs(angle_deg) / 360) + 1)
    angs = np.linspace(0, angle_deg, n, endpoint=not full)
    G = np.stack([xform_pts(rotation(a, axis_dir, axis_pt), s) for a in angs], axis=0)
    return grid_mesh(G, closed_u=full, closed_v=False, source="Revolve", **kw)


def pipe(curve: Geom, radius, sides=16, cap=True, **kw):
    s = curve.sample(2)
    if len(s) < 2:
        raise ValueError("Curve too short")
    rings = []
    prev_u = None
    for i in range(len(s)):
        t = s[min(i + 1, len(s) - 1)] - s[max(i - 1, 0)]
        t /= np.linalg.norm(t) or 1
        if prev_u is None:
            u, _ = plane_axes(t)
        else:  # parallel transport
            u = prev_u - np.dot(prev_u, t) * t
            u /= np.linalg.norm(u) or 1
        v = np.cross(t, u)
        prev_u = u
        rings.append([s[i] + radius * (math.cos(a) * u + math.sin(a) * v)
                      for a in np.linspace(0, TAU, sides, endpoint=False)])
    G = np.array(rings)
    m = grid_mesh(G, closed_u=False, closed_v=True, source="Pipe", **kw)
    if cap and not curve.closed:
        F = m.mesh_faces()
        n = len(s)
        F.append(list(range(sides))[::-1])
        F.append([(n - 1) * sides + k for k in range(sides)])
        m = mesh(m.V, F, source="Pipe", **kw)
    return m


def offset_polyline_2d(g: Geom, dist: float):
    """Planar (XY) offset for polylines/curves; positive = left of direction."""
    s = g.sample(2)
    closed = g.closed
    if closed:
        s = s[:-1]
    n = len(s)
    out = []
    for i in range(n):
        if closed:
            a, b, c = s[i - 1], s[i], s[(i + 1) % n]
        else:
            a = s[max(i - 1, 0)]
            b = s[i]
            c = s[min(i + 1, n - 1)]
        d1 = b - a if np.linalg.norm(b - a) > 1e-12 else c - b
        d2 = c - b if np.linalg.norm(c - b) > 1e-12 else b - a
        n1 = np.array([-d1[1], d1[0], 0]) / (np.linalg.norm(d1[:2]) or 1)
        n2 = np.array([-d2[1], d2[0], 0]) / (np.linalg.norm(d2[:2]) or 1)
        m = n1 + n2
        m /= np.linalg.norm(m) or 1
        cosh = np.dot(m, n1) or 1
        out.append(b + m * dist / max(cosh, 0.2))
    if g.kind == "polyline":
        return polyline(out, closed=closed, layer=g.layer, space=g.space)
    if g.kind == "ellipse" and g.closed:
        ru, rv = np.linalg.norm(g.u), np.linalg.norm(g.v)
        if abs(ru - rv) < 1e-9:
            sign = 1 if np.cross(g.u, g.v)[2] > 0 else -1
            r = ru - sign * dist
            return circle(g.c, abs(r), normal=np.cross(g.u, g.v), layer=g.layer, space=g.space)
    return interp_curve(out + ([out[0]] if closed else []), layer=g.layer, space=g.space) if len(out) > 3 else polyline(out, closed, layer=g.layer, space=g.space)


def segment_intersection_2d(a, b, c, d):
    r, s = b[:2] - a[:2], d[:2] - c[:2]
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-12:
        return None
    t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / den
    u = ((c[0] - a[0]) * r[1] - (c[1] - a[1]) * r[0]) / den
    if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
        return a + t * (b - a)
    return None


def curve_intersections_2d(g1: Geom, g2: Geom, limit=64):
    s1, s2 = g1.sample(), g2.sample()
    out = []
    for i in range(len(s1) - 1):
        a, b = s1[i], s1[i + 1]
        for j in range(len(s2) - 1):
            p = segment_intersection_2d(a, b, s2[j], s2[j + 1])
            if p is not None:
                out.append(p)
                if len(out) >= limit:
                    return out
    return out


def join_curves(curves, tol=1e-6):
    """Join touching curves into polylines (curves are sampled)."""
    chains = [list(c.sample()) for c in curves if c.is_curve and not c.closed]
    merged = True
    while merged:
        merged = False
        for i in range(len(chains)):
            for j in range(i + 1, len(chains)):
                a, b = chains[i], chains[j]
                if np.linalg.norm(a[-1] - b[0]) < tol:
                    chains[i] = a + b[1:]
                elif np.linalg.norm(a[-1] - b[-1]) < tol:
                    chains[i] = a + b[::-1][1:]
                elif np.linalg.norm(a[0] - b[-1]) < tol:
                    chains[i] = b + a[1:]
                elif np.linalg.norm(a[0] - b[0]) < tol:
                    chains[i] = b[::-1] + a[1:]
                else:
                    continue
                chains.pop(j)
                merged = True
                break
            if merged:
                break
    out = []
    for ch in chains:
        closed = len(ch) > 2 and np.linalg.norm(ch[0] - ch[-1]) < tol
        out.append(polyline(ch[:-1] if closed else ch, closed=closed))
    return out


def explode(g: Geom):
    if g.kind == "polyline":
        pts = g.pts
        segs = list(zip(pts[:-1], pts[1:]))
        if g.closed:
            segs.append((pts[-1], pts[0]))
        return [line(a, b, layer=g.layer, space=g.space) for a, b in segs]
    if g.kind == "mesh":
        return [mesh(g.V, [f], layer=g.layer, space=g.space, source="Mesh face") for f in g.mesh_faces()]
    if g.kind == "brep":
        return _brep().explode(g)
    return [g]


def feature_edges(m: Geom, angle_deg=30.0):
    """Boundary edges and sharp edges of a mesh (for wireframe/Make2D). Breps use their true edges."""
    if m.kind == "brep":
        segs = []
        for pl in _brep().tessellate(m)[3]:
            segs += list(zip(pl[:-1], pl[1:]))
        return segs
    V = m.V
    faces = m.mesh_faces()
    normals = []
    for f in faces:
        n = polygon_normal(V[f])
        normals.append(n / (np.linalg.norm(n) or 1))
    edges = {}
    for fi, f in enumerate(faces):
        for a, b in zip(f, f[1:] + f[:1]):
            key = (min(a, b), max(a, b))
            edges.setdefault(key, []).append(fi)
    cos_t = math.cos(math.radians(angle_deg))
    out = []
    for (a, b), fs in edges.items():
        if len(fs) == 1 or (len(fs) == 2 and np.dot(normals[fs[0]], normals[fs[1]]) < cos_t):
            out.append((V[a], V[b]))
    return out


def project_to_plane_2d(geoms, view="top"):
    """Make2D-style projection: curves and mesh feature edges flattened into the 2D space."""
    proj = {"top": lambda p: np.array([p[0], p[1], 0.0]),
            "front": lambda p: np.array([p[0], p[2], 0.0]),
            "right": lambda p: np.array([p[1], p[2], 0.0])}[view]
    out = []
    for g in geoms:
        if g.is_curve:
            out.append(polyline([proj(p) for p in g.sample()], layer=g.layer, space="2d"))
        elif g.kind == "point":
            out.append(point(proj(g.p), layer=g.layer, space="2d"))
        elif g.kind == "brep":
            for pl in _brep().tessellate(g)[3]:
                out.append(polyline([proj(p) for p in pl], layer=g.layer, space="2d"))
        elif g.kind == "mesh":
            segs = feature_edges(g)
            for a, b in segs:
                pa, pb = proj(a), proj(b)
                if np.linalg.norm(pa - pb) > 1e-9:
                    out.append(line(pa, pb, layer=g.layer, space="2d"))
    return out


def as_geom_list(x):
    """Flatten nested lists into a list of Geom."""
    out = []
    if isinstance(x, Geom):
        return [x]
    if isinstance(x, (list, tuple)):
        for y in x:
            out += as_geom_list(y)
    return out
