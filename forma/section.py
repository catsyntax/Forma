"""Section planes: cut the model with a rectangular plane and draw the result in 2D, from either side.

A section plane is an ordinary flat surface object marked data["section"] = True, so it can be moved,
rotated and resized with the gumball like anything else. The section views show
  - cut lines (bold) where the plane slices through solids, surfaces and meshes, with solid cuts filled,
  - everything beyond the plane (thin), seen square-on,
for the side the view looks towards.
"""
from __future__ import annotations

import numpy as np

from . import geometry as G
from .geometry import Geom


def _B():
    from . import brep
    return brep


def make_plane(a, b, z0, z1, layer="Sections", name="Section"):
    """Vertical rectangular section plane through plan points a, b, from height z0 to z1."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pts = [(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)]
    g = _B().planar_surface(G.polyline(pts, closed=True), layer=layer)
    g.data["section"] = True
    g.data["source"] = "Section plane"
    g.name = name
    return g


def plane_frame(g: Geom):
    """(origin, normal) of a section plane object, in world space."""
    V, _, T = _B().tessellate(g)[:3]
    V = np.asarray(V, float)
    c = V.mean(0)
    q = V - c
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    n = vt[-1]
    n /= np.linalg.norm(n) or 1
    return c, n


def view_basis(n):
    """Rows: screen right, screen up, toward the viewer (= n). Up is world Z when possible."""
    n = np.asarray(n, float)
    z = np.array([0, 0, 1.0])
    u = np.cross(z, n)
    if np.linalg.norm(u) < 1e-6:            # horizontal plane (plan cut): x to the right
        u = np.array([1.0, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    if v[2] < -1e-9:
        u, v = -u, -v
    return np.array([u, v, n])


def _clip_beyond(A, B, origin, n):
    """Keep the parts of segments A->B that lie beyond the plane (signed distance < 0 along n)."""
    da = (A - origin) @ n
    db = (B - origin) @ n
    keep_a, keep_b = da < 0, db < 0
    both = keep_a & keep_b
    one = keep_a ^ keep_b
    A1, B1 = A[both], B[both]
    if one.any():
        a, b, d1, d2 = A[one], B[one], da[one], db[one]
        t = (d1 / (d1 - d2))[:, None]
        X = a + (b - a) * t
        inside_a = (d1 < 0)[:, None]
        A2 = np.where(inside_a, a, X)
        B2 = np.where(inside_a, X, b)
        A1, B1 = np.vstack([A1, A2]), np.vstack([B1, B2])
    return A1, B1


def _slice_mesh(V, T, origin, n):
    """Intersection segments of a triangle mesh with the plane."""
    d = (V - origin) @ n
    out = []
    for tri in T:
        dd = d[tri]
        if (dd > 0).all() or (dd < 0).all():
            continue
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = tri[i], tri[j]
            if (d[a] > 0) != (d[b] > 0):
                t = d[a] / (d[a] - d[b])
                pts.append(V[a] + (V[b] - V[a]) * t)
        if len(pts) == 2:
            out.append(pts)
    return out


def _loops(segs, tol):
    """Chain cut segments into polylines (closed where they meet)."""
    segs = [(np.asarray(a), np.asarray(b)) for a, b in segs if np.linalg.norm(np.asarray(a) - np.asarray(b)) > tol * 1e-3]
    key = lambda p: tuple(np.round(p / tol).astype(np.int64))
    ends = {}
    for i, (a, b) in enumerate(segs):
        ends.setdefault(key(a), []).append((i, 0))
        ends.setdefault(key(b), []).append((i, 1))
    used = [False] * len(segs)
    loops = []
    for i in range(len(segs)):
        if used[i]:
            continue
        used[i] = True
        chain = [segs[i][0], segs[i][1]]
        for direction in (1, 0):
            while True:
                tip = chain[-1] if direction else chain[0]
                nxt = None
                for j, e in ends.get(key(tip), []):
                    if not used[j]:
                        nxt = (j, e)
                        break
                if nxt is None:
                    break
                j, e = nxt
                used[j] = True
                other = segs[j][1 - e]
                if direction:
                    chain.append(other)
                else:
                    chain.insert(0, other)
        closed = np.linalg.norm(chain[0] - chain[-1]) <= tol
        loops.append((np.array(chain), closed))
    return loops


def compute(doc, section: Geom, side=1, cache=None):
    """Linework of one section view. side=1 looks from the plane's front, -1 from its back.

    Returns dict with 'R' (view basis), 'origin', 'cut' (list of (N,2) polylines),
    'fill' (list of closed (N,2) loops of cut solids) and 'beyond' (M,2,2 segments)."""
    B = _B()
    origin, n = plane_frame(section)
    n = n * side
    R = view_basis(n)
    to2 = lambda P: (np.asarray(P, float) - origin) @ R[:2].T
    lo, hi = None, None
    for owner, g, col, dim in doc.display("3d"):
        try:
            l, h = g.bbox()
        except Exception:
            continue
        lo = np.minimum(lo, l) if lo is not None else np.array(l, float)
        hi = np.maximum(hi, h) if hi is not None else np.array(h, float)
    span = float(np.linalg.norm(hi - lo)) if lo is not None else 1000.0
    tol = max(span * 1e-6, 1e-6)
    cut, fill, beyond = [], [], []
    faces, shades, depths = [], [], []
    cache = cache if cache is not None else {}
    for owner, g, col, dim in doc.display("3d"):
        if g.data.get("section") or owner == section.id:
            continue
        ck = (owner, id(g), tuple(np.round(origin, 3)), tuple(np.round(n, 6)))
        if ck in cache:
            c_, f_, b_, fc = cache[ck]
            cut += c_
            fill += f_
            beyond.append(b_)
            if fc is not None:
                faces.append(fc[0]); shades.append(fc[1]); depths.append(fc[2])
            continue
        fc = None
        c_, f_ = [], []
        A = Bv = None
        try:
            if g.kind == "brep":
                V, _, T, E = B.tessellate(g)[:4]
                segs = [(pl[:-1], pl[1:]) for pl in (np.asarray(e, float) for e in E) if len(pl) > 1]
                if segs:
                    A = np.vstack([s[0] for s in segs])
                    Bv = np.vstack([s[1] for s in segs])
                fc = _beyond_faces(np.asarray(V, float), np.asarray(T, int), origin, n, R)
                cs = _slice_mesh(np.asarray(V, float), np.asarray(T, int), origin, n)
                loops = _loops(cs, max(tol * 50, span * 2e-5))
                for pts, closed in loops:
                    c_.append(to2(pts))
                    if closed and len(pts) > 3:
                        f_.append(to2(pts))
            elif g.kind == "mesh":
                fe = G.feature_edges(g)
                if fe:
                    A = np.array([a for a, b in fe], float)
                    Bv = np.array([b for a, b in fe], float)
                fc = _beyond_faces(np.asarray(g.V, float), g.triangles(), origin, n, R)
                for pts, closed in _loops(_slice_mesh(np.asarray(g.V, float), g.triangles(), origin, n),
                                          max(tol * 50, span * 2e-5)):
                    c_.append(to2(pts))
            elif g.is_curve:
                sm = np.asarray(g.sample(), float)
                if len(sm) > 1:
                    A, Bv = sm[:-1], sm[1:]
        except Exception:
            pass
        if A is not None and len(A):
            A1, B1 = _clip_beyond(A, Bv, origin, n)
            b_ = np.stack([to2(A1), to2(B1)], axis=1) if len(A1) else np.zeros((0, 2, 2))
        else:
            b_ = np.zeros((0, 2, 2))
        cache[ck] = (c_, f_, b_, fc)
        cut += c_
        fill += f_
        beyond.append(b_)
        if fc is not None:
            faces.append(fc[0]); shades.append(fc[1]); depths.append(fc[2])
    beyond = np.concatenate(beyond) if beyond else np.zeros((0, 2, 2))
    if faces:
        F = np.concatenate(faces)
        S = np.concatenate(shades)
        D = np.concatenate(depths)
        order = np.argsort(D)                # far first (painter's order)
        F, S = F[order], S[order]
        if len(F) > 120000:                  # keep huge scenes responsive
            keep = np.linspace(0, len(F) - 1, 120000).astype(int)
            F, S = F[keep], S[keep]
    else:
        F, S = np.zeros((0, 3, 2)), np.zeros(0)
    return {"R": R, "origin": origin, "cut": cut, "fill": fill, "beyond": beyond, "normal": n,
            "faces": F, "shade": S}


def _beyond_faces(V, T, origin, n, R):
    """Triangles fully beyond the plane, projected, with a light shade (elevation look) and depth."""
    if not len(T):
        return None
    d = (V - origin) @ n
    keep = (d[T] < 0).all(axis=1)
    T = T[keep]
    if not len(T):
        return None
    P = V[T]
    nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    L = np.linalg.norm(nrm, axis=1)
    ok = L > 0
    P, nrm, L = P[ok], nrm[ok], L[ok]
    nrm = nrm / L[:, None]
    light = np.array([0.35, 0.45, 0.82])
    light /= np.linalg.norm(light)
    shade = 0.55 + 0.45 * np.abs(nrm @ light)
    depth = ((P.mean(axis=1) - origin) @ n)
    P2 = (P - origin) @ R[:2].T
    return P2, shade, depth
