"""Linework for curved solids and hidden-line display.

* edge_lines(g): true edges + isocurves (breps) or feature edges (meshes), in world coordinates.
* silhouettes(g, direction | eye): outline edges of the tessellation for a view direction (parallel) or an
  eye point (perspective): mesh edges whose two faces turn one towards and one away from the viewer.
* DepthClassifier: renders the occluding surfaces into a depth buffer (VTK, off screen) and splits polylines
  into visible and hidden runs, so back lines can be drawn dashed or left out.
* dashes(): turns hidden runs into dash segments for the 3D view, where line stipple isn't available.
"""
from __future__ import annotations

import numpy as np

from . import geometry as G


# ------------------------------------------------------------------ geometry helpers
def surface_mesh(g):
    """(V, T) triangles of a brep / surface mesh in world coordinates, or None."""
    if g.kind == "brep":
        from . import brep as B
        V, _, T, _ = B.tessellate(g)
        return np.asarray(V, float), np.asarray(T, int)
    if g.kind == "mesh" and g.data.get("axes") is None:
        hit = g.data.get("_trimesh")
        if hit is not None and hit[0] == id(g.data.get("V")):
            return hit[1]
        V = np.asarray(g.V, float)
        tris = []
        for f in g.mesh_faces():
            for k in range(1, len(f) - 1):
                tris.append((f[0], f[k], f[k + 1]))
        out = (V, np.asarray(tris, int).reshape(-1, 3))
        g.data["_trimesh"] = (id(g.data.get("V")), out)
        return out
    return None


def edge_lines(g):
    """Polylines (list of (n,3) arrays) that always show: brep edges + isocurves, mesh feature edges, curves."""
    if g.kind == "brep":
        from . import brep as B
        return list(B.tessellate(g)[3]) + list(B.isocurves(g))
    if g.kind == "mesh":
        return [np.array([a, b]) for a, b in G.feature_edges(g)]
    if g.kind == "point":
        return []
    return [np.asarray(g.sample(), float)]


def _adjacency(g):
    """Merged-vertex edge list with the two faces on each edge, plus face normals (cached on the geom)."""
    hit = g.data.get("_hl_adj")
    sm = surface_mesh(g)
    if sm is None:
        return None
    V, T = sm
    ckey = (id(V), len(V), len(T))
    if hit is not None and hit[0] == ckey:
        return hit[1]
    if not len(T):
        return None
    span = float(np.ptp(V, axis=0).max()) or 1.0
    q = np.round(V / (span * 1e-6)).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.ravel()
    TT = inv[T]
    keep = (TT[:, 0] != TT[:, 1]) & (TT[:, 1] != TT[:, 2]) & (TT[:, 0] != TT[:, 2])
    TT, Tk = TT[keep], T[keep]
    fn = np.cross(V[Tk[:, 1]] - V[Tk[:, 0]], V[Tk[:, 2]] - V[Tk[:, 0]])
    ln = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / np.where(ln > 0, ln, 1)
    cen = V[Tk].mean(1)
    E = np.concatenate([TT[:, [0, 1]], TT[:, [1, 2]], TT[:, [2, 0]]])
    F = np.tile(np.arange(len(TT)), 3)
    P = np.concatenate([Tk[:, [0, 1]], Tk[:, [1, 2]], Tk[:, [2, 0]]])   # original vertex ids for positions
    key = np.sort(E, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    key, F, P = key[order], F[order], P[order]
    same = np.all(key[1:] == key[:-1], axis=1)
    i = np.nonzero(same)[0]
    # only edges shared by exactly two faces
    if len(i):
        ok = np.ones(len(i), bool)
        ok[1:] &= i[1:] != i[:-1] + 1
        ok[:-1] &= i[:-1] + 1 != i[1:]
        i = i[ok]
    pairs = np.stack([F[i], F[i + 1]], 1)
    segs = V[P[i]]
    # skip edges between nearly coplanar faces (they can never be an outline worth drawing)
    data = (pairs, segs, fn, cen)
    g.data["_hl_adj"] = (ckey, data)
    return data


def silhouettes(g, direction=None, eye=None):
    """Outline segments ((k,2,3) array) of a surface for a parallel view direction or a perspective eye."""
    adj = _adjacency(g)
    if adj is None:
        return np.zeros((0, 2, 3))
    pairs, segs, fn, cen = adj
    if not len(pairs):
        return np.zeros((0, 2, 3))
    if eye is not None:
        s = np.einsum("ij,ij->i", fn, cen - np.asarray(eye, float))
    else:
        s = fn @ np.asarray(direction, float)
    a, b = s[pairs[:, 0]], s[pairs[:, 1]]
    mask = (a * b) < 0
    # ignore near-flat folds (tessellation noise on almost planar faces)
    dot = np.einsum("ij,ij->i", fn[pairs[:, 0]], fn[pairs[:, 1]])
    mask &= dot < 0.99999
    return segs[mask]


def curved(g):
    """True when the object has curved surfaces (so silhouettes matter)."""
    if g.kind == "brep":
        from . import brep as B
        return bool(B.isocurves(g))
    return g.kind == "mesh" and not g.data.get("no_edges")


# ------------------------------------------------------------------ depth classification (2D)
class DepthClassifier:
    """Off-screen depth buffer for a parallel view (view coordinates: x right, y up, z towards the viewer)."""

    def __init__(self):
        self._rw = None

    def _window(self, w, h):
        import vtk
        if self._rw is None:
            rw = vtk.vtkRenderWindow()
            rw.SetOffScreenRendering(1)
            rw.SetMultiSamples(0)
            ren = vtk.vtkRenderer()
            ren.SetBackground(1, 1, 1)
            rw.AddRenderer(ren)
            self._rw, self._ren = rw, ren
        self._rw.SetSize(w, h)
        return self._rw, self._ren

    def depth(self, tris, bounds, max_px=2048):
        """Render triangles (n,3,3, view coords). Returns (zbuf (H,W) as view z, origin, pixel size)."""
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
        lo, hi = bounds
        ext = np.maximum(hi[:2] - lo[:2], 1e-9)
        ps = float(max(ext) / max_px)
        W = int(min(max_px, max(16, np.ceil(ext[0] / ps) + 4)))
        H = int(min(max_px, max(16, np.ceil(ext[1] / ps) + 4)))
        origin = np.array([lo[0] - 2 * ps, lo[1] - 2 * ps])
        zmin, zmax = float(lo[2]), float(hi[2])
        rw, ren = self._window(W, H)
        ren.RemoveAllViewProps()
        if len(tris):
            P = tris.reshape(-1, 3).astype(np.float64)
            pts = vtk.vtkPoints()
            pts.SetData(numpy_to_vtk(P, deep=True))
            n = len(tris)
            cells = np.hstack([np.full((n, 1), 3), np.arange(3 * n).reshape(-1, 3)]).astype(np.int64).ravel()
            ca = vtk.vtkCellArray()
            ca.SetCells(n, numpy_to_vtkIdTypeArray(cells, deep=True))
            pd = vtk.vtkPolyData()
            pd.SetPoints(pts)
            pd.SetPolys(ca)
            m = vtk.vtkPolyDataMapper()
            m.SetInputData(pd)
            a = vtk.vtkActor()
            a.SetMapper(m)
            a.GetProperty().LightingOff()
            ren.AddActor(a)
        cam = ren.GetActiveCamera()
        cam.ParallelProjectionOn()
        cx, cy = origin[0] + W * ps / 2, origin[1] + H * ps / 2
        L = (zmax - zmin) + 10 * ps + 1.0
        cam.SetPosition(cx, cy, zmax + L)
        cam.SetFocalPoint(cx, cy, zmin)
        cam.SetViewUp(0, 1, 0)
        cam.SetParallelScale(H * ps / 2)
        near, far = L * 0.5, L + (zmax - zmin) + L * 0.5
        cam.SetClippingRange(near, far)
        rw.Render()
        zb = vtk.vtkFloatArray()
        rw.GetZbufferData(0, 0, W - 1, H - 1, zb)
        d = vtk_to_numpy(zb).reshape(H, W)
        z = (zmax + L) - (near + d * (far - near))   # linear for parallel projection
        z[d >= 0.999999] = -np.inf                     # background: nothing in front
        return z, origin, ps

    def depth_camera(self, tris, camera, W, H):
        """Depth buffer (H,W, 0..1) of the triangles seen through a copy of `camera`. Returns (buffer, camera used)."""
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
        rw, ren = self._window(W, H)
        ren.RemoveAllViewProps()
        if len(tris):
            P = tris.reshape(-1, 3).astype(np.float64)
            pts = vtk.vtkPoints()
            pts.SetData(numpy_to_vtk(P, deep=True))
            n = len(tris)
            cells = np.hstack([np.full((n, 1), 3), np.arange(3 * n).reshape(-1, 3)]).astype(np.int64).ravel()
            ca = vtk.vtkCellArray()
            ca.SetCells(n, numpy_to_vtkIdTypeArray(cells, deep=True))
            pd = vtk.vtkPolyData()
            pd.SetPoints(pts)
            pd.SetPolys(ca)
            m = vtk.vtkPolyDataMapper()
            m.SetInputData(pd)
            a = vtk.vtkActor()
            a.SetMapper(m)
            a.GetProperty().LightingOff()
            ren.AddActor(a)
        cam = ren.GetActiveCamera()
        cam.DeepCopy(camera)
        ren.ResetCameraClippingRange()
        rw.Render()
        zb = vtk.vtkFloatArray()
        rw.GetZbufferData(0, 0, W - 1, H - 1, zb)
        return vtk_to_numpy(zb).reshape(H, W).copy(), cam

    def classify(self, tris, polylines, bounds, max_px=2048):
        """Split polylines (view coords) into runs: returns list of (points (k,3), visible bool) per polyline."""
        from scipy.ndimage import minimum_filter
        try:
            z, origin, ps = self.depth(tris, bounds, max_px)
        except Exception as e:   # no off-screen GL here: draw everything as visible
            print("[forma] hidden lines unavailable:", e, flush=True)
            return [[(np.asarray(pl, float), True)] for pl in polylines]
        zf = minimum_filter(z, size=3)    # farthest nearby surface: forgiving at outlines and on the lines' own surface
        H, W = z.shape
        eps = 2.5 * ps
        out = []
        for pl in polylines:
            pl = np.asarray(pl, float)
            if len(pl) < 2:
                out.append([])
                continue
            # resample so no step is longer than a pixel
            seg = np.linalg.norm(np.diff(pl[:, :2], axis=0), axis=1)
            pts = [pl[0]]
            for k in range(len(pl) - 1):
                n = max(1, int(np.ceil(seg[k] / ps)))
                t = np.linspace(0, 1, n + 1)[1:, None]
                pts.append(pl[k] + (pl[k + 1] - pl[k]) * t)
            P = np.vstack([np.atleast_2d(p) for p in pts])
            i = np.clip(((P[:, 0] - origin[0]) / ps).astype(int), 0, W - 1)
            j = np.clip(((P[:, 1] - origin[1]) / ps).astype(int), 0, H - 1)
            vis = P[:, 2] >= zf[j, i] - eps
            runs = []
            start = 0
            for k in range(1, len(P) + 1):
                if k == len(P) or vis[k] != vis[start]:
                    seg_pts = P[max(0, start - 1):k]
                    if len(seg_pts) >= 2:
                        runs.append((seg_pts, bool(vis[start])))
                    start = k
            out.append(runs)
        return out


# ------------------------------------------------------------------ 3D helpers
def dashes(P, on_px, off_px, px_per_unit):
    """Cut a polyline into dash segments ((k,2,3)) given how many pixels one model unit covers at each
    point (array, same length as P)."""
    P = np.asarray(P, float)
    if len(P) < 2:
        return np.zeros((0, 2, 3))
    d = np.linalg.norm(np.diff(P, axis=0), axis=1) * (px_per_unit[:-1] + px_per_unit[1:]) / 2
    s = np.concatenate([[0], np.cumsum(d)])
    total = s[-1]
    if total <= 0:
        return np.zeros((0, 2, 3))
    period = on_px + off_px
    starts = np.arange(0, total, period)
    ends = np.minimum(starts + on_px, total)

    def at(x):
        k = np.clip(np.searchsorted(s, x) - 1, 0, len(P) - 2)
        t = (x - s[k]) / np.maximum(s[k + 1] - s[k], 1e-12)
        return P[k] + (P[k + 1] - P[k]) * t[:, None]
    return np.stack([at(starts), at(ends)], 1)
