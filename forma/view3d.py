"""Model workspace viewport (VTK via pyvista/pyvistaqt).

Mouse: left = select / pick points, left-drag = window select (right-to-left = crossing),
right-drag = orbit, Shift+right-drag or middle-drag = pan, wheel = zoom, right-click = Enter.
"""
from __future__ import annotations

import math

import numpy as np
import pyvista as pv
import vtk
from pyvistaqt import QtInteractor
from PySide6 import QtCore, QtWidgets

from . import geometry as G
from .geometry import P

SNAP_PX = 12
pv.global_theme.allow_empty_mesh = True


def geom_polydata(g):
    if g.kind == "brep":
        from . import brep as B
        V, N, T, _ = B.tessellate(g)
        pd = pv.PolyData(np.asarray(V, float), np.hstack([np.full((len(T), 1), 3), T]).ravel()) if len(T) else pv.PolyData(np.asarray(V, float))
        if len(T):
            pd.point_data["Normals"] = N
        return pd
    if g.kind == "mesh":
        F = g.F
        return pv.PolyData(np.asarray(g.V, float), np.asarray(F, dtype=np.int64)) if len(F) else pv.PolyData(np.asarray(g.V, float))
    if g.kind == "point":
        return pv.PolyData(g.p.reshape(1, 3))
    s = g.sample()
    if len(s) < 2:
        return pv.PolyData(s)
    return pv.lines_from_points(s)


# Material colours for parametric-architecture previews (node results), so they read like a model.
ROLE_COLORS = {"pitch": "#3f8f4e", "seating": "#8a2f3a", "roof": "#f2f2f0", "facade": "#c9ced6", "ribs": "#ffffff",
               "trusses": "#e8e8e8", "glass": "#6f9fbf", "slabs": "#d8d8d8", "core": "#b8b2a8", "diagrid": "#f4f4f4",
               "crown": "#e0e6ea", "shell": "#f5f5f2", "columns": "#eeeeea", "ground": "#bfc3c7",
               # modular plan generator room types
               "living": "#c48422", "bedroom": "#2e74b5", "kitchen": "#28946a", "storage": "#7a708c",
               "corridor": "#bc5252", "bathroom": "#3898a8", "envelope": "#5a5a5a"}


class ViewCube:
    """Clickable navigation cube in the top-right corner. Click a face, edge or corner to look from there."""
    SIZE = 0.15  # fraction of viewport width

    def __init__(self, view):
        self.view = view
        rw = view.plotter.ren_win
        rw.SetNumberOfLayers(max(2, rw.GetNumberOfLayers()))
        self.ren = vtk.vtkRenderer()
        self.ren.SetLayer(1)
        self.ren.InteractiveOff()
        self.ren.SetBackgroundAlpha(0.0)
        rw.AddRenderer(self.ren)
        self.cube = vtk.vtkAnnotatedCubeActor()
        self.cube.SetXPlusFaceText("Right")
        self.cube.SetXMinusFaceText("Left")
        self.cube.SetYPlusFaceText("Back")
        self.cube.SetYMinusFaceText("Front")
        self.cube.SetZPlusFaceText("Top")
        self.cube.SetZMinusFaceText("Bottom")
        self.cube.SetFaceTextScale(0.2)
        self.cube.SetXFaceTextRotation(0)
        self.cube.SetYFaceTextRotation(0)
        self.cube.SetZFaceTextRotation(180)
        for get in (self.cube.GetXPlusFaceProperty, self.cube.GetXMinusFaceProperty, self.cube.GetYPlusFaceProperty,
                    self.cube.GetYMinusFaceProperty, self.cube.GetZPlusFaceProperty, self.cube.GetZMinusFaceProperty):
            get().SetInterpolationToFlat()
        self.cube.GetTextEdgesProperty().SetLineWidth(1)
        self.ren.AddActor(self.cube)
        self.highlight = None
        self.apply_theme()
        self.layout()
        view.ren.AddObserver("StartEvent", lambda o, e: self.sync())

    def apply_theme(self):
        c = self.view.settings.palette()
        light = c["mode"] == "light"
        face = (0.93, 0.94, 0.96) if light else (0.30, 0.32, 0.36)
        text = (0.15, 0.15, 0.17) if light else (0.92, 0.92, 0.94)
        cp = self.cube.GetCubeProperty()
        cp.SetColor(*face)
        cp.SetAmbient(0.6)
        cp.SetDiffuse(0.5)
        cp.SetEdgeVisibility(True)
        cp.SetEdgeColor(*(0.55, 0.57, 0.6))
        for get in (self.cube.GetXPlusFaceProperty, self.cube.GetXMinusFaceProperty, self.cube.GetYPlusFaceProperty,
                    self.cube.GetYMinusFaceProperty, self.cube.GetZPlusFaceProperty, self.cube.GetZMinusFaceProperty):
            get().SetColor(*text)
        self.cube.GetTextEdgesProperty().SetColor(*text)
        self.cube.SetTextEdgesVisibility(False)

    def layout(self):
        w, h = self.view.ren.GetSize()
        w, h = max(w, 1), max(h, 1)
        px = max(90, min(150, int(w * self.SIZE)))
        fx, fy = px / w, px / h
        self.ren.SetViewport(1 - fx - 6 / w, 1 - fy - 6 / h, 1 - 6 / w, 1 - 6 / h)

    def sync(self):
        self.layout()
        cam = self.view.ren.GetActiveCamera()
        c2 = self.ren.GetActiveCamera()
        dop = np.array(cam.GetDirectionOfProjection())
        c2.SetFocalPoint(0, 0, 0)
        c2.SetPosition(*(-dop * 3.2))
        c2.SetViewUp(*cam.GetViewUp())
        c2.SetParallelProjection(True)
        c2.SetParallelScale(0.95)
        self.ren.ResetCameraClippingRange()

    def contains(self, x, y):
        vp = self.ren.GetViewport()
        w, h = self.view.ren.GetSize()
        return vp[0] * w <= x <= vp[2] * w and vp[1] * h <= y <= vp[3] * h

    def direction_at(self, x, y):
        """Face / edge / corner direction under the cursor (26 possible views), by ray-casting the unit cube."""
        r = self.ren
        r.SetDisplayPoint(x, y, 0.0)
        r.DisplayToWorld()
        a = np.array(r.GetWorldPoint())
        r.SetDisplayPoint(x, y, 1.0)
        r.DisplayToWorld()
        b = np.array(r.GetWorldPoint())
        if not a[3] or not b[3]:
            return None
        o = a[:3] / a[3]
        d = b[:3] / b[3] - o
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-0.5 - o) / d
            t2 = (0.5 - o) / d
        tmin = np.nanmax(np.minimum(t1, t2))
        tmax = np.nanmin(np.maximum(t1, t2))
        if not np.isfinite(tmin) or tmax < tmin:
            return None
        p = o + d * tmin
        dirn = np.where(p > 0.3, 1.0, np.where(p < -0.3, -1.0, 0.0))
        if not dirn.any():
            k = int(np.argmax(np.abs(p)))
            dirn[k] = np.sign(p[k])
        return dirn


class View3D(QtWidgets.QWidget):
    cursor_moved = QtCore.Signal(object, str)
    component_dropped = QtCore.Signal(str, object)

    def __init__(self, doc, ix, settings, cmdline=None, parent=None, preview_only=False, source=None):
        """source: optional callable returning list of (owner, geom, color, dim) instead of the document."""
        super().__init__(parent)
        self.doc, self.ix, self.settings, self.cmdline = doc, ix, settings, cmdline
        self.preview_only = preview_only
        self.source = source
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self, auto_update=False, multi_samples=4)
        lay.addWidget(self.plotter.interactor)
        try:
            self.plotter.enable_anti_aliasing("msaa", multi_samples=4)
        except Exception:
            try:
                self.plotter.enable_anti_aliasing("fxaa")
            except Exception:
                pass
        self.ren = self.plotter.renderer
        # Fit the near/far planes to everything visible right before each draw, so previews,
        # big moves and overlays never get cut off by a stale clipping range.
        self.ren.SetNearClippingPlaneTolerance(0.001)
        self.ren.AddObserver("StartEvent", lambda *_: self.ren.ResetCameraClippingRange())
        self.shade = "shaded"
        self._pd_cache = {}
        self.preview_actors = []
        self.grid_actors = []
        self.snap = None
        self.snap_actor = None
        self.snap_text = None
        self.cursor_world = None
        self._rb_actor = None
        self._press = None
        self._rpress = None
        self._rmoved = False
        self.extra_actors = []  # node previews etc.
        self.extra_batches = []
        self.batches = []
        self.batch_by_actor = {}
        self._owner_list = []
        self._snap_cache = (None, None, None)
        self._move_pending = False
        self._anim = None
        self._setup_style()
        self.apply_appearance()
        self.viewcube = ViewCube(self)
        from .gumball3d import Gumball3D
        self.gumball = None if preview_only else Gumball3D(self)
        self._gb_drag = False
        self._grip_drag = None
        self._gb_prev = []
        self._grip_actors = []
        if not preview_only:
            doc.subscribe(self._doc_changed)
            ix.preview_changed.connect(self.update_preview)
            ix.changed.connect(self._ix_changed)
        self.plotter.interactor.installEventFilter(self)
        self.plotter.interactor.setAcceptDrops(True)
        self.view_persp(reset=True)
        self.rebuild()

    # ------------------------------------------------------------ appearance
    def apply_appearance(self):
        c = self.settings.palette()
        if self.settings.gradient_3d:
            self.plotter.set_background(c["view3d_bottom"], top=c["view3d_top"])
        else:
            self.plotter.set_background(c["view3d_bottom"])
        self._build_grid()
        if hasattr(self, "viewcube"):
            self.viewcube.apply_theme()
        self.rebuild()

    def _build_grid(self):
        for a in self.grid_actors:
            self.plotter.remove_actor(a, render=False)
        self.grid_actors = []
        if not self.settings.show_grid:
            return
        c = self.settings.palette()
        sp = self.settings.grid_spacing
        n = int(self.settings.grid_extent)
        ext = sp * n
        minor, major = [], []
        for i in range(-n, n + 1):
            v = i * sp
            (major if i % 5 == 0 else minor).append(((v, -ext, 0), (v, ext, 0)))
            (major if i % 5 == 0 else minor).append(((-ext, v, 0), (ext, v, 0)))
        for segs, col, w in ((minor, c["grid"], 1), (major, c["grid_major"], 1)):
            pts = np.array([p for s in segs for p in s], float)
            lines = np.hstack([[2, 2 * k, 2 * k + 1] for k in range(len(segs))])
            pd = pv.PolyData(pts, lines=lines)
            a = self.plotter.add_mesh(pd, color=col, line_width=w, pickable=False, reset_camera=False, render=False,
                                      lighting=False)
            self.grid_actors.append(a)
        for (a, b), col in (((( -ext, 0, 0), (ext, 0, 0)), "#c8463c"), (((0, -ext, 0), (0, ext, 0)), "#46aa50")):
            act = self.plotter.add_mesh(pv.Line(a, b), color=col, line_width=1.5, pickable=False, reset_camera=False,
                                        render=False, lighting=False)
            self.grid_actors.append(act)

    # ------------------------------------------------------------ build scene
    # The scene is drawn with a handful of batched actors (surfaces, edges, curves, points) instead of
    # one actor per object. Per-object colour lives in a per-point RGB array, so selection highlighting
    # only rewrites colours. Picking maps the picked cell back to its object.
    def _items(self):
        if self.source is not None:
            return self.source()
        return self.doc.display("3d")

    def _doc_changed(self, what):
        if what in ("changed", "loaded", "layers", "definitions", "editing"):
            self.rebuild()
        elif what == "selection":
            if self.gumball:
                self.gumball.refresh()
            self.recolor()
        elif what == "grips":
            self._build_grips()
            if self.gumball:
                self.gumball.refresh()
            self.plotter.render()

    def _ix_changed(self):
        if self.gumball is not None:
            was = self.gumball.visible
            self.gumball.refresh()
            if was != self.gumball.visible:
                self.plotter.render()

    # ------------------------------------------------------------ control points
    def _build_grips(self):
        if self.gumball is None:
            return
        gren = self.gumball.ren
        for a in self._grip_actors:
            gren.RemoveActor(a)
        self._grip_actors = []
        gp = self.doc.grip_points()
        if not gp:
            return
        from . import editpoints as E
        c = self.settings.palette()
        lines = []
        for oid in self.doc.points_on:
            g = self.doc.get(oid)
            if g is not None:
                lines += [l for l in E.grid_lines(g) if len(l) >= 2]
        if lines:
            pts = np.vstack(lines)
            cells, k = [], 0
            for l in lines:
                cells += [len(l)] + list(range(k, k + len(l)))
                k += len(l)
            pd = pv.PolyData(pts, lines=np.array(cells))
            m = vtk.vtkPolyDataMapper()
            m.SetInputData(pd)
            a = vtk.vtkActor()
            a.SetMapper(m)
            a.GetProperty().SetColor(pv.Color(c["muted"]).float_rgb)
            a.PickableOff()
            gren.AddActor(a)
            self._grip_actors.append(a)
        P = np.array([p for _, _, p in gp])
        sel = self.doc.grip_sel
        cols = np.array([pv.Color(c["selection"]).int_rgb if (o, i) in sel else (245, 245, 245) for o, i, _ in gp], np.uint8)
        for size, colors in ((11, np.zeros_like(cols)), (8, cols)):
            pd = pv.PolyData(P)
            pd.point_data["rgb"] = colors
            m = vtk.vtkPolyDataMapper()
            m.SetInputData(pd)
            m.SetScalarModeToUsePointData()
            m.SetColorModeToDirectScalars()
            m.SelectColorArray("rgb")
            m.ScalarVisibilityOn()
            a = vtk.vtkActor()
            a.SetMapper(m)
            a.GetProperty().SetPointSize(size)
            a.GetProperty().SetRepresentationToPoints()
            a.PickableOff()
            gren.AddActor(a)
            self._grip_actors.append(a)

    def _grip_at(self, x, y):
        gp = self.doc.grip_points()
        if not gp:
            return None
        P = np.array([p for _, _, p in gp])
        scr = self.project(P)
        d = np.hypot(scr[:, 0] - x, scr[:, 1] - y)
        if not np.isfinite(d).any():
            return None
        i = int(np.nanargmin(d))
        r = self.plotter.interactor.devicePixelRatioF()
        if d[i] <= 9 * r:
            return (gp[i][0], gp[i][1])
        return None

    def _grips_in_rect(self, x0, y0, x1, y1):
        gp = self.doc.grip_points()
        if not gp:
            return []
        scr = self.project(np.array([p for _, _, p in gp]))
        lo, hi = (min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))
        ins = (scr[:, 0] >= lo[0]) & (scr[:, 0] <= hi[0]) & (scr[:, 1] >= lo[1]) & (scr[:, 1] <= hi[1])
        return [(gp[i][0], gp[i][1]) for i in np.nonzero(ins)[0]]

    def _show_drag_preview(self, M, label=""):
        for a in self._gb_prev:
            self.plotter.remove_actor(a, render=False)
        self._gb_prev = []
        col = self.settings.color("preview")
        for g in self.doc.preview_targets(M):
            try:
                self._gb_prev += self._add_geom(g, col, False, None)
            except Exception:
                pass
        if label:
            self.cursor_moved.emit(self.cursor_world, label)
        self.plotter.render()

    def _clear_drag_preview(self):
        for a in self._gb_prev:
            self.plotter.remove_actor(a, render=False)
        self._gb_prev = []

    def _prep(self, g):
        """Per-geometry render arrays, cached by object identity (geoms are immutable)."""
        key = id(g)
        hit = self._pd_cache.get(key)
        if hit is not None and hit[0] is g:
            return hit[1]
        out = {}
        if g.kind == "brep":
            from . import brep as B
            V, N, T, edges = B.tessellate(g)
            out["pts"] = np.asarray(V, np.float32)
            out["nrm"] = np.asarray(N, np.float32)
            out["tris"] = np.asarray(T, int)
            segs = [np.repeat(e, 2, axis=0)[1:-1] for e in edges if len(e) >= 2]
            if segs:
                out["segs"] = np.vstack(segs).astype(np.float32)
        elif g.kind == "mesh":
            pd = geom_polydata(g)
            if pd.n_cells:
                smooth = bool(g.data.get("no_edges"))
                try:
                    n = pd.compute_normals(split_vertices=not smooth, feature_angle=35 if not smooth else 180,
                                           auto_orient_normals=False,
                                           consistent_normals=False, cell_normals=False).triangulate()
                except Exception:
                    n = pd.triangulate()
                    n = n.compute_normals(cell_normals=False)
                tris = n.faces.reshape(-1, 4)[:, 1:] if n.n_cells else np.zeros((0, 3), int)
                out["pts"] = np.asarray(n.points, np.float32)
                out["nrm"] = np.asarray(n.point_data["Normals"], np.float32)
                out["tris"] = tris
                fe = None if smooth else pd.extract_feature_edges(35, boundary_edges=True, non_manifold_edges=False,
                                                                   manifold_edges=False)
                if fe is not None and fe.n_cells:
                    L = fe.lines.reshape(-1, 3)[:, 1:]
                    out["segs"] = np.asarray(fe.points, np.float32)[L].reshape(-1, 3)
        elif g.kind == "point":
            out["pt"] = np.asarray(g.p, np.float32).reshape(1, 3)
        else:
            out["line"] = np.asarray(g.sample(), np.float32)
        if len(self._pd_cache) > 20000:
            self._pd_cache.clear()
        self._pd_cache[key] = (g, out)
        return out

    @staticmethod
    def _rgb(hexcol):
        return np.array(pv.Color(hexcol).int_rgb, np.uint8)

    def _build_batches(self, items, pickable=True, uniform=None, opacity_scale=1.0):
        """items: list of (owner, geom, colour, dim). Returns list of (actor, meta)."""
        c = self.settings.palette()
        light = c["mode"] == "light"
        edge_rgb = self._rgb("#111111" if light else "#0c0c0e")
        groups = {}
        for owner, g, col, dim in items:
            if light and col.lower() == "#d0d0d0" and g.kind not in ("mesh", "brep"):
                col = "#2a2a2e"
            rgb = self._rgb(uniform or col)
            prep = self._prep(g)
            groups.setdefault(bool(dim), []).append((owner, prep, rgb))
        out = []
        for dim, entries in groups.items():
            op = (0.25 if dim else 1.0) * opacity_scale
            surf_p, surf_n, surf_t, surf_c, surf_o, surf_co = [], [], [], [], [], []
            seg_p, seg_c, seg_o, seg_co = [], [], [], []
            ln_p, ln_cells, ln_c, ln_o, ln_co = [], [], [], [], []
            pt_p, pt_c, pt_o = [], [], []
            ns = ne = nl = 0
            for owner, prep, rgb in entries:
                if "pts" in prep and self.shade != "wire":
                    k = len(prep["pts"])
                    surf_p.append(prep["pts"])
                    surf_n.append(prep["nrm"])
                    surf_t.append(prep["tris"] + ns)
                    surf_c.append(np.broadcast_to(rgb, (k, 3)))
                    surf_o.append(np.full(k, len(self._owner_list), np.int32))
                    surf_co.append(np.full(len(prep["tris"]), len(self._owner_list), np.int32))
                    ns += k
                if "segs" in prep:
                    k = len(prep["segs"])
                    seg_p.append(prep["segs"])
                    seg_c.append(np.broadcast_to(rgb if (self.shade == "wire" or uniform) else edge_rgb, (k, 3)))
                    seg_o.append(np.full(k, len(self._owner_list), np.int32))
                    seg_co.append(np.full(k // 2, len(self._owner_list), np.int32))
                    ne += k
                if "line" in prep:
                    L = prep["line"]
                    k = len(L)
                    if k >= 2:
                        ln_p.append(L)
                        ln_cells.append(np.concatenate([[k], np.arange(nl, nl + k)]))
                        ln_c.append(np.broadcast_to(rgb, (k, 3)))
                        ln_o.append(np.full(k, len(self._owner_list), np.int32))
                        ln_co.append(len(self._owner_list))
                        nl += k
                if "pt" in prep:
                    pt_p.append(prep["pt"])
                    pt_c.append(rgb.reshape(1, 3))
                    pt_o.append(np.full(1, len(self._owner_list), np.int32))
                self._owner_list.append(owner)
            if surf_p:
                T = np.vstack(surf_t)
                # build with faces directly: pv.PolyData(points) alone would add one vertex cell per point,
                # shifting cell ids so picking maps to the wrong object
                pd = pv.PolyData(np.vstack(surf_p), faces=np.hstack([np.full((len(T), 1), 3), T]).ravel())
                pd.point_data["Normals"] = np.vstack(surf_n)
                pd.point_data.active_normals_name = "Normals"
                base = np.vstack(surf_c).copy()
                pd.point_data["rgb"] = base.copy()
                a = self.plotter.add_mesh(pd, scalars="rgb", rgb=True, reset_camera=False, render=False,
                                          opacity=(0.35 if self.shade == "ghost" else 1.0) * op,
                                          specular=0.2, specular_power=20, ambient=0.2, diffuse=0.85,
                                          pickable=pickable, show_scalar_bar=False)
                a.GetProperty().SetInterpolationToPhong()
                if self.shade == "ghost" or op < 1:
                    a.GetProperty().SetBackfaceCulling(False)
                out.append((a, {"pd": pd, "base": base, "pown": np.concatenate(surf_o), "cown": np.concatenate(surf_co)}))
            if seg_p:
                P_ = np.vstack(seg_p)
                n = len(P_) // 2
                pd = pv.PolyData(P_, lines=np.hstack([np.full((n, 1), 2), np.arange(2 * n).reshape(-1, 2)]).ravel())
                base = np.vstack(seg_c).copy()
                pd.point_data["rgb"] = base.copy()
                a = self.plotter.add_mesh(pd, scalars="rgb", rgb=True, line_width=1.6 if self.shade == "wire" else 1.2,
                                          reset_camera=False, render=False, lighting=False, opacity=op,
                                          pickable=pickable, show_scalar_bar=False)
                out.append((a, {"pd": pd, "base": base, "pown": np.concatenate(seg_o), "cown": np.concatenate(seg_co),
                                "edges": True}))
            if ln_p:
                pd = pv.PolyData(np.vstack(ln_p), lines=np.concatenate(ln_cells))
                base = np.vstack(ln_c).copy()
                pd.point_data["rgb"] = base.copy()
                a = self.plotter.add_mesh(pd, scalars="rgb", rgb=True, line_width=2.2, reset_camera=False, render=False,
                                          lighting=False, opacity=op, pickable=pickable, show_scalar_bar=False)
                out.append((a, {"pd": pd, "base": base, "pown": np.concatenate(ln_o), "cown": np.array(ln_co, np.int32)}))
            if pt_p:
                pd = pv.PolyData(np.vstack(pt_p))
                base = np.vstack(pt_c).copy()
                pd.point_data["rgb"] = base.copy()
                a = self.plotter.add_mesh(pd, scalars="rgb", rgb=True, point_size=9, render_points_as_spheres=True,
                                          reset_camera=False, render=False, opacity=op, pickable=pickable,
                                          show_scalar_bar=False)
                own = np.concatenate(pt_o)
                out.append((a, {"pd": pd, "base": base, "pown": own, "cown": own}))
        return out

    def rebuild(self):
        for a, _ in self.batches:
            self.plotter.remove_actor(a, render=False)
        self._owner_list = []
        self.batches = self._build_batches(self._items(), pickable=not self.preview_only)
        self.batch_by_actor = {a: m for a, m in self.batches}
        self.recolor(render=False)
        self._update_ssao()
        if getattr(self, "gumball", None) is not None:
            self._build_grips()
            self.gumball.refresh()
        self.plotter.render()

    def _update_ssao(self):
        """Soft ambient-occlusion shading, scaled to the size of the scene."""
        on = bool(getattr(self.settings, "ambient_occlusion", False)) and not self.preview_only
        try:
            if not on:
                self.ren.UseSSAOOff()
                return
            b = np.array(self.ren.ComputeVisiblePropBounds(), float)
            if not np.isfinite(b).all() or b[0] > b[1]:
                return
            diag = float(np.linalg.norm(b[1::2] - b[0::2])) or 1.0
            self.ren.SetSSAORadius(diag * 0.006)
            self.ren.SetSSAOBias(diag * 0.0006)
            self.ren.SetSSAOKernelSize(128)
            self.ren.SSAOBlurOn()
            self.ren.UseSSAOOn()
        except Exception:
            pass

    def recolor(self, render=True):
        sel = self.doc.selection
        sel_rgb = self._rgb(self.settings.color("selection"))
        sel_idx = np.array([i for i, o in enumerate(self._owner_list) if o in sel], np.int32)
        for a, m in self.batches:
            rgb = m["base"].copy()
            if len(sel_idx):
                mask = np.isin(m["pown"], sel_idx)
                if mask.any():
                    rgb[mask] = sel_rgb
            m["pd"].point_data["rgb"][:] = rgb
            m["pd"].Modified()
        if render:
            self.plotter.render()

    def set_extra(self, geoms, color=None):
        """Node previews: drawn batched in the preview colour, not pickable."""
        for a, _ in self.extra_batches:
            self.plotter.remove_actor(a, render=False)
        for a in getattr(self, "heat_actors", []):
            self.plotter.remove_actor(a, render=False)
        self.heat_actors = []
        legends = []
        plain = []
        for g in geoms:
            fc = g.data.get("fcolors") if g.kind == "mesh" else None
            if fc is None:
                plain.append(g)
                continue
            # analysis result: faces painted with a heat map
            V = np.asarray(g.V, float)
            T = g.triangles()
            if not len(T):
                continue
            pdh = pv.PolyData(V, np.hstack([np.full((len(T), 1), 3), T]).ravel())
            pdh.cell_data["rgb"] = (np.asarray(fc)[:len(T)] * 255).astype(np.uint8)
            act = self.plotter.add_mesh(pdh, scalars="rgb", rgb=True, reset_camera=False, render=False,
                                        pickable=False, lighting=False, show_scalar_bar=False)
            self.heat_actors.append(act)
            if g.data.get("legend"):
                legends.append(g.data["legend"])
        if getattr(self, "_legend_actor", None) is not None:
            self.plotter.remove_actor(self._legend_actor, render=False)
            self._legend_actor = None
        if legends:
            self._legend_actor = self.plotter.add_text("\n".join(dict.fromkeys(legends)) + "\nblue = low, red = high",
                                                       position="lower_left", font_size=7,
                                                       color=self.settings.color("text"))
        geoms = plain
        saved = self._owner_list
        self._owner_list = []
        prev = color or self.settings.color("preview")
        items = [(None, g, ROLE_COLORS.get(g.data.get("role"), prev), False) for g in geoms]
        has_roles = any(g.data.get("role") in ROLE_COLORS for g in geoms)
        self.extra_batches = self._build_batches(items, pickable=False, uniform=None if has_roles else prev,
                                                 opacity_scale=1.0)
        self._owner_list = saved
        self.extra_actors = [a for a, _ in self.extra_batches]
        self.plotter.render()

    def _add_geom(self, g, col, dim, owner, preview=False):
        """Single-geometry actors (used for transient previews)."""
        pd = geom_polydata(g)
        if g.kind in ("mesh", "brep"):
            return [self.plotter.add_mesh(pd, color=col, reset_camera=False, render=False, opacity=0.45,
                                          pickable=False, smooth_shading=pd.n_cells > 12)]
        if g.kind == "point":
            return [self.plotter.add_mesh(pd, color=col, point_size=9, render_points_as_spheres=True,
                                          reset_camera=False, render=False, pickable=False)]
        return [self.plotter.add_mesh(pd, color=col, line_width=2.2, reset_camera=False, render=False,
                                      lighting=False, pickable=False)]

    # ------------------------------------------------------------ preview
    def update_preview(self):
        for a in self.preview_actors:
            self.plotter.remove_actor(a, render=False)
        self.preview_actors = []
        if self.ix.space == "3d":
            col = self.settings.color("preview")
            for g in self.ix.preview():
                for part in self.doc.expand(g):
                    try:
                        acts = self._add_geom(part.copy(keep_id=False), col, False, None)
                    except Exception:
                        continue
                    for a in acts:
                        a.SetPickable(False)
                        if part.kind in ("mesh", "brep"):
                            a.GetProperty().SetOpacity(0.45)
                    self.preview_actors += acts
            r = self.ix.request
            base = getattr(r, "base", None) if r else None
            if base is not None and self.cursor_world is not None and np.linalg.norm(self.cursor_world - base) > 1e-9:
                a = self.plotter.add_mesh(pv.Line(base, self.cursor_world), color=self.settings.color("muted"),
                                          line_width=1, reset_camera=False, render=False, lighting=False, pickable=False)
                self.preview_actors.append(a)
        self._update_snap_marker()
        self.plotter.render()

    def _update_snap_marker(self):
        if self.snap_actor is not None:
            self.plotter.remove_actor(self.snap_actor, render=False)
            self.snap_actor = None
        if self.snap_text is not None:
            self.plotter.remove_actor(self.snap_text, render=False)
            self.snap_text = None
        if self.snap is not None and self.ix.wants_point():
            p, lbl = self.snap
            self.snap_actor = self.plotter.add_mesh(pv.PolyData(np.array([p])), color=self.settings.color("accent"),
                                                    point_size=14, render_points_as_spheres=False, reset_camera=False,
                                                    render=False, pickable=False, lighting=False)
            self.snap_text = self.plotter.add_text(lbl, position="lower_left", font_size=7,
                                                   color=self.settings.color("accent"), font="arial")

    # ------------------------------------------------------------ geometry utils
    def _display_size(self):
        return self.ren.GetSize()

    def project(self, pts):
        """World Nx3 -> display Nx2 (VTK display coords, origin bottom-left)."""
        pts = np.asarray(pts, float).reshape(-1, 3)
        w, h = self._display_size()
        cam = self.ren.GetActiveCamera()
        M = cam.GetCompositeProjectionTransformMatrix(w / max(h, 1), -1, 1)
        A = np.array([[M.GetElement(i, j) for j in range(4)] for i in range(4)])
        hp = np.hstack([pts, np.ones((len(pts), 1))]) @ A.T
        wv = hp[:, 3:4]
        wv[np.abs(wv) < 1e-12] = 1e-12
        ndc = hp[:, :3] / wv
        vp = self.ren.GetViewport()
        x = (ndc[:, 0] + 1) / 2 * w
        y = (ndc[:, 1] + 1) / 2 * h
        behind = hp[:, 3] <= 0
        x[behind] = np.nan
        return np.c_[x, y]

    def ray(self, x, y):
        self.ren.SetDisplayPoint(x, y, 0.0)
        self.ren.DisplayToWorld()
        a = np.array(self.ren.GetWorldPoint())
        self.ren.SetDisplayPoint(x, y, 1.0)
        self.ren.DisplayToWorld()
        b = np.array(self.ren.GetWorldPoint())
        a = a[:3] / (a[3] or 1)
        b = b[:3] / (b[3] or 1)
        return a, b - a

    def cplane_point(self, x, y, z=0.0):
        o, d = self.ray(x, y)
        if abs(d[2]) < 1e-12:
            # looking parallel to the plane: use the view plane through the origin
            cam = self.ren.GetActiveCamera()
            n = np.array(cam.GetDirectionOfProjection())
            t = -np.dot(o, n) / (np.dot(d, n) or 1)
            return o + t * d
        t = (z - o[2]) / d[2]
        return o + t * d

    def resolve(self, x, y):
        self.snap = None
        wants = self.ix.wants_point()
        if wants and self.settings.osnap:
            ver = getattr(self.doc, "version", 0)
            if self._snap_cache[0] != ver:
                sp = self.doc.snap_points("3d")
                pts = np.array([p for p, _ in sp]) if sp else np.zeros((0, 3))
                labels = [l for _, l in sp]
                bonus = np.array([4.0 if l == "Connector" else 0.0 for l in labels])
                self._snap_cache = (ver, pts, (labels, bonus))
            _, pts, (labels, bonus) = self._snap_cache
            if len(pts):
                scr = self.project(pts)
                d = np.hypot(scr[:, 0] - x, scr[:, 1] - y)
                d = d - bonus
                i = int(np.nanargmin(d)) if np.isfinite(d).any() else -1
                if i >= 0 and d[i] < SNAP_PX:
                    self.snap = (P(pts[i]), labels[i])
                    return self.snap[0]
        if wants and self.settings.osnap:
            hit = self._geometry_snap(x, y)
            if hit is not None:
                self.snap = hit
                return self.ix.constrain(hit[0])
        p = self.cplane_point(x, y)
        if wants:
            p = self.ix.grid_snap(p)
        return self.ix.constrain(p)

    def _pick_edge(self, x, y):
        """Cmd/Ctrl+Option+Shift-click: select one edge (its two end points) of a flat-faced object."""
        from . import editpoints as EP
        hit = self._geometry_snap(x, y)
        if hit is None:
            return False
        p = np.asarray(hit[0], float)
        best, bd = None, float("inf")
        # polyline segments first
        for g in self.doc.selectable("3d"):
            if g.kind != "polyline":
                continue
            pts = list(g.pts) + ([g.pts[0]] if g.closed else [])
            for k in range(len(pts) - 1):
                a, b = np.asarray(pts[k]), np.asarray(pts[k + 1])
                ab = b - a
                t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-18), 0, 1)
                d = float(np.linalg.norm(a + ab * t - p))
                if d < bd:
                    best, bd = ("poly", g, k, (k + 1) % len(g.pts)), d
        if best is not None and bd < 1e-3 * (1 + float(np.linalg.norm(p))):
            _, g, i, j = best
            self.doc.selection = {g.id}
            self.doc.face_sel = None
            self.doc.points_on = {g.id}
            self.doc.grip_sel = {(g.id, i), (g.id, j)}
            self.doc.edge_sel = (g.id, (i, j))
            self.doc.notify("selection")
            self.doc.notify("grips")
            self.ix.message.emit("Segment selected. Move it with the gumball, or Delete to cut it out.")
            return True
        best, bd = None, float("inf")
        for g in self.doc.selectable("3d"):
            if g.kind != "brep":
                continue
            lo, hi = g.bbox()
            if np.any(p < lo - 1e-3 * (1 + np.abs(lo))) or np.any(p > hi + 1e-3 * (1 + np.abs(hi))):
                continue
            from . import brep as BR
            V, _, T, E = BR.tessellate(g)[:4]
            for pl in E:
                pl = np.asarray(pl, float)
                if len(pl) < 2:
                    continue
                a, b = pl[:-1], pl[1:]
                ab = b - a
                t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18), 0, 1)
                d = float(np.min(np.linalg.norm(a + ab * t[:, None] - p, axis=1)))
                if d < bd:
                    best, bd = (g, pl[0], pl[-1]), d
        if best is None:
            return False
        g, a, b = best
        idx = EP.edge_grips(g, a, b)
        if not idx:
            self.ix.message.emit("Edges can be picked on flat-faced objects. For curved objects, Explode or use DupEdge.")
            return False
        self.doc.selection = {g.id}
        self.doc.face_sel = None
        self.doc.points_on = {g.id}
        self.doc.grip_sel = {(g.id, i) for i in idx}
        self.doc.edge_sel = (g.id, tuple(idx))
        self.doc.notify("selection")
        self.doc.notify("grips")
        self.ix.message.emit("Edge selected. Drag the gumball (or run Move) to move it; the faces next to it follow.")
        return True

    def _pick_face(self, x, y):
        """Cmd/Ctrl+Shift-click: select one face of a solid or polysurface (sub-object selection)."""
        from . import brep as BR
        oid = self.pick(x, y)
        g = self.doc.objects.get(oid) if oid else None
        if g is None or g.kind != "brep":
            return False
        hit = self._geometry_snap(x, y, surface_only=True)
        if hit is None:
            return False
        fi = BR.face_index_at(g, hit[0])
        if fi is None:
            return False
        self.doc.select_face(oid, fi)
        planar = BR.face_info(g, fi)[0]
        self.ix.message.emit(
            f"Face {fi + 1} selected. " + ("Drag the gumball to move it, Cmd/Ctrl-drag an arrow to push or pull it, "
                                            "or run ExtractSrf to copy it out." if planar else
                                            "Curved face: Cmd/Ctrl-drag an arrow to push or pull it, or ExtractSrf."))
        return True

    # ------------------------------------------------------------ snapping onto edges, curves and surfaces
    def _scene_geometry(self):
        """Edge segments (A, B) and surface triangles (V, T) of everything visible, cached per document version."""
        key = (getattr(self.doc, "version", 0), id(self.doc.isolated) if self.doc.isolated is not None else None,
               len(self.doc.objects))
        c = getattr(self, "_scene_cache", None)
        if c is not None and c[0] == key:
            return c[1]
        A, Bs, Vs, Ts, n = [], [], [], [], 0
        for owner, g, col, dim in self.doc.display("3d"):
            try:
                if g.kind == "brep":
                    from . import brep as BR
                    V, _, T, edges = BR.tessellate(g)[:4]
                    for pl in edges:
                        pl = np.asarray(pl, float)
                        if len(pl) > 1:
                            A.append(pl[:-1])
                            Bs.append(pl[1:])
                elif g.kind == "mesh":
                    V, T = g.V, g.triangles()
                    for a_, b_ in G.feature_edges(g):
                        A.append(np.asarray(a_, float)[None])
                        Bs.append(np.asarray(b_, float)[None])
                elif g.is_curve:
                    sm = g.sample()
                    if len(sm) > 1:
                        A.append(sm[:-1])
                        Bs.append(sm[1:])
                    continue
                else:
                    continue
                V = np.asarray(V, float)
                T = np.asarray(T, int).reshape(-1, 3)
                if len(T):
                    Vs.append(V)
                    Ts.append(T + n)
                    n += len(V)
            except Exception:
                continue
        segs = (np.vstack(A), np.vstack(Bs)) if A else (np.zeros((0, 3)), np.zeros((0, 3)))
        tris = (np.vstack(Vs), np.vstack(Ts)) if Vs else (np.zeros((0, 3)), np.zeros((0, 3), int))
        if len(tris[1]) > 400000:   # very heavy scenes: skip surface snapping to stay fast
            tris = (np.zeros((0, 3)), np.zeros((0, 3), int))
        self._scene_cache = (key, (segs, tris))
        return segs, tris

    def _geometry_snap(self, x, y, surface_only=False):
        """Closest point on an edge or curve near the cursor ("On edge"), else the point on the surface
        under the cursor ("On surface")."""
        (A, Bs), (V, T) = self._scene_geometry()
        o, d = self.ray(x, y)
        dn = d / (np.linalg.norm(d) or 1)
        # surface under the cursor (nearest hit along the view ray)
        surf, t_surf = None, np.inf
        if len(T):
            v0 = V[T[:, 0]]
            e1, e2 = V[T[:, 1]] - v0, V[T[:, 2]] - v0
            pv = np.cross(dn, e2)
            det = np.einsum("ij,ij->i", e1, pv)
            ok = np.abs(det) > 1e-12
            inv = np.where(ok, 1.0 / np.where(ok, det, 1), 0)
            tv = o - v0
            u = np.einsum("ij,ij->i", tv, pv) * inv
            qv = np.cross(tv, e1)
            v = (qv @ dn) * inv
            tt = np.einsum("ij,ij->i", qv, e2) * inv
            hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (tt > 0)
            if hit.any():
                j = np.where(hit)[0][np.argmin(tt[hit])]
                t_surf = float(tt[j])
                surf = (P(o + dn * t_surf), "On surface")
        # edges and curves near the cursor, skipping ones hidden behind a surface
        if len(A) and not surface_only:
            sa, sb = self.project(A), self.project(Bs)
            ab = sb - sa
            L2 = np.einsum("ij,ij->i", ab, ab)
            q = np.array([x, y])
            t = np.clip(np.einsum("ij,ij->i", q - sa, ab) / np.maximum(L2, 1e-12), 0, 1)
            dist = np.linalg.norm(sa + ab * t[:, None] - q, axis=1)
            dist = np.where(np.isfinite(dist), dist, np.inf)
            cand = np.where(dist < SNAP_PX * 0.6)[0]
            span = float(np.linalg.norm(V.max(0) - V.min(0))) if len(V) else 1.0
            for i in cand[np.argsort(dist[cand])][:12]:
                p0, uu = A[i], Bs[i] - A[i]
                w0 = p0 - o
                a_, b_ = np.dot(uu, uu), np.dot(uu, dn)
                d_, e_ = np.dot(uu, w0), np.dot(dn, w0)
                den = a_ - b_ * b_
                s_ = np.clip((b_ * e_ - d_) / den, 0, 1) if abs(den) > 1e-12 else t[i]
                pt = p0 + s_ * uu
                if np.dot(pt - o, dn) <= t_surf + max(span * 1e-3, 1e-6):
                    return (P(pt), "On edge")
        return surf

    # ------------------------------------------------------------ interaction
    def _setup_style(self):
        style = vtk.vtkInteractorStyleTrackballCamera()
        pv_iren = self.plotter.iren
        # Register the style with pyvista too. Otherwise pyvista re-applies its own default style
        # (left = rotate, right = zoom, no selection) whenever it resets, e.g. after a double-click.
        try:
            pv_iren.style = style
        except Exception:
            pv_iren.interactor.SetInteractorStyle(style)
        for side in ("left", "right"):
            try:
                pv_iren.untrack_click_position(side=side)   # pyvista's chart double-click toggle
            except Exception:
                pass
        pv_iren.interactor.SetInteractorStyle(style)
        self.style = style
        style.AddObserver("LeftButtonPressEvent", self._on_left_down)
        style.AddObserver("LeftButtonReleaseEvent", self._on_left_up)
        style.AddObserver("RightButtonPressEvent", self._on_right_down)
        style.AddObserver("RightButtonReleaseEvent", self._on_right_up)
        style.AddObserver("MouseMoveEvent", self._on_move)
        style.AddObserver("MouseWheelForwardEvent", lambda o, e: self._zoom(1.12))
        style.AddObserver("MouseWheelBackwardEvent", lambda o, e: self._zoom(1 / 1.12))
        for ev in ("KeyPressEvent", "CharEvent", "KeyReleaseEvent"):
            style.AddObserver(ev, lambda o, e: None)

    def _pos(self):
        return self.plotter.iren.interactor.GetEventPosition()

    def _qt_to_vtk(self, pos):
        w = self.plotter.interactor
        r = w.devicePixelRatioF()
        return pos.x() * r, (w.height() - pos.y()) * r

    def orbit_pivot(self):
        """Centre of the selected objects or points (orbiting turns around them), else None."""
        if self.preview_only or not getattr(self.settings, "orbit_selection", True):
            return None
        try:
            if self.doc.grip_sel or self.doc.selected("3d"):
                return np.asarray(self.doc.target_center(), float)
        except Exception:
            pass
        return None

    def _orbit(self, dx, dy, pivot=None):
        """Turntable orbit keeping world Z up. Turns around 'pivot' (the selection) when given,
        otherwise around the view centre. The view does not jump: camera and target both turn."""
        cam = self.ren.GetActiveCamera()
        fp = np.array(cam.GetFocalPoint())
        pos = np.array(cam.GetPosition())
        c = fp if pivot is None else np.asarray(pivot, float)
        # azimuth about the vertical axis through the pivot
        a = math.radians(-dx * 0.35)
        ca, sa = math.cos(a), math.sin(a)
        Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1.0]])
        pos = c + Rz @ (pos - c)
        fp = c + Rz @ (fp - c)
        # elevation about the camera's horizontal axis through the pivot, clamped short of straight up/down
        d = fp - pos
        r = np.linalg.norm(d) or 1e-9
        el = math.asin(max(-1.0, min(1.0, -d[2] / r)))
        new_el = max(-math.radians(89), min(math.radians(89), el + math.radians(dy * 0.35)))
        da = new_el - el
        right = np.cross(d, (0, 0, 1.0))
        if np.linalg.norm(right) > 1e-12 and abs(da) > 1e-12:
            right /= np.linalg.norm(right)
            K = np.array([[0, -right[2], right[1]], [right[2], 0, -right[0]], [-right[1], right[0], 0]])
            for sign in (1.0, -1.0):
                t = sign * da
                R = np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * (K @ K)
                p2 = c + R @ (pos - c)
                f2 = c + R @ (fp - c)
                d2 = f2 - p2
                e2 = math.asin(max(-1.0, min(1.0, -d2[2] / (np.linalg.norm(d2) or 1e-9))))
                if abs(e2 - new_el) < 1e-6 or sign < 0:
                    pos, fp = p2, f2
                    break
        if cam.GetParallelProjection() and pivot is None:
            cam.SetParallelProjection(False)
        cam.SetPosition(*pos)
        cam.SetFocalPoint(*fp)
        cam.SetViewUp(0, 0, 1)
        self.ren.ResetCameraClippingRange()
        if self.gumball:
            self.gumball.refresh()
        self.plotter.render()

    def _pan_pixels(self, dx, dy):
        cam = self.ren.GetActiveCamera()
        fp = np.array(cam.GetFocalPoint())
        self.ren.SetWorldPoint(*fp, 1.0)
        self.ren.WorldToDisplay()
        fx, fy, fz = self.ren.GetDisplayPoint()
        r = self.plotter.interactor.devicePixelRatioF()
        self.ren.SetDisplayPoint(fx - dx * r, fy + dy * r, fz)
        self.ren.DisplayToWorld()
        w = np.array(self.ren.GetWorldPoint())
        d = w[:3] / (w[3] or 1) - fp
        cam.SetFocalPoint(*(fp + d))
        cam.SetPosition(*(np.array(cam.GetPosition()) + d))
        self.ren.ResetCameraClippingRange()
        self.plotter.render()

    def _zoom(self, f, x=None, y=None):
        """Zoom toward the cursor."""
        if x is None:
            x, y = self._pos()
        cam = self.ren.GetActiveCamera()
        target = self.cplane_point(x, y)
        if cam.GetParallelProjection():
            cam.SetParallelScale(cam.GetParallelScale() / f)
            fp = np.array(cam.GetFocalPoint())
            pos = np.array(cam.GetPosition())
            shift = (target - fp) * (1 - 1 / f)
            cam.SetFocalPoint(*(fp + shift))
            cam.SetPosition(*(pos + shift))
        else:
            pos = np.array(cam.GetPosition())
            fp = np.array(cam.GetFocalPoint())
            dist = np.linalg.norm(pos - fp)
            to_t = target - pos
            if np.linalg.norm(to_t) > 1e9 or not np.isfinite(to_t).all():
                cam.Dolly(f)
            else:
                move = to_t * (1 - 1 / f)
                cam.SetPosition(*(pos + move))
                cam.SetFocalPoint(*(fp + move * (1 if dist > 1e-9 else 0)))
        self.ren.ResetCameraClippingRange()
        self.plotter.render()

    def _on_left_down(self, obj, ev):
        x, y = self._pos()
        if self.viewcube.contains(x, y):
            d = self.viewcube.direction_at(x, y)
            if d is not None:
                self.look_from(d)
            self._press = None
            self._cube_click = True
            return
        self._cube_click = False
        mods = getattr(self, "_qt_mods", QtCore.Qt.NoModifier)
        if not self.preview_only and not self.ix.active and self.gumball is not None:
            h = self.gumball.pick(x, y)
            if h:
                self._alt_orbit = False
                self._gb_mods = mods
                self.gumball.start(h, x, y)
                self._gb_drag = True
                self._press = None
                return
        subobj = bool(mods & QtCore.Qt.ShiftModifier) and bool(mods & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier))
        if self.preview_only or (bool(mods & QtCore.Qt.AltModifier) and not subobj):
            self._alt_orbit = True   # Option/Alt-drag orbits (handy on a trackpad)
            self._orbit_last = (x, y)
            self._press = None
            return
        self._alt_orbit = False
        self.setFocus()
        self.plotter.interactor.setFocus()
        if not self.ix.active and self.gumball is not None:
            h = self.gumball.pick(x, y)
            if h:
                self.gumball.start(h, x, y)
                self._gb_drag = True
                self._press = None
                return
        if not self.ix.wants_point() and self.doc.points_on:
            g = self._grip_at(x, y)
            if g is not None:
                shift = self.plotter.iren.interactor.GetShiftKey()
                if shift:
                    self.doc.toggle_grip(g)
                elif g not in self.doc.grip_sel:
                    self.doc.select_grips([g])
                pt = dict(((o, i), p) for o, i, p in self.doc.grip_points()).get(g)
                o, d = self.ray(x, y)
                n = np.array([0, 0, 1.0]) if abs(d[2]) > 0.2 * np.linalg.norm(d) else -d / np.linalg.norm(d)
                self._grip_drag = {"p": pt, "n": n, "x0": x, "y0": y, "moved": False, "M": np.eye(4)}
                self._press = None
                return
        self._press = self._pos()

    def _on_left_up(self, obj, ev):
        if getattr(self, "_cube_click", False):
            self._cube_click = False
            return
        if getattr(self, "_alt_orbit", False):
            self._alt_orbit = False
            return
        if self.preview_only:
            return
        if self._gb_drag:
            self._gb_drag = False
            st = self.gumball.finish()
            self._clear_drag_preview()
            if st["moved"]:
                mods = getattr(self, "_qt_mods", QtCore.Qt.NoModifier) | getattr(self, "_gb_mods", QtCore.Qt.NoModifier)
                extrude = bool(mods & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier))
                copy = bool(mods & QtCore.Qt.AltModifier)
                if st["kind"] == "extrude":
                    extrude = True
                if extrude and self.doc.face_sel and st["kind"] in ("move", "plane", "free", "extrude"):
                    try:
                        self.doc.push_pull_face(st["M"][:3, 3])
                        self.ix.message.emit("Face pushed/pulled.")
                    except Exception as e:
                        self.ix.message.emit(f"Push/pull failed: {e}")
                elif extrude and st["kind"] in ("move", "plane", "free", "extrude") and not self.doc.grip_sel:
                    made, skipped = self.doc.extrude_targets(st["M"][:3, 3], "Gumball extrude")
                    msg = f"Extruded {made} object(s)."
                    if skipped:
                        msg += f" {skipped} could not be extruded."
                    self.ix.message.emit(msg)
                elif copy:
                    n = self.doc.transform_targets(st["M"], "Gumball copy", copy=True)
                    self.ix.message.emit(f"Copied {n} object(s).")
                else:
                    self.doc.transform_targets(st["M"], "Gumball " + st["kind"])
            elif st["kind"] in ("move", "rotate", "scale"):
                self.ix.start("GumballValue", "3d", kind=st["kind"], axis=tuple(st["a"]), center=tuple(st["c"]))
            self.plotter.render()
            return
        if self._grip_drag is not None:
            st, self._grip_drag = self._grip_drag, None
            self._clear_drag_preview()
            if st["moved"]:
                self.doc.transform_targets(st["M"], "Move points")
            self.plotter.render()
            return
        if self._press is None:
            return
        x0, y0 = self._press
        x1, y1 = self._pos()
        self._press = None
        self._clear_rubber()
        iren = self.plotter.iren.interactor
        shift, ctrl = iren.GetShiftKey(), iren.GetControlKey()
        if self.ix.wants_window() and abs(x1 - x0) + abs(y1 - y0) > 5:
            lx, hx, ly, hy = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)

            def inside(p, lx=lx, hx=hx, ly=ly, hy=hy):
                q = self.project(np.asarray(p, float).reshape(1, 3))[0]
                return bool(np.isfinite(q).all() and lx <= q[0] <= hx and ly <= q[1] <= hy)
            self.ix.window_pick(inside, x1 < x0)
            return
        if self.ix.wants_point():
            self.ix.click_point(self.resolve(x1, y1))
            return
        if abs(x1 - x0) + abs(y1 - y0) > 5 and self.doc.points_on and not self.ix.wants_objects():
            items = self._grips_in_rect(x0, y0, x1, y1)
            if items or not shift:
                if ctrl:
                    self.doc.grip_sel -= set(items)
                    self.doc.notify("grips")
                else:
                    self.doc.select_grips(items, add=shift)
                self.ix.changed.emit()
                return
        if abs(x1 - x0) + abs(y1 - y0) <= 5 and self.doc.grip_sel and not shift and not self.ix.active:
            self.doc.clear_grips()
        if abs(x1 - x0) + abs(y1 - y0) > 5:
            ids = self._window_select(x0, y0, x1, y1)
            if ctrl:
                self.doc.selection -= set(ids)
                self.doc.notify("selection")
            else:
                self.doc.select(ids, add=shift or self.ix.wants_objects())
        else:
            qm = getattr(self, "_qt_mods", QtCore.Qt.NoModifier)
            if (qm & QtCore.Qt.ShiftModifier) and (qm & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier)) \
                    and not self.ix.active:
                picked = self._pick_edge(x1, y1) if (qm & QtCore.Qt.AltModifier) else self._pick_face(x1, y1)
                if picked:
                    self.ix.changed.emit()
                    return
            oid = self.pick(x1, y1)
            if oid is None:
                if not shift and not ctrl and not self.ix.wants_objects():
                    self.doc.clear_selection()
                    if not self.ix.active and not getattr(View3D, "_hinted", False):
                        View3D._hinted = True
                        self.ix.message.emit("Tip: pick a tool (Line, Box...) or type a command, then click in the view. "
                                             "Trackpad: two-finger swipe orbits, Shift+swipe pans, pinch zooms; Option-drag orbits.")
            elif ctrl:
                self.doc.selection.discard(oid)
                self.doc.notify("selection")
            elif shift:
                self.doc.toggle_select(oid)
            else:
                self.doc.select([oid], add=self.ix.wants_objects())
            if oid is not None:
                self.ix.object_clicked_in_select_mode()
        self.ix.changed.emit()

    def _on_right_down(self, obj, ev):
        self._rpress = self._pos()
        self._rmoved = False
        iren = self.plotter.iren.interactor
        if iren.GetShiftKey():
            self.style.OnMiddleButtonDown()
            self._rmode = "pan"
        else:
            self._rmode = "rotate"
            self._rlast = self._rpress
            self._rpivot = self.orbit_pivot()

    def _on_right_up(self, obj, ev):
        if self._rmode == "pan":
            self.style.OnMiddleButtonUp()
        if not self._rmoved and not self.preview_only:
            self.ix.submit_empty()
        self._rpress = None

    def _on_move(self, obj, ev):
        x, y = self._pos()
        if self._rpress is not None:
            if abs(x - self._rpress[0]) + abs(y - self._rpress[1]) > 3:
                self._rmoved = True
            if self._rmode == "rotate":
                lx, ly = self._rlast
                self._rlast = (x, y)
                r = self.plotter.interactor.devicePixelRatioF()
                if self._rmoved:
                    self._orbit((x - lx) / r, -(y - ly) / r, self._rpivot)
                return
            self.style.OnMouseMove()
            return
        self.style.OnMouseMove()  # handles middle-button pan
        if self.preview_only:
            return
        if self._gb_drag:
            shift = self.plotter.iren.interactor.GetShiftKey()
            M = self.gumball.update(x, y, shift)
            if M is not None:
                mods = getattr(self, "_qt_mods", QtCore.Qt.NoModifier) | getattr(self, "_gb_mods", QtCore.Qt.NoModifier)
                tag = ("  extrude" if mods & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier)
                       else "  copy" if mods & QtCore.Qt.AltModifier else "")
                self._show_drag_preview(M, self.gumball.drag.get("label", "") + tag)
            return
        if self._grip_drag is not None:
            st = self._grip_drag
            if abs(x - st["x0"]) + abs(y - st["y0"]) > 3:
                st["moved"] = True
            if st["moved"] and st["p"] is not None:
                o, d = self.ray(x, y)
                den = np.dot(d, st["n"])
                if abs(den) > 1e-12:
                    hit = o + d * (np.dot(st["p"] - o, st["n"]) / den)
                    target = hit
                    if self.settings.osnap:
                        sp = self.doc.snap_points("3d")
                        if sp:
                            pts = np.array([q for q, _ in sp])
                            scr = self.project(pts)
                            dd = np.hypot(scr[:, 0] - x, scr[:, 1] - y)
                            i = int(np.nanargmin(dd)) if np.isfinite(dd).any() else -1
                            if i >= 0 and dd[i] < 12 and np.linalg.norm(pts[i] - st["p"]) > 1e-9:
                                target = pts[i]
                    if target is hit and self.settings.snap:
                        target = self.ix.grid_snap(hit)
                        target[2] = hit[2]
                    st["M"] = G.translation(target - st["p"])
                    self._show_drag_preview(st["M"])
            return
        if not self.ix.active and self.gumball is not None and self.gumball.visible and self._press is None:
            if self.gumball.set_hover(self.gumball.pick(x, y)):
                self.plotter.render()
        if self._press is not None and (not self.ix.wants_point() or self.ix.wants_window()):
            self._draw_rubber(self._press, (x, y))
            return
        self._move_xy = (x, y)
        if not self._move_pending:  # coalesce bursts of mouse events into one update per frame
            self._move_pending = True
            QtCore.QTimer.singleShot(0, self._process_move)

    def _process_move(self):
        self._move_pending = False
        x, y = self._move_xy
        p = self.resolve(x, y)
        self.cursor_world = p
        self.ix.cursor = p
        self.cursor_moved.emit(p, self.snap[1] if self.snap else "")
        if self.ix.active:
            self.update_preview()
        elif self.snap_actor is not None:
            self._update_snap_marker()
            self.plotter.render()

    def _draw_rubber(self, a, b):
        self._clear_rubber(render=False)
        pts = vtk.vtkPoints()
        for p in ((a[0], a[1]), (b[0], a[1]), (b[0], b[1]), (a[0], b[1]), (a[0], a[1])):
            pts.InsertNextPoint(p[0], p[1], 0)
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(5)
        for i in range(5):
            line.GetPointIds().SetId(i, i)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(line)
        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetLines(cells)
        m = vtk.vtkPolyDataMapper2D()
        m.SetInputData(pd)
        act = vtk.vtkActor2D()
        act.SetMapper(m)
        crossing = b[0] < a[0]
        col = pv.Color(self.settings.color("preview" if crossing else "accent")).float_rgb
        act.GetProperty().SetColor(col)
        act.GetProperty().SetLineWidth(1.5)
        if crossing:
            act.GetProperty().SetLineStipplePattern(0xF0F0) if hasattr(act.GetProperty(), "SetLineStipplePattern") else None
        self.ren.AddActor2D(act)
        self._rb_actor = act
        self.plotter.render()

    def _clear_rubber(self, render=True):
        if self._rb_actor is not None:
            self.ren.RemoveActor2D(self._rb_actor)
            self._rb_actor = None
            if render:
                self.plotter.render()

    def pick(self, x, y):
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.006)
        picker.Pick(x, y, 0, self.ren)
        act = picker.GetActor()
        filt = self.ix.selection_filter()
        oid = None
        m = self.batch_by_actor.get(act)
        cid = picker.GetCellId()
        if m is not None and 0 <= cid < len(m["cown"]):
            oid = self._owner_list[int(m["cown"][cid])]
        if oid is not None:
            g = self.doc.get(oid)
            sel_ok = {x.id for x in self.doc.selectable("3d")}
            if g is not None and oid in sel_ok and (filt is None or filt(g)):
                return oid
        # fallback: nearest curve/point in screen space
        best, bd = None, 8.0
        sel_ok = self.doc.selectable("3d")
        for g in sel_ok:
            if filt and not filt(g):
                continue
            for part in self.doc.expand(g):
                if part.kind in ("mesh", "brep"):
                    continue
                scr = self.project(part.sample())
                if len(scr) == 1:
                    d = float(np.hypot(scr[0, 0] - x, scr[0, 1] - y))
                else:
                    a, b = scr[:-1], scr[1:]
                    ab = b - a
                    t = np.clip(np.einsum("ij,ij->i", np.array([x, y]) - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0, 1)
                    proj = a + ab * t[:, None]
                    dd = np.hypot(proj[:, 0] - x, proj[:, 1] - y)
                    d = float(np.nanmin(dd)) if np.isfinite(dd).any() else 1e9
                if d < bd:
                    best, bd = g.id, d
        return best

    def _window_select(self, x0, y0, x1, y1):
        lo = (min(x0, x1), min(y0, y1))
        hi = (max(x0, x1), max(y0, y1))
        crossing = x1 < x0
        filt = self.ix.selection_filter()
        out = []
        for g in self.doc.selectable("3d"):
            if filt and not filt(g):
                continue
            pts = []
            for part in self.doc.expand(g):
                pts.append(part.sample())
            if not pts:
                continue
            scr = self.project(np.vstack(pts))
            inside = (scr[:, 0] >= lo[0]) & (scr[:, 0] <= hi[0]) & (scr[:, 1] >= lo[1]) & (scr[:, 1] <= hi[1])
            if (crossing and inside.any()) or (not crossing and inside.all()):
                out.append(g.id)
        return out

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseMove, QtCore.QEvent.MouseButtonRelease):
            self._qt_mods = ev.modifiers()
            if t == QtCore.QEvent.MouseMove and getattr(self, "_alt_orbit", False):
                x, y = self._qt_to_vtk(ev.position())
                lx, ly = getattr(self, "_orbit_last", (x, y))
                self._orbit_last = (x, y)
                r = self.plotter.interactor.devicePixelRatioF()
                self._orbit((x - lx) / r, -(y - ly) / r, self.orbit_pivot())
                return True
        if t == QtCore.QEvent.Wheel:
            pd, ad = ev.pixelDelta(), ev.angleDelta()
            mods = ev.modifiers()
            trackpad = not pd.isNull() and ev.phase() != QtCore.Qt.NoScrollPhase if hasattr(ev, "phase") else not pd.isNull()
            if trackpad and not (mods & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier)):
                if mods & QtCore.Qt.ShiftModifier:
                    self._pan_pixels(pd.x(), pd.y())
                else:
                    self._orbit(pd.x(), pd.y(), self.orbit_pivot())
                return True
            x, y = self._qt_to_vtk(ev.position())
            steps = (ad.y() or ad.x()) / 120.0 if not ad.isNull() else pd.y() / 50.0
            if steps:
                self._zoom(1.12 ** steps, x, y)
            return True
        if t == QtCore.QEvent.NativeGesture:
            if ev.gestureType() == QtCore.Qt.ZoomNativeGesture:
                x, y = self._qt_to_vtk(ev.position())
                self._zoom(1.0 + ev.value(), x, y)
                return True
            if ev.gestureType() == QtCore.Qt.SmartZoomNativeGesture:
                self.zoom_extents()
                return True
        if t in (QtCore.QEvent.DragEnter, QtCore.QEvent.DragMove) and not self.preview_only:
            if ev.mimeData().hasText() and ev.mimeData().text().startswith("forma-component:"):
                ev.acceptProposedAction()
                return True
        if t == QtCore.QEvent.Drop and not self.preview_only:
            txt = ev.mimeData().text() if ev.mimeData().hasText() else ""
            if txt.startswith("forma-component:"):
                x, y = self._qt_to_vtk(ev.position())
                self.component_dropped.emit(txt.split(":", 1)[1], self.ix.grid_snap(self.cplane_point(x, y)))
                ev.acceptProposedAction()
                return True
        if ev.type() == QtCore.QEvent.KeyPress and not self.preview_only:
            k = ev.key()
            typing = bool(self.cmdline and self.cmdline.edit.text())
            if (k == QtCore.Qt.Key_Delete or (k == QtCore.Qt.Key_Backspace and not typing)) and not self.ix.active:
                try:
                    msg = self.doc.delete_subobjects()
                except Exception as e:
                    msg = f"Could not delete: {e}"
                if msg:
                    self.ix.message.emit(msg)
                    return True
                if self.doc.selected("3d"):
                    self.ix.start("Delete", "3d")
                return True
            if k == QtCore.Qt.Key_Escape and not self.ix.active:
                if self.doc.grip_sel:
                    self.doc.clear_grips()
                else:
                    self.doc.clear_selection()
                return True
            if ev.modifiers() & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier):
                return False  # let app shortcuts work
            if self.cmdline and self.cmdline.forward_key(ev):
                return True
            return True  # swallow VTK default keys (e.g. 'q' would close)
        if ev.type() == QtCore.QEvent.MouseButtonDblClick and not self.preview_only:
            x, y = self._pos()
            oid = self.pick(x, y)
            if oid and self.doc.objects[oid].kind == "instance" and not self.ix.active:
                self.doc.select([oid])
                self.ix.start("EditComponent", "3d")
                return True
        return False

    # ------------------------------------------------------------ cameras
    def _bounds(self, selected=False):
        if self.source is not None:
            items = self.source()
        else:
            items = [(o, g, c, d) for (o, g, c, d) in self.doc.display("3d") if not selected or o in self.doc.selection]
        pts = []
        for _, g, _, _ in items:
            lo, hi = g.bbox()
            pts += [lo, hi]
        for a in self.extra_actors if not selected else []:
            b = a.GetBounds()
            if np.isfinite(b).all() and b[1] >= b[0]:
                pts += [(b[0], b[2], b[4]), (b[1], b[3], b[5])]
        if not pts:
            s = self.settings.grid_spacing * 10
            return (-s, s, -s, s, 0, s / 2)
        a = np.array(pts)
        lo, hi = a.min(0), a.max(0)
        pad = max((hi - lo).max() * 0.05, 1e-3)
        return (lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad, lo[2] - pad, hi[2] + pad)

    def zoom_extents(self, selected=False):
        self.ren.ResetCamera(self._bounds(selected))
        self.ren.ResetCameraClippingRange()
        self.plotter.render()

    def _set_view(self, direction, up, parallel):
        cam = self.ren.GetActiveCamera()
        cam.SetParallelProjection(parallel)
        cam.SetFocalPoint(0, 0, 0)
        cam.SetPosition(*direction)
        cam.SetViewUp(*up)
        self.zoom_extents()

    def look_from(self, direction, up=None, animate=True):
        """Orbit the camera to look from a direction (used by the ViewCube); animated, ~180 ms."""
        d = np.asarray(direction, float)
        d /= np.linalg.norm(d)
        cam = self.ren.GetActiveCamera()
        fp = np.array(cam.GetFocalPoint())
        dist = np.linalg.norm(np.array(cam.GetPosition()) - fp)
        if up is None:
            up = np.array([0, 1.0, 0]) if abs(d[2]) > 0.99 else np.array([0, 0, 1.0])
            if abs(d[2]) > 0.99 and d[2] < 0:
                up = np.array([0, -1.0, 0])
        ortho_face = int(np.count_nonzero(np.abs(direction) > 0.5)) == 1
        start_dir = np.array(cam.GetPosition()) - fp
        start_dir /= np.linalg.norm(start_dir) or 1
        start_up = np.array(cam.GetViewUp())
        steps = 10 if animate else 1
        if self._anim is not None:
            self._anim.stop()
        state = {"i": 0}

        def step():
            state["i"] += 1
            t = state["i"] / steps
            t = t * t * (3 - 2 * t)
            v = start_dir * (1 - t) + d * t
            if np.linalg.norm(v) < 1e-6:
                v = d
            v /= np.linalg.norm(v)
            u = start_up * (1 - t) + up * t
            u = u - np.dot(u, v) * v
            if np.linalg.norm(u) < 1e-6:
                u = up
            cam.SetPosition(*(fp + v * dist))
            cam.SetViewUp(*(u / np.linalg.norm(u)))
            if state["i"] >= steps:
                self._anim.stop()
                cam.SetParallelProjection(ortho_face)
                if ortho_face:
                    cam.SetParallelScale(dist * math.tan(math.radians(cam.GetViewAngle() / 2)))
            self.ren.ResetCameraClippingRange()
            self.plotter.render()

        self._anim = QtCore.QTimer(self)
        self._anim.setInterval(16)
        self._anim.timeout.connect(step)
        self._anim.start()

    def view_top(self):
        self._set_view((0, 0, 1), (0, 1, 0), True)

    def view_front(self):
        self._set_view((0, -1, 0), (0, 0, 1), True)

    def view_right(self):
        self._set_view((1, 0, 0), (0, 0, 1), True)

    def view_persp(self, reset=False):
        self._set_view((1, -1.3, 0.9), (0, 0, 1), False)

    ISO = {"SE": (1, -1, 1), "SW": (-1, -1, 1), "NE": (1, 1, 1), "NW": (-1, 1, 1)}

    def view_iso(self, corner="SE"):
        """True isometric (parallel, 35.26 degree elevation) from a corner."""
        self._set_view(self.ISO.get(corner, (1, -1, 1)), (0, 0, 1), True)

    def view_two_point(self):
        """Eye-level perspective with vertical lines kept vertical."""
        cam = self.ren.GetActiveCamera()
        d = np.array(cam.GetPosition()) - np.array(cam.GetFocalPoint())
        d[2] = abs(np.linalg.norm(d[:2])) * 0.08 + 1e-9
        self._set_view(tuple(d), (0, 0, 1), False)

    def set_parallel(self, on):
        cam = self.ren.GetActiveCamera()
        if bool(cam.GetParallelProjection()) == bool(on):
            return
        fp = np.array(cam.GetFocalPoint())
        dist = np.linalg.norm(np.array(cam.GetPosition()) - fp)
        if on:
            cam.SetParallelScale(dist * math.tan(math.radians(cam.GetViewAngle() / 2)))
        cam.SetParallelProjection(bool(on))
        self.ren.ResetCameraClippingRange()
        self.plotter.render()

    def set_lens(self, degrees):
        """Field of view for perspective (wide 60, normal 35, telephoto 15)."""
        cam = self.ren.GetActiveCamera()
        cam.SetParallelProjection(False)
        cam.SetViewAngle(float(degrees))
        self.ren.ResetCameraClippingRange()
        self.plotter.render()

    def set_shade(self, mode):
        self.shade = mode
        self.rebuild()

    def screenshot(self, path):
        self.plotter.render()
        self.plotter.screenshot(path)

    def closeEvent(self, ev):
        self.plotter.close()
        super().closeEvent(ev)
