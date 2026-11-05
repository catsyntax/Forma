"""3D gumball and control-point (grip) interaction for the Model viewport.

The gumball is drawn in an overlay renderer that shares the main camera and is always on top.
Handles: X/Y/Z arrows (move along axis), plane squares (move in plane), arcs (rotate),
end cubes (scale along axis; Shift = uniform). A click on a handle without dragging asks
for an exact value in the command line.
"""
from __future__ import annotations

import math

import numpy as np
import pyvista as pv
import vtk

from . import geometry as G

AXES = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}
COLORS = {"x": "#e5534b", "y": "#3fb37f", "z": "#4c8dff"}
HOVER = "#ffd23f"
PX_SIZE = 95      # gumball arm length in screen pixels
GRIP_PX = 9       # grip pick radius in pixels


def _arc(axis, r=0.62, a0=10, a1=80, n=24):
    u, v = {"x": ((0, 1, 0), (0, 0, 1)), "y": ((0, 0, 1), (1, 0, 0)), "z": ((1, 0, 0), (0, 1, 0))}[axis]
    u, v = np.array(u, float), np.array(v, float)
    t = np.radians(np.linspace(a0, a1, n))
    pts = np.outer(np.cos(t), u) * r + np.outer(np.sin(t), v) * r
    return pv.lines_from_points(pts).tube(radius=0.018, n_sides=8)


def _plane_square(axis, off=0.28, s=0.13):
    u, v = {"x": ((0, 1, 0), (0, 0, 1)), "y": ((0, 0, 1), (1, 0, 0)), "z": ((1, 0, 0), (0, 1, 0))}[axis]
    u, v = np.array(u, float), np.array(v, float)
    c = (u + v) * off
    pts = [c - u * s / 2 - v * s / 2, c + u * s / 2 - v * s / 2, c + u * s / 2 + v * s / 2, c - u * s / 2 + v * s / 2]
    return pv.PolyData(np.array(pts), faces=[4, 0, 1, 2, 3])


class Gumball3D:
    def __init__(self, view):
        self.view = view
        rw = view.plotter.ren_win
        rw.SetNumberOfLayers(max(3, rw.GetNumberOfLayers()))
        self.ren = vtk.vtkRenderer()
        self.ren.SetLayer(2)
        self.ren.InteractiveOff()
        # Own camera, copied from the main one before every draw. Sharing the camera let this
        # overlay shrink the clipping range to the gumball and cut the scene off.
        self.cam = vtk.vtkCamera()
        self.ren.SetActiveCamera(self.cam)
        self.ren.AddObserver("StartEvent", lambda *_: self.sync_camera())
        self.ren.SetBackgroundAlpha(0.0)
        light = vtk.vtkLight()
        light.SetLightTypeToHeadlight()
        self.ren.AddLight(light)
        rw.AddRenderer(self.ren)
        self.handles = {}   # name -> actor
        self.kind = {}      # name -> (kind, axis)
        self._build()
        self.center = None
        self.size = 1.0
        self.visible = False
        self.hover = None
        self.drag = None
        self.preview_actors = []
        self.grip_actors = []
        view.ren.AddObserver("StartEvent", lambda o, e: self._resize())

    # ------------------------------------------------------------ geometry
    def _add(self, name, mesh, color, kind, axis):
        m = vtk.vtkPolyDataMapper()
        m.SetInputData(mesh)
        a = vtk.vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(pv.Color(color).float_rgb)
        a.GetProperty().SetAmbient(0.5)
        a.GetProperty().SetDiffuse(0.6)
        a.VisibilityOff()
        a._base = color
        self.ren.AddActor(a)
        self.handles[name] = a
        self.kind[name] = (kind, axis)

    def _build(self):
        for ax, d in AXES.items():
            arrow = pv.Arrow(start=(0, 0, 0), direction=d, tip_length=0.22, tip_radius=0.06, shaft_radius=0.018, scale=1.0)
            self._add("move_" + ax, arrow, COLORS[ax], "move", ax)
            self._add("scale_" + ax, pv.Cube(center=d * 1.12, x_length=0.09, y_length=0.09, z_length=0.09), COLORS[ax], "scale", ax)
            # extrude handle: a small ball half-way along the arrow (curves -> surfaces, surfaces -> solids,
            # solids -> push/pull of the face pointing that way)
            self._add("extrude_" + ax, pv.Sphere(radius=0.052, center=d * 0.6), COLORS[ax], "extrude", ax)
            self._add("rot_" + ax, _arc(ax), COLORS[ax], "rotate", ax)
            self._add("plane_" + ax, _plane_square(ax), COLORS[ax], "plane", ax)
        self._add("center", pv.Sphere(radius=0.06), "#dddddd", "free", "z")

    # ------------------------------------------------------------ state
    def refresh(self):
        v = self.view
        st = v.settings
        doc = v.doc
        show = (st.gumball and not v.preview_only and not v.ix.active and self.drag is None
                and (bool(doc.grip_sel) or bool(doc.selected("3d"))))
        self.center = doc.target_center() if show else None
        self.visible = self.center is not None
        for a in self.handles.values():
            a.SetVisibility(self.visible)
        self._resize()

    def _resize(self):
        if not self.visible or self.center is None:
            return
        v = self.view
        c = self.center
        ren = v.ren
        ren.SetWorldPoint(*c, 1.0)
        ren.WorldToDisplay()
        dx, dy, dz = ren.GetDisplayPoint()
        r = v.plotter.interactor.devicePixelRatioF() if hasattr(v.plotter.interactor, "devicePixelRatioF") else 1.0
        ren.SetDisplayPoint(dx + PX_SIZE * r, dy, dz)
        ren.DisplayToWorld()
        w = np.array(ren.GetWorldPoint())
        w = w[:3] / (w[3] or 1)
        self.size = float(np.linalg.norm(w - c)) or 1.0
        M = vtk.vtkMatrix4x4()
        for i in range(3):
            M.SetElement(i, i, self.size)
            M.SetElement(i, 3, float(c[i]))
        for a in self.handles.values():
            a.SetUserMatrix(M)
        self.sync_camera()

    def sync_camera(self):
        main = self.view.ren.GetActiveCamera()
        self.cam.SetPosition(main.GetPosition())
        self.cam.SetFocalPoint(main.GetFocalPoint())
        self.cam.SetViewUp(main.GetViewUp())
        self.cam.SetViewAngle(main.GetViewAngle())
        self.cam.SetParallelProjection(main.GetParallelProjection())
        self.cam.SetParallelScale(main.GetParallelScale())
        self.ren.ResetCameraClippingRange()

    def pick(self, x, y):
        if not self.visible:
            return None
        self.sync_camera()
        p = vtk.vtkCellPicker()
        p.SetTolerance(0.004)
        p.PickFromListOn()
        for a in self.handles.values():
            p.AddPickList(a)
        if not p.Pick(x, y, 0, self.ren):
            return None
        act = p.GetActor()
        for n, a in self.handles.items():
            if a is act:
                return n
        return None

    def set_hover(self, name):
        if name == self.hover:
            return False
        if self.hover in self.handles:
            a = self.handles[self.hover]
            a.GetProperty().SetColor(pv.Color(a._base).float_rgb)
        self.hover = name
        if name in self.handles:
            self.handles[name].GetProperty().SetColor(pv.Color(HOVER).float_rgb)
        return True

    # ------------------------------------------------------------ dragging
    def _ray(self, x, y):
        return self.view.ray(x, y)

    @staticmethod
    def _closest_on_axis(o, d, c, a):
        """Parameter t along axis line (c + t a) closest to the ray (o + s d)."""
        w0 = c - o
        b = np.dot(a, d)
        dd = np.dot(d, d)
        den = dd - b * b
        if abs(den) < 1e-12:
            return None
        return (b * np.dot(d, w0) - dd * np.dot(a, w0)) / den

    @staticmethod
    def _hit_plane(o, d, c, n):
        den = np.dot(d, n)
        if abs(den) < 1e-12:
            return None
        return o + d * (np.dot(c - o, n) / den)

    def start(self, name, x, y):
        kind, ax = self.kind[name]
        a = AXES[ax]
        o, d = self._ray(x, y)
        c = self.center.copy()
        st = {"name": name, "kind": kind, "axis": ax, "a": a, "c": c, "x0": x, "y0": y, "M": np.eye(4), "moved": False}
        if kind in ("move", "scale", "extrude"):
            st["t0"] = self._closest_on_axis(o, d, c, a)
        elif kind in ("plane", "rotate"):
            st["p0"] = self._hit_plane(o, d, c, a)
        elif kind == "free":
            cam = self.view.ren.GetActiveCamera()
            st["n"] = np.array(cam.GetDirectionOfProjection())
            st["p0"] = self._hit_plane(o, d, c, st["n"])
        self.drag = st
        for h, act in self.handles.items():
            act.SetVisibility(h == name)

    def update(self, x, y, shift=False):
        st = self.drag
        if st is None:
            return None
        if abs(x - st["x0"]) + abs(y - st["y0"]) > 3:
            st["moved"] = True
        o, d = self._ray(x, y)
        a, c, kind = st["a"], st["c"], st["kind"]
        settings = self.view.settings
        snap = settings.snap
        grid = settings.grid_spacing / 10 if snap else 0
        M = np.eye(4)
        if kind in ("move", "extrude") and st.get("t0") is not None:
            t = self._closest_on_axis(o, d, c, a)
            if t is not None:
                dist = t - st["t0"]
                if grid:
                    dist = round(dist / grid) * grid
                M = G.translation(a * dist)
                st["label"] = f"{dist:.4g}" + ("  extrude" if kind == "extrude" else "")
        elif kind in ("plane", "free") and st.get("p0") is not None:
            n = a if kind == "plane" else st["n"]
            p = self._hit_plane(o, d, c, n)
            if p is not None:
                dv = p - st["p0"]
                if grid:
                    dv = np.round(dv / grid) * grid
                M = G.translation(dv)
                st["label"] = ", ".join(f"{v:.4g}" for v in dv)
        elif kind == "rotate" and st.get("p0") is not None:
            p = self._hit_plane(o, d, c, a)
            if p is not None:
                v0, v1 = st["p0"] - c, p - c
                ang = math.degrees(math.atan2(np.dot(np.cross(v0, v1), a), np.dot(v0, v1)))
                step = 15 if shift else (5 if snap else 0)
                if step:
                    ang = round(ang / step) * step
                M = G.rotation(ang, a, c)
                st["label"] = f"{ang:.1f}°"
        elif kind == "scale" and st.get("t0"):
            t = self._closest_on_axis(o, d, c, a)
            if t is not None and abs(st["t0"]) > 1e-12:
                f = t / st["t0"]
                if snap:
                    f = round(f * 20) / 20 or (0.05 if f >= 0 else -0.05)
                if abs(f) < 1e-3:          # never flatten to zero (passing through the centre mirrors)
                    f = 1e-3 if f >= 0 else -1e-3
                f = float(np.clip(f, -1e4, 1e4))
                if shift:
                    M = G.scaling(f, c)
                else:
                    S = np.eye(3) + (f - 1) * np.outer(a, a)
                    M = np.eye(4)
                    M[:3, :3] = S
                    M[:3, 3] = c - S @ c
                st["label"] = f"x{f:.3g}"
        st["M"] = M
        return M

    def finish(self):
        st, self.drag = self.drag, None
        self.refresh()
        return st
