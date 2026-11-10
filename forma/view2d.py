"""Draft workspace canvas: 2D drafting with grid, snaps, ortho and window selection."""
from __future__ import annotations

import math
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import geometry as G
from .document import display_px
from .geometry import P

SNAP_PX = 12
PICK_PX = 7

# Draft views: rows are screen-right, screen-up and toward-the-viewer, in world axes.
VIEWS = {
    "Top":    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float),
    "Bottom": np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float),
    "Front":  np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float),
    "Back":   np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], float),
    "Right":  np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float),
    "Left":   np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], float),
}


def _p3(p):
    a = np.zeros(3)
    p = np.asarray(p, float).ravel()
    a[:len(p)] = p[:3]
    return a


class View2D(QtWidgets.QWidget):
    cursor_moved = QtCore.Signal(object, str)
    component_dropped = QtCore.Signal(str, object)
    view_changed = QtCore.Signal(str)

    def __init__(self, doc, ix, settings, cmdline=None, parent=None):
        super().__init__(parent)
        self.doc, self.ix, self.settings, self.cmdline = doc, ix, settings, cmdline
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMinimumSize(300, 200)
        self.scale = 4.0             # pixels per unit
        self.center = np.array([60.0, 40.0])
        self.cursor_world = None
        self.snap = None             # (point, label)
        self.drag_start = None
        self.drag_now = None
        self._gb = None
        self._drag_M = None
        self.pan_last = None
        self.pan_moved = False
        self._paths = {}
        self.view_name = "Top"
        self.R = VIEWS["Top"]
        self._vcache = {}
        doc.subscribe(self._doc_changed)
        ix.preview_changed.connect(self.update)
        ix.changed.connect(self.update)

    # ------------------------------------------------------------ coords
    def set_view(self, name):
        """Look at the model from Top, Bottom, Front, Back, Left or Right. Drawing happens on the
        plane through the origin facing you."""
        if name not in VIEWS:
            return
        self.view_name = name
        self.R = VIEWS[name]
        self.ix.plane_normal = self.R[2].copy()
        self.ix.view_R = self.R.copy()
        self._vcache.clear()
        self.zoom_extents()
        self.view_changed.emit(name)

    def _vg(self, g, cache=True):
        """The object as seen from the current view (view coordinates: x right, y up)."""
        if self.view_name == "Top":
            return g
        key = (g.id, id(g))
        if cache and key in self._vcache:
            return self._vcache[key]
        M = np.eye(4)
        M[:3, :3] = self.R
        try:
            v = g.transformed(M, keep_id=True)
        except Exception:
            v = g
        if cache:
            if len(self._vcache) > 20000:
                self._vcache.clear()
            self._vcache[key] = v
        return v

    def _v2s(self, v):
        return QtCore.QPointF(self.width() / 2 + (v[0] - self.center[0]) * self.scale,
                              self.height() / 2 - (v[1] - self.center[1]) * self.scale)

    def _s2v(self, q):
        return np.array([self.center[0] + (q.x() - self.width() / 2) / self.scale,
                         self.center[1] - (q.y() - self.height() / 2) / self.scale, 0.0])

    def w2s(self, p):
        return self._v2s(self.R @ _p3(p))

    def s2w(self, q):
        return self.R.T @ self._s2v(q)

    def world_transform(self):
        t = QtGui.QTransform()
        t.translate(self.width() / 2, self.height() / 2)
        t.scale(self.scale, -self.scale)
        t.translate(-self.center[0], -self.center[1])
        return t

    def _doc_changed(self, what):
        if what in ("changed", "loaded", "layers", "definitions"):
            self._paths = {k: v for k, v in self._paths.items() if k[0] in self.doc.objects}
            self._vcache = {k: v for k, v in self._vcache.items() if k[0] in self.doc.objects}
        self.update()

    # ------------------------------------------------------------ paint
    def palette_colors(self):
        return self.settings.palette()

    def path_for(self, g):
        key = (g.id, id(g))
        p = self._paths.get(key)
        if p is None:
            p = QtGui.QPainterPath()
            if g.kind == "point":
                pass
            elif g.kind in ("mesh", "brep"):
                for a, b in G.feature_edges(g):
                    p.moveTo(a[0], a[1])
                    p.lineTo(b[0], b[1])
            else:
                s = g.sample()
                if len(s):
                    p.moveTo(s[0][0], s[0][1])
                    for q in s[1:]:
                        p.lineTo(q[0], q[1])
            if len(self._paths) > 20000:
                self._paths.clear()
            self._paths[key] = p
        return p

    def paintEvent(self, ev):
        c = self.palette_colors()
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        qp.fillRect(self.rect(), QtGui.QColor(c["view2d"]))
        if self.settings.show_grid:
            self._draw_grid(qp, c)
        T = self.world_transform()
        sel = self.doc.selection
        # objects
        for owner, g, col, dim in self.doc.display("2d"):
            color = QtGui.QColor(c["selection"] if owner in sel else col)
            if c["mode"] == "light" and col.lower() in ("#d0d0d0", "#ffffff"):
                color = QtGui.QColor(c["selection"] if owner in sel else "#1b1b1d")
            if dim:
                color.setAlphaF(0.25)
            lw = display_px(self.doc.weight_of(owner), 1.4)
            pen = QtGui.QPen(color, max(lw, 2.0) if owner in sel else lw)
            pen.setCosmetic(True)
            qp.setPen(pen)
            if g.kind == "point":
                s = self.w2s(g.p)
                qp.setTransform(QtGui.QTransform())
                qp.drawRect(QtCore.QRectF(s.x() - 3, s.y() - 3, 6, 6))
                continue
            qp.setTransform(T)
            qp.drawPath(self.path_for(self._vg(g)))
        qp.setTransform(QtGui.QTransform())
        # preview
        prev = self.ix.preview()
        if prev:
            pen = QtGui.QPen(QtGui.QColor(c["preview"]), 1.4, QtCore.Qt.DashLine)
            pen.setCosmetic(True)
            qp.setPen(pen)
            qp.setTransform(T)
            for g in prev:
                for part in self.doc.expand(g):
                    if part.kind != "point":
                        qp.drawPath(self.path_for(self._vg(part.copy(keep_id=False), cache=False)))
            qp.setTransform(QtGui.QTransform())
        # base rubber line
        r = self.ix.request
        base = getattr(r, "base", None) if r else None
        if base is not None and self.cursor_world is not None:
            pen = QtGui.QPen(QtGui.QColor(c["muted"]), 1, QtCore.Qt.DotLine)
            qp.setPen(pen)
            qp.drawLine(self.w2s(base), self.w2s(self.cursor_world))
        # snap marker
        if self.snap is not None and self.ix.wants_point():
            s = self.w2s(self.snap[0])
            qp.setPen(QtGui.QPen(QtGui.QColor(c["accent"]), 2))
            qp.setBrush(QtCore.Qt.NoBrush)
            if self.snap[1] == "Connector":
                poly = QtGui.QPolygonF([s + QtCore.QPointF(0, -7), s + QtCore.QPointF(7, 0), s + QtCore.QPointF(0, 7), s + QtCore.QPointF(-7, 0)])
                qp.drawPolygon(poly)
            else:
                qp.drawRect(QtCore.QRectF(s.x() - 5, s.y() - 5, 10, 10))
            qp.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Bold))
            qp.drawText(s + QtCore.QPointF(9, -8), self.snap[1])
        # crosshair
        if self.cursor_world is not None and self.underMouse():
            s = self.w2s(self.cursor_world)
            pen = QtGui.QPen(QtGui.QColor(c["muted"]), 1)
            qp.setPen(pen)
            qp.drawLine(QtCore.QPointF(s.x() - 12, s.y()), QtCore.QPointF(s.x() + 12, s.y()))
            qp.drawLine(QtCore.QPointF(s.x(), s.y() - 12), QtCore.QPointF(s.x(), s.y() + 12))
        # window selection rectangle
        if self.drag_start is not None and self.drag_now is not None:
            rect = QtCore.QRectF(self.drag_start, self.drag_now).normalized()
            crossing = self.drag_now.x() < self.drag_start.x()
            col = QtGui.QColor(c["preview"] if crossing else c["accent"])
            pen = QtGui.QPen(col, 1, QtCore.Qt.DashLine if crossing else QtCore.Qt.SolidLine)
            qp.setPen(pen)
            col.setAlphaF(0.08)
            qp.setBrush(col)
            qp.drawRect(rect)
        # live drag preview (gumball / control points)
        if self._drag_M is not None:
            pen = QtGui.QPen(QtGui.QColor(c["preview"]), 1.6)
            pen.setCosmetic(True)
            qp.setPen(pen)
            qp.setTransform(T)
            for g in self.doc.preview_targets(self._drag_M):
                if g.kind != "point":
                    qp.drawPath(self.path_for(self._vg(g.copy(keep_id=False), cache=False)))
            qp.setTransform(QtGui.QTransform())
        self._paint_grips(qp, c)
        self._paint_gumball(qp, c)
        # view name
        qp.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        qp.setPen(QtGui.QColor(c["muted"]))
        qp.drawText(QtCore.QPointF(12, self.height() - 12), self.view_name)
        # editing banner
        if self.doc.editing:
            self._banner(qp, c)
        qp.end()

    # ------------------------------------------------------------ control points & gumball (2D)
    GB_LEN = 78

    def _gumball_visible(self):
        return (self.settings.gumball and not self.ix.active and
                (bool(self.doc.grip_sel) or bool(self.doc.selected("2d"))))

    def _gumball_handles(self):
        """Screen-space handle shapes: name -> (kind, axis vector, QRectF/point)."""
        c = self.doc.target_center()
        if c is None:
            return None, {}
        s = self.w2s(c)
        L = self.GB_LEN
        u, v, n = self.R[0].copy(), self.R[1].copy(), self.R[2].copy()
        h = {
            "move_x": ("move", u, QtCore.QPointF(s.x() + L, s.y())),
            "move_y": ("move", v, QtCore.QPointF(s.x(), s.y() - L)),
            "scale_x": ("scale", u, QtCore.QPointF(s.x() + L + 16, s.y())),
            "scale_y": ("scale", v, QtCore.QPointF(s.x(), s.y() - L - 16)),
            "rot_z": ("rotate", n, QtCore.QPointF(s.x() + L * 0.62 * 0.7071, s.y() - L * 0.62 * 0.7071)),
            "free": ("free", n, s),
        }
        return c, h

    def _gumball_hit(self, q):
        if not self._gumball_visible():
            return None
        c, h = self._gumball_handles()
        best, bd = None, 10.0
        for name, (kind, ax, pt) in h.items():
            if kind == "move":   # whole arrow shaft
                s = self.w2s(c)
                a, b = np.array([s.x(), s.y()]), np.array([pt.x(), pt.y()])
                t = np.clip(np.dot(np.array([q.x(), q.y()]) - a, b - a) / max(np.dot(b - a, b - a), 1e-9), 0.18, 1.0)
                d = float(np.linalg.norm(a + t * (b - a) - np.array([q.x(), q.y()])))
            else:
                d = math.hypot(pt.x() - q.x(), pt.y() - q.y())
            if d < bd:
                best, bd = name, d
        return best

    def _paint_gumball(self, qp, c):
        if not self._gumball_visible() or self._gb is not None and False:
            return
        cen, h = self._gumball_handles()
        if cen is None:
            return
        s = self.w2s(cen)
        hover = getattr(self, "_gb_hover", None)
        active = self._gb["name"] if self._gb else None
        cols = {"x": QtGui.QColor("#e5534b"), "y": QtGui.QColor("#3fb37f"), "z": QtGui.QColor("#4c8dff")}
        for name, (kind, ax, pt) in h.items():
            if active and name != active:
                continue
            axn = "xyz"[int(np.argmax(np.abs(ax)))]
            col = QtGui.QColor("#ffd23f") if name in (hover, active) else cols[axn]
            qp.setPen(QtGui.QPen(col, 2.2))
            qp.setBrush(col)
            if kind == "move":
                qp.drawLine(s, pt)
                d = np.array([pt.x() - s.x(), pt.y() - s.y()])
                d /= np.linalg.norm(d) or 1
                n = np.array([-d[1], d[0]])
                tip = np.array([pt.x(), pt.y()])
                poly = [tip + d * 10, tip + n * 5, tip - n * 5]
                qp.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(*p) for p in poly]))
            elif kind == "scale":
                qp.drawRect(QtCore.QRectF(pt.x() - 4, pt.y() - 4, 8, 8))
            elif kind == "rotate":
                qp.setBrush(QtCore.Qt.NoBrush)
                r = self.GB_LEN * 0.62
                qp.drawArc(QtCore.QRectF(s.x() - r, s.y() - r, 2 * r, 2 * r), 15 * 16, 60 * 16)
            elif kind == "free":
                col2 = QtGui.QColor("#ffd23f") if name in (hover, active) else QtGui.QColor("#dddddd")
                qp.setPen(QtGui.QPen(col2, 1.5))
                qp.setBrush(QtCore.Qt.NoBrush)
                qp.drawRect(QtCore.QRectF(s.x() - 6, s.y() - 6, 12, 12))
        if self._gb and self._gb.get("label"):
            qp.setPen(QtGui.QColor(c["text"]))
            qp.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
            qp.drawText(s + QtCore.QPointF(14, 22), self._gb["label"])

    def _paint_grips(self, qp, c):
        gp = self.doc.grip_points()
        if not gp:
            return
        from . import editpoints as E
        pen = QtGui.QPen(QtGui.QColor(c["muted"]), 1, QtCore.Qt.DashLine)
        qp.setPen(pen)
        for oid in self.doc.points_on:
            g = self.doc.get(oid)
            if g is None:
                continue
            for line in E.grid_lines(g):
                pts = [self.w2s(p) for p in line]
                for a, b in zip(pts[:-1], pts[1:]):
                    qp.drawLine(a, b)
        sel = self.doc.grip_sel
        for o, i, p in gp:
            q = self.w2s(p)
            on = (o, i) in sel
            qp.setPen(QtGui.QPen(QtGui.QColor("#111111"), 1))
            qp.setBrush(QtGui.QColor(c["selection"] if on else "#f5f5f5"))
            qp.drawRect(QtCore.QRectF(q.x() - 3.5, q.y() - 3.5, 7, 7))

    def _grip_at(self, q):
        gp = self.doc.grip_points()
        best, bd = None, 9.0
        for o, i, p in gp:
            s = self.w2s(p)
            d = math.hypot(s.x() - q.x(), s.y() - q.y())
            if d < bd:
                best, bd = (o, i), d
        return best

    def _drag_matrix(self, q, shift):
        st = self._gb
        p = self.s2w(q)
        snap = self.settings.snap
        grid = self.settings.grid_spacing / 10 if snap else 0
        c = st["c"]
        kind, ax = st["kind"], st["axis"]
        if kind in ("move", "free", "grip"):
            dv = p - st["p0"]
            if kind == "move":
                dv = ax * np.dot(dv, ax)
            if kind == "grip":
                tgt = self.resolve(q)  # snaps to object snaps / grid like drawing
                dv = tgt - st["p0"]
                n = self.R[2]
                dv = dv - n * np.dot(dv, n)   # stay at the point's own depth
            elif grid:
                dv = np.round(dv / grid) * grid
            st["label"] = ", ".join(f"{v:.4g}" for v in (self.R @ dv)[:2])
            return G.translation(dv)
        if kind == "rotate":
            v0, v1 = self.R @ (st["p0"] - c), self.R @ (p - c)
            ang = math.degrees(math.atan2(v0[0] * v1[1] - v0[1] * v1[0], v0[0] * v1[0] + v0[1] * v1[1]))
            step = 15 if shift else (5 if snap else 0)
            if step:
                ang = round(ang / step) * step
            st["label"] = f"{ang:.1f}°"
            return G.rotation(ang, tuple(self.R[2]), c)
        if kind == "scale":
            d0 = np.dot(st["p0"] - c, ax)
            f = np.dot(p - c, ax) / d0 if abs(d0) > 1e-12 else 1.0
            if snap:
                f = round(f * 20) / 20 or 0.05
            if abs(f) < 1e-3:              # never flatten to zero (passing through the centre mirrors)
                f = 1e-3 if f >= 0 else -1e-3
            f = float(np.clip(f, -1e4, 1e4))
            st["label"] = f"x{f:.3g}"
            if shift:
                return G.scaling(f, c)
            S = np.eye(3) + (f - 1) * np.outer(ax, ax)
            M = np.eye(4)
            M[:3, :3] = S
            M[:3, 3] = c - S @ c
            return M
        return np.eye(4)

    def _banner(self, qp, c):
        d = self.doc.definitions.get(self.doc.editing[0])
        txt = f"Editing component: {d.name if d else ''}   (CloseComponent to finish)"
        qp.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        qp.setPen(QtGui.QColor(c["accent"]))
        qp.drawText(QtCore.QPointF(12, 22), txt)

    def _draw_grid(self, qp, c):
        sp = self.settings.grid_spacing
        while sp * self.scale < 8:
            sp *= 5
        while sp * self.scale > 80:
            sp /= 5
        lo = self._s2v(QtCore.QPointF(0, self.height()))
        hi = self._s2v(QtCore.QPointF(self.width(), 0))
        minor = QtGui.QPen(QtGui.QColor(c["grid"]), 1)
        major = QtGui.QPen(QtGui.QColor(c["grid_major"]), 1)
        i0, i1 = int(math.floor(lo[0] / sp)), int(math.ceil(hi[0] / sp))
        j0, j1 = int(math.floor(lo[1] / sp)), int(math.ceil(hi[1] / sp))
        if (i1 - i0) > 600 or (j1 - j0) > 600:
            return
        for i in range(i0, i1 + 1):
            qp.setPen(major if i % 5 == 0 else minor)
            x = self._v2s((i * sp, 0, 0)).x()
            qp.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, self.height()))
        for j in range(j0, j1 + 1):
            qp.setPen(major if j % 5 == 0 else minor)
            y = self._v2s((0, j * sp, 0)).y()
            qp.drawLine(QtCore.QPointF(0, y), QtCore.QPointF(self.width(), y))
        o = self._v2s((0, 0, 0))
        axc = [QtGui.QColor(200, 70, 60, 160), QtGui.QColor(70, 170, 80, 160), QtGui.QColor(70, 120, 220, 170)]
        qp.setPen(QtGui.QPen(axc[int(np.argmax(np.abs(self.R[0])))], 1.2))
        qp.drawLine(QtCore.QPointF(0, o.y()), QtCore.QPointF(self.width(), o.y()))
        qp.setPen(QtGui.QPen(axc[int(np.argmax(np.abs(self.R[1])))], 1.2))
        qp.drawLine(QtCore.QPointF(o.x(), 0), QtCore.QPointF(o.x(), self.height()))

    # ------------------------------------------------------------ snapping
    def resolve(self, q: QtCore.QPointF):
        raw = self.s2w(q)
        self.snap = None
        r = self.ix.request
        if self.settings.osnap and self.ix.wants_point():
            best, bd = None, SNAP_PX
            ver = getattr(self.doc, "version", 0)
            if getattr(self, "_snap_ver", None) != ver:
                sp = self.doc.snap_points("2d")
                self._snap_pts = np.array([p for p, _ in sp]).reshape(-1, 3)
                self._snap_lbl = [l for _, l in sp]
                self._snap_ver = ver
            if len(self._snap_pts):
                pv = self._snap_pts @ self.R.T
                sx = self.width() / 2 + (pv[:, 0] - self.center[0]) * self.scale
                sy = self.height() / 2 - (pv[:, 1] - self.center[1]) * self.scale
                dd = np.hypot(sx - q.x(), sy - q.y()) - np.array([4.0 if l == "Connector" else 0.0 for l in self._snap_lbl])
                i = int(np.argmin(dd))
                if dd[i] < bd:
                    best, bd = (P(self._snap_pts[i]), self._snap_lbl[i]), dd[i]
            # intersections of nearby curves
            if best is None:
                tol = SNAP_PX / self.scale
                rv = self.R @ raw
                near = [self._vg(g) for g in self.doc.visible("2d") if g.is_curve and self._dist_to(self._vg(g), rv) < tol * 3][:6]
                for i in range(len(near)):
                    for j in range(i + 1, len(near)):
                        for pv_ in G.curve_intersections_2d(near[i], near[j], 16):
                            p = self.R.T @ _p3(pv_)
                            s = self.w2s(p)
                            d = math.hypot(s.x() - q.x(), s.y() - q.y())
                            if d < bd:
                                best, bd = (P(p), "Int"), d
            if best is not None:
                self.snap = best
                return best[0]
        p = self.ix.grid_snap(raw) if self.ix.wants_point() else raw
        return self.ix.constrain(p)

    def _dist_to(self, g, p):
        if g.kind == "point":
            return float(np.linalg.norm(g.p[:2] - p[:2]))
        if g.kind in ("brep", "mesh"):
            # clicking anywhere inside a surface (seen from above) selects it
            try:
                V = g.sample()[:, :2] if g.kind == "brep" else g.V[:, :2]
                T = g.triangles()
                if len(T):
                    A, B_, C_ = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
                    q = p[:2]
                    d1 = (q[0] - B_[:, 0]) * (A[:, 1] - B_[:, 1]) - (A[:, 0] - B_[:, 0]) * (q[1] - B_[:, 1])
                    d2 = (q[0] - C_[:, 0]) * (B_[:, 1] - C_[:, 1]) - (B_[:, 0] - C_[:, 0]) * (q[1] - C_[:, 1])
                    d3 = (q[0] - A[:, 0]) * (C_[:, 1] - A[:, 1]) - (C_[:, 0] - A[:, 0]) * (q[1] - A[:, 1])
                    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
                    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
                    if np.any(~(neg & pos)):
                        return 0.0
            except Exception:
                pass
            segs = G.feature_edges(g)
            if not segs:
                return 1e18
            A = np.array([a[:2] for a, b in segs])
            B = np.array([b[:2] for a, b in segs])
            AB = B - A
            t = np.clip(np.einsum("ij,ij->i", p[:2] - A, AB) / np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-18), 0, 1)
            return float(np.min(np.linalg.norm(A + AB * t[:, None] - p[:2], axis=1)))
        s = g.sample()[:, :2]
        if len(s) == 1:
            return float(np.linalg.norm(s[0] - p[:2]))
        a, b = s[:-1], s[1:]
        ab = b - a
        t = np.clip(np.einsum("ij,ij->i", p[:2] - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18), 0, 1)
        proj = a + ab * t[:, None]
        return float(np.min(np.linalg.norm(proj - p[:2], axis=1)))

    def pick(self, q):
        p = self.s2w(q)
        tol = PICK_PX / self.scale
        filt = self.ix.selection_filter()
        best, bd = None, tol
        for g in self.doc.selectable("2d"):
            if filt and not filt(g):
                continue
            for part in self.doc.expand(g):
                d = self._dist_to(self._vg(part), self.R @ p)
                if d < bd:
                    best, bd = g.id, d
        return best

    # ------------------------------------------------------------ mouse
    def mousePressEvent(self, ev):
        self.setFocus()
        if ev.button() in (QtCore.Qt.MiddleButton, QtCore.Qt.RightButton):
            self.pan_last = ev.position()
            self.pan_moved = False
            return
        if ev.button() == QtCore.Qt.LeftButton:
            if self.ix.wants_window():
                self.drag_start = ev.position()   # click, or drag a window (handled on release)
                self.drag_now = None
                return
            if self.ix.wants_point():
                self.ix.click_point(self.resolve(ev.position()))
                return
            q = ev.position()
            h = self._gumball_hit(q)
            if h:
                c, handles = self._gumball_handles()
                kind, ax, _ = handles[h]
                self._gb = {"name": h, "kind": kind, "axis": ax, "c": c, "p0": self.s2w(q), "q0": q, "moved": False}
                self._drag_M = None
                return
            if self.doc.points_on:
                g = self._grip_at(q)
                if g is not None:
                    if ev.modifiers() & QtCore.Qt.ShiftModifier:
                        self.doc.toggle_grip(g)
                    elif g not in self.doc.grip_sel:
                        self.doc.select_grips([g])
                    p0 = dict(((o, i), p) for o, i, p in self.doc.grip_points())[g].copy()
                    self._gb = {"name": None, "kind": "grip", "axis": None, "c": p0, "p0": p0, "q0": q, "moved": False}
                    self._drag_M = None
                    return
            self.drag_start = ev.position()
            self.drag_now = None

    def mouseMoveEvent(self, ev):
        q = ev.position()
        if self.pan_last is not None:
            d = q - self.pan_last
            if abs(d.x()) + abs(d.y()) > 2:
                self.pan_moved = True
            self.center -= np.array([d.x(), -d.y()]) / self.scale
            self.pan_last = q
            self.update()
            return
        if self._gb is not None:
            if (q - self._gb["q0"]).manhattanLength() > 3:
                self._gb["moved"] = True
            if self._gb["moved"]:
                self._drag_M = self._drag_matrix(q, bool(ev.modifiers() & QtCore.Qt.ShiftModifier))
            self.cursor_world = self.s2w(q)
            self.update()
            return
        if self.drag_start is None and not self.ix.active:
            hv = self._gumball_hit(q)
            if hv != getattr(self, "_gb_hover", None):
                self._gb_hover = hv
        if self.drag_start is not None:
            if (q - self.drag_start).manhattanLength() > 4:
                self.drag_now = q
        self.cursor_world = self.resolve(q)
        self.ix.cursor = self.cursor_world
        self.cursor_moved.emit(self.cursor_world, self.snap[1] if self.snap else "")
        self.update()

    def mouseReleaseEvent(self, ev):
        if ev.button() in (QtCore.Qt.MiddleButton, QtCore.Qt.RightButton):
            if ev.button() == QtCore.Qt.RightButton and not self.pan_moved:
                self.ix.submit_empty()   # right click = Enter / repeat last command
            self.pan_last = None
            return
        if ev.button() == QtCore.Qt.LeftButton and self._gb is not None:
            st, self._gb = self._gb, None
            M, self._drag_M = self._drag_M, None
            if st["moved"] and M is not None:
                if st["kind"] != "grip" and ev.modifiers() & QtCore.Qt.AltModifier:
                    self.doc.transform_targets(M, "Gumball copy", copy=True)
                else:
                    self.doc.transform_targets(M, "Move points" if st["kind"] == "grip" else "Gumball " + st["kind"])
            elif not st["moved"] and st["kind"] in ("move", "rotate", "scale"):
                self.ix.start("GumballValue", "2d", kind=st["kind"], axis=tuple(st["axis"]), center=tuple(st["c"]))
            self.update()
            return
        if ev.button() != QtCore.Qt.LeftButton or self.drag_start is None:
            return
        if self.ix.wants_window():
            a_, b_ = self.drag_start, self.drag_now
            self.drag_start = self.drag_now = None
            if b_ is None:
                self.ix.click_point(self.resolve(ev.position()))
            else:
                rect = QtCore.QRectF(a_, b_).normalized()
                self.ix.window_pick(lambda p, r=rect: r.contains(self.w2s(p)), b_.x() < a_.x())
            self.update()
            return
        add = bool(ev.modifiers() & QtCore.Qt.ShiftModifier)
        rem = bool(ev.modifiers() & QtCore.Qt.ControlModifier)
        if self.drag_now is not None and self.doc.points_on and not self.ix.wants_objects():
            rect = QtCore.QRectF(self.drag_start, self.drag_now).normalized()
            items = [(o, i) for o, i, p in self.doc.grip_points() if rect.contains(self.w2s(p))]
            if items or not add:
                if rem:
                    self.doc.grip_sel -= set(items)
                    self.doc.notify("grips")
                else:
                    self.doc.select_grips(items, add=add)
                self.drag_start = self.drag_now = None
                self.update()
                return
        if self.drag_now is None and self.doc.grip_sel and not add and not self.ix.active:
            self.doc.clear_grips()
        if self.drag_now is not None:
            ids = self._window_select(self.drag_start, self.drag_now)
            if rem:
                self.doc.selection -= set(ids)
                self.doc.notify("selection")
            else:
                self.doc.select(ids, add=add or self.ix.wants_objects())
        else:
            oid = self.pick(ev.position())
            if oid is None:
                if not add and not rem and not self.ix.wants_objects():
                    self.doc.clear_selection()
            elif rem:
                self.doc.selection.discard(oid)
                self.doc.notify("selection")
            elif add or self.ix.wants_objects():
                self.doc.toggle_select(oid) if add else self.doc.select([oid], add=True)
            else:
                self.doc.select([oid])
            if oid is not None:
                self.ix.object_clicked_in_select_mode()
        self.drag_start = self.drag_now = None
        self.ix.changed.emit()
        self.update()

    def mouseDoubleClickEvent(self, ev):
        oid = self.pick(ev.position())
        if oid and self.doc.objects[oid].kind == "instance" and not self.ix.active:
            self.doc.select([oid])
            self.ix.start("EditComponent", "2d")

    def _window_select(self, a, b):
        rect = QtCore.QRectF(a, b).normalized()
        crossing = b.x() < a.x()
        lo, hi = self._s2v(rect.bottomLeft()), self._s2v(rect.topRight())
        filt = self.ix.selection_filter()
        out = []
        for g in self.doc.selectable("2d"):
            if filt and not filt(g):
                continue
            pts = np.vstack([part.sample(2) if part.kind not in ("mesh",) else part.V for part in self.doc.expand(g)] or [np.zeros((0, 3))])
            if not len(pts):
                continue
            pts = pts @ self.R.T
            inside = (pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0]) & (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1])
            if (crossing and inside.any()) or (not crossing and inside.all()):
                out.append(g.id)
        return out

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasText() and ev.mimeData().text().startswith("forma-component:"):
            ev.acceptProposedAction()

    def dragMoveEvent(self, ev):
        ev.acceptProposedAction()

    def dropEvent(self, ev):
        txt = ev.mimeData().text()
        if txt.startswith("forma-component:"):
            self.component_dropped.emit(txt.split(":", 1)[1], self.ix.grid_snap(self.s2w(ev.position())))
            ev.acceptProposedAction()

    def event(self, ev):
        if ev.type() == QtCore.QEvent.NativeGesture and ev.gestureType() == QtCore.Qt.ZoomNativeGesture:
            q = ev.position()
            before = self._s2v(q)
            self.scale = float(np.clip(self.scale * (1 + ev.value()), 1e-4, 1e5))
            self.center += (before - self._s2v(q))[:2]
            self.update()
            return True
        return super().event(ev)

    def wheelEvent(self, ev):
        """Scroll zooms at the cursor (mouse wheel or two-finger trackpad scroll).
        Shift+scroll (or Option+scroll) pans; pinch also zooms."""
        q = ev.position()
        pd, ad = ev.pixelDelta(), ev.angleDelta()
        mods = ev.modifiers()
        if mods & (QtCore.Qt.ShiftModifier | QtCore.Qt.AltModifier):
            if not pd.isNull():
                dx, dy = pd.x(), pd.y()
            else:
                dx, dy = ad.x() / 4, ad.y() / 4
            self.center -= np.array([dx, -dy]) / self.scale
            self.update()
            return
        if not pd.isNull():
            steps = pd.y() / 60.0          # smooth trackpad scrolling
        else:
            steps = ad.y() / 120.0 if ad.y() else ad.x() / 120.0
        if steps == 0:
            return
        before = self._s2v(q)
        f = 1.15 ** steps
        self.scale = float(np.clip(self.scale * f, 1e-4, 1e5))
        after = self._s2v(q)
        self.center += (before - after)[:2]
        self.update()

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Delete or ev.key() == QtCore.Qt.Key_Backspace and not (self.cmdline and self.cmdline.edit.text()):
            if not self.ix.active:
                try:
                    msg = self.doc.delete_subobjects()
                except Exception as e:
                    msg = f"Could not delete: {e}"
                if msg:
                    self.ix.message.emit(msg)
                    return
            if self.doc.selected("2d") and not self.ix.active:
                self.ix.start("Delete", "2d")
                return
        if ev.key() == QtCore.Qt.Key_Escape and not self.ix.active:
            if self.doc.grip_sel:
                self.doc.clear_grips()
            else:
                self.doc.clear_selection()
            return
        if self.cmdline and self.cmdline.forward_key(ev):
            return
        super().keyPressEvent(ev)

    def leaveEvent(self, ev):
        self.update()

    # ------------------------------------------------------------ view actions
    def zoom_extents(self, selected=False):
        objs = self.doc.selected("2d") if selected else self.doc.visible("2d")
        pts = []
        for g in objs:
            for part in self.doc.expand(g):
                lo, hi = part.bbox()
                corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                pts += list(corners @ self.R.T)
        if not pts:
            self.center = np.array([0.0, 0.0])
            self.scale = 4.0
        else:
            a = np.array(pts)
            lo, hi = a.min(0), a.max(0)
            self.center = ((lo + hi) / 2)[:2]
            size = max(hi[0] - lo[0], 1e-6), max(hi[1] - lo[1], 1e-6)
            self.scale = 0.85 * min(self.width() / size[0], self.height() / size[1])
        self.update()
