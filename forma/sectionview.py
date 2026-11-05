"""2D section viewports for the Draft workspace (shown only while a section plane exists)."""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import section as S


class SectionView(QtWidgets.QWidget):
    """Read-only 2D view of the model cut by a section plane, looking from one side."""

    def __init__(self, doc, settings, side=1, parent=None):
        super().__init__(parent)
        self.doc, self.settings, self.side = doc, settings, side
        self.data = None
        self.section_id = None
        self.scale = None
        self.center = np.zeros(2)
        self._pan = None
        self._cache = {}
        self.setMinimumSize(240, 160)
        self.setMouseTracking(True)
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self.recompute)

    # ------------------------------------------------------------ data
    def schedule(self):
        self._timer.start()

    def pick_section(self):
        secs = self.doc.sections()
        if not secs:
            return None
        sel = [g for g in secs if g.id in self.doc.selection]
        return (sel or secs)[-1]

    def recompute(self):
        sec = self.pick_section()
        if sec is None:
            self.data = None
            self.update()
            return
        first = self.data is None or sec.id != self.section_id
        if len(self._cache) > 4000:
            self._cache.clear()
        try:
            self.data = S.compute(self.doc, sec, self.side, self._cache)
        except Exception as e:
            self.data = None
            print("[forma] section view:", e, flush=True)
        self.section_id = sec.id
        if first or self.scale is None:
            self.zoom_extents()
        self.update()

    # ------------------------------------------------------------ view
    def zoom_extents(self):
        if not self.data:
            return
        pts = [p for pl in self.data["cut"] for p in pl]
        if len(self.data["beyond"]):
            pts += list(self.data["beyond"].reshape(-1, 2))
        if len(self.data["faces"]):
            pts += list(self.data["faces"].reshape(-1, 2)[::7])
        if not pts:
            self.scale, self.center = 0.01, np.zeros(2)
            return
        a = np.array(pts)
        lo, hi = a.min(0), a.max(0)
        self.center = (lo + hi) / 2
        size = np.maximum(hi - lo, 1e-6)
        self.scale = 0.88 * min(self.width() / size[0], self.height() / size[1])
        self.update()

    def _T(self):
        t = QtGui.QTransform()
        t.translate(self.width() / 2, self.height() / 2)
        t.scale(self.scale, -self.scale)
        t.translate(-self.center[0], -self.center[1])
        return t

    def paintEvent(self, ev):
        c = self.settings.palette()
        light = c.get("mode") == "light"
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        qp.fillRect(self.rect(), QtGui.QColor(c["view2d"]))
        title = ("Section A" if self.side > 0 else "Section B")
        if not self.data:
            qp.setPen(QtGui.QColor(c["muted"]))
            qp.drawText(self.rect(), QtCore.Qt.AlignCenter, "Add a section plane (SectionPlane) to see a cut here")
            qp.end()
            return
        if self.scale is None:
            self.zoom_extents()
        qp.setTransform(self._T())
        base = np.array([238, 238, 235]) if light else np.array([178, 182, 188])
        F, Sh = self.data["faces"], self.data["shade"]
        qp.setPen(QtCore.Qt.NoPen)
        for tri, sh in zip(F, Sh):
            col = (base * sh).clip(0, 255).astype(int)
            qp.setBrush(QtGui.QColor(*col))
            qp.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(*p) for p in tri]))
        pen = QtGui.QPen(QtGui.QColor("#5a5f66" if light else "#2a2c30"), 1.0)
        pen.setCosmetic(True)
        qp.setPen(pen)
        path = QtGui.QPainterPath()
        for a, b in self.data["beyond"]:
            path.moveTo(*a)
            path.lineTo(*b)
        qp.drawPath(path)
        # poché of cut solids
        fillp = QtGui.QPainterPath()
        fillp.setFillRule(QtCore.Qt.OddEvenFill)
        for loop in self.data["fill"]:
            fillp.moveTo(*loop[0])
            for p in loop[1:]:
                fillp.lineTo(*p)
            fillp.closeSubpath()
        qp.setPen(QtCore.Qt.NoPen)
        qp.setBrush(QtGui.QColor("#1d1f23" if light else "#0e0f11"))
        qp.drawPath(fillp)
        pen = QtGui.QPen(QtGui.QColor("#d0303a"), 2.4)
        pen.setCosmetic(True)
        qp.setPen(pen)
        qp.setBrush(QtCore.Qt.NoBrush)
        cutp = QtGui.QPainterPath()
        for pl in self.data["cut"]:
            cutp.moveTo(*pl[0])
            for p in pl[1:]:
                cutp.lineTo(*p)
        qp.drawPath(cutp)
        qp.setTransform(QtGui.QTransform())
        n = self.data["normal"]
        heading = (np.degrees(np.arctan2(-n[0], -n[1])) % 360)
        qp.setPen(QtGui.QColor(c["text"]))
        qp.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        qp.drawText(QtCore.QPointF(10, 18), f"{title}  ·  looking {self._compass(heading)}")
        qp.end()

    @staticmethod
    def _compass(deg):
        return ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"][int((deg + 22.5) // 45) % 8]

    # ------------------------------------------------------------ navigation
    def wheelEvent(self, ev):
        if self.scale is None:
            return
        pd, ad = ev.pixelDelta(), ev.angleDelta()
        steps = pd.y() / 60.0 if not pd.isNull() else ad.y() / 120.0
        q = ev.position()
        inv, _ = self._T().inverted()
        before = inv.map(q)
        self.scale *= 1.15 ** steps
        inv, _ = self._T().inverted()
        after = inv.map(q)
        self.center += np.array([before.x() - after.x(), before.y() - after.y()])
        self.update()

    def mousePressEvent(self, ev):
        self._pan = ev.position()

    def mouseMoveEvent(self, ev):
        if self._pan is not None and self.scale:
            d = ev.position() - self._pan
            self._pan = ev.position()
            self.center -= np.array([d.x(), -d.y()]) / self.scale
            self.update()

    def mouseReleaseEvent(self, ev):
        self._pan = None

    def mouseDoubleClickEvent(self, ev):
        self.zoom_extents()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self.data is not None and self.scale is None:
            self.zoom_extents()
