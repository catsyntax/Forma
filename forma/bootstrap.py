"""Lightweight startup helpers (no heavy imports), so a splash can show instantly."""
from __future__ import annotations

import os
import sys

from PySide6 import QtCore, QtGui, QtWidgets

from . import APP_NAME


class _FileOpenFilter(QtCore.QObject):
    """macOS sends double-clicked documents as FileOpen events, not argv."""

    def __init__(self):
        super().__init__()
        self.window = None
        self.pending = []

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.FileOpen:
            path = ev.file()
            if self.window is not None:
                QtCore.QTimer.singleShot(0, lambda p=path: self.window.file_open(p))
            else:
                self.pending.append(path)
            return True
        return False


def app_icon():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, "resources", "forma_icon.png"),
              os.path.join(os.path.dirname(here), "packaging", "forma_icon.png")):
        if os.path.exists(p):
            return QtGui.QIcon(p)
    return QtGui.QIcon()


def _mac_app_name():
    """When launched through Forma.app, make the menu bar say Forma instead of Python."""
    if sys.platform != "darwin":
        return
    try:
        from Foundation import NSBundle
        info = NSBundle.mainBundle().localizedInfoDictionary() or NSBundle.mainBundle().infoDictionary()
        if info is not None:
            info["CFBundleName"] = APP_NAME
    except Exception:
        pass


class FormaStyle(QtWidgets.QProxyStyle):
    """Fusion-based style that draws its own small standard icons.

    Qt 6.12 on macOS fetches standard icons (toolbar overflow arrow, message-box icons...) from
    SF Symbols through QAppleIconEngine, which crashes on some macOS versions (EXC_BREAKPOINT
    in -[NSImageSymbolRepProvider ...]). Drawing them ourselves avoids that code path entirely.
    """

    def __init__(self):
        super().__init__(QtWidgets.QStyleFactory.create("Fusion"))
        self._cache = {}

    def standardIcon(self, sp, opt=None, widget=None):
        if sp in self._cache:
            return self._cache[sp]
        icon = QtGui.QIcon(self._draw(sp))
        self._cache[sp] = icon
        return icon

    def standardPixmap(self, sp, opt=None, widget=None):
        return self._draw(sp)

    @staticmethod
    def _draw(sp, size=32):
        SP = QtWidgets.QStyle.StandardPixmap
        pm = QtGui.QPixmap(size, size)
        pm.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        col = QtGui.QColor("#9aa0a8")
        pen = QtGui.QPen(col, size * 0.09, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
        p.setPen(pen)
        c = size / 2
        k = size * 0.18

        def chevron(dx, dy, off=0.0):
            ox, oy = off * dx, off * dy
            if dx:
                p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(c - dx * k + ox, c - 2 * k), QtCore.QPointF(c + dx * k + ox, c),
                                                QtCore.QPointF(c - dx * k + ox, c + 2 * k)]))
            else:
                p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(c - 2 * k, c - dy * k + oy), QtCore.QPointF(c, c + dy * k + oy),
                                                QtCore.QPointF(c + 2 * k, c - dy * k + oy)]))
        arrows = {getattr(SP, "SP_ArrowRight", None): (1, 0), getattr(SP, "SP_ArrowForward", None): (1, 0),
                  getattr(SP, "SP_ArrowLeft", None): (-1, 0), getattr(SP, "SP_ArrowBack", None): (-1, 0),
                  getattr(SP, "SP_ArrowUp", None): (0, -1), getattr(SP, "SP_ArrowDown", None): (0, 1),
                  getattr(SP, "SP_TitleBarUnshadeButton", None): (0, 1), getattr(SP, "SP_TitleBarShadeButton", None): (0, -1)}
        ext = {getattr(SP, "SP_ToolBarHorizontalExtensionButton", None): (1, 0),
               getattr(SP, "SP_ToolBarVerticalExtensionButton", None): (0, 1)}
        marks = {getattr(SP, "SP_MessageBoxQuestion", None): "?", getattr(SP, "SP_MessageBoxWarning", None): "!",
                 getattr(SP, "SP_MessageBoxCritical", None): "!", getattr(SP, "SP_MessageBoxInformation", None): "i"}
        closes = {getattr(SP, n, None) for n in ("SP_TitleBarCloseButton", "SP_DockWidgetCloseButton",
                                                 "SP_LineEditClearButton", "SP_DialogCloseButton", "SP_DialogCancelButton")}
        if sp in ext:
            chevron(*ext[sp], off=-k * 0.9)
            chevron(*ext[sp], off=k * 0.9)
        elif sp in arrows:
            chevron(*arrows[sp])
        elif sp in marks:
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor("#4c8dff" if marks[sp] != "!" else "#e5a33b"))
            p.drawEllipse(QtCore.QRectF(2, 2, size - 4, size - 4))
            p.setPen(QtGui.QColor("white"))
            f = QtGui.QFont("Arial", int(size * 0.5), QtGui.QFont.Bold)
            p.setFont(f)
            p.drawText(pm.rect(), QtCore.Qt.AlignCenter, marks[sp])
        elif sp in closes:
            p.drawLine(QtCore.QPointF(c - k, c - k), QtCore.QPointF(c + k, c + k))
            p.drawLine(QtCore.QPointF(c + k, c - k), QtCore.QPointF(c - k, c + k))
        p.end()
        return pm


def create_qapp(argv=None):
    """Create (or return) the QApplication. Safe to call before the heavy imports."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        _mac_app_name()
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
        app = QtWidgets.QApplication(argv if argv is not None else sys.argv)
        app.setApplicationName(APP_NAME)
        app.setApplicationDisplayName(APP_NAME)
        app.setStyle(FormaStyle())
    return app


