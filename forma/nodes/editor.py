"""Node editor (the Nodes workspace): canvas, nodes, wires, search, preview and bake."""
from __future__ import annotations

import os

import math
import uuid

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import geometry as G
from ..geometry import Geom
from .registry import NODE_TYPES, DEFAULT_PY_CODE, categories, flatten
from .solver import ports, downstream, topo_order
from .compute import ComputeManager

NODE_W = 170
TITLE_H = 24
PORT_H = 20
PORT_R = 5


def summarize(v, depth=0):
    if isinstance(v, list):
        if not v:
            return "[] empty"
        head = ", ".join(summarize(x, depth + 1) for x in v[:4])
        return f"[{len(v)}] " + head + (" ..." if len(v) > 4 else "")
    if isinstance(v, Geom):
        return v.kind if depth else v.describe() if v.kind != "instance" else "Component"
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, np.ndarray):
        return "(" + ", ".join(f"{x:.3g}" for x in v.ravel()[:3]) + ")"
    if isinstance(v, str):
        return v if len(v) < 60 else v[:57] + "..."
    return str(v)


def panel_text(v, indent=0):
    if isinstance(v, list):
        lines = []
        for i, x in enumerate(v[:200]):
            lines.append(f"{i}: {panel_text(x)}")
        if len(v) > 200:
            lines.append(f"... {len(v) - 200} more")
        return "\n".join(lines) if lines else "(empty)"
    return summarize(v, 1 if isinstance(v, Geom) else 0) if not isinstance(v, Geom) else v.describe() if v.kind != "instance" else "Component instance"


def parse_value(text):
    t = text.strip()
    if not t:
        return None
    if t.lower() in ("true", "false"):
        return t.lower() == "true"
    if "," in t:
        return tuple(float(x) for x in t.strip("()[] ").split(","))
    try:
        v = float(t)
        return int(v) if v.is_integer() and "." not in t else v
    except ValueError:
        return t


def fmt_value(v):
    if v is None:
        return ""
    if isinstance(v, tuple):
        return ",".join(f"{x:g}" for x in v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


# ===================================================================== graphics items
class PortItem(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, node, name, is_input, index):
        super().__init__(-PORT_R, -PORT_R, 2 * PORT_R, 2 * PORT_R, node)
        self.node, self.name, self.is_input, self.index = node, name, is_input, index
        self.setAcceptHoverEvents(True)
        self.setZValue(2)
        self.setCursor(QtCore.Qt.CrossCursor)

    def scene_pos(self):
        return self.mapToScene(QtCore.QPointF(0, 0))

    def hoverEnterEvent(self, ev):
        ed = self.node.editor
        if self.is_input:
            v = ed.graph["nodes"][self.node.nid].get("values", {}).get(self.name)
            self.setToolTip(f"{self.name}  (double-click to set a value)\nValue: {fmt_value(v) or 'default'}")
        else:
            r = ed.results.get(self.node.nid, {})
            self.setToolTip(f"{self.name}\n{summarize(r.get(self.name)) if self.name in r else 'No data'}")
        super().hoverEnterEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        if self.is_input:
            self.node.editor.edit_input_value(self.node.nid, self.name)


class WireItem(QtWidgets.QGraphicsPathItem):
    def __init__(self, editor, wire=None):
        super().__init__()
        self.editor, self.wire = editor, wire
        self.setZValue(-1)
        self.a = self.b = QtCore.QPointF()
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, wire is not None)

    def set_ends(self, a, b):
        self.a, self.b = a, b
        dx = max(40.0, abs(b.x() - a.x()) * 0.5)
        p = QtGui.QPainterPath(a)
        p.cubicTo(a + QtCore.QPointF(dx, 0), b - QtCore.QPointF(dx, 0), b)
        self.setPath(p)

    def paint(self, qp, opt, w=None):
        c = self.editor.colors
        col = QtGui.QColor(c["accent"] if self.isSelected() else c["muted"])
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        qp.setPen(QtGui.QPen(col, 2.2 if self.isSelected() else 1.6))
        qp.setBrush(QtCore.Qt.NoBrush)
        qp.drawPath(self.path())

    def shape(self):
        s = QtGui.QPainterPathStroker()
        s.setWidth(10)
        return s.createStroke(self.path())


class NodeItem(QtWidgets.QGraphicsItem):
    def __init__(self, editor, nid):
        super().__init__()
        self.editor, self.nid = editor, nid
        self.setFlags(QtWidgets.QGraphicsItem.ItemIsMovable | QtWidgets.QGraphicsItem.ItemIsSelectable |
                      QtWidgets.QGraphicsItem.ItemSendsGeometryChanges)
        self.in_ports, self.out_ports = {}, {}
        self.proxy = None
        self.widget_h = 0
        self.status = "dirty"
        self.message = ""
        self.progress = None
        self.build()

    @property
    def data(self):
        return self.editor.graph["nodes"][self.nid]

    @property
    def ntype(self):
        return NODE_TYPES[self.data["type"]]

    def build(self):
        for p in list(self.in_ports.values()) + list(self.out_ports.values()):
            p.setParentItem(None)
            if p.scene():
                p.scene().removeItem(p)
        if self.proxy is not None:
            self.proxy.setWidget(None)
            self.proxy.setParentItem(None)
            if self.proxy.scene():
                self.proxy.scene().removeItem(self.proxy)
            self.proxy = None
        ins, outs = ports(self.data)
        self.in_ports = {p.name: PortItem(self, p.name, True, i) for i, p in enumerate(ins)}
        self.out_ports = {o: PortItem(self, o, False, i) for i, o in enumerate(outs)}
        w = self.editor.make_widget(self)
        self.widget_h = 0
        if w is not None:
            self.proxy = QtWidgets.QGraphicsProxyWidget(self)
            self.proxy.setWidget(w)
            w.setFixedWidth(NODE_W - 12)
            self.widget_h = w.sizeHint().height() + 6
        rows = max(len(ins), len(outs), 1)
        self.h = TITLE_H + rows * PORT_H + self.widget_h + 8
        for name, p in self.in_ports.items():
            p.setPos(0, TITLE_H + 6 + p.index * PORT_H + PORT_H / 2)
        for name, p in self.out_ports.items():
            p.setPos(NODE_W, TITLE_H + 6 + p.index * PORT_H + PORT_H / 2)
        if self.proxy is not None:
            self.proxy.setPos(6, TITLE_H + 6 + rows * PORT_H)
        self.prepareGeometryChange()
        self.update()

    def boundingRect(self):
        return QtCore.QRectF(-PORT_R - 1, -1, NODE_W + 2 * PORT_R + 2, self.h + 2)

    def paint(self, qp, opt, w=None):
        c = self.editor.colors
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        r = QtCore.QRectF(0, 0, NODE_W, self.h)
        status_col = {"ok": "#3fb37f", "error": "#e5534b", "running": c["accent"], "dirty": c["muted"],
                      "disabled": "#6b6e75", "cancelled": "#d29922"}.get(self.status, c["muted"])
        border = QtGui.QColor(c["accent"] if self.isSelected() else c["border"])
        qp.setPen(QtGui.QPen(border, 2 if self.isSelected() else 1))
        bg = QtGui.QColor(c["panel"])
        if self.data.get("disabled"):
            bg.setAlphaF(0.55)
        qp.setBrush(bg)
        qp.drawRoundedRect(r, 6, 6)
        # title
        qp.setPen(QtCore.Qt.NoPen)
        qp.setBrush(QtGui.QColor(c["field"]))
        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(1, 1, NODE_W - 2, TITLE_H), 5, 5)
        qp.drawPath(path)
        qp.setBrush(QtGui.QColor(status_col))
        qp.drawRect(QtCore.QRectF(1, TITLE_H - 2, NODE_W - 2, 2))
        if self.status == "running" and self.progress is not None:
            qp.setBrush(QtGui.QColor(c["accent"]))
            qp.drawRect(QtCore.QRectF(1, TITLE_H - 3, (NODE_W - 2) * self.progress, 3))
        qp.setPen(QtGui.QColor(c["text"]))
        f = QtGui.QFont("Arial", 9, QtGui.QFont.Bold)
        qp.setFont(f)
        title = self.data.get("label") or self.data["type"]
        if not self.data.get("preview", True):
            title += "  (hidden)"
        if self.data.get("disabled"):
            title += "  (off)"
        qp.drawText(QtCore.QRectF(8, 0, NODE_W - 16, TITLE_H), QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, title)
        # ports
        qp.setFont(QtGui.QFont("Arial", 8))
        vals = self.data.get("values", {})
        connected = {(d, i) for s, o, d, i in self.editor.graph["wires"]}
        for name, p in self.in_ports.items():
            y = p.pos().y()
            label = name
            if (self.nid, name) not in connected and vals.get(name) is not None:
                label += " = " + fmt_value(vals[name])
            qp.setPen(QtGui.QColor(c["text"]))
            qp.drawText(QtCore.QRectF(10, y - PORT_H / 2, NODE_W / 2 + 30, PORT_H), QtCore.Qt.AlignVCenter, label)
        for name, p in self.out_ports.items():
            y = p.pos().y()
            qp.setPen(QtGui.QColor(c["text"]))
            qp.drawText(QtCore.QRectF(NODE_W / 2 - 30, y - PORT_H / 2, NODE_W / 2 + 20, PORT_H),
                        QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight, name)
        for p in list(self.in_ports.values()) + list(self.out_ports.values()):
            p.setBrush(QtGui.QColor(c["accent"] if not p.is_input else c["panel"]))
            p.setPen(QtGui.QPen(QtGui.QColor(c["accent"]), 1.5))
        # error text
        if self.status == "error" and self.message:
            qp.setPen(QtGui.QColor("#e5534b"))
            qp.setFont(QtGui.QFont("Arial", 8))
            qp.drawText(QtCore.QRectF(0, self.h + 2, NODE_W + 80, 30), QtCore.Qt.TextWordWrap, self.message[:120])

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            self.data["pos"] = [self.pos().x(), self.pos().y()]
            self.editor.update_wires(self.nid)
        return super().itemChange(change, value)

    def contextMenuEvent(self, ev):
        self.editor.node_menu(self, ev.screenPos())

    def mouseDoubleClickEvent(self, ev):
        if self.ntype.widget == "code":
            self.editor.edit_code(self.nid)
        else:
            super().mouseDoubleClickEvent(ev)


# ===================================================================== search popup
class NodeSearch(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint)
        self.resize(320, 360)
        lay = QtWidgets.QVBoxLayout(self)
        self.edit = QtWidgets.QLineEdit()
        self.edit.setPlaceholderText("Search nodes...")
        self.list = QtWidgets.QListWidget()
        lay.addWidget(self.edit)
        lay.addWidget(self.list)
        self.choice = None
        self.edit.textChanged.connect(self.fill)
        self.edit.returnPressed.connect(self.done_)
        self.list.itemActivated.connect(lambda _: self.done_())
        self.edit.installEventFilter(self)
        self.fill("")

    def fill(self, t):
        self.list.clear()
        t = t.lower()
        for name, nt in sorted(NODE_TYPES.items(), key=lambda kv: (kv[1].category, kv[0])):
            if not t or t in name.lower() or t in nt.category.lower() or t in nt.desc.lower():
                it = QtWidgets.QListWidgetItem(f"{name}   ·  {nt.category}")
                it.setData(QtCore.Qt.UserRole, name)
                it.setToolTip(nt.desc)
                self.list.addItem(it)
        if self.list.count():
            self.list.setCurrentRow(0)

    def eventFilter(self, o, e):
        if e.type() == QtCore.QEvent.KeyPress and e.key() in (QtCore.Qt.Key_Down, QtCore.Qt.Key_Up):
            r = self.list.currentRow() + (1 if e.key() == QtCore.Qt.Key_Down else -1)
            self.list.setCurrentRow(max(0, min(self.list.count() - 1, r)))
            return True
        return super().eventFilter(o, e)

    def done_(self):
        it = self.list.currentItem()
        self.choice = it.data(QtCore.Qt.UserRole) if it else None
        self.accept()


# ===================================================================== canvas
class NodeView(QtWidgets.QGraphicsView):
    def __init__(self, editor):
        super().__init__(editor.scene)
        self.editor = editor
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setSceneRect(-20000, -20000, 40000, 40000)
        self.drag_wire = None
        self.drag_from = None
        self._pan = None
        self.setAcceptDrops(True)

    def drawBackground(self, qp, rect):
        c = self.editor.colors
        qp.fillRect(rect, QtGui.QColor(c["view2d"]))
        step = 24
        left, top = int(math.floor(rect.left() / step) * step), int(math.floor(rect.top() / step) * step)
        qp.setPen(QtGui.QPen(QtGui.QColor(c["grid"]), 1))
        x = left
        while x < rect.right():
            qp.drawLine(QtCore.QPointF(x, rect.top()), QtCore.QPointF(x, rect.bottom()))
            x += step
        y = top
        while y < rect.bottom():
            qp.drawLine(QtCore.QPointF(rect.left(), y), QtCore.QPointF(rect.right(), y))
            y += step

    def wheelEvent(self, ev):
        f = 1.15 ** (ev.angleDelta().y() / 120)
        s = self.transform().m11() * f
        if 0.15 < s < 3:
            self.scale(f, f)

    def _port_at(self, pos):
        for it in self.items(pos):
            if isinstance(it, PortItem):
                return it
        return None

    def mousePressEvent(self, ev):
        if ev.button() in (QtCore.Qt.MiddleButton, QtCore.Qt.RightButton) and not isinstance(self.itemAt(ev.position().toPoint()), (NodeItem, PortItem)):
            self._pan = ev.position()
            self._pan_moved = False
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            return
        port = self._port_at(ev.position().toPoint())
        if ev.button() == QtCore.Qt.LeftButton and port is not None:
            ed = self.editor
            if port.is_input:
                existing = [w for w in ed.graph["wires"] if w[2] == port.node.nid and w[3] == port.name]
                if existing:  # pick up the wire from its input
                    w = existing[-1]
                    ed.remove_wire(w)
                    src = ed.items[w[0]].out_ports[w[1]]
                    self.drag_from = src
                else:
                    self.drag_from = port
            else:
                self.drag_from = port
            self.drag_wire = WireItem(self.editor)
            self.scene().addItem(self.drag_wire)
            p = self.drag_from.scene_pos()
            self.drag_wire.set_ends(p, p)
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._pan is not None:
            d = ev.position() - self._pan
            self._pan = ev.position()
            if d.manhattanLength() > 1:
                self._pan_moved = True
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        if self.drag_wire is not None:
            p = self.mapToScene(ev.position().toPoint())
            a = self.drag_from.scene_pos()
            if self.drag_from.is_input:
                self.drag_wire.set_ends(p, a)
            else:
                self.drag_wire.set_ends(a, p)
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._pan is not None:
            self._pan = None
            self.setCursor(QtCore.Qt.ArrowCursor)
            if ev.button() == QtCore.Qt.RightButton and not self._pan_moved:
                self.editor.open_search(self.mapToScene(ev.position().toPoint()))
            return
        if self.drag_wire is not None:
            self.scene().removeItem(self.drag_wire)
            self.drag_wire = None
            tgt = self._port_at(ev.position().toPoint())
            src = self.drag_from
            self.drag_from = None
            if tgt is not None and tgt.is_input != src.is_input and tgt.node is not src.node:
                out_p, in_p = (src, tgt) if not src.is_input else (tgt, src)
                add = bool(ev.modifiers() & QtCore.Qt.ShiftModifier)
                self.editor.connect(out_p.node.nid, out_p.name, in_p.node.nid, in_p.name, replace=not add)
            elif tgt is None:
                self.editor.refresh_after_wire_change()
            return
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        if self.itemAt(ev.position().toPoint()) is None:
            self.editor.open_search(self.mapToScene(ev.position().toPoint()))
            return
        super().mouseDoubleClickEvent(ev)

    def keyPressEvent(self, ev):
        if isinstance(self.scene().focusItem(), QtWidgets.QGraphicsProxyWidget):
            return super().keyPressEvent(ev)
        if ev.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            self.editor.delete_selected()
            return
        if ev.key() == QtCore.Qt.Key_Space or (ev.key() == QtCore.Qt.Key_Tab):
            self.editor.open_search(self.mapToScene(self.mapFromGlobal(QtGui.QCursor.pos())))
            return
        super().keyPressEvent(ev)

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasText():
            ev.acceptProposedAction()

    def dragMoveEvent(self, ev):
        ev.acceptProposedAction()

    def dropEvent(self, ev):
        name = ev.mimeData().text()
        if name in NODE_TYPES:
            self.editor.add_node(name, self.mapToScene(ev.position().toPoint()))


class NodeLibraryTree(QtWidgets.QTreeWidget):
    def __init__(self, editor):
        super().__init__()
        self.editor = editor
        self.setHeaderHidden(True)
        self.setDragEnabled(True)
        for cat, names in sorted(categories().items()):
            top = QtWidgets.QTreeWidgetItem([cat])
            f = top.font(0)
            f.setBold(True)
            top.setFont(0, f)
            for n in sorted(names):
                it = QtWidgets.QTreeWidgetItem([n])
                it.setToolTip(0, NODE_TYPES[n].desc)
                top.addChild(it)
            self.addTopLevelItem(top)
        self.itemDoubleClicked.connect(self._add)

    def _add(self, it, col):
        if it.parent() is not None:
            v = self.editor.view
            self.editor.add_node(it.text(0), v.mapToScene(v.viewport().rect().center()))

    def mimeData(self, items):
        md = QtCore.QMimeData()
        if items and items[0].parent() is not None:
            md.setText(items[0].text(0))
        return md


# ===================================================================== editor
class NodeEditor(QtWidgets.QWidget):
    preview_changed = QtCore.Signal(list)
    message = QtCore.Signal(str)

    def __init__(self, doc, settings, parent=None):
        super().__init__(parent)
        self.doc, self.settings = doc, settings
        self.colors = settings.palette()
        self.graph = {"nodes": {}, "wires": []}
        self.items: dict[str, NodeItem] = {}
        self.wire_items = {}
        self.results = {}
        self.dirty = set()
        self.auto = True
        self.compute = ComputeManager(settings, self)
        self.compute.node_started.connect(self._on_start)
        self.compute.node_progress.connect(self._on_progress)
        self.compute.node_result.connect(self._on_result)
        self.compute.node_error.connect(self._on_error)
        self.compute.finished.connect(self._on_finished)
        self.compute.disable_node.connect(self.disable_node)
        self.scene = QtWidgets.QGraphicsScene(self)
        self.view = NodeView(self)
        self.solve_timer = QtCore.QTimer(self)
        self.solve_timer.setSingleShot(True)
        self.solve_timer.setInterval(120)
        self.solve_timer.timeout.connect(self.solve)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar = QtWidgets.QToolBar()
        self.auto_btn = QtWidgets.QToolButton()
        self.auto_btn.setText("Auto solve")
        self.auto_btn.setCheckable(True)
        self.auto_btn.setChecked(True)
        self.auto_btn.toggled.connect(self._set_auto)
        bar.addWidget(self.auto_btn)
        for text, fn in (("Solve (F5)", self.solve_all), ("Add node (Space)", lambda: self.open_search(None)),
                         ("Bake selected", self.bake_selected), ("Frame all", self.frame_all), ("Clear", self.clear_confirm)):
            b = QtWidgets.QToolButton()
            b.setText(text)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addSeparator()
        bar.addWidget(QtWidgets.QLabel(" Bake to "))
        self.bake_target = QtWidgets.QComboBox()
        self.bake_target.addItems(["Model (3D)", "Draft (2D)"])
        bar.addWidget(self.bake_target)
        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("Muted")
        sp = QtWidgets.QWidget()
        sp.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        bar.addWidget(sp)
        bar.addWidget(self.status)
        lay.addWidget(bar)
        lay.addWidget(self.view)
        QtGui.QShortcut(QtGui.QKeySequence("F5"), self, self.solve_all)
        doc.subscribe(self._doc_changed)

    # ------------------------------------------------------------ appearance
    def apply_appearance(self):
        self.colors = self.settings.palette()
        self.scene.update()
        self.view.viewport().update()

    # ------------------------------------------------------------ graph editing
    def add_node(self, type_name, pos=None, props=None, values=None, nid=None, solve=True):
        nid = nid or uuid.uuid4().hex[:8]
        t = NODE_TYPES[type_name]
        props = dict(props or {})
        if t.widget == "slider":
            props.setdefault("value", 5.0)
            props.setdefault("min", 0.0)
            props.setdefault("max", 10.0)
            props.setdefault("decimals", 1)
        elif t.widget in ("number", "toggle", "text"):
            props.setdefault("value", {"number": 0.0, "toggle": False, "text": ""}[t.widget])
        elif t.widget == "code":
            props.setdefault("code", DEFAULT_PY_CODE)
            props.setdefault("inputs", ["x", "y"])
            props.setdefault("outputs", ["a"])
        elif t.widget == "expr":
            props.setdefault("expr", "x * 2")
        elif t.widget == "ref":
            props.setdefault("ref_ids", [])
        elif t.widget == "component":
            if "def_id" not in props:
                defs = list(self.doc.definitions.values())
                if not defs:
                    from .. import components as C
                    for d in C.library():
                        self.doc.definitions.setdefault(d.id, d)
                    defs = list(self.doc.definitions.values())
                props["def_id"] = defs[0].id if defs else None
        p = pos if pos is not None else self.view.mapToScene(self.view.viewport().rect().center())
        self.graph["nodes"][nid] = {"type": type_name, "props": props, "values": dict(values or {}),
                                    "pos": [p.x(), p.y()] if hasattr(p, "x") else list(p), "preview": True}
        item = NodeItem(self, nid)
        self.items[nid] = item
        self.scene.addItem(item)
        item.setPos(*self.graph["nodes"][nid]["pos"])
        self.mark_dirty([nid], solve=solve)
        return nid

    def connect(self, s, o, d, i, replace=True):
        if s == d or s in downstream(self.graph, [d]) and d in self._upstream(s) or d in self._upstream(s):
            self.message.emit("That connection would create a loop.")
            return
        if replace:
            for w in [w for w in self.graph["wires"] if w[2] == d and w[3] == i]:
                self.remove_wire(w, refresh=False)
        w = [s, o, d, i]
        if w in self.graph["wires"]:
            return
        self.graph["wires"].append(w)
        self._add_wire_item(w)
        self.mark_dirty([d])

    def _upstream(self, nid):
        out, frontier = set(), {nid}
        while frontier:
            nxt = set()
            for s, _, d, _ in self.graph["wires"]:
                if d in frontier and s not in out:
                    out.add(s)
                    nxt.add(s)
            frontier = nxt
        return out

    def _add_wire_item(self, w):
        it = WireItem(self, w)
        self.scene.addItem(it)
        self.wire_items[tuple(w)] = it
        self._place_wire(w)

    def _place_wire(self, w):
        it = self.wire_items.get(tuple(w))
        if it is None or w[0] not in self.items or w[2] not in self.items:
            return
        a = self.items[w[0]].out_ports.get(w[1])
        b = self.items[w[2]].in_ports.get(w[3])
        if a is None or b is None:
            it.setVisible(False)
            return
        it.setVisible(True)
        it.set_ends(a.scene_pos(), b.scene_pos())

    def update_wires(self, nid):
        for w in self.graph["wires"]:
            if nid in (w[0], w[2]):
                self._place_wire(w)

    def remove_wire(self, w, refresh=True):
        if w in self.graph["wires"]:
            self.graph["wires"].remove(w)
        it = self.wire_items.pop(tuple(w), None)
        if it is not None:
            self.scene.removeItem(it)
        self.mark_dirty([w[2]], solve=refresh)

    def refresh_after_wire_change(self):
        for it in self.items.values():
            it.update()

    def delete_selected(self):
        for it in self.scene.selectedItems():
            if isinstance(it, WireItem) and it.wire in self.graph["wires"]:
                self.remove_wire(it.wire)
        nids = [it.nid for it in self.scene.selectedItems() if isinstance(it, NodeItem)]
        self.delete_nodes(nids)

    def delete_nodes(self, nids):
        if not nids:
            return
        affected = set()
        for nid in nids:
            for w in [w for w in self.graph["wires"] if nid in (w[0], w[2])]:
                if w[0] == nid:
                    affected.add(w[2])
                self.remove_wire(w, refresh=False)
            it = self.items.pop(nid, None)
            if it is not None:
                self.scene.removeItem(it)
            self.graph["nodes"].pop(nid, None)
            self.results.pop(nid, None)
            self.dirty.discard(nid)
        self.mark_dirty(affected - set(nids))
        self.emit_preview()

    def clear_confirm(self):
        if QtWidgets.QMessageBox.question(self, "Clear", "Remove all nodes?") == QtWidgets.QMessageBox.Yes:
            self.delete_nodes(list(self.graph["nodes"]))

    # ------------------------------------------------------------ widgets on nodes
    def make_widget(self, item):
        t = item.ntype
        d = item.data
        props = d["props"]
        if t.widget == "slider":
            w = QtWidgets.QWidget()
            l = QtWidgets.QVBoxLayout(w)
            l.setContentsMargins(0, 0, 0, 0)
            l.setSpacing(2)
            row = QtWidgets.QHBoxLayout()
            val = QtWidgets.QLabel()
            val.setObjectName("Heading")
            cfg = QtWidgets.QToolButton()
            cfg.setText("Range")
            row.addWidget(val, 1)
            row.addWidget(cfg)
            sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            sl.setRange(0, 1000)
            lo, hi = float(props["min"]), float(props["max"])
            sl.setValue(int((float(props["value"]) - lo) / ((hi - lo) or 1) * 1000))
            dec = int(props.get("decimals", 1))
            val.setText(f"{float(props['value']):.{dec}f}")

            def moved(v, nid=item.nid):
                pr = self.graph["nodes"][nid]["props"]
                lo, hi = float(pr["min"]), float(pr["max"])
                x = lo + (hi - lo) * v / 1000
                dd = int(pr.get("decimals", 1))
                x = round(x, dd) if dd > 0 else int(round(x))
                pr["value"] = x
                val.setText(f"{float(x):.{dd}f}")
                self.mark_dirty([nid])
            sl.valueChanged.connect(moved)
            cfg.clicked.connect(lambda _=False, nid=item.nid: self.edit_slider(nid))
            l.addLayout(row)
            l.addWidget(sl)
            return w
        if t.widget == "number":
            w = QtWidgets.QDoubleSpinBox()
            w.setRange(-1e9, 1e9)
            w.setDecimals(4)
            w.setValue(float(props.get("value", 0)))
            w.setKeyboardTracking(False)
            w.valueChanged.connect(lambda v, nid=item.nid: self._set_prop(nid, "value", v))
            return w
        if t.widget == "toggle":
            w = QtWidgets.QCheckBox("True")
            w.setChecked(bool(props.get("value")))
            w.toggled.connect(lambda v, nid=item.nid: self._set_prop(nid, "value", v))
            return w
        if t.widget == "text":
            txt = str(props.get("value", ""))
            if "\n" in txt or props.get("multiline"):
                w = QtWidgets.QPlainTextEdit(txt)
                w.setFixedSize(300, 150)
                w.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
                timer = QtCore.QTimer(w)
                timer.setSingleShot(True)
                timer.setInterval(700)
                timer.timeout.connect(lambda nid=item.nid, w=w: self._set_prop(nid, "value", w.toPlainText()))
                w.textChanged.connect(timer.start)
                return w
            w = QtWidgets.QLineEdit(txt)
            w.editingFinished.connect(lambda nid=item.nid, w=w: self._set_prop(nid, "value", w.text()))
            return w
        if t.widget == "expr":
            w = QtWidgets.QLineEdit(str(props.get("expr", "x")))
            w.setToolTip("Math expression using x, y, z")
            w.editingFinished.connect(lambda nid=item.nid, w=w: self._set_prop(nid, "expr", w.text()))
            return w
        if t.widget == "panel":
            w = QtWidgets.QPlainTextEdit()
            w.setReadOnly(True)
            w.setFixedHeight(90)
            w.setPlainText("(no data)")
            item.panel_widget = w
            return w
        if t.widget == "code":
            w = QtWidgets.QPushButton("Edit code")
            w.clicked.connect(lambda _=False, nid=item.nid: self.edit_code(nid))
            return w
        if t.widget == "ref":
            w = QtWidgets.QWidget()
            l = QtWidgets.QVBoxLayout(w)
            l.setContentsMargins(0, 0, 0, 0)
            lab = QtWidgets.QLabel(f"{len(props.get('ref_ids', []))} objects referenced")
            lab.setObjectName("Muted")
            b = QtWidgets.QPushButton("Set from selection")
            b.clicked.connect(lambda _=False, nid=item.nid: self.set_refs(nid))
            l.addWidget(lab)
            l.addWidget(b)
            item.ref_label = lab
            return w
        if t.widget == "component":
            w = QtWidgets.QComboBox()
            defs = sorted(self.doc.definitions.values(), key=lambda d: d.name)
            for dd in defs:
                w.addItem(dd.name, dd.id)
            i = w.findData(props.get("def_id"))
            if i >= 0:
                w.setCurrentIndex(i)
            w.activated.connect(lambda i, nid=item.nid, w=w: self._set_prop(nid, "def_id", w.itemData(i)))
            return w
        return None

    def _set_prop(self, nid, key, v):
        self.graph["nodes"][nid]["props"][key] = v
        self.mark_dirty([nid])

    def edit_slider(self, nid):
        pr = self.graph["nodes"][nid]["props"]
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Slider range")
        f = QtWidgets.QFormLayout(dlg)
        mn, mx, dec = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox(), QtWidgets.QSpinBox()
        for s in (mn, mx):
            s.setRange(-1e9, 1e9)
            s.setDecimals(4)
        mn.setValue(float(pr["min"]))
        mx.setValue(float(pr["max"]))
        dec.setRange(0, 6)
        dec.setValue(int(pr.get("decimals", 1)))
        f.addRow("Min", mn)
        f.addRow("Max", mx)
        f.addRow("Decimals (0 = integer)", dec)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        f.addRow(bb)
        if dlg.exec():
            pr["min"], pr["max"], pr["decimals"] = mn.value(), mx.value(), dec.value()
            pr["value"] = min(max(float(pr["value"]), pr["min"]), pr["max"])
            self.items[nid].build()
            self.update_wires(nid)
            self.mark_dirty([nid])

    def edit_input_value(self, nid, port):
        cur = self.graph["nodes"][nid].get("values", {}).get(port)
        text, ok = QtWidgets.QInputDialog.getText(self, f"Set {port}", "Value (number, x,y,z, true/false or text; empty = default):",
                                                  text=fmt_value(cur))
        if ok:
            v = parse_value(text)
            vals = self.graph["nodes"][nid].setdefault("values", {})
            if v is None:
                vals.pop(port, None)
            else:
                vals[port] = v
            self.items[nid].update()
            self.mark_dirty([nid])

    def edit_code(self, nid):
        node = self.graph["nodes"][nid]
        pr = node["props"]
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Python node")
        dlg.resize(720, 560)
        lay = QtWidgets.QVBoxLayout(dlg)
        form = QtWidgets.QFormLayout()
        ins = QtWidgets.QLineEdit(", ".join(pr.get("inputs", [])))
        outs = QtWidgets.QLineEdit(", ".join(pr.get("outputs", [])))
        lists = QtWidgets.QLineEdit(", ".join(pr.get("list_inputs", [])))
        lists.setPlaceholderText("inputs that receive whole lists (others run once per item)")
        form.addRow("Inputs", ins)
        form.addRow("Outputs", outs)
        form.addRow("List inputs", lists)
        lay.addLayout(form)
        ed = QtWidgets.QPlainTextEdit(pr.get("code", ""))
        font = QtGui.QFont("Menlo")
        font.setStyleHint(QtGui.QFont.Monospace)
        ed.setFont(font)
        ed.setTabStopDistance(28)
        lay.addWidget(ed, 1)
        note = QtWidgets.QLabel("Runs in a separate process. Long loops can call progress(fraction). "
                                "If it hangs, use Cancel in the calculation popup.")
        note.setObjectName("Muted")
        note.setWordWrap(True)
        lay.addWidget(note)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec():
            split = lambda s: [x.strip() for x in s.split(",") if x.strip().isidentifier()]
            pr["inputs"] = split(ins.text()) or ["x"]
            pr["outputs"] = [o for o in split(outs.text()) if o != "out"] or ["a"]
            pr["list_inputs"] = split(lists.text())
            pr["code"] = ed.toPlainText()
            valid_in = set(pr["inputs"])
            valid_out = set(pr["outputs"]) | {"out"}
            for w in list(self.graph["wires"]):
                if (w[2] == nid and w[3] not in valid_in) or (w[0] == nid and w[1] not in valid_out):
                    self.remove_wire(w, refresh=False)
            self.items[nid].build()
            self.update_wires(nid)
            self.mark_dirty([nid])

    def set_refs(self, nid):
        ids = sorted(self.doc.selection)
        self.graph["nodes"][nid]["props"]["ref_ids"] = ids
        it = self.items[nid]
        if hasattr(it, "ref_label"):
            it.ref_label.setText(f"{len(ids)} objects referenced")
        if not ids:
            self.message.emit("Nothing selected. Select objects in Draft or Model first, then click 'Set from selection'.")
        self.mark_dirty([nid])

    # ------------------------------------------------------------ menus
    def node_menu(self, item, screen_pos):
        nid = item.nid
        node = self.graph["nodes"][nid]
        m = QtWidgets.QMenu(self)
        a_prev = m.addAction("Preview")
        a_prev.setCheckable(True)
        a_prev.setChecked(node.get("preview", True))
        a_en = m.addAction("Enabled")
        a_en.setCheckable(True)
        a_en.setChecked(not node.get("disabled"))
        a_bake = m.addAction("Bake")
        a_ren = m.addAction("Rename...")
        a_dup = m.addAction("Duplicate")
        m.addSeparator()
        a_del = m.addAction("Delete")
        act = m.exec(screen_pos)
        if act is a_prev:
            node["preview"] = not node.get("preview", True)
            item.update()
            self.emit_preview()
        elif act is a_en:
            node["disabled"] = not node.get("disabled")
            item.update()
            self.mark_dirty([nid])
        elif act is a_bake:
            self.bake([nid])
        elif act is a_ren:
            t, ok = QtWidgets.QInputDialog.getText(self, "Rename node", "Label:", text=node.get("label") or node["type"])
            if ok:
                node["label"] = t
                item.update()
        elif act is a_dup:
            p = item.pos() + QtCore.QPointF(30, 30)
            self.add_node(node["type"], p, props=dict(node["props"]), values=dict(node.get("values", {})))
        elif act is a_del:
            self.delete_nodes([nid])

    def open_search(self, scene_pos):
        dlg = NodeSearch(self)
        dlg.move(QtGui.QCursor.pos() - QtCore.QPoint(20, 20))
        dlg.edit.setFocus()
        if dlg.exec() and dlg.choice:
            self.add_node(dlg.choice, scene_pos)

    def frame_all(self):
        if self.items:
            r = QtCore.QRectF()
            for it in self.items.values():
                r = r.united(it.sceneBoundingRect())
            self.view.fitInView(r.adjusted(-60, -60, 60, 60), QtCore.Qt.KeepAspectRatio)
            s = self.view.transform().m11()
            if s > 1.2:
                self.view.resetTransform()
                self.view.centerOn(r.center())

    # ------------------------------------------------------------ solving
    def _set_auto(self, v):
        self.auto = v
        if v:
            self.solve_timer.start()

    def mark_dirty(self, nids, solve=True):
        d = downstream(self.graph, [n for n in nids if n in self.graph["nodes"]])
        self.dirty |= d
        for n in d:
            if n in self.items and self.items[n].status != "running":
                self.items[n].status = "dirty"
                self.items[n].update()
        if solve and self.auto:
            self.solve_timer.start()

    def solve_all(self):
        self.mark_dirty(list(self.graph["nodes"]), solve=False)
        self.solve()

    def _refs(self):
        refs = {}
        for nid, n in self.graph["nodes"].items():
            if NODE_TYPES[n["type"]].widget == "ref":
                gs = []
                for i in n["props"].get("ref_ids", []):
                    g = self.doc.get(i)
                    if g is not None:
                        gs.append(g)
                refs[nid] = gs
        return refs

    def solve(self):
        if not self.dirty:
            return
        try:
            topo_order(self.graph)
        except ValueError as e:
            self.message.emit(str(e))
            return
        dirty = set(self.dirty)
        needed = {s for s, _, d, _ in self.graph["wires"] if d in dirty and s not in dirty}
        cache = {n: self.results[n] for n in needed if n in self.results}
        names = {nid: (n.get("label") or n["type"]) for nid, n in self.graph["nodes"].items()}
        self.dirty -= dirty
        self._solving = dirty
        self._t0 = QtCore.QElapsedTimer()
        self._t0.start()
        self.status.setText("Calculating...")
        self.compute.solve(self.graph, cache, dirty, self._refs(), dict(self.doc.definitions), names)

    def _on_start(self, nid):
        it = self.items.get(nid)
        if it:
            it.status, it.progress = "running", 0.0
            it.update()

    def _on_progress(self, nid, f):
        it = self.items.get(nid)
        if it:
            it.progress = f
            it.update()

    def _on_result(self, nid, out, elapsed):
        self.results[nid] = out
        it = self.items.get(nid)
        if it:
            it.status, it.message, it.progress = "ok", "", None
            it.setToolTip(f"Computed in {elapsed:.3f}s")
            if getattr(it, "panel_widget", None) is not None:
                it.panel_widget.setPlainText(panel_text(out.get("Data")))
            if it.data["type"] == "Python" and out.get("out"):
                it.setToolTip(f"Computed in {elapsed:.3f}s\n\nOutput:\n{summarize(out.get('out'))}")
            it.update()

    def _on_error(self, nid, msg):
        self.results.pop(nid, None)
        it = self.items.get(nid)
        if it:
            disabled = msg == "Disabled"
            it.status = "disabled" if disabled else "error"
            it.message = "" if disabled else msg
            it.progress = None
            it.setToolTip(msg)
            it.update()
        elif msg:
            self.message.emit(msg)

    def _on_finished(self, cancelled):
        if cancelled:
            for nid in getattr(self, "_solving", set()):
                it = self.items.get(nid)
                if it and it.status in ("running", "dirty"):
                    it.status, it.message, it.progress = "cancelled", "Cancelled", None
                    it.update()
            self.status.setText("Calculation cancelled")
            self.message.emit("Node calculation cancelled.")
        else:
            ms = self._t0.elapsed() if hasattr(self, "_t0") else 0
            self.status.setText(f"Solved in {ms / 1000:.2f}s")
        self.emit_preview()
        if self.dirty and self.auto and not cancelled:
            self.solve_timer.start()

    def disable_node(self, nid):
        if nid in self.graph["nodes"]:
            self.graph["nodes"][nid]["disabled"] = True
            self.items[nid].status = "disabled"
            self.items[nid].update()
            self.message.emit(f"Node '{self.graph['nodes'][nid]['type']}' disabled. Re-enable it from its right-click menu.")

    # ------------------------------------------------------------ preview / bake
    def node_geoms(self, nid):
        out = []
        for v in self.results.get(nid, {}).values():
            out += [g for g in flatten(v) if isinstance(g, Geom)]
        return out

    def emit_preview(self):
        gs = []
        for nid, n in self.graph["nodes"].items():
            t = NODE_TYPES[n["type"]]
            if not n.get("preview", True) or t.widget == "ref" or n.get("disabled"):
                continue
            gs += self.node_geoms(nid)
        self.preview_changed.emit(gs)

    def bake_selected(self):
        nids = [it.nid for it in self.scene.selectedItems() if isinstance(it, NodeItem)]
        if not nids:
            self.message.emit("Select one or more nodes to bake.")
            return
        self.bake(nids)

    def bake(self, nids):
        space = "3d" if self.bake_target.currentIndex() == 0 else "2d"
        gs = []
        for nid in nids:
            for g in self.node_geoms(nid):
                c = g.copy(keep_id=False, space=space, layer=self.doc.current_layer)
                if space == "2d" and c.kind != "instance":
                    gs += G.project_to_plane_2d([c]) if c.kind in ("mesh", "brep") else [c.transformed(np.diag([1, 1, 0, 1.0]))]
                else:
                    gs.append(c)
        if not gs:
            self.message.emit("Nothing to bake (no geometry on those nodes).")
            return
        for g in gs:
            g.space = space
            g.layer = self.doc.current_layer
        self.doc.checkpoint("Bake")
        self.doc.add(gs)
        self.message.emit(f"Baked {len(gs)} objects to the {'Model' if space == '3d' else 'Draft'} workspace.")

    # ------------------------------------------------------------ doc sync / io
    def _doc_changed(self, what):
        if what == "changed":
            stale = []
            for nid, n in self.graph["nodes"].items():
                if NODE_TYPES[n["type"]].widget == "ref" and n["props"].get("ref_ids"):
                    stale.append(nid)
            if stale:
                self.mark_dirty(stale)
        elif what == "definitions":
            comp = [nid for nid, n in self.graph["nodes"].items() if NODE_TYPES[n["type"]].widget == "component"]
            if comp:
                self.mark_dirty(comp)
        elif what == "loaded":
            self.load_graph(self.doc.node_graph)

    def to_dict(self):
        return {"nodes": self.graph["nodes"], "wires": self.graph["wires"]}

    def load_graph(self, g):
        for it in list(self.items.values()):
            self.scene.removeItem(it)
        for it in list(self.wire_items.values()):
            self.scene.removeItem(it)
        self.items, self.wire_items, self.results, self.dirty = {}, {}, {}, set()
        self.graph = {"nodes": {}, "wires": []}
        if not g:
            self.emit_preview()
            return
        for nid, n in g.get("nodes", {}).items():
            if n["type"] not in NODE_TYPES:
                continue
            vals = {k: (tuple(v) if isinstance(v, list) else v) for k, v in n.get("values", {}).items()}
            self.graph["nodes"][nid] = dict(n, values=vals)
            item = NodeItem(self, nid)
            self.items[nid] = item
            self.scene.addItem(item)
            item.setPos(*n.get("pos", [0, 0]))
        for w in g.get("wires", []):
            if w[0] in self.items and w[2] in self.items:
                self.graph["wires"].append(list(w))
                self._add_wire_item(list(w))
        self.mark_dirty(list(self.graph["nodes"]))
        QtCore.QTimer.singleShot(50, self.frame_all)

    def shutdown(self):
        self.compute.shutdown()

    # ------------------------------------------------------------ Grasshopper
    GH_UNITS = [("Metres", "m"), ("Millimetres", "mm"), ("Centimetres", "cm"), ("Feet", "ft"), ("Inches", "in")]

    def import_grasshopper(self, path, doc_units="mm"):
        from .. import ghimport
        from .userlib import UNIT_TO_M
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Import Grasshopper definition")
        lay = QtWidgets.QFormLayout(dlg)
        lay.addRow(QtWidgets.QLabel(f"<b>{os.path.basename(path)}</b>"))
        units = QtWidgets.QComboBox()
        for label, u in self.GH_UNITS:
            units.addItem(label, u)
        units.setToolTip("Units of the Rhino file this definition was made in. Length sliders that feed Forma's "
                         "built-in tools are converted to this model's units.")
        lay.addRow("Lengths in the Grasshopper file", units)
        mode = QtWidgets.QComboBox()
        mode.addItems(["Add beside the current graph", "Replace the current graph"])
        if self.graph["nodes"]:
            lay.addRow("Graph", mode)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addRow(bb)
        if not dlg.exec():
            return
        scale = UNIT_TO_M.get(units.currentData(), 1.0) / UNIT_TO_M.get(doc_units, 0.001)
        try:
            g, report = ghimport.convert(path, scale)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Grasshopper import failed", str(e))
            return
        if mode.currentIndex() == 1 and self.graph["nodes"]:
            self.delete_nodes(list(self.graph["nodes"]))
        dx = dy = 0.0
        if self.graph["nodes"] and g["nodes"]:
            xs = [n["pos"][0] for n in self.graph["nodes"].values()]
            ys = [n["pos"][1] for n in self.graph["nodes"].values()]
            gx = [n["pos"][0] for n in g["nodes"].values()]
            gy = [n["pos"][1] for n in g["nodes"].values()]
            dx, dy = max(xs) + 400 - min(gx), min(ys) - min(gy)
        for nid, n in g["nodes"].items():
            props = n["props"]
            new = self.add_node(n["type"], QtCore.QPointF(n["pos"][0] + dx, n["pos"][1] + dy), props=props,
                                values=n.get("values"), nid=nid, solve=False)
            self.graph["nodes"][new]["preview"] = n.get("preview", True)
            if n.get("label"):
                self.graph["nodes"][new]["label"] = n["label"]
            self.items[new].build()
            self.items[new].update()
        for s, o, d, i in g["wires"]:
            self.connect(s, o, d, i)
        self.mark_dirty(list(g["nodes"]))
        QtCore.QTimer.singleShot(60, self.frame_all)
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Grasshopper import")
        box.setText(report[0])
        box.setInformativeText("\n\n".join(report[1:]) or "Everything came across.")
        box.exec()

    # ------------------------------------------------------------ demo
    EXAMPLES = ["Parametric stadium", "Twisting diagrid tower", "Free-form pavilion", "Tower: solar radiation",
                "Sun hours + energy", "Daylight factor (room)", "Shadow study", "Solar gains by orientation",
                "Space syntax (plan)", "Walking distance (plan)"]

    def load_example(self, name):
        """Ready-made analysis graphs from the Climate, Daylight, Energy and Space Syntax libraries."""
        if self.graph["nodes"]:
            r = QtWidgets.QMessageBox.question(self, "Load example",
                                               "Replace the current graph with this example?\n(No adds it beside your nodes.)",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel)
            if r == QtWidgets.QMessageBox.Cancel:
                return
            if r == QtWidgets.QMessageBox.Yes:
                self.delete_nodes(list(self.graph["nodes"]))
        if name == "Sun hours + energy":
            return self.load_analysis_demo()
        c = lambda x, y: QtCore.QPointF(x, y)
        add = lambda t, x, y, **kw: self.add_node(t, c(x, y), solve=False, **kw)
        N = self.graph["nodes"]

        def label(nid, text):
            N[nid]["label"] = text

        def hide(*ids):
            for i in ids:
                N[i]["preview"] = False
        def slider(x, y, text, value, lo, hi, dec=0):
            nid = add("Number Slider", x, y, props={"value": value, "min": lo, "max": hi, "decimals": dec})
            label(nid, text)
            return nid
        if name == "Parametric stadium":
            st = add("Stadium", -300, -40)
            for k, (lab, port, v, lo, hi, dec) in enumerate((
                    ("Length (mm)", "Length", 230000, 150000, 320000, 0), ("Width (mm)", "Width", 180000, 120000, 260000, 0),
                    ("Roof height", "Height", 48000, 30000, 70000, 0), ("Seating tiers", "Tiers", 3, 1, 5, 0),
                    ("Facade ribs", "Ribs", 56, 16, 120, 0), ("Roof saddle", "Saddle", 9000, 0, 25000, 0),
                    ("Squareness", "Squareness", 2.6, 2.0, 5.0, 1))):
                sl = slider(-640, -300 + k * 90, lab, v, lo, hi, dec)
                self.connect(sl, "Value", st, port)
        elif name == "Twisting diagrid tower":
            tw = add("Diagrid Tower", -300, -40)
            for k, (lab, port, v, lo, hi, dec) in enumerate((
                    ("Height (mm)", "Height", 240000, 80000, 400000, 0), ("Twist (deg)", "Twist", 110, 0, 360, 0),
                    ("Taper", "Taper", 0.62, 0.3, 1.0, 2), ("Lobes", "Lobes", 3, 2, 8, 0),
                    ("Diagrid modules", "Modules", 10, 4, 24, 0), ("Columns", "Columns", 24, 8, 48, 0))):
                sl = slider(-640, -260 + k * 90, lab, v, lo, hi, dec)
                self.connect(sl, "Value", tw, port)
        elif name == "Free-form pavilion":
            pv_ = add("Free-form Pavilion", -300, -40)
            for k, (lab, port, v, lo, hi, dec) in enumerate((
                    ("Length (mm)", "Length", 110000, 50000, 200000, 0), ("Height (mm)", "Height", 19000, 8000, 35000, 0),
                    ("Waves", "Waves", 3, 1, 6, 1), ("Columns", "Columns", 5, 2, 9, 0))):
                sl = slider(-640, -180 + k * 90, lab, v, lo, hi, dec)
                self.connect(sl, "Value", pv_, port)
        elif name == "Tower: solar radiation":
            loc = add("Location", -660, -260, values={"City": "Adelaide"})
            tw = add("Diagrid Tower", -660, -60, values={"Height": 200000, "Twist": 90})
            st = add("Stadium", -660, 200, values={"Length": 200000, "Width": 160000})
            mv = add("Move", -420, 220, values={"Vector": (0, -190000, 0)})
            rad = add("Radiation Map", -200, -100, values={"Grid Size": 7000})
            leg = add("Panel", 80, -100)
            self.connect(st, "Roof", mv, "Geometry")
            for o in ("Latitude", "Longitude", "Time Zone"):
                self.connect(loc, o, rad, o)
            self.connect(tw, "Skin", rad, "Surfaces")
            self.connect(mv, "Geometry", rad, "Context")
            self.connect(rad, "Legend", leg, "Data")
            for n_ in (tw,):
                N[n_]["preview"] = False
        elif name == "Daylight factor (room)":
            outer = add("Box", -600, -220, values={"X": 6000, "Y": 5000, "Z": 3000})
            inner = add("Box", -600, -60, values={"Corner": (200, 200, 200), "X": 5600, "Y": 4600, "Z": 2600})
            win = add("Box", -600, 100, values={"Corner": (1500, -100, 900), "X": 3000, "Y": 400, "Z": 1500})
            label(win, "Window opening")
            room = add("Difference", -360, -60)
            floor = add("Rectangle", -360, 180, values={"Center": (3000, 2500, 0), "Width": 5600, "Height": 4600})
            df = add("Daylight Factor", -120, 0, values={"Grid Size": 400, "Work Plane Height": 850})
            leg = add("Panel", 160, 0)
            self.connect(outer, "Box", room, "A")
            self.connect(inner, "Box", room, "B")
            self.connect(win, "Box", room, "B", replace=False)
            self.connect(floor, "Rectangle", df, "Floor")
            self.connect(room, "Result", df, "Context")
            self.connect(df, "Legend", leg, "Data")
            hide(outer, inner, win, floor)
        elif name == "Shadow study":
            loc = add("Location", -620, -220, values={"City": "Adelaide"})
            hour = add("Number Slider", -620, -60, props={"value": 10, "min": 6, "max": 18, "decimals": 1})
            label(hour, "Hour")
            month = add("Number Slider", -620, 40, props={"value": 6, "min": 1, "max": 12, "decimals": 0})
            label(month, "Month")
            b1 = add("Box", -620, 140, values={"X": 12000, "Y": 8000, "Z": 9000})
            b2 = add("Box", -620, 280, values={"Corner": (16000, 4000, 0), "X": 6000, "Y": 6000, "Z": 24000})
            sh = add("Shadow Study", -300, -40)
            for o in ("Latitude", "Longitude", "Time Zone"):
                self.connect(loc, o, sh, o)
            self.connect(hour, "Value", sh, "Hour")
            self.connect(month, "Value", sh, "Month")
            self.connect(b1, "Box", sh, "Geometry")
            self.connect(b2, "Box", sh, "Geometry", replace=False)
        elif name == "Solar gains by orientation":
            loc = add("Location", -620, -200, values={"City": "Adelaide"})
            b = add("Box", -620, -20, values={"X": 15000, "Y": 9000, "Z": 6000})
            ctx = add("Box", -620, 140, values={"Corner": (-2000, 13000, 0), "X": 20000, "Y": 4000, "Z": 12000})
            label(ctx, "Neighbour")
            sg = add("Solar Gains", -300, -60, values={"Window Ratio": 0.35, "SHGC": 0.5})
            env = add("Envelope by Orientation", -300, 240)
            p1 = add("Panel", 0, -60)
            p2 = add("Panel", 0, 240)
            for o in ("Latitude", "Longitude", "Time Zone"):
                self.connect(loc, o, sg, o)
            self.connect(b, "Box", sg, "Building")
            self.connect(ctx, "Box", sg, "Context")
            self.connect(b, "Box", env, "Building")
            self.connect(sg, "Report", p1, "Data")
            self.connect(env, "Report", p2, "Data")
            hide(b, env)
        elif name in ("Space syntax (plan)", "Walking distance (plan)"):
            bd = add("Rectangle", -620, -220, values={"Center": (10000, 6000, 0), "Width": 20000, "Height": 12000})
            label(bd, "Plan outline")
            w1 = add("Line", -620, -60, values={"Start": (8000, 0, 0), "End": (8000, 8000, 0)})
            w2 = add("Line", -620, 60, values={"Start": (8000, 8000, 0), "End": (14000, 8000, 0)})
            w3 = add("Line", -620, 180, values={"Start": (14000, 4000, 0), "End": (20000, 4000, 0)})
            walls = [w1, w2, w3]
            if name == "Space syntax (plan)":
                an = add("Visibility Graph", -300, -60, values={"Grid Size": 500, "Metric": "Integration"})
                self.connect(bd, "Rectangle", an, "Boundary")
            else:
                an = add("Walking Distance", -300, -60, values={"Grid Size": 400})
                door = add("Point", -620, 300, values={"X": 1000, "Y": 1000, "Z": 0})
                label(door, "Entrance")
                self.connect(bd, "Rectangle", an, "Boundary")
                self.connect(door, "Point", an, "Origins")
            for k, wl in enumerate(walls):
                self.connect(wl, "Line", an, "Obstacles", replace=k == 0)
            leg = add("Panel", 0, -60)
            self.connect(an, "Legend", leg, "Data")
        else:
            return
        QtCore.QTimer.singleShot(50, self.frame_all)
        self.mark_dirty(list(N))
        self.message.emit(f"Example loaded: {name}")

    def load_analysis_demo(self):
        """Sun hours on a building shaded by a tower, a visibility graph of a plan, and an energy report."""
        c = lambda x, y: QtCore.QPointF(x, y)
        month = self.add_node("Number Slider", c(-560, -200), props={"value": 6, "min": 1, "max": 12, "decimals": 0}, solve=False)
        self.graph["nodes"][month]["label"] = "Month"
        bld = self.add_node("Box", c(-560, -40), values={"X": 12000, "Y": 8000, "Z": 6000}, solve=False)
        tower = self.add_node("Box", c(-560, 120), values={"Corner": (16000, -2000, 0), "X": 6000, "Y": 6000, "Z": 24000}, solve=False)
        sun = self.add_node("Sun Exposure", c(-200, -120), values={"Grid Size": 1500}, solve=False)
        en = self.add_node("Energy Balance", c(-200, 200), solve=False)
        rep = self.add_node("Panel", c(120, 200), solve=False)
        leg = self.add_node("Panel", c(120, -160), solve=False)
        self.connect(month, "Value", sun, "Month")
        self.connect(bld, "Box", sun, "Surfaces")
        self.connect(tower, "Box", sun, "Context")
        self.connect(bld, "Box", en, "Building")
        self.connect(en, "Report", rep, "Data")
        self.connect(sun, "Legend", leg, "Data")
        self.graph["nodes"][bld]["preview"] = False
        self.graph["nodes"][en]["preview"] = False   # its heat map would sit on top of the sun map
        QtCore.QTimer.singleShot(50, self.frame_all)
        self.mark_dirty(list(self.graph["nodes"]))

    def load_demo(self):
        """A small starter graph: slider-driven twisted tower of rectangles, lofted."""
        c = lambda x, y: QtCore.QPointF(x, y)
        s1 = self.add_node("Number Slider", c(-520, -120), props={"value": 12, "min": 2, "max": 40, "decimals": 0}, solve=False)
        self.graph["nodes"][s1]["label"] = "Floors"
        s2 = self.add_node("Number Slider", c(-520, 10), props={"value": 4.0, "min": 0, "max": 15, "decimals": 1}, solve=False)
        self.graph["nodes"][s2]["label"] = "Twist per floor"
        rng = self.add_node("Series", c(-300, -140), values={"Start": 0, "Step": 30}, solve=False)
        rect = self.add_node("Rectangle", c(-300, 60), values={"Width": 200, "Height": 120}, solve=False)
        mv = self.add_node("Vector", c(-90, -160), values={"X": 0, "Y": 0}, solve=False)
        mul = self.add_node("Expression", c(-90, 60), props={"expr": "x * y"}, solve=False)
        rot = self.add_node("Rotate", c(120, 40), solve=False)
        move = self.add_node("Move", c(320, -60), solve=False)
        loft = self.add_node("Loft", c(520, -40), solve=False)
        self.connect(s1, "Value", rng, "Count")
        self.connect(rng, "Numbers", mv, "Z")
        self.connect(rect, "Rectangle", rot, "Geometry")
        self.connect(rng, "Numbers", mul, "x")
        self.connect(s2, "Value", mul, "y")
        self.connect(mul, "Result", rot, "Angle")
        self.connect(rot, "Geometry", move, "Geometry")
        self.connect(mv, "Vector", move, "Vector")
        self.connect(move, "Geometry", loft, "Curves")
        for n in (rng, mul):
            self.graph["nodes"][n]["preview"] = False
        self.graph["nodes"][rect]["preview"] = False
        self.graph["nodes"][rot]["preview"] = False
        self.items[rot].update()
        QtCore.QTimer.singleShot(50, self.frame_all)
        self.mark_dirty(list(self.graph["nodes"]))
