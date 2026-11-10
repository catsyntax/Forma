"""Plan Designer: the visual brief editor of the Modular Plan Generator.

A Qt port of the web window from the Grasshopper script "Modular Plan Generator": module size, room cards
(size by modules or by area, attached kitchen / bathroom / storage, warnings against the room standards)
and a layered adjacency graph (link / unlink, layers, lock to layer, auto arrange). Generate writes the
brief, links, module and plan count into the node's inputs and computes; plans can also be baked into
GH_PLANS::PLAN_xx::<room> layers like the original.
"""
from __future__ import annotations

import math
import re

from PySide6 import QtCore, QtGui, QtWidgets

from .nodes.userlib import UNIT_TO_M, parse_brief, parse_links, default_edges, engine, DEFAULT_BRIEF

ADD_TYPES = [("living", "Living Room"), ("bedroom", "Bedroom"), ("kitchen", "Kitchen"), ("storage", "Storage"),
             ("corridor", "Corridor")]
LETTER = {"living": "L", "bedroom": "B", "kitchen": "K", "storage": "S", "corridor": "C", "bathroom": "BA"}
SUFFIX = {"kitchen": "KI", "bathroom": "BA", "storage": "ST"}


def att_keys(t):
    if t == "living":
        return [("kitchen", "Attached kitchen"), ("bathroom", "Attached bathroom"), ("storage", "Attached storage")]
    if t in ("bedroom", "corridor"):
        return [("bathroom", "Attached bathroom"), ("storage", "Attached storage")]
    return []


def nn(v):
    return ("%g" % round(v, 1))


class Model:
    """Same state as the original web interface (S)."""

    def __init__(self, std, state=None):
        self.std = std
        s = state or {}
        self.mod = dict(s.get("mod", {"a": 2.4, "b": 3.6, "h": 2.7}))
        self.spaces = [dict(x, att={k: dict(v) for k, v in x.get("att", {}).items()}) for x in s.get("spaces", [])]
        self.counters = dict(s.get("counters", {k: 0 for k, _ in ADD_TYPES}))
        self.layers = list(s.get("layers", ["Layer 1", "Layer 2", "Layer 3"]))
        self.pos = {k: dict(v) for k, v in s.get("pos", {}).items()}
        self.edges = [list(e) for e in s.get("edges", [])]
        self.known = dict(s.get("known", {}))
        self.uid = int(s.get("uid", 0))
        self.plans = int(s.get("plans", 5))

    def state(self):
        return {"mod": self.mod, "spaces": self.spaces, "counters": self.counters, "layers": self.layers,
                "pos": self.pos, "edges": self.edges, "known": self.known, "uid": self.uid, "plans": self.plans}

    # ---- sizes
    def short(self):
        return min(self.mod["a"], self.mod["b"])

    def long(self):
        return max(self.mod["a"], self.mod["b"])

    def mod_area(self):
        return self.short() * self.long()

    def foot(self, t, n):
        s, l = self.short(), self.long()
        return (s, n * l) if t == "corridor" else (n * s, l)

    def min_modules(self, t):
        st = self.std.get(t)
        if not st:
            return 1
        n = 0.5
        while n <= 12:
            w, d = self.foot(t, n)
            if w * d + 1e-9 >= st["min_area"] and min(w, d) + 1e-9 >= st["min_dim"]:
                return n
            n += 0.5
        return 1

    def check(self, t, n):
        st = self.std.get(t)
        if not st:
            return []
        w, d = self.foot(t, n)
        a, md, msgs = w * d, min(w, d), []
        if a + 1e-9 < st["min_area"]:
            msgs.append(f"Area {a:.2f} m² is under the {st['min_area']:.2f} m² working minimum.")
        if md + 1e-9 < st["min_dim"]:
            msgs.append(f"Least dimension {md:.2f} m is under the {st['min_dim']:.2f} m minimum" +
                        (" clear circulation width." if t == "corridor" else "."))
        if self.mod["h"] + 1e-9 < st["min_height"]:
            msgs.append(f"Module height {self.mod['h']:.2f} m is under the {st['min_height']:.2f} m ceiling height for " +
                        ("a habitable room." if st.get("habitable") else "this space."))
        return msgs

    def eff(self, sp):
        if sp.get("mode") == "custom":
            return max(0.5, math.ceil((sp["custom"] / self.mod_area()) * 2) / 2)
        return sp["modules"]

    # ---- spaces
    def add_space(self, t):
        self.counters[t] = self.counters.get(t, 0) + 1
        self.uid += 1
        sp = {"id": f"sp{self.uid}", "type": t, "code": f"{LETTER.get(t, 'X')}{self.counters[t]}",
              "modules": max(1, self.min_modules(t)), "mode": "modules", "custom": round(self.mod_area(), 2), "att": {}}
        for k, _ in att_keys(t):
            sp["att"][k] = {"on": False, "modules": self.min_modules(k)}
        self.spaces.append(sp)
        self.sync()
        return sp

    def remove_space(self, sid):
        self.spaces = [s for s in self.spaces if s["id"] != sid]
        self.sync()

    def find(self, sid):
        return next((s for s in self.spaces if s["id"] == sid), None)

    # ---- graph
    def node_list(self):
        out = []
        for sp in self.spaces:
            out.append({"id": sp["id"], "code": sp["code"], "type": sp["type"], "modules": self.eff(sp), "parent": None})
            for k, _ in att_keys(sp["type"]):
                a = sp["att"].get(k)
                if a and a["on"]:
                    out.append({"id": sp["id"] + "#" + k, "code": sp["code"] + "_" + SUFFIX[k], "type": k,
                                "modules": a["modules"], "parent": sp["id"]})
        return out

    @staticmethod
    def ekey(a, b):
        return (a, b) if a < b else (b, a)

    def has_edge(self, a, b):
        return any(self.ekey(*e) == self.ekey(a, b) for e in self.edges)

    def first_living(self):
        for s in self.spaces:
            if s["type"] == "living":
                return s["id"]
        return self.spaces[0]["id"] if self.spaces else None

    def sync(self):
        nodes = self.node_list()
        ids = {n["id"] for n in nodes}
        root = self.first_living()
        nl = len(self.layers)
        for n in nodes:
            if not self.known.get(n["id"]):
                self.known[n["id"]] = True
                if n["parent"]:
                    pl = self.pos.get(n["parent"], {}).get("layer", 1)
                    lay = pl + 1 if pl + 1 <= nl else pl
                else:
                    lay = 1 if n["id"] == root else min(2, nl)
                taken = [v["x"] for v in self.pos.values() if v["layer"] == lay]
                cands = [0.06 + i * 0.02 for i in range(45)]
                x = max(cands, key=lambda c: min([abs(c - t) for t in taken] or [1.0]) - 0.001 * c)
                self.pos[n["id"]] = {"layer": lay, "x": x}
                if n["parent"]:
                    self.edges.append([n["parent"], n["id"]])
                elif root and n["id"] != root and not self.has_edge(root, n["id"]):
                    self.edges.append([root, n["id"]])
        self.pos = {k: v for k, v in self.pos.items() if k in ids}
        self.known = {k: v for k, v in self.known.items() if k in ids}
        self.edges = [e for e in self.edges if e[0] in ids and e[1] in ids]
        for v in self.pos.values():
            v["layer"] = min(v["layer"], nl)

    def auto_arrange(self):
        """Breadth-first layers from the living room; siblings spread evenly."""
        nodes = self.node_list()
        if not nodes:
            return
        adj = {n["id"]: set() for n in nodes}
        for a, b in self.edges:
            adj[a].add(b)
            adj[b].add(a)
        root = self.first_living() or nodes[0]["id"]
        depth = {root: 0}
        queue = [root]
        while queue:
            u = queue.pop(0)
            for v in sorted(adj[u]):
                if v not in depth:
                    depth[v] = depth[u] + 1
                    queue.append(v)
        far = max(depth.values()) + 1
        for n in nodes:
            depth.setdefault(n["id"], far)
        need = max(depth.values()) + 1
        while len(self.layers) < need:
            self.layers.append(f"Layer {len(self.layers) + 1}")
        rows = {}
        for n in nodes:
            rows.setdefault(depth[n["id"]], []).append(n["id"])
        for d, ids in rows.items():
            for i, nid in enumerate(ids):
                self.pos[nid] = {"layer": d + 1, "x": (i + 1) / (len(ids) + 1)}

    # ---- brief text for the node
    def brief_text(self):
        nodes = self.node_list()
        code = {n["id"]: n["code"] for n in nodes}
        lines = ["# code  type  modules  attached-to   (written by the Plan Designer)"]
        for n in nodes:
            lines.append(f"{n['code']:<8}{n['type']:<10}{nn(n['modules']):<6}" + (code[n['parent']] if n["parent"] else ""))
        links = ", ".join(f"{code[a]}-{code[b]}" for a, b in self.edges if a in code and b in code)
        return "\n".join(lines) + "\n", links

    @staticmethod
    def from_brief(std, brief, links, mod, plans):
        """Rebuild a designer state from the node's text brief (used when the designer opens the first time)."""
        m = Model(std, {"mod": mod, "plans": plans})
        try:
            rows = parse_brief(brief or DEFAULT_BRIEF)
        except ValueError:
            rows = parse_brief(DEFAULT_BRIEF)
        by_code = {}
        for r in rows:
            par = by_code.get(r["parent"]) if r["parent"] else None
            if par is not None and r["type"] in dict(att_keys(par["type"])):
                par["att"].setdefault(r["type"], {})
                par["att"][r["type"]] = {"on": True, "modules": r["modules"]}
                by_code[r["code"]] = {"id": par["id"] + "#" + r["type"], "type": r["type"], "att": {}}
                continue
            m.uid += 1
            sp = {"id": f"sp{m.uid}", "type": r["type"], "code": r["code"], "modules": r["modules"], "mode": "modules",
                  "custom": round(m.mod_area(), 2), "att": {}}
            for k, _ in att_keys(r["type"]):
                sp["att"][k] = {"on": False, "modules": m.min_modules(k)}
            num = re.findall(r"\d+", r["code"])
            if r["type"] in m.counters and num:
                m.counters[r["type"]] = max(m.counters[r["type"]], int(num[-1]))
            m.spaces.append(sp)
            by_code[r["code"]] = sp
        nodes = m.node_list()
        ids_by_code = {n["code"]: n["id"] for n in nodes}
        for n in nodes:
            m.known[n["id"]] = True
        try:
            rows2 = [dict(r, id=r["code"]) for r in rows]
            edges = parse_links(links, rows2) if str(links or "").strip() else default_edges(rows2)
        except ValueError:
            edges = []
        m.edges = [[ids_by_code[a], ids_by_code[b]] for a, b in edges if a in ids_by_code and b in ids_by_code]
        m.auto_arrange()
        return m


# ===================================================================== graph view
class GraphView(QtWidgets.QWidget):
    changed = QtCore.Signal()
    status = QtCore.Signal(str)
    BAND, PAD, R = 112, 16, 23

    def __init__(self, model, tint, parent=None):
        super().__init__(parent)
        self.m, self.tint = model, tint
        self.link_mode = False
        self.lock_layer = True
        self.pick = None
        self.sel_edge = -1
        self.drag = None
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMinimumWidth(360)

    def sizeHint(self):
        return QtCore.QSize(600, self.PAD * 2 + len(self.m.layers) * self.BAND)

    def refit(self):
        self.setMinimumHeight(self.PAD * 2 + len(self.m.layers) * self.BAND)
        self.update()

    def node_xy(self, nid):
        p = self.m.pos[nid]
        W = self.width()
        return QtCore.QPointF(self.PAD + 34 + p["x"] * (W - self.PAD * 2 - 68), self.PAD + (p["layer"] - 0.5) * self.BAND)

    def _edge_path(self, e):
        a, b = self.node_xy(e[0]), self.node_xy(e[1])
        my = (a.y() + b.y()) / 2
        path = QtGui.QPainterPath(a)
        path.cubicTo(QtCore.QPointF(a.x(), my), QtCore.QPointF(b.x(), my), b)
        return path

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor("#ffffff"))
        W = self.width()
        f = p.font()
        for i, name in enumerate(self.m.layers):
            r = QtCore.QRectF(self.PAD, self.PAD + i * self.BAND, W - self.PAD * 2, self.BAND)
            p.setPen(QtGui.QPen(QtGui.QColor("#e4e9ee"), 1))
            p.setBrush(QtGui.QColor("#f7f9fb" if i % 2 else "#fbfcfd"))
            p.drawRoundedRect(r, 8, 8)
            p.setPen(QtGui.QColor("#8a94a2"))
            f.setPointSizeF(8.5)
            f.setBold(False)
            p.setFont(f)
            p.drawText(r.adjusted(10, 6, 0, 0), QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, name.upper())
        for i, e in enumerate(self.m.edges):
            if e[0] not in self.m.pos or e[1] not in self.m.pos:
                continue
            sel = i == self.sel_edge
            p.setPen(QtGui.QPen(QtGui.QColor("#0d6d7c" if sel else "#9aa6b4"), 3.2 if sel else 1.8))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawPath(self._edge_path(e))
        for n in self.m.node_list():
            if n["id"] not in self.m.pos:
                continue
            c = self.node_xy(n["id"])
            if self.pick == n["id"]:
                p.setPen(QtGui.QPen(QtGui.QColor("#0d6d7c"), 3, QtCore.Qt.DashLine))
                p.setBrush(QtCore.Qt.NoBrush)
                p.drawEllipse(c, self.R + 6, self.R + 6)
            p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 2))
            p.setBrush(QtGui.QColor(self.tint.get(n["type"], "#888")))
            p.drawEllipse(c, self.R, self.R)
            f.setPointSizeF(8.5 if len(n["code"]) < 5 else 7)
            f.setBold(True)
            p.setFont(f)
            p.setPen(QtGui.QColor("#ffffff"))
            p.drawText(QtCore.QRectF(c.x() - self.R, c.y() - self.R, 2 * self.R, 2 * self.R), QtCore.Qt.AlignCenter, n["code"])
            f.setBold(False)
            f.setPointSizeF(8)
            p.setFont(f)
            p.setPen(QtGui.QColor("#5a6472"))
            p.drawText(QtCore.QRectF(c.x() - 40, c.y() + self.R + 2, 80, 16), QtCore.Qt.AlignCenter, nn(n["modules"]) + " mod")
        if not self.m.spaces:
            p.setPen(QtGui.QColor("#8a94a2"))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, "Add spaces with the buttons above")
        p.end()

    def _node_at(self, pt):
        for n in reversed(self.m.node_list()):
            if n["id"] in self.m.pos and QtCore.QLineF(self.node_xy(n["id"]), pt).length() <= self.R + 2:
                return n["id"]
        return None

    def _edge_at(self, pt):
        stroker = QtGui.QPainterPathStroker()
        stroker.setWidth(12)
        for i, e in enumerate(self.m.edges):
            if e[0] in self.m.pos and e[1] in self.m.pos and stroker.createStroke(self._edge_path(e)).contains(pt):
                return i
        return -1

    def label(self, nid):
        n = next((x for x in self.m.node_list() if x["id"] == nid), None)
        return n["code"] if n else nid

    def mousePressEvent(self, ev):
        pt = ev.position()
        nid = self._node_at(pt)
        if nid is None:
            i = self._edge_at(pt)
            if i >= 0:
                self.sel_edge = -1 if self.sel_edge == i else i
                self.status.emit("Line deselected." if self.sel_edge < 0 else
                                 "Line selected. Press Delete or use Unlink selected.")
            else:
                self.sel_edge, self.pick = -1, None
            self.update()
            return
        if self.link_mode:
            if self.pick is None:
                self.pick = nid
                self.status.emit(f"Linking from {self.label(nid)}. Now click a second node.")
            elif self.pick == nid:
                self.pick = None
                self.status.emit("Link cancelled.")
            else:
                k = self.m.ekey(self.pick, nid)
                before = len(self.m.edges)
                self.m.edges = [e for e in self.m.edges if self.m.ekey(*e) != k]
                if len(self.m.edges) == before:
                    self.m.edges.append([self.pick, nid])
                    self.status.emit(f"Linked {self.label(self.pick)} to {self.label(nid)}.")
                else:
                    self.status.emit(f"Unlinked {self.label(self.pick)} from {self.label(nid)}.")
                self.pick, self.sel_edge = None, -1
                self.changed.emit()
            self.update()
            return
        self.drag = nid

    def mouseMoveEvent(self, ev):
        if not self.drag:
            n = self._node_at(ev.position())
            self.setCursor(QtCore.Qt.PointingHandCursor if (n or self._edge_at(ev.position()) >= 0) else QtCore.Qt.ArrowCursor)
            return
        p = self.m.pos.get(self.drag)
        if p is None:
            self.drag = None
            return
        W = self.width()
        p["x"] = max(0.0, min(1.0, (ev.position().x() - self.PAD - 34) / max(1, W - self.PAD * 2 - 68)))
        if not self.lock_layer:
            lay = int((ev.position().y() - self.PAD) // self.BAND) + 1
            p["layer"] = max(1, min(len(self.m.layers), lay))
        self.update()

    def mouseReleaseEvent(self, ev):
        if self.drag:
            self.drag = None
            self.changed.emit()

    def keyPressEvent(self, ev):
        if ev.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace) and self.sel_edge >= 0:
            self.delete_edge()
            return
        super().keyPressEvent(ev)

    def delete_edge(self):
        if self.sel_edge < 0:
            self.status.emit("Click a connection line first to select it.")
            return
        e = self.m.edges.pop(self.sel_edge)
        self.status.emit(f"Unlinked {self.label(e[0])} and {self.label(e[1])}.")
        self.sel_edge = -1
        self.changed.emit()
        self.update()


# ===================================================================== window
CSS = """
QDialog#PlanDesigner, QWidget#pd_left, QScrollArea#pd_scroll > QWidget > QWidget { background: #f4f6f8; color: #1b2028; }
QWidget#pd_head, QWidget#pd_gbar { background: #ffffff; border-bottom: 1px solid #d3dae2; }
QLabel { color: #1b2028; }
QLabel#pd_field { color: #8a94a2; font-size: 10px; letter-spacing: 1px; }
QLabel#pd_derived { color: #0d6d7c; background: #e2f2f4; border: 1px solid #bfe0e4; border-radius: 6px; padding: 4px 9px; }
QLabel#pd_hint, QLabel#pd_status { color: #5a6472; font-size: 11px; }
QWidget#pd_statusbar { background: #fbfcfd; border-top: 1px solid #d3dae2; }
QPushButton { background: #ffffff; border: 1px solid #d3dae2; color: #1b2028; border-radius: 7px; padding: 6px 11px; }
QPushButton:hover { border-color: #0d6d7c; color: #0d6d7c; }
QPushButton:checked { background: #e2f2f4; border-color: #0d6d7c; color: #0d6d7c; }
QPushButton#pd_go { background: #0d6d7c; border-color: #0d6d7c; color: #ffffff; font-weight: 600; }
QPushButton#pd_go:hover { background: #0b5b68; color: #ffffff; }
QDoubleSpinBox, QSpinBox, QComboBox { background: #ffffff; border: 1px solid #d3dae2; color: #1b2028; border-radius: 6px;
    padding: 3px 6px; }
QComboBox QAbstractItemView { background: #ffffff; color: #1b2028; selection-background-color: #e2f2f4; selection-color: #0d6d7c; }
QCheckBox { color: #1b2028; }
QFrame#pd_card { background: #ffffff; border: 1px solid #d3dae2; border-radius: 10px; }
QFrame#pd_card[bad="true"] { border-color: #e3cd9b; }
QLabel#pd_notice { background: #fdf3df; border: 1px solid #e3cd9b; color: #8a5a10; border-radius: 7px; padding: 6px 8px; }
QLabel#pd_dims { color: #5a6472; }
"""


class PlanDesigner(QtWidgets.QDialog):
    def __init__(self, editor, nid):
        super().__init__(editor.window())
        self.editor, self.nid = editor, nid
        self.setObjectName("PlanDesigner")
        self.setWindowTitle("Modular Plan Generator — Plan Designer")
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, True)
        self.resize(1320, 820)
        self.setStyleSheet(CSS)
        node = editor.graph["nodes"][nid]
        self.PL = engine("planlib", node["props"])
        self.std = self.PL.STANDARDS
        self.tint = {k: "#%02x%02x%02x" % tuple(v) for k, v in getattr(self.PL, "TYPE_COLOUR", {}).items()}
        self.units = getattr(editor.doc, "units", "mm")
        self.k = UNIT_TO_M.get(self.units, 0.001)   # model units -> metres
        state = node["props"].get("design")
        if state:
            self.m = Model(self.std, state)
        else:
            vals = self._inputs()
            self.m = Model.from_brief(self.std, vals.get("Brief"), vals.get("Links"),
                                      {"a": vals["Module A"] * self.k, "b": vals["Module B"] * self.k,
                                       "h": vals["Module Height"] * self.k}, int(vals.get("Plans", 5)))
        self._build()
        self.render()

    # ---- node values (a wired Text / Slider source wins over the node's own value)
    def _source(self, port):
        for s, o, d, i in self.editor.graph["wires"]:
            if d == self.nid and i == port:
                return s
        return None

    def _inputs(self):
        from .nodes.registry import NODE_TYPES
        node = self.editor.graph["nodes"][self.nid]
        out = {}
        for p in NODE_TYPES[node["type"]].inputs:
            src = self._source(p.name)
            v = None
            if src is not None:
                sn = self.editor.graph["nodes"][src]
                v = sn["props"].get("value")
            if v is None:
                v = node.get("values", {}).get(p.name, p.default)
            out[p.name] = v
        for k in ("Module A", "Module B", "Module Height", "Plans", "Gap"):
            try:
                out[k] = float(out[k])
            except (TypeError, ValueError):
                out[k] = {"Module A": 2400, "Module B": 3600, "Module Height": 2700, "Plans": 5, "Gap": 4000}[k]
        return out

    def _set_input(self, port, value):
        ed = self.editor
        src = self._source(port)
        if src is not None:
            sn = ed.graph["nodes"][src]
            if "value" in sn["props"] or sn["type"] in ("Text", "Number Slider", "Number", "Toggle"):
                if sn["type"] == "Number Slider":
                    pr = sn["props"]
                    pr["min"] = min(pr.get("min", value), value)
                    pr["max"] = max(pr.get("max", value), value)
                sn["props"]["value"] = value
                if src in ed.items:
                    ed.items[src].build()
                ed.mark_dirty([src])
                return
        ed.graph["nodes"][self.nid].setdefault("values", {})[port] = value
        ed.mark_dirty([self.nid])

    # ---- UI
    def _field(self, label, w):
        box = QtWidgets.QHBoxLayout()
        l = QtWidgets.QLabel(label.upper())
        l.setObjectName("pd_field")
        box.addWidget(l)
        box.addWidget(w)
        return box

    def _spin(self, v, lo, step=0.1):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, 50)
        s.setDecimals(2)
        s.setSingleStep(step)
        s.setValue(v)
        s.setFixedWidth(80)
        s.setKeyboardTracking(False)
        s.valueChanged.connect(self._mod_changed)
        return s

    def _build(self):
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        head = QtWidgets.QWidget()
        head.setObjectName("pd_head")
        hl = QtWidgets.QVBoxLayout(head)
        hl.setContentsMargins(14, 10, 14, 9)
        r1 = QtWidgets.QHBoxLayout()
        self.sa = self._spin(self.m.mod["a"], 0.5)
        self.sb = self._spin(self.m.mod["b"], 0.5)
        self.sh = self._spin(self.m.mod["h"], 1.8)
        r1.addLayout(self._field("Module side A", self.sa))
        r1.addSpacing(10)
        r1.addLayout(self._field("Module side B", self.sb))
        r1.addSpacing(10)
        r1.addLayout(self._field("Module height", self.sh))
        r1.addStretch(1)
        self.derived = QtWidgets.QLabel()
        self.derived.setObjectName("pd_derived")
        r1.addWidget(self.derived)
        hl.addLayout(r1)
        r2 = QtWidgets.QHBoxLayout()
        for t, label in ADD_TYPES:
            b = QtWidgets.QPushButton("Add " + label)
            b.clicked.connect(lambda _=False, t=t: self._add(t))
            r2.addWidget(b)
        r2.addSpacing(12)
        self.nplans = QtWidgets.QSpinBox()
        self.nplans.setRange(1, 12)
        self.nplans.setValue(self.m.plans)
        self.nplans.setFixedWidth(60)
        r2.addLayout(self._field("Plans", self.nplans))
        self.bake = QtWidgets.QCheckBox("Bake to model (replace previous)")
        self.bake.setToolTip("Also add the plans to the model on GH_PLANS::PLAN_xx::<room> layers, "
                             "replacing plans baked before. Without it, plans show as the node's preview.")
        r2.addWidget(self.bake)
        go = QtWidgets.QPushButton("Generate plans")
        go.setObjectName("pd_go")
        go.clicked.connect(self.generate)
        r2.addWidget(go)
        r2.addStretch(1)
        hl.addLayout(r2)
        lay.addWidget(head)
        body = QtWidgets.QSplitter()
        body.setHandleWidth(1)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setObjectName("pd_scroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setMinimumWidth(470)
        self.left = QtWidgets.QWidget()
        self.left.setObjectName("pd_left")
        self.cards = QtWidgets.QVBoxLayout(self.left)
        self.cards.setContentsMargins(12, 12, 12, 12)
        self.cards.setSpacing(10)
        self.scroll.setWidget(self.left)
        body.addWidget(self.scroll)
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        gbar = QtWidgets.QWidget()
        gbar.setObjectName("pd_gbar")
        gl = QtWidgets.QHBoxLayout(gbar)
        gl.setContentsMargins(12, 8, 12, 8)
        self.graph = GraphView(self.m, self.tint)
        self.b_link = QtWidgets.QPushButton("Link nodes")
        self.b_link.setCheckable(True)
        self.b_link.toggled.connect(self._link_mode)
        gl.addWidget(self.b_link)
        b = QtWidgets.QPushButton("Unlink selected")
        b.clicked.connect(self.graph.delete_edge)
        gl.addWidget(b)
        gl.addSpacing(8)
        for txt, fn in (("Add layer", self._add_layer), ("Remove last layer", self._remove_layer)):
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(fn)
            gl.addWidget(b)
        self.b_lock = QtWidgets.QPushButton("Locked to layer")
        self.b_lock.setCheckable(True)
        self.b_lock.setChecked(True)
        self.b_lock.toggled.connect(self._lock)
        gl.addWidget(self.b_lock)
        b = QtWidgets.QPushButton("Auto arrange")
        b.clicked.connect(self._arrange)
        gl.addWidget(b)
        hint = QtWidgets.QLabel("Drag a node sideways to reorder. Click a line to select it.\n"
                                "Unlock to drag nodes between layers. Delete removes a selected line.")
        hint.setObjectName("pd_hint")
        gl.addWidget(hint, 1)
        rl.addWidget(gbar)
        gs = QtWidgets.QScrollArea()
        gs.setWidgetResizable(True)
        gs.setStyleSheet("QScrollArea { background: #ffffff; border: none; }")
        gs.setWidget(self.graph)
        rl.addWidget(gs, 1)
        body.addWidget(right)
        body.setSizes([490, 830])
        lay.addWidget(body, 1)
        sb = QtWidgets.QWidget()
        sb.setObjectName("pd_statusbar")
        sl = QtWidgets.QHBoxLayout(sb)
        sl.setContentsMargins(14, 6, 14, 6)
        self.status_lbl = QtWidgets.QLabel("Ready. Set the module, add spaces, then generate.")
        self.status_lbl.setObjectName("pd_status")
        sl.addWidget(self.status_lbl)
        lay.addWidget(sb)
        self.graph.status.connect(self.status_lbl.setText)
        self.graph.changed.connect(self._save_state)

    # ---- actions
    def _mod_changed(self):
        self.m.mod = {"a": self.sa.value(), "b": self.sb.value(), "h": self.sh.value()}
        self.render()

    def _add(self, t):
        sp = self.m.add_space(t)
        self.render()
        self.status_lbl.setText(f"Added {sp['code']} ({self.std.get(t, {}).get('label', t)}).")

    def _link_mode(self, on):
        self.graph.link_mode = on
        self.graph.pick = None
        self.status_lbl.setText("Link mode: click two nodes to link or unlink them." if on else "Link mode off.")
        self.graph.update()

    def _lock(self, on):
        self.graph.lock_layer = on
        self.b_lock.setText("Locked to layer" if on else "Free to move between layers")

    def _add_layer(self):
        self.m.layers.append(f"Layer {len(self.m.layers) + 1}")
        self.graph.refit()
        self._save_state()

    def _remove_layer(self):
        if len(self.m.layers) <= 1:
            self.status_lbl.setText("At least one layer is needed.")
            return
        self.m.layers.pop()
        self.m.sync()
        self.graph.refit()
        self._save_state()

    def _arrange(self):
        self.m.auto_arrange()
        self.graph.refit()
        self._save_state()
        self.status_lbl.setText("Arranged by distance from the living room.")

    def _save_state(self):
        self.m.plans = self.nplans.value()
        self.editor.graph["nodes"][self.nid]["props"]["design"] = self.m.state()

    # ---- cards
    def render(self):
        m = self.m
        self.derived.setText(f"module {m.short():.2f} × {m.long():.2f} m  |  {m.mod_area():.2f} m²  |  h {m.mod['h']:.2f} m")
        while self.cards.count():
            it = self.cards.takeAt(0)
            if it.widget():
                it.widget().hide()
                it.widget().deleteLater()
        if not m.spaces:
            e = QtWidgets.QLabel("No spaces yet.\nUse the buttons above to add a room.")
            e.setAlignment(QtCore.Qt.AlignCenter)
            e.setObjectName("pd_notice")
            self.cards.addWidget(e)
        for sp in m.spaces:
            self.cards.addWidget(self._card(sp))
        self.cards.addStretch(1)
        m.sync()
        self.graph.refit()
        self._save_state()

    def _modules_combo(self, sel, lo, hi, cb):
        c = QtWidgets.QComboBox()
        c.setFixedWidth(108)
        n = lo
        while n <= hi + 1e-9:
            c.addItem("1 module" if n == 1 else f"{nn(n)} modules", n)
            if abs(n - sel) < 1e-9:
                c.setCurrentIndex(c.count() - 1)
            n += 0.5
        c.activated.connect(lambda i: cb(c.itemData(i)))
        return c

    def _card(self, sp):
        m = self.m
        n = m.eff(sp)
        w, d = m.foot(sp["type"], n)
        msgs = m.check(sp["type"], n)
        st = self.std.get(sp["type"], {"label": sp["type"].title()})
        card = QtWidgets.QFrame()
        card.setObjectName("pd_card")
        card.setProperty("bad", bool(msgs))
        cl = QtWidgets.QVBoxLayout(card)
        cl.setContentsMargins(12, 10, 12, 11)
        head = QtWidgets.QHBoxLayout()
        chip = QtWidgets.QLabel(sp["code"])
        chip.setStyleSheet(f"background:{self.tint.get(sp['type'], '#888')}; color:white; font-weight:700; "
                           f"border-radius:5px; padding:2px 7px;")
        head.addWidget(chip)
        nm = QtWidgets.QLabel(st["label"])
        nm.setStyleSheet("font-weight:600;")
        head.addWidget(nm)
        head.addStretch(1)
        kill = QtWidgets.QPushButton("×")
        kill.setFixedSize(26, 24)
        kill.setStyleSheet("QPushButton { padding: 0px; font-size: 16px; color: #8a94a2; border-color: transparent; }"
                           "QPushButton:hover { color: #b03a3a; border-color: #b03a3a; }")
        kill.setToolTip("Remove")
        kill.clicked.connect(lambda: (m.remove_space(sp["id"]), self.render()))
        head.addWidget(kill)
        cl.addLayout(head)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Size"))
        mode = QtWidgets.QComboBox()
        mode.addItems(["By module", "By area m²"])
        mode.setCurrentIndex(1 if sp.get("mode") == "custom" else 0)
        mode.activated.connect(lambda i: (sp.update(mode="custom" if i else "modules"), self.render()))
        row.addWidget(mode)
        if sp.get("mode") == "custom":
            area = QtWidgets.QDoubleSpinBox()
            area.setRange(1, 1000)
            area.setDecimals(1)
            area.setSingleStep(0.5)
            area.setValue(sp["custom"])
            area.setKeyboardTracking(False)
            area.setFixedWidth(76)
            area.valueChanged.connect(lambda v: (sp.update(custom=max(0.1, v)), self.render()))
            row.addWidget(area)
            row.addWidget(QtWidgets.QLabel(f"m² → {nn(n)} mod"))
        else:
            row.addWidget(self._modules_combo(sp["modules"], 1, 10, lambda v: (sp.update(modules=v), self.render())))
        row.addStretch(1)
        dims = QtWidgets.QLabel(f"<b>{w:.2f} × {d:.2f} m</b><br><small>{w * d:.2f} m² · {nn(n)} mod</small>")
        dims.setObjectName("pd_dims")
        dims.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        dims.setMinimumWidth(96)
        row.addWidget(dims)
        cl.addLayout(row)
        if msgs:
            note = QtWidgets.QLabel(f"<b>Below the working minimum for a {st['label'].lower()}</b><br>• " + "<br>• ".join(msgs))
            note.setObjectName("pd_notice")
            note.setWordWrap(True)
            cl.addWidget(note)
        for key, label in att_keys(sp["type"]):
            a = sp["att"].setdefault(key, {"on": False, "modules": m.min_modules(key)})
            ar = QtWidgets.QHBoxLayout()
            chk = QtWidgets.QCheckBox(label)
            chk.setChecked(a["on"])
            chk.toggled.connect(lambda on, a=a: (a.update(on=on), self.render()))
            ar.addWidget(chk)
            ar.addStretch(1)
            combo = self._modules_combo(a["modules"], 0.5, 4, lambda v, a=a: (a.update(modules=v), self.render()))
            combo.setEnabled(a["on"])
            ar.addWidget(combo)
            aw, ad = m.foot(key, a["modules"])
            am = m.check(key, a["modules"])
            dl = QtWidgets.QLabel(f"{aw:.2f}×{ad:.2f}" + ("  ⚠" if am else ""))
            dl.setObjectName("pd_dims")
            dl.setMinimumWidth(78)
            dl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            ar.addWidget(dl)
            cl.addLayout(ar)
            if a["on"] and am:
                note = QtWidgets.QLabel(label + ": " + " ".join(am))
                note.setObjectName("pd_notice")
                note.setWordWrap(True)
                cl.addWidget(note)
        return card

    # ---- generate
    def generate(self):
        m = self.m
        if not m.spaces:
            self.status_lbl.setText("Add at least one space first.")
            return
        m.plans = self.nplans.value()
        brief, links = m.brief_text()
        k = self.k
        self._set_input("Module A", round(m.mod["a"] / k, 6))
        self._set_input("Module B", round(m.mod["b"] / k, 6))
        self._set_input("Module Height", round(m.mod["h"] / k, 6))
        self._set_input("Brief", brief)
        self._set_input("Links", links)
        self._set_input("Plans", m.plans)
        self._save_state()
        ed = self.editor
        it = ed.items.get(self.nid)
        if it is not None:
            it.build()
            it.update()
        ed.mark_dirty([self.nid])
        if not ed.auto:
            ed.solve()
        self.status_lbl.setText(f"Generating {m.plans} plan(s) from {len(m.node_list())} spaces…")
        self._ticks = 0
        self._wait = QtCore.QTimer(self)
        self._wait.setInterval(150)
        self._wait.timeout.connect(self._done)
        self._wait.start()

    def _done(self):
        self._ticks += 1
        ed = self.editor
        it = ed.items.get(self.nid)
        if it is None or (self._ticks < 400 and (it.status in ("dirty", "running") or ed.compute.busy)):
            return
        self._wait.stop()
        if it.status == "error":
            self.status_lbl.setText("Error: " + (it.message or ""))
            return
        res = ed.results.get(self.nid, {})
        rep = (res.get("Report") or "").splitlines()
        if self.bake.isChecked():
            root = getattr(self.PL, "ROOT_LAYER", "GH_PLANS")
            ed.bake([self.nid], outputs=("Outlines", "Envelopes"), replace_root=root)
            self.status_lbl.setText(f"Baked {max(0, len(rep) - 1)} plan(s) to {root}. " + (rep[1] if len(rep) > 1 else ""))
        else:
            self.status_lbl.setText(f"Generated {max(0, len(rep) - 1)} plan(s) — see the node preview in the Model view. "
                                    + (rep[1] if len(rep) > 1 else ""))
