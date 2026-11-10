"""Side panels: Layers, Properties (with component parameters), command palette, settings dialog."""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import commands as K
from . import settings as S
from .document import LAYER_COLORS


def heading(text):
    l = QtWidgets.QLabel(text)
    l.setObjectName("Heading")
    return l


def muted(text):
    l = QtWidgets.QLabel(text)
    l.setObjectName("Muted")
    l.setWordWrap(True)
    return l


class ColorButton(QtWidgets.QPushButton):
    color_changed = QtCore.Signal(str)

    def __init__(self, color="#ffffff", parent=None):
        super().__init__(parent)
        self.setFixedSize(34, 20)
        self.set_color(color)
        self.clicked.connect(self._pick)

    def set_color(self, c):
        self.color = c
        self.setStyleSheet(f"QPushButton {{ background: {c}; border: 1px solid #777; border-radius: 3px; }}")

    def _pick(self):
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.color), self)
        if c.isValid():
            self.set_color(c.name())
            self.color_changed.emit(c.name())


# ===================================================================== layers
class _LayerTree(QtWidgets.QTreeWidget):
    """Tree that hands drops to the panel instead of moving items itself."""
    dropped = QtCore.Signal(object, object)   # (dragged items, target item or None)

    def dropEvent(self, ev):
        target = self.itemAt(ev.position().toPoint())
        self.dropped.emit(self.selectedItems(), target)
        ev.setDropAction(QtCore.Qt.IgnoreAction)
        ev.accept()


class LayersPanel(QtWidgets.QWidget):
    """Layers (with sublayers) and the objects on them, plus components.

    Columns: name, shown, locked, colour, line weight. Drag objects onto a layer to move them,
    drag a layer onto another to make it a sublayer (or onto empty space for top level).
    Right-click for more."""
    MAX_OBJECTS = 400
    ROLE = QtCore.Qt.UserRole

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.setObjectName("Panel")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        bar = QtWidgets.QHBoxLayout()
        for txt, tip, fn in (("+ Layer", "New layer", self._add), ("+ Sub", "New sublayer of the selected layer", self._add_sub),
                             ("Delete", "Delete the selected layer", self._delete)):
            b = QtWidgets.QPushButton(txt)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            bar.addWidget(b)
        self.show_objects = QtWidgets.QCheckBox("Objects")
        self.show_objects.setChecked(True)
        self.show_objects.setToolTip("List the objects on each layer")
        self.show_objects.toggled.connect(self.refresh)
        bar.addWidget(self.show_objects)
        lay.addLayout(bar)
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("Filter layers and objects…")
        self.filter.textChanged.connect(self.refresh)
        lay.addWidget(self.filter)
        self.tree = _LayerTree()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["Name", "On", "Lock", "Color", "mm"])
        hd = self.tree.header()
        hd.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for i, w in ((1, 30), (2, 40), (3, 44), (4, 40)):
            hd.setSectionResizeMode(i, QtWidgets.QHeaderView.Fixed)
            hd.resizeSection(i, w)
        hd.setStretchLastSection(False)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tree.setDragDropMode(QtWidgets.QAbstractItemView.DragDrop)
        self.tree.setDragEnabled(True)
        self.tree.setAcceptDrops(True)
        self.tree.setDropIndicatorShown(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setIndentation(12)
        self.tree.setIconSize(QtCore.QSize(26, 14))
        hd.setMinimumSectionSize(24)
        self.tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._menu)
        self.tree.itemClicked.connect(self._clicked)
        self.tree.itemDoubleClicked.connect(self._double)
        self.tree.itemChanged.connect(self._renamed)
        self.tree.itemExpanded.connect(lambda it: self._expand(it, True))
        self.tree.itemCollapsed.connect(lambda it: self._expand(it, False))
        self.tree.itemSelectionChanged.connect(self._tree_selection)
        self.tree.dropped.connect(self._drop)
        lay.addWidget(self.tree, 1)
        lay.addWidget(muted("Double-click a layer to make it current (bold). Click the On / Lock / Color / Weight "
                            "cells to change them. Drag objects onto a layer to move them; drag a layer onto "
                            "another to nest it. Right-click for more."))
        self._busy = False
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self.refresh)
        doc.subscribe(self._doc_changed)
        self.refresh()

    # ------------------------------------------------------------ building
    def _doc_changed(self, what):
        if what == "selection":
            self._sync_selection()
        elif what in ("layers", "loaded", "changed", "definitions"):
            self._timer.start()

    @staticmethod
    def _swatch(color, faded=False):
        pm = QtGui.QPixmap(26, 14)
        pm.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pm)
        c = QtGui.QColor(color)
        if faded:
            c.setAlpha(110)
        p.setBrush(c)
        p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 90), 1, QtCore.Qt.DashLine if faded else QtCore.Qt.SolidLine))
        p.drawRoundedRect(1, 1, 24, 12, 3, 3)
        p.end()
        return QtGui.QIcon(pm)

    def _set_flags(self, it, shown, locked):
        it.setText(1, "●" if shown else "○")
        it.setText(2, "🔒" if locked else "·")
        it.setTextAlignment(1, QtCore.Qt.AlignCenter)
        it.setTextAlignment(2, QtCore.Qt.AlignCenter)
        it.setTextAlignment(4, QtCore.Qt.AlignCenter)
        it.setToolTip(1, "Click to show / hide")
        it.setToolTip(2, "Click to lock / unlock")

    def refresh(self):
        self._busy = True
        scroll = self.tree.verticalScrollBar().value()
        self.tree.clear()
        doc = self.doc
        flt = self.filter.text().strip().lower()
        items = {}
        by_layer = {}
        if self.show_objects.isChecked():
            for g in doc.objects.values():
                by_layer.setdefault(g.layer, []).append(g)
        bold = QtGui.QFont()
        bold.setBold(True)
        for l in doc.layers:
            parent = items.get(l.parent)
            it = QtWidgets.QTreeWidgetItem(parent if parent is not None else self.tree)
            it.setData(0, self.ROLE, ("layer", l.name))
            it.setText(0, l.short)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsDragEnabled | QtCore.Qt.ItemIsDropEnabled)
            if l.name == doc.current_layer:
                it.setFont(0, bold)
                it.setToolTip(0, "Current layer: new objects go here")
            self._set_flags(it, l.visible, l.locked)
            it.setIcon(3, self._swatch(l.color))
            it.setText(4, f"{l.weight:g}" if l.weight else "def")
            it.setToolTip(4, "Line weight in mm (screen and print). Click to change.")
            objs = by_layer.get(l.name, [])
            if objs:
                it.setText(0, f"{l.short}  ({len(objs)})")
            for g in objs[:self.MAX_OBJECTS]:
                label = g.name or doc.describe(g).split(",")[0].split(" at ")[0]
                if flt and flt not in label.lower() and flt not in l.name.lower():
                    continue
                o = QtWidgets.QTreeWidgetItem(it)
                o.setData(0, self.ROLE, ("object", g.id))
                o.setText(0, label)
                o.setFlags((o.flags() | QtCore.Qt.ItemIsDragEnabled) & ~QtCore.Qt.ItemIsDropEnabled)
                self._set_flags(o, not g.data.get("hidden"), bool(g.data.get("locked")))
                oc = g.data.get("attr_color")
                o.setIcon(3, self._swatch(oc or l.color, faded=not oc))
                o.setToolTip(3, "Object colour" if oc else "By layer (click to set an object colour)")
                w = g.data.get("attr_weight")
                o.setText(4, f"{w:g}" if w else "layer")
                if g.data.get("hidden"):
                    o.setForeground(0, QtGui.QBrush(QtGui.QColor(140, 140, 140)))
            if len(objs) > self.MAX_OBJECTS:
                more = QtWidgets.QTreeWidgetItem(it)
                more.setText(0, f"… {len(objs) - self.MAX_OBJECTS} more (select via right-click → Select objects)")
                more.setFlags(QtCore.Qt.ItemIsEnabled)
            items[l.name] = it
            if flt and flt not in l.name.lower() and it.childCount() == 0:
                it.setHidden(True)
        # components
        if doc.definitions:
            root = QtWidgets.QTreeWidgetItem(self.tree)
            root.setText(0, "Components")
            root.setFont(0, bold)
            root.setFlags(QtCore.Qt.ItemIsEnabled)
            for d in doc.definitions.values():
                n = len(doc.instances_of(d.id))
                c = QtWidgets.QTreeWidgetItem(root)
                c.setData(0, self.ROLE, ("component", d.id))
                c.setText(0, f"{d.name}  ({n})")
                c.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                self._set_flags(c, not d.hidden, d.locked)
                c.setToolTip(0, "Hide or lock every instance of this component")
            root.setExpanded(True)
        for l in doc.layers:
            it = items.get(l.name)
            if it is not None:
                it.setExpanded(l.expanded and (it.childCount() < 60 or bool(flt)))
        self._busy = False
        self._sync_selection()
        self.tree.verticalScrollBar().setValue(scroll)

    def _sync_selection(self):
        if self._busy:
            return
        self._busy = True
        sel = self.doc.selection
        it = QtWidgets.QTreeWidgetItemIterator(self.tree)
        while it.value():
            item = it.value()
            d = item.data(0, self.ROLE)
            if d and d[0] == "object":
                item.setSelected(d[1] in sel)
            it += 1
        self._busy = False

    # ------------------------------------------------------------ helpers
    def _kind(self, item):
        return item.data(0, self.ROLE) if item is not None else None

    def _selected(self, kind):
        return [d[1] for it in self.tree.selectedItems() if (d := self._kind(it)) and d[0] == kind]

    def _current_layer_item(self):
        it = self.tree.currentItem()
        d = self._kind(it)
        if d and d[0] == "object":
            g = self.doc.objects.get(d[1])
            return g.layer if g else None
        return d[1] if d and d[0] == "layer" else None

    # ------------------------------------------------------------ editing
    def _clicked(self, item, col):
        d = self._kind(item)
        if not d:
            return
        kind, key = d
        doc = self.doc
        if kind == "layer":
            l = doc.layer(key)
            if col == 1:
                l.visible = not l.visible
                doc.selection = {i for i in doc.selection if doc.objects[i].layer != key}
                doc.notify("layers"); doc.notify("changed")
            elif col == 2:
                l.locked = not l.locked
                doc.selection = {i for i in doc.selection if doc.objects[i].layer != key}
                doc.notify("layers"); doc.notify("changed")
            elif col == 3:
                c = QtWidgets.QColorDialog.getColor(QtGui.QColor(l.color), self, f"Colour of layer {l.short}")
                if c.isValid():
                    doc.checkpoint("Layer colour")
                    l.color = c.name()
                    doc.notify("layers"); doc.notify("changed")
            elif col == 4:
                w = self._pick_weight(l.weight, "Default")
                if w is not None:
                    doc.checkpoint("Layer line weight")
                    l.weight = w
                    doc.notify("layers"); doc.notify("changed")
        elif kind == "object":
            g = doc.objects.get(key)
            if g is None:
                return
            ids = self._selected("object") if key in self._selected("object") else [key]
            if col == 1:
                doc.set_attr(ids, "hidden", not g.data.get("hidden"), "Hide" if not g.data.get("hidden") else "Show")
            elif col == 2:
                doc.set_attr(ids, "locked", not g.data.get("locked"), "Lock")
            elif col == 3:
                self._object_color(ids)
            elif col == 4:
                w = self._pick_weight(g.data.get("attr_weight", 0.0), "By layer")
                if w is not None:
                    doc.set_attr(ids, "attr_weight", w or None, "Line weight")
        elif kind == "component":
            dd = doc.definitions.get(key)
            if dd is None:
                return
            if col == 1:
                dd.hidden = not dd.hidden
            elif col == 2:
                dd.locked = not dd.locked
            else:
                return
            doc.selection = {i for i in doc.selection if getattr(doc.objects[i], "def_id", None) != key}
            doc.notify("definitions"); doc.notify("layers"); doc.notify("changed")

    def _object_color(self, ids):
        g = self.doc.objects[ids[0]]
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.doc.color_of(g)), self, "Object colour")
        if c.isValid():
            self.doc.set_attr(ids, "attr_color", c.name(), "Object colour")

    def _pick_weight(self, current, zero_label):
        from .document import WEIGHTS
        m = QtWidgets.QMenu(self)
        acts = {}
        for w in WEIGHTS:
            a = m.addAction((zero_label if w == 0 else f"{w:g} mm"))
            a.setCheckable(True)
            a.setChecked(abs((current or 0) - w) < 1e-9)
            acts[a] = w
        a = m.exec(QtGui.QCursor.pos())
        return acts.get(a)

    def _double(self, item, col):
        d = self._kind(item)
        if d and d[0] == "layer" and col == 0:
            self.doc.current_layer = d[1]
            self.refresh()
        elif d and d[0] == "object":
            self.doc.select([d[1]])

    def _renamed(self, item, col):
        if self._busy or col != 0:
            return
        d = self._kind(item)
        if not d or d[0] != "layer":
            return
        new = item.text(0).split("  (")[0].strip()
        if not self.doc.rename_layer(d[1], new):
            self._timer.start()

    def _expand(self, item, state):
        d = self._kind(item)
        if d and d[0] == "layer" and not self._busy:
            l = self.doc.layer(d[1])
            if l:
                l.expanded = state

    def _tree_selection(self):
        if self._busy:
            return
        ids = self._selected("object")
        if ids or any(self._kind(i) and self._kind(i)[0] == "object" for i in self.tree.selectedItems()):
            self.doc.select([i for i in ids if i in {g.id for g in self.doc.selectable(None)}])

    def _drop(self, dragged, target):
        d = self._kind(target)
        tlayer = None
        if d and d[0] == "layer":
            tlayer = d[1]
        elif d and d[0] == "object":
            g = self.doc.objects.get(d[1])
            tlayer = g.layer if g else None
        objs = [x[1] for it in dragged if (x := self._kind(it)) and x[0] == "object"]
        lays = [x[1] for it in dragged if (x := self._kind(it)) and x[0] == "layer"]
        if objs and tlayer:
            self.doc.move_to_layer(objs, tlayer)
        for name in lays:
            if name != tlayer:
                self.doc.move_layer(name, tlayer)

    # ------------------------------------------------------------ buttons / menu
    def _add(self):
        self.doc.checkpoint("New layer")
        l = self.doc.add_layer()
        self._edit_new(l.name)

    def _add_sub(self):
        par = self._current_layer_item() or self.doc.current_layer
        self.doc.checkpoint("New sublayer")
        l = self.doc.add_layer(parent=par)
        self._edit_new(l.name)

    def _edit_new(self, name):
        self.refresh()
        it = QtWidgets.QTreeWidgetItemIterator(self.tree)
        while it.value():
            d = self._kind(it.value())
            if d == ("layer", name):
                self.tree.scrollToItem(it.value())
                self.tree.editItem(it.value(), 0)
                return
            it += 1

    def _delete(self, layer=None):
        name = layer or self._current_layer_item()
        if not name:
            return
        if len(self.doc.layers) <= 1 + len(self.doc.sublayers(name)):
            QtWidgets.QMessageBox.information(self, "Delete layer", "A model needs at least one layer.")
            return
        n = sum(1 for g in self.doc.objects.values() if g.layer == name or g.layer.startswith(name + "::"))
        delete_objects = False
        if n:
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Delete layer")
            box.setText(f"Layer '{name}' (and its sublayers) has {n} object(s).")
            mv = box.addButton("Move objects to parent layer", QtWidgets.QMessageBox.AcceptRole)
            de = box.addButton("Delete objects too", QtWidgets.QMessageBox.DestructiveRole)
            box.addButton(QtWidgets.QMessageBox.Cancel)
            box.exec()
            if box.clickedButton() not in (mv, de):
                return
            delete_objects = box.clickedButton() is de
        self.doc.delete_layer(name, delete_objects)

    def _menu(self, pos):
        item = self.tree.itemAt(pos)
        d = self._kind(item)
        m = QtWidgets.QMenu(self)
        doc = self.doc
        if d and d[0] == "layer":
            name = d[1]
            m.addAction("Set as current layer", lambda: (setattr(doc, "current_layer", name), self.refresh()))
            m.addAction("New sublayer", lambda: (doc.checkpoint("New sublayer"), self._edit_new(doc.add_layer(parent=name).name)))
            m.addAction("Rename", lambda: self.tree.editItem(item, 0))
            m.addSeparator()
            m.addAction("Select objects", lambda: doc.select([g.id for g in doc.selectable(None)
                                                              if g.layer == name or g.layer.startswith(name + "::")]))
            if doc.selection:
                m.addAction(f"Move selected objects here ({len(doc.selection)})",
                            lambda: doc.move_to_layer(list(doc.selection), name))
            m.addSeparator()
            m.addAction("Show only this layer", lambda: self._solo(name, "visible"))
            m.addAction("Lock all other layers", lambda: self._solo(name, "locked"))
            m.addAction("Show and unlock all layers", self._all_on)
            if doc.layer(name).parent:
                m.addAction("Move to top level", lambda: doc.move_layer(name, None))
            m.addSeparator()
            m.addAction("Delete layer…", lambda: self._delete(name))
        elif d and d[0] == "object":
            ids = self._selected("object") or [d[1]]
            m.addAction(f"Select ({len(ids)})", lambda: doc.select(ids))
            m.addAction("Hide", lambda: doc.set_attr(ids, "hidden", True, "Hide"))
            m.addAction("Show", lambda: doc.set_attr(ids, "hidden", None, "Show"))
            m.addAction("Lock", lambda: doc.set_attr(ids, "locked", True, "Lock"))
            m.addAction("Unlock", lambda: doc.set_attr(ids, "locked", None, "Unlock"))
            m.addSeparator()
            m.addAction("Object colour…", lambda: self._object_color(ids))
            m.addAction("Colour by layer", lambda: doc.set_attr(ids, "attr_color", None, "Colour by layer"))
            m.addAction("Line weight by layer", lambda: doc.set_attr(ids, "attr_weight", None, "Weight by layer"))
            lm = m.addMenu("Move to layer")
            for l in doc.layers:
                lm.addAction(l.name.replace("::", " › "), lambda n=l.name: doc.move_to_layer(ids, n))
        else:
            m.addAction("New layer", self._add)
            m.addAction("Show and unlock all layers", self._all_on)
            if doc.hidden_objects():
                m.addAction(f"Show hidden objects ({len(doc.hidden_objects())})",
                            lambda: doc.set_attr([g.id for g in doc.hidden_objects()], "hidden", None, "Show"))
        m.exec(self.tree.viewport().mapToGlobal(pos))

    def _solo(self, name, attr):
        keep = {name} | {l.name for l in self.doc.sublayers(name)}
        n = name
        while "::" in n:   # parents must stay on for the layer to show
            n = n.rsplit("::", 1)[0]
            keep.add(n)
        for l in self.doc.layers:
            if attr == "visible":
                l.visible = l.name in keep
            else:
                l.locked = l.name not in keep
        if attr == "visible" and self.doc.current_layer not in keep:
            self.doc.current_layer = name
        self.doc.selection = set()
        self.doc.notify("selection"); self.doc.notify("layers"); self.doc.notify("changed")

    def _all_on(self):
        for l in self.doc.layers:
            l.visible, l.locked = True, False
        self.doc.notify("layers"); self.doc.notify("changed")


# ===================================================================== properties
class PropertiesPanel(QtWidgets.QWidget):
    def __init__(self, doc, app=None, parent=None):
        super().__init__(parent)
        self.doc, self.app = doc, app
        self.setObjectName("Panel")
        self.lay = QtWidgets.QVBoxLayout(self)
        self.lay.setContentsMargins(8, 8, 8, 8)
        self.lay.setAlignment(QtCore.Qt.AlignTop)
        self.body = QtWidgets.QWidget()
        self.lay.addWidget(self.body)
        self.space = "3d"
        doc.subscribe(self._changed)
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self.refresh)
        self.refresh()

    def _changed(self, what):
        if what in ("selection", "changed", "loaded", "definitions"):
            self._timer.start()

    def set_space(self, space):
        self.space = space
        self.refresh()

    def refresh(self):
        self.body.setParent(None)
        self.body = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(self.body)
        form.setContentsMargins(0, 0, 0, 0)
        sel = self.doc.selected(self.space)
        if not sel:
            form.addWidget(heading("No selection"))
            n = len(self.doc.in_space(self.space))
            form.addWidget(muted(f"{n} objects in this workspace.\nClick to select, drag for window select "
                                 f"(drag right-to-left for crossing). Shift adds, Ctrl removes."))
        elif len(sel) > 1:
            form.addWidget(heading(f"{len(sel)} objects selected"))
            kinds = {}
            for g in sel:
                k = self.doc.describe(g).split(",")[0].split(" at ")[0]
                kinds[k] = kinds.get(k, 0) + 1
            for k, v in sorted(kinds.items()):
                form.addWidget(muted(f"{v} × {k}"))
            self._layer_row(form, sel)
        else:
            g = sel[0]
            form.addWidget(heading(self.doc.describe(g).split(",")[0].split(" at ")[0]))
            form.addWidget(muted(self.doc.describe(g)))
            name = QtWidgets.QLineEdit(g.name)
            name.setPlaceholderText("Object name")
            name.editingFinished.connect(lambda: self._set_name(g.id, name.text()))
            form.addWidget(name)
            self._layer_row(form, sel)
            if g.kind == "instance":
                self._instance_ui(form, g)
        self.lay.addWidget(self.body)

    def _layer_row(self, form, sel):
        from .document import WEIGHTS
        ids = [g.id for g in sel]
        grid = QtWidgets.QFormLayout()
        cb = QtWidgets.QComboBox()
        names = [l.name for l in self.doc.layers]
        cb.addItems([n.replace("::", " › ") for n in names])
        cur = {g.layer for g in sel}
        if len(cur) == 1:
            cb.setCurrentIndex(names.index(next(iter(cur))) if next(iter(cur)) in names else 0)
        cb.activated.connect(lambda i: self._set_layer(ids, names[i]))
        grid.addRow("Layer", cb)
        # colour: by layer or own
        crow = QtWidgets.QHBoxLayout()
        own = {g.data.get("attr_color") for g in sel}
        mode = QtWidgets.QComboBox()
        mode.addItems(["By layer", "Object colour"])
        mode.setCurrentIndex(0 if own == {None} else 1)
        btn = ColorButton(next(iter(own)) if len(own) == 1 and None not in own else self.doc.color_of(sel[0]))
        btn.setEnabled(mode.currentIndex() == 1)
        mode.activated.connect(lambda i: self.doc.set_attr(ids, "attr_color", None if i == 0 else btn.color, "Object colour"))
        btn.color_changed.connect(lambda c: self.doc.set_attr(ids, "attr_color", c, "Object colour"))
        crow.addWidget(mode, 1)
        crow.addWidget(btn)
        grid.addRow("Colour", crow)
        wcb = QtWidgets.QComboBox()
        wcb.addItems(["By layer"] + [f"{w:g} mm" for w in WEIGHTS[1:]])
        ws = {g.data.get("attr_weight") for g in sel}
        if len(ws) == 1 and next(iter(ws)):
            w0 = next(iter(ws))
            wcb.setCurrentIndex(min(range(1, len(WEIGHTS)), key=lambda k: abs(WEIGHTS[k] - w0)))
        wcb.activated.connect(lambda i: self.doc.set_attr(ids, "attr_weight", None if i == 0 else WEIGHTS[i], "Line weight"))
        grid.addRow("Line weight", wcb)
        flags = QtWidgets.QHBoxLayout()
        hide = QtWidgets.QPushButton("Hide")
        hide.setToolTip("Hide these objects (Show, or the Layers panel, brings them back)")
        hide.clicked.connect(lambda: self.doc.set_attr(ids, "hidden", True, "Hide"))
        lock = QtWidgets.QPushButton("Lock")
        lock.setToolTip("Lock these objects: visible, but they can't be selected")
        lock.clicked.connect(lambda: self.doc.set_attr(ids, "locked", True, "Lock"))
        flags.addWidget(hide)
        flags.addWidget(lock)
        grid.addRow("", flags)
        form.addLayout(grid)

    def _instance_ui(self, form, g):
        d = self.doc.definitions.get(g.def_id)
        if d is None:
            return
        n = len(self.doc.instances_of(d.id))
        form.addWidget(muted(f"Definition '{d.name}' is used {n} time(s)." + (f"\n{d.description}" if d.description else "")))
        err = g.data.get("_error")
        if err:
            form.addWidget(muted("Build error: " + err))
        if d.parametric and d.params:
            form.addSpacing(6)
            form.addWidget(heading("Parameters"))
            grid = QtWidgets.QFormLayout()
            vals = d.merged_params(g.params)
            for k, v in vals.items():
                w = self._param_widget(v)
                self._bind(w, lambda val, key=k, gid=g.id: self._set_param(gid, key, val))
                grid.addRow(k.replace("_", " ").capitalize(), w)
            form.addLayout(grid)
            reset = QtWidgets.QPushButton("Reset to defaults")
            reset.clicked.connect(lambda: self._reset_params(g.id))
            form.addWidget(reset)
        row = QtWidgets.QHBoxLayout()
        for txt, cmd in (("Edit", "EditComponent"), ("Make unique", "MakeUnique"), ("Connect", "Connect"), ("Explode", "Explode")):
            if txt == "Edit" and d.parametric:
                continue
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(lambda _=False, c=cmd: self.app and self.app.run_command(c))
            row.addWidget(b)
        form.addLayout(row)

    @staticmethod
    def _param_widget(v):
        if isinstance(v, bool):
            w = QtWidgets.QCheckBox()
            w.setChecked(v)
        elif isinstance(v, int):
            w = QtWidgets.QSpinBox()
            w.setRange(-100000, 100000)
            w.setValue(v)
        elif isinstance(v, float):
            w = QtWidgets.QDoubleSpinBox()
            w.setRange(-1e9, 1e9)
            w.setDecimals(3)
            w.setValue(v)
            w.setSingleStep(max(abs(v) / 20, 0.1))
        else:
            w = QtWidgets.QLineEdit(str(v))
        return w

    @staticmethod
    def _bind(w, fn):
        if isinstance(w, QtWidgets.QCheckBox):
            w.toggled.connect(fn)
        elif isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            w.setKeyboardTracking(False)
            w.valueChanged.connect(fn)
        else:
            w.editingFinished.connect(lambda: fn(w.text()))

    def _set_param(self, gid, key, val):
        g = self.doc.get(gid)
        if g is None:
            return
        self.doc.checkpoint(f"Set {key}")
        p = dict(g.params)
        p[key] = val
        self.doc.replace(g.copy(params=p))

    def _reset_params(self, gid):
        g = self.doc.get(gid)
        self.doc.checkpoint("Reset parameters")
        self.doc.replace(g.copy(params={}))

    def _set_name(self, gid, name):
        g = self.doc.get(gid)
        if g and g.name != name:
            self.doc.checkpoint("Rename")
            self.doc.replace(g.copy(name=name))

    def _set_layer(self, ids, layer):
        self.doc.checkpoint("Change layer")
        for i in ids:
            self.doc.replace(self.doc.objects[i].copy(layer=layer), notify=False)
        self.doc.notify("changed")


# ===================================================================== command palette
class CommandPalette(QtWidgets.QDialog):
    def __init__(self, space, parent=None):
        super().__init__(parent, QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint)
        self.space = space
        self.choice = None
        self.resize(520, 420)
        lay = QtWidgets.QVBoxLayout(self)
        self.edit = QtWidgets.QLineEdit()
        self.edit.setPlaceholderText("Search commands...")
        self.list = QtWidgets.QListWidget()
        lay.addWidget(self.edit)
        lay.addWidget(self.list)
        self.edit.textChanged.connect(self.fill)
        self.edit.returnPressed.connect(self.accept_current)
        self.list.itemActivated.connect(lambda it: self.accept_current())
        self.edit.installEventFilter(self)
        self.fill("")

    def fill(self, t):
        self.list.clear()
        t = t.lower().strip()
        rows = []
        for c in K.COMMANDS.values():
            if self.space not in c.spaces:
                continue
            hay = (c.name + " " + " ".join(c.aliases) + " " + c.help + " " + c.group).lower()
            if not t or all(w in hay for w in t.split()):
                score = 0 if c.name.lower().startswith(t) else 1
                rows.append((score, c.group, c.name, c))
        for _, grp, name, c in sorted(rows, key=lambda r: (r[0], r[1], r[2])):
            it = QtWidgets.QListWidgetItem(f"{name}    —  {c.help}" if c.help else name)
            it.setData(QtCore.Qt.UserRole, name)
            it.setToolTip(f"{c.group}. Aliases: {', '.join(c.aliases) or '-'}")
            self.list.addItem(it)
        if self.list.count():
            self.list.setCurrentRow(0)

    def eventFilter(self, o, e):
        if e.type() == QtCore.QEvent.KeyPress and e.key() in (QtCore.Qt.Key_Down, QtCore.Qt.Key_Up):
            r = self.list.currentRow() + (1 if e.key() == QtCore.Qt.Key_Down else -1)
            self.list.setCurrentRow(max(0, min(self.list.count() - 1, r)))
            return True
        return super().eventFilter(o, e)

    def accept_current(self):
        it = self.list.currentItem()
        if it:
            self.choice = it.data(QtCore.Qt.UserRole)
        self.accept()


# ===================================================================== settings
class SettingsDialog(QtWidgets.QDialog):
    applied = QtCore.Signal()

    COLOR_KEYS = [("view3d_top", "3D background (top)"), ("view3d_bottom", "3D background (bottom)"),
                  ("view2d", "2D canvas background"), ("grid", "Grid (minor)"), ("grid_major", "Grid (major)"),
                  ("accent", "Accent"), ("selection", "Selection"), ("preview", "Preview / node output")]

    def __init__(self, settings: S.Settings, parent=None):
        super().__init__(parent)
        self.s = settings
        self.setWindowTitle("Settings")
        self.resize(460, 560)
        lay = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        lay.addWidget(tabs)

        # Appearance
        ap = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(ap)
        self.theme = QtWidgets.QComboBox()
        self.theme.addItems(list(S.PRESETS))
        self.theme.setCurrentText(settings.theme)
        self.theme.currentTextChanged.connect(self._theme_changed)
        f.addRow(heading("Theme"), self.theme)
        self.grad = QtWidgets.QCheckBox("Gradient 3D background")
        self.grad.setChecked(settings.gradient_3d)
        f.addRow("", self.grad)
        self.ssao = QtWidgets.QCheckBox("Ambient occlusion (soft shadows in 3D)")
        self.ssao.setChecked(getattr(settings, "ambient_occlusion", False))
        f.addRow("", self.ssao)
        self.font = QtWidgets.QSpinBox()
        self.font.setRange(9, 20)
        self.font.setValue(settings.font_size)
        f.addRow("UI font size (Arial)", self.font)
        self.btns = {}
        for key, label in self.COLOR_KEYS:
            b = ColorButton(settings.color(key))
            self.btns[key] = b
            f.addRow(label, b)
        reset = QtWidgets.QPushButton("Reset colors to theme")
        reset.clicked.connect(self._reset_colors)
        f.addRow("", reset)
        tabs.addTab(ap, "Appearance")

        # Modeling
        mo = QtWidgets.QWidget()
        f2 = QtWidgets.QFormLayout(mo)
        self.units = QtWidgets.QComboBox()
        self.units.addItems(["mm", "cm", "m", "in", "ft"])
        self.units.setCurrentText(settings.units)
        f2.addRow("Units", self.units)
        self.grid = QtWidgets.QDoubleSpinBox()
        self.grid.setRange(0.001, 100000)
        self.grid.setDecimals(3)
        self.grid.setValue(settings.grid_spacing)
        f2.addRow("Grid spacing", self.grid)
        self.extent = QtWidgets.QSpinBox()
        self.extent.setRange(5, 500)
        self.extent.setValue(settings.grid_extent)
        f2.addRow("Grid lines (each side)", self.extent)
        self.delay = QtWidgets.QSpinBox()
        self.delay.setRange(0, 5000)
        self.delay.setSuffix(" ms")
        self.delay.setValue(settings.compute_dialog_delay_ms)
        f2.addRow("Show compute dialog after", self.delay)
        self.always = QtWidgets.QCheckBox("Always show the compute dialog")
        self.always.setChecked(settings.always_show_compute_dialog)
        f2.addRow("", self.always)
        tabs.addTab(mo, "Modeling")

        # Keyboard shortcuts
        self.sc_rows = []
        acts = parent.shortcut_actions() if parent is not None and hasattr(parent, "shortcut_actions") else []
        if acts:
            kb = QtWidgets.QWidget()
            kl = QtWidgets.QVBoxLayout(kb)
            self.sc_filter = QtWidgets.QLineEdit()
            self.sc_filter.setPlaceholderText("Search commands...")
            kl.addWidget(self.sc_filter)
            self.sc_table = QtWidgets.QTableWidget(len(acts), 3)
            self.sc_table.setHorizontalHeaderLabels(["Command", "Group", "Shortcut"])
            self.sc_table.verticalHeader().setVisible(False)
            self.sc_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
            self.sc_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
            custom = getattr(settings, "shortcuts", {}) or {}
            for r, (aid, label, group, default, fn) in enumerate(acts):
                self.sc_table.setItem(r, 0, QtWidgets.QTableWidgetItem(label))
                self.sc_table.setItem(r, 1, QtWidgets.QTableWidgetItem(group))
                for c in (0, 1):
                    self.sc_table.item(r, c).setFlags(QtCore.Qt.ItemIsEnabled)
                ed = QtWidgets.QKeySequenceEdit(QtGui.QKeySequence(custom.get(aid, default)))
                try:
                    ed.setMaximumSequenceLength(1)
                    ed.setClearButtonEnabled(True)
                except Exception:
                    pass
                ed.keySequenceChanged.connect(lambda *_: self._check_conflicts())
                self.sc_table.setCellWidget(r, 2, ed)
                self.sc_rows.append((aid, default, ed, r))
            kl.addWidget(self.sc_table, 1)
            self.sc_note = QtWidgets.QLabel("Click a shortcut cell and press the keys. Tip: use F-keys or combinations "
                                            "with Ctrl/Cmd, Alt/Option or Shift; plain letters are used for typing commands.")
            self.sc_note.setWordWrap(True)
            kl.addWidget(self.sc_note)
            rb = QtWidgets.QPushButton("Reset all shortcuts")
            rb.clicked.connect(self._reset_shortcuts)
            kl.addWidget(rb, 0, QtCore.Qt.AlignLeft)
            self.sc_filter.textChanged.connect(self._filter_shortcuts)
            tabs.addTab(kb, "Shortcuts")
            self.resize(560, 640)

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Apply | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(lambda: (self.apply(), self.accept()))
        bb.rejected.connect(self.reject)
        bb.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(self.apply)
        lay.addWidget(bb)

    def _filter_shortcuts(self, text):
        t = text.lower()
        for aid, default, ed, r in self.sc_rows:
            label = self.sc_table.item(r, 0).text().lower() + " " + self.sc_table.item(r, 1).text().lower()
            self.sc_table.setRowHidden(r, bool(t) and t not in label)

    def _reset_shortcuts(self):
        for aid, default, ed, r in self.sc_rows:
            ed.setKeySequence(QtGui.QKeySequence(default))
        self._check_conflicts()

    def _check_conflicts(self):
        seen = {}
        for aid, default, ed, r in self.sc_rows:
            k = ed.keySequence().toString()
            if k:
                seen.setdefault(k, []).append(r)
        clash = {r for rows in seen.values() if len(rows) > 1 for r in rows}
        for aid, default, ed, r in self.sc_rows:
            ed.setStyleSheet("background:#5a1f24;" if r in clash else "")
        if hasattr(self, "sc_note"):
            self.sc_note.setText(("Some shortcuts are used twice (marked red); only one of them will work. " if clash else "") +
                                 "Click a shortcut cell and press the keys. Tip: use F-keys or combinations with "
                                 "Ctrl/Cmd, Alt/Option or Shift; plain letters are used for typing commands.")
        return clash

    def _theme_changed(self, name):
        self.s.colors = {}
        for key, b in self.btns.items():
            b.set_color(S.PRESETS[name][key])

    def _reset_colors(self):
        self._theme_changed(self.theme.currentText())

    def apply(self):
        s = self.s
        s.theme = self.theme.currentText()
        preset = S.PRESETS[s.theme]
        s.colors = {k: b.color for k, b in self.btns.items() if b.color.lower() != preset[k].lower()}
        s.gradient_3d = self.grad.isChecked()
        s.ambient_occlusion = self.ssao.isChecked()
        s.font_size = self.font.value()
        s.units = self.units.currentText()
        s.grid_spacing = self.grid.value()
        s.grid_extent = self.extent.value()
        s.compute_dialog_delay_ms = self.delay.value()
        s.always_show_compute_dialog = self.always.isChecked()
        if self.sc_rows:
            s.shortcuts = {}
            for aid, default, ed, r in self.sc_rows:
                k = ed.keySequence().toString(QtGui.QKeySequence.PortableText)
                if k != default:
                    s.shortcuts[aid] = k
        s.save()
        self.applied.emit()
