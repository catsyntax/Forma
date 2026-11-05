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
class LayersPanel(QtWidgets.QWidget):
    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.setObjectName("Panel")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Layer", "On", "Lock", "Color"])
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for i in (1, 2, 3):
            self.table.horizontalHeader().setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setShowGrid(False)
        self.table.itemChanged.connect(self._renamed)
        self.table.cellDoubleClicked.connect(self._make_current)
        lay.addWidget(self.table)
        row = QtWidgets.QHBoxLayout()
        for txt, fn in (("New", self._add), ("Set current", self._set_current), ("Delete", self._delete)):
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        lay.addWidget(muted("Double-click a layer to make it current. New objects go on the current layer (bold)."))
        doc.subscribe(lambda w: self.refresh() if w in ("layers", "loaded") else None)
        self._busy = False
        self.refresh()

    def refresh(self):
        self._busy = True
        self.table.setRowCount(len(self.doc.layers))
        for r, l in enumerate(self.doc.layers):
            it = QtWidgets.QTableWidgetItem(l.name)
            f = it.font()
            f.setBold(l.name == self.doc.current_layer)
            it.setFont(f)
            self.table.setItem(r, 0, it)
            for col, attr in ((1, "visible"), (2, "locked")):
                cb = QtWidgets.QCheckBox()
                cb.setChecked(getattr(l, attr))
                cb.toggled.connect(lambda v, name=l.name, a=attr: self._toggle(name, a, v))
                w = QtWidgets.QWidget()
                hl = QtWidgets.QHBoxLayout(w)
                hl.setContentsMargins(6, 0, 6, 0)
                hl.addWidget(cb)
                self.table.setCellWidget(r, col, w)
            btn = ColorButton(l.color)
            btn.color_changed.connect(lambda c, name=l.name: self._color(name, c))
            self.table.setCellWidget(r, 3, btn)
        self._busy = False

    def _toggle(self, name, attr, v):
        l = self.doc.layer(name)
        setattr(l, attr, v)
        if attr == "visible" or attr == "locked":
            self.doc.selection = {i for i in self.doc.selection if self.doc.objects[i].layer != name}
        self.doc.notify("changed")

    def _color(self, name, c):
        self.doc.layer(name).color = c
        self.doc.notify("changed")

    def _renamed(self, item):
        if self._busy or item.column() != 0:
            return
        old = self.doc.layers[item.row()].name
        new = item.text().strip()
        if not new or new == old or self.doc.layer(new):
            self.refresh()
            return
        self.doc.checkpoint("Rename layer")
        self.doc.layers[item.row()].name = new
        for g in list(self.doc.objects.values()):
            if g.layer == old:
                self.doc.objects[g.id] = g.copy(layer=new)
        if self.doc.current_layer == old:
            self.doc.current_layer = new
        self.doc.notify("layers")
        self.doc.notify("changed")

    def _make_current(self, r, c):
        if c == 0 or True:
            self.doc.current_layer = self.doc.layers[r].name
            self.refresh()

    def _set_current(self):
        r = self.table.currentRow()
        if r >= 0:
            self._make_current(r, 0)

    def _add(self):
        self.doc.checkpoint("New layer")
        self.doc.add_layer()

    def _delete(self):
        r = self.table.currentRow()
        if r < 0 or len(self.doc.layers) == 1:
            return
        name = self.doc.layers[r].name
        used = [g for g in self.doc.objects.values() if g.layer == name]
        if used:
            QtWidgets.QMessageBox.information(self, "Layer in use", f"Layer '{name}' has {len(used)} objects. Move or delete them first.")
            return
        self.doc.checkpoint("Delete layer")
        self.doc.layers.pop(r)
        if self.doc.current_layer == name:
            self.doc.current_layer = self.doc.layers[0].name
        self.doc.notify("layers")


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
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Layer"))
        cb = QtWidgets.QComboBox()
        names = [l.name for l in self.doc.layers]
        cb.addItems(names)
        cur = {g.layer for g in sel}
        if len(cur) == 1:
            cb.setCurrentText(cur.pop())
        cb.activated.connect(lambda i: self._set_layer([g.id for g in sel], names[i]))
        row.addWidget(cb, 1)
        form.addLayout(row)

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

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Apply | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(lambda: (self.apply(), self.accept()))
        bb.rejected.connect(self.reject)
        bb.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(self.apply)
        lay.addWidget(bb)

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
        s.font_size = self.font.value()
        s.units = self.units.currentText()
        s.grid_spacing = self.grid.value()
        s.grid_extent = self.extent.value()
        s.compute_dialog_delay_ms = self.delay.value()
        s.always_show_compute_dialog = self.always.isChecked()
        s.save()
        self.applied.emit()
