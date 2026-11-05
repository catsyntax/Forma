"""Components workspace: library browser, live preview and definition editor."""
from __future__ import annotations

import math
import json

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import components as C
from . import geometry as G
from .panels import heading, muted, PropertiesPanel

THUMB = 96
_thumb_cache = {}


def thumbnail(d: C.ComponentDef, color="#d0d0d0", bg=None, size=THUMB):
    key = (d.id, d.version, color, bg, size, json.dumps(d.params, sort_keys=True, default=str))
    if key in _thumb_cache:
        return _thumb_cache[key]
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtGui.QColor(bg) if bg else QtCore.Qt.transparent)
    try:
        geoms = d.build()
    except Exception:
        geoms = []
    segs = []
    c30, s30 = math.cos(math.radians(30)), math.sin(math.radians(30))
    proj = lambda p: ((p[0] - p[1]) * c30, -((p[0] + p[1]) * s30 + p[2]))
    for g in geoms:
        if g.kind in ("mesh", "brep"):
            fe = G.feature_edges(g, 35)
            if len(fe) < 24:
                fe = G.feature_edges(g, 14)
            for a, b in fe:
                segs.append((proj(a), proj(b)))
        elif g.is_curve:
            s = g.sample()
            for a, b in zip(s[:-1], s[1:]):
                segs.append((proj(a), proj(b)))
    qp = QtGui.QPainter(pm)
    qp.setRenderHint(QtGui.QPainter.Antialiasing)
    if segs:
        pts = np.array([p for s in segs for p in s])
        lo, hi = pts.min(0), pts.max(0)
        span = max(hi[0] - lo[0], hi[1] - lo[1], 1e-9)
        sc = (size - 14) / span
        off = (np.array([size, size]) - (hi - lo) * sc) / 2
        qp.setPen(QtGui.QPen(QtGui.QColor(color), 1.0))
        for a, b in segs[:6000]:
            pa = (np.array(a) - lo) * sc + off
            pb = (np.array(b) - lo) * sc + off
            qp.drawLine(QtCore.QPointF(*pa), QtCore.QPointF(*pb))
    else:
        qp.setPen(QtGui.QColor(color))
        qp.drawText(pm.rect(), QtCore.Qt.AlignCenter, "Empty")
    qp.end()
    _thumb_cache[key] = pm
    return pm


class LibraryList(QtWidgets.QListWidget):
    """Thumbnails you can drag straight into the Draft or Model view to place a component."""

    def __init__(self):
        super().__init__()
        self.setDragEnabled(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragOnly)

    def mimeData(self, items):
        md = QtCore.QMimeData()
        if items:
            md.setText("forma-component:" + items[0].data(QtCore.Qt.UserRole))
        return md

    def startDrag(self, actions):
        it = self.currentItem()
        if it is None:
            return
        drag = QtGui.QDrag(self)
        drag.setMimeData(self.mimeData([it]))
        drag.setPixmap(it.icon().pixmap(64, 64))
        drag.exec(QtCore.Qt.CopyAction)


class ComponentsWorkspace(QtWidgets.QWidget):
    place_requested = QtCore.Signal(str, str)  # def_id, space

    def __init__(self, doc, settings, make_preview_view, parent=None):
        super().__init__(parent)
        self.doc, self.settings = doc, settings
        self.current_id = None
        self.preview_params = {}
        # The workspace is two panels placed beside the shared 3D view by the main window:
        # self.library_panel (left) and self.details_panel (right).

        # ---- library
        left = QtWidgets.QWidget()
        left.setObjectName("Panel")
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(10, 10, 10, 10)
        ll.addWidget(heading("Library"))
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search components...")
        self.search.textChanged.connect(self.refresh_list)
        self.cat = QtWidgets.QComboBox()
        self.cat.currentTextChanged.connect(self.refresh_list)
        ll.addWidget(self.search)
        ll.addWidget(self.cat)
        self.list = LibraryList()
        self.list.setViewMode(QtWidgets.QListView.IconMode)
        self.list.setIconSize(QtCore.QSize(THUMB, THUMB))
        self.list.setGridSize(QtCore.QSize(THUMB + 24, THUMB + 36))
        self.list.setResizeMode(QtWidgets.QListView.Adjust)
        self.list.setMovement(QtWidgets.QListView.Static)
        self.list.setWordWrap(True)
        self.list.currentItemChanged.connect(self._selected)
        self.list.itemDoubleClicked.connect(lambda it: self.place("3d"))
        ll.addWidget(self.list, 1)
        row = QtWidgets.QHBoxLayout()
        b_new = QtWidgets.QPushButton("New parametric")
        b_new.clicked.connect(self.new_parametric)
        b_imp = QtWidgets.QPushButton("Import...")
        b_imp.clicked.connect(self.import_lib)
        row.addWidget(b_new)
        row.addWidget(b_imp)
        ll.addLayout(row)
        ll.addWidget(muted("Drag a component into the view to place it, or double-click to place with the mouse. "
                           "Make your own: select objects and run MakeComponent. Snap them together with Connect or Chain."))
        left.setMinimumWidth(250)
        self.library_panel = left

        # ---- details (small live preview on top, then parameters and actions)
        right = QtWidgets.QWidget()
        right.setObjectName("Panel")
        right.setMinimumWidth(280)
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        self.preview = make_preview_view(self._preview_items)
        self.preview.setFixedHeight(220)
        rl.addWidget(self.preview)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self.details = QtWidgets.QWidget()
        self.details.setObjectName("Panel")
        scroll.setWidget(self.details)
        rl.addWidget(scroll, 1)
        self.details_panel = right

        doc.subscribe(self._doc_changed)
        self.refresh_list()

    # ------------------------------------------------------------ data
    def all_defs(self):
        seen = {}
        for d in self.doc.definitions.values():
            seen[d.id] = (d, True)
        for d in C.library():
            if d.id not in seen:
                seen[d.id] = (d, False)
        return list(seen.values())

    def get_def(self, did):
        if did in self.doc.definitions:
            return self.doc.definitions[did]
        for d in C.library():
            if d.id == did:
                return d
        return None

    def ensure_in_doc(self, did):
        d = self.get_def(did)
        if d is not None and did not in self.doc.definitions:
            self.doc.add_definition(d)
        return self.doc.definitions.get(did)

    def _doc_changed(self, what):
        if what in ("definitions", "loaded"):
            self.refresh_list()

    def refresh_list(self):
        cur = self.current_id
        cats = sorted({d.category for d, _ in self.all_defs()})
        self.cat.blockSignals(True)
        sel = self.cat.currentText()
        self.cat.clear()
        self.cat.addItems(["All categories", "In this document"] + cats)
        if sel:
            i = self.cat.findText(sel)
            self.cat.setCurrentIndex(max(0, i))
        self.cat.blockSignals(False)
        self.list.blockSignals(True)
        self.list.clear()
        q = self.search.text().lower()
        cat = self.cat.currentText()
        col = self.settings.color("text")
        for d, in_doc in sorted(self.all_defs(), key=lambda x: (not x[1], x[0].category, x[0].name)):
            if q and q not in (d.name + d.category + d.description).lower():
                continue
            if cat == "In this document" and not in_doc:
                continue
            if cat not in ("All categories", "In this document", "") and d.category != cat:
                continue
            label = d.name + ("" if not in_doc else f"\n({len(self.doc.instances_of(d.id))} used)")
            it = QtWidgets.QListWidgetItem(QtGui.QIcon(thumbnail(d, col)), label)
            it.setData(QtCore.Qt.UserRole, d.id)
            it.setToolTip(f"{d.name} — {d.category}\n{d.description}" + ("\nParametric" if d.parametric else ""))
            self.list.addItem(it)
            if d.id == cur:
                self.list.setCurrentItem(it)
        self.list.blockSignals(False)
        if cur and self.get_def(cur):
            self.show_details(cur)
        elif self.list.count():
            self.list.setCurrentRow(0)

    def _selected(self, it, prev=None):
        if it is None:
            return
        self.current_id = it.data(QtCore.Qt.UserRole)
        self.preview_params = {}
        self.show_details(self.current_id)
        self.preview.rebuild()
        self.preview.zoom_extents()

    def _preview_items(self):
        d = self.get_def(self.current_id) if self.current_id else None
        if d is None:
            return []
        try:
            geoms = d.build(self.preview_params)
        except Exception as e:
            return []
        out = [("__preview__", g, "#c9ccd2" if g.kind in ("mesh", "brep") else self.settings.color("text"), False) for g in geoms
               if g.kind != "instance"]
        # connectors as points
        for c in d.get_connectors(self.preview_params):
            out.append(("__conn__", G.point(c["p"]), self.settings.color("accent"), False))
            L = max(1.0, np.linalg.norm(np.ptp(np.vstack([g.bbox() for g in geoms]), axis=0)) * 0.08) if geoms else 1
            out.append(("__conn__", G.line(c["p"], c["p"] + c["n"] * L), self.settings.color("accent"), False))
        return out

    # ------------------------------------------------------------ details panel
    def show_details(self, did):
        d = self.get_def(did)
        old = self.details.layout()
        if old is not None:
            QtWidgets.QWidget().setLayout(old)
        lay = QtWidgets.QVBoxLayout(self.details)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setAlignment(QtCore.Qt.AlignTop)
        if d is None:
            lay.addWidget(heading("No component selected"))
            return
        in_doc = did in self.doc.definitions
        lay.addWidget(heading(d.name))
        lay.addWidget(muted(f"{d.category}" + ("  ·  Parametric" if d.parametric else "  ·  Geometry") +
                            (f"  ·  used {len(self.doc.instances_of(did))}×" if in_doc else "  ·  in library")))
        if d.description:
            lay.addWidget(muted(d.description))

        form = QtWidgets.QFormLayout()
        name = QtWidgets.QLineEdit(d.name)
        cat = QtWidgets.QLineEdit(d.category)
        desc = QtWidgets.QLineEdit(d.description)
        editable = not did.startswith("builtin-") or in_doc
        for w in (name, cat, desc):
            w.setEnabled(editable)
        name.editingFinished.connect(lambda: self._update_def(did, name=name.text()))
        cat.editingFinished.connect(lambda: self._update_def(did, category=cat.text()))
        desc.editingFinished.connect(lambda: self._update_def(did, description=desc.text()))
        form.addRow("Name", name)
        form.addRow("Category", cat)
        form.addRow("Description", desc)
        lay.addLayout(form)

        if d.parametric and d.params:
            lay.addSpacing(8)
            lay.addWidget(heading("Parameters"))
            lay.addWidget(muted("Try values here (preview only), or set new defaults for future placements."))
            pf = QtWidgets.QFormLayout()
            vals = d.merged_params(self.preview_params)
            for k, v in vals.items():
                w = PropertiesPanel._param_widget(v)
                PropertiesPanel._bind(w, lambda val, key=k: self._preview_param(key, val))
                pf.addRow(k.replace("_", " ").capitalize(), w)
            lay.addLayout(pf)
            b = QtWidgets.QPushButton("Use these values as defaults")
            b.clicked.connect(lambda: self._set_defaults(did))
            lay.addWidget(b)

        conns = d.get_connectors(self.preview_params) if True else []
        lay.addSpacing(8)
        lay.addWidget(heading(f"Connectors ({len(conns)})"))
        lay.addWidget(muted(", ".join(c["name"] for c in conns) or "None. Plain components get one per bounding-box face."))

        lay.addSpacing(10)
        grid = QtWidgets.QGridLayout()
        b1 = QtWidgets.QPushButton("Place in Model")
        b1.setObjectName("Primary")
        b1.clicked.connect(lambda: self.place("3d"))
        b2 = QtWidgets.QPushButton("Place in Draft")
        b2.clicked.connect(lambda: self.place("2d"))
        b3 = QtWidgets.QPushButton("Edit code" if d.parametric else "Convert to parametric")
        b3.clicked.connect(lambda: self.edit_code(did))
        b4 = QtWidgets.QPushButton("Duplicate")
        b4.clicked.connect(lambda: self.duplicate(did))
        b5 = QtWidgets.QPushButton("Save to my library")
        b5.clicked.connect(lambda: self.save_to_library(did))
        b7 = QtWidgets.QPushButton(f"Select placed copies ({len(self.doc.instances_of(did))})")
        b7.setEnabled(in_doc and bool(self.doc.instances_of(did)))
        b7.clicked.connect(lambda: self.select_copies(did))
        b6 = QtWidgets.QPushButton("Remove from document")
        b6.setEnabled(in_doc)
        b6.clicked.connect(lambda: self.remove_from_doc(did))
        for i, b in enumerate((b1, b2, b3, b4, b5, b6, b7)):
            grid.addWidget(b, i // 2, i % 2)
        lay.addLayout(grid)
        lay.addSpacing(8)
        lay.addWidget(muted("Tips: in the Model, run Connect to snap one component's connector to another's, "
                            "Chain to grow a row of modules, and double-click a component to edit it in place."))

    def select_copies(self, did):
        self.doc.select([g.id for g in self.doc.instances_of(did)])
        self.place_requested.emit("", "select")

    def show_definition(self, did):
        """Called when you switch here with a placed component selected."""
        if did and self.get_def(did):
            self.current_id = did
            self.preview_params = {}
            self.search.clear()
            self.cat.setCurrentIndex(0)
            self.refresh_list()
            self.preview.rebuild()
            self.preview.zoom_extents()

    def _preview_param(self, key, val):
        self.preview_params[key] = val
        self.preview.rebuild()

    def _set_defaults(self, did):
        d = self.ensure_in_doc(did)
        self.doc.checkpoint("Component defaults")
        self.doc.definitions[did] = d.clone(params=d.merged_params(self.preview_params))
        self.preview_params = {}
        self.doc.notify("definitions")
        self.doc.notify("changed")

    def _update_def(self, did, **kw):
        d = self.ensure_in_doc(did)
        if d is None or all(getattr(d, k) == v for k, v in kw.items()):
            return
        self.doc.checkpoint("Edit component")
        self.doc.definitions[did] = d.clone(**kw)
        self.doc.notify("definitions")
        self.doc.notify("changed")

    def place(self, space):
        if not self.current_id:
            return
        self.ensure_in_doc(self.current_id)
        self.place_requested.emit(self.current_id, space)

    def duplicate(self, did):
        d = self.get_def(did)
        nd = d.clone(id=C.new_id(), name=d.name + " copy", version=0,
                     category=d.category if not did.startswith("builtin-") else "My Components")
        self.doc.checkpoint("Duplicate component")
        self.doc.add_definition(nd)
        self.current_id = nd.id
        self.refresh_list()

    def remove_from_doc(self, did):
        n = len(self.doc.instances_of(did))
        if n and QtWidgets.QMessageBox.question(self, "Remove component",
                                                f"This deletes {n} placed copies too. Continue?") != QtWidgets.QMessageBox.Yes:
            return
        self.doc.checkpoint("Remove component")
        self.doc.remove([g.id for g in self.doc.instances_of(did)], notify=False)
        self.doc.definitions.pop(did, None)
        self.doc.notify("definitions")
        self.doc.notify("changed")

    def save_to_library(self, did):
        d = self.get_def(did)
        lib = C.load_user_library()
        lib = [x for x in lib if x.id != d.id] + [d]
        C.save_user_library(lib)
        QtWidgets.QMessageBox.information(self, "Saved", f"'{d.name}' saved to your library:\n{C.USER_LIBRARY}")
        self.refresh_list()

    def import_lib(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import components", "", "Forma components (*.json);;Forma project (*.forma)")
        if not path:
            return
        with open(path) as f:
            data = json.load(f)
        defs = data.get("definitions", []) if isinstance(data, dict) else data
        self.doc.checkpoint("Import components")
        for dd in defs:
            self.doc.add_definition(C.ComponentDef.from_dict(dd))
        self.refresh_list()

    def new_parametric(self):
        d = C.ComponentDef("New component", code=C.NEW_COMPONENT_TEMPLATE, params=dict(C.NEW_COMPONENT_PARAMS),
                           category="My Components", description="Custom parametric component.")
        if self.edit_code(None, d):
            self.current_id = d.id
            self.refresh_list()

    def edit_code(self, did, new_def=None):
        d = new_def or self.get_def(did)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Parametric component: {d.name}")
        dlg.resize(780, 620)
        lay = QtWidgets.QVBoxLayout(dlg)
        form = QtWidgets.QFormLayout()
        name = QtWidgets.QLineEdit(d.name)
        params = QtWidgets.QLineEdit(json.dumps(d.params) if d.params else json.dumps(C.NEW_COMPONENT_PARAMS))
        params.setToolTip('JSON, e.g. {"width": 800.0, "count": 4, "round": true}')
        form.addRow("Name", name)
        form.addRow("Parameters (JSON)", params)
        lay.addLayout(form)
        ed = QtWidgets.QPlainTextEdit(d.code or C.NEW_COMPONENT_TEMPLATE)
        font = QtGui.QFont("Menlo")
        font.setStyleHint(QtGui.QFont.Monospace)
        ed.setFont(font)
        lay.addWidget(ed, 1)
        err = QtWidgets.QLabel("")
        err.setObjectName("Muted")
        err.setWordWrap(True)
        lay.addWidget(err)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        lay.addWidget(bb)
        bb.rejected.connect(dlg.reject)

        def try_accept():
            try:
                p = json.loads(params.text() or "{}")
                test = C.ComponentDef(name.text(), code=ed.toPlainText(), params=p)
                test.build()
                test.get_connectors()
            except Exception as e:
                err.setText(f"Error: {type(e).__name__}: {e}")
                return
            dlg.accept()
        bb.accepted.connect(try_accept)
        if not dlg.exec():
            return False
        p = json.loads(params.text() or "{}")
        self.doc.checkpoint("Edit component code")
        if new_def is not None:
            new_def.name, new_def.code, new_def.params = name.text(), ed.toPlainText(), p
            self.doc.add_definition(new_def)
        else:
            base = self.ensure_in_doc(did)
            if did.startswith("builtin-"):
                base = base.clone(id=C.new_id(), version=0, category="My Components")
            self.doc.definitions[base.id] = base.clone(name=name.text(), code=ed.toPlainText(), params=p)
            self.current_id = base.id
            self.doc.notify("definitions")
            self.doc.notify("changed")
        return True
