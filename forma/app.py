"""Forma main window: workspace switcher (Draft / Model / Components / Nodes), menus, shortcuts."""
from __future__ import annotations

import os
import re
import sys

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import APP_NAME, __version__
from . import commands as K
from . import geometry as G
from .document import Document
from .interaction import Interaction, CommandLine
from .settings import Settings, stylesheet, FONT
from .panels import LayersPanel, PropertiesPanel, CommandPalette, SettingsDialog, heading, muted
from .view2d import View2D
from .view3d import View3D
from .components_ui import ComponentsWorkspace
from .nodes.editor import NodeEditor, NodeLibraryTree
from .io3dm import export_3dm, import_3dm
from .bootstrap import _FileOpenFilter, app_icon, create_qapp

PERSONAS = ["Draft", "Model", "Components", "Nodes"]


def _trace(msg):
    if os.environ.get("FORMA_TRACE"):
        print(f"[forma] {msg}", flush=True)

DRAFT_TOOLS = ["Line", "Polyline", "Rectangle", "Circle", "Arc", "Ellipse", "Polygon", "InterpCrv", "|",
               "Move", "Copy", "Rotate", "Scale", "Mirror", "Offset", "Trim", "Split", "Array", "Join", "Explode", "|",
               "PointsOn", "PointsOff", "|", "SectionPlane", "Insert", "MakeComponent", "Connect", "|", "Distance", "Length", "Area"]
MODEL_TOOLS = ["Line", "Polyline", "Circle", "Rectangle", "Curve", "InterpCrv", "|",
               "Box", "Sphere", "Cylinder", "Cone", "Torus", "Plane", "|",
               "Extrude", "Loft", "Revolve", "Pipe", "PlanarSrf", "|",
               "Move", "Copy", "Rotate", "Scale", "Mirror", "Array", "||",
               "Trim", "Split", "|", "BooleanUnion", "BooleanDifference", "BooleanIntersection", "FilletEdge", "ChamferEdge", "Sweep1", "Intersect", "DupEdge", "JoinSrf", "Cap", "|",
               "PointsOn", "PointsOff", "SelAllPoints", "|",
               "Insert", "MakeComponent", "Connect", "Chain", "|", "SectionPlane", "Make2D"]
COMPONENT_TOOLS = ["Insert", "MakeComponent", "EditComponent", "CloseComponent", "MakeUnique", "|",
                   "Connect", "Chain", "|", "Move", "Copy", "Rotate", "Mirror", "Array", "|", "Explode", "Delete"]
NODE_TOOLS = ["Move", "Copy", "Rotate", "Scale", "|", "Box", "Extrude", "Loft", "|", "Distance", "Area", "Volume"]
DRAFT_VIEWS = ["Top", "Bottom", "Front", "Back", "Left", "Right"]


CMD_KEYS = {"Hide": "Ctrl+H", "Show": "Ctrl+Alt+H", "Lock": "Ctrl+L", "Unlock": "Ctrl+Alt+L"}   # Rhino's keys
if sys.platform == "darwin":
    CMD_KEYS["Hide"] = "Ctrl+Shift+Alt+H"   # Cmd+H is macOS "Hide application"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings: Settings | None = None):
        super().__init__()
        self.settings = settings or Settings.load()
        self.doc = Document()
        self.doc.units = self.settings.units
        self.ix = Interaction(self.doc, self.settings, app=self)
        self.clipboard = []
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 920)
        self._apply_style()

        central = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        self.setCentralWidget(central)

        # ---------- workspace switcher bar
        bar = QtWidgets.QWidget()
        bar.setObjectName("PersonaBar")
        bl = QtWidgets.QHBoxLayout(bar)
        bl.setContentsMargins(12, 0, 8, 0)
        bl.setSpacing(0)
        brand = QtWidgets.QLabel(APP_NAME)
        brand.setObjectName("Heading")
        brand.setStyleSheet("font-size: 15px; padding-right: 18px;")
        bl.addWidget(brand)
        self.persona_buttons = []
        grp = QtWidgets.QButtonGroup(self)
        grp.setExclusive(True)
        for i, name in enumerate(PERSONAS):
            b = QtWidgets.QPushButton(name)
            b.setObjectName("Persona")
            b.setCheckable(True)
            b.setToolTip(f"{name} workspace (Ctrl+{i + 1})")
            b.clicked.connect(lambda _=False, i=i: self.set_persona(i))
            bf = QtGui.QFont(FONT, self.settings.font_size)
            bf.setBold(True)
            b.setMinimumWidth(QtGui.QFontMetrics(bf).horizontalAdvance(name) + 36)
            grp.addButton(b)
            bl.addWidget(b)
            self.persona_buttons.append(b)
        bl.addStretch()
        self.toggle_buttons = {}
        for key, label, tip in (("snap", "Snap", "Grid snap (F9)"), ("ortho", "Ortho", "Ortho (F8)"),
                                ("osnap", "Osnap", "Object snaps (F3)"), ("show_grid", "Grid", "Show grid (F7)"),
                                ("gumball", "Gumball", "Show the gumball on selected objects or points")):
            b = QtWidgets.QToolButton()
            b.setText(label)
            b.setCheckable(True)
            b.setChecked(getattr(self.settings, key))
            b.setToolTip(tip)
            b.toggled.connect(lambda v, k=key: self.set_toggle(k, v))
            bl.addWidget(b)
            self.toggle_buttons[key] = b
        sbtn = QtWidgets.QToolButton()
        sbtn.setText("Settings")
        sbtn.clicked.connect(self.open_settings)
        bl.addSpacing(10)
        bl.addWidget(sbtn)
        cl.addWidget(bar)

        # ---------- shared widgets
        self.cmdline = CommandLine(self.ix)
        self.layers = LayersPanel(self.doc)
        self.props = PropertiesPanel(self.doc, app=self)
        self.sidebar_host = {}

        # ---------- pages
        self.stack = QtWidgets.QStackedWidget()
        cl.addWidget(self.stack, 1)
        cl.addWidget(self.cmdline)

        _trace("creating 2D view")
        self.view2d = View2D(self.doc, self.ix, self.settings, self.cmdline)
        _trace("creating 3D view")
        self.view3d = View3D(self.doc, self.ix, self.settings, self.cmdline)
        self.stack.addWidget(self._modeling_page("2d", self.view2d, DRAFT_TOOLS))

        _trace("creating Components workspace")
        self.components = ComponentsWorkspace(self.doc, self.settings, self._make_preview_view)
        self.components.place_requested.connect(self.place_component)

        _trace("creating Nodes workspace")
        self.nodes = NodeEditor(self.doc, self.settings)
        self.nodes.preview_changed.connect(self._node_preview)
        self.nodes.message.connect(self.message)
        self.nodes.compute.dialog = None

        # Model, Components and Nodes share one 3D page: the same scene, with different panels around it.
        self.stack.addWidget(self._scene_page())

        # ---------- status bar
        self.coord = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self.coord)
        self.layer_lbl = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self.layer_lbl)
        self.view2d.cursor_moved.connect(self._cursor)
        self.view2d.component_dropped.connect(lambda d, p: self.drop_component(d, p, "2d"))
        self.view3d.component_dropped.connect(lambda d, p: self.drop_component(d, p, "3d"))
        self.view3d.cursor_moved.connect(self._cursor)
        self.doc.subscribe(self._doc_event)
        self.ix.message.connect(lambda t: self.statusBar().showMessage(t, 4000))

        _trace("building menus")
        self._menus()
        self._shortcuts()
        self.set_persona(1)
        self._update_title()
        self.message(f"{APP_NAME} {__version__}. Type a command or press Ctrl+K to search commands. "
                     f"Right-drag orbits, Shift+right-drag pans, right-click repeats the last command.")

    # ================================================================ layout
    def _apply_style(self):
        app = QtWidgets.QApplication.instance()
        f = QtGui.QFont(FONT, self.settings.font_size)
        app.setFont(f)
        app.setStyleSheet(stylesheet(self.settings))

    def _toolbar(self, tools, space):
        if "||" in tools:  # two rows
            i = tools.index("||")
            box = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(box)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)
            self._last_row = self._toolbar(tools[i + 1:], space)
            first = self._toolbar(tools[:i], space)
            lay.addWidget(first)
            lay.addWidget(self._last_row)
            box.addSeparator = self._last_row.addSeparator
            box.addAction = self._last_row.addAction
            box.addWidget = self._last_row.addWidget
            return box
        tb = QtWidgets.QToolBar()
        tb.setMovable(False)
        for t in tools:
            if t == "|":
                tb.addSeparator()
                continue
            cmd = K.find_command(t)
            a = tb.addAction(t)
            a.setToolTip(f"{t}: {cmd.help}" if cmd else t)
            a.triggered.connect(lambda _=False, t=t, s=space: self.run_command(t, s))
        return tb

    def _modeling_page(self, space, view, tools):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        top = self._toolbar(tools, space)
        if space == "3d":
            top.addSeparator()
            for label, act in (("Top", "view_top"), ("Front", "view_front"), ("Right", "view_right"),
                               ("Persp", "view_persp"), ("Zoom all", "zoom_extents")):
                a = top.addAction(label)
                a.triggered.connect(lambda _=False, a_=act: self.view_action(a_))
            self.shade_combo = QtWidgets.QComboBox()
            self.shade_combo.addItems(["Shaded", "Wireframe", "Ghosted"])
            self.shade_combo.currentIndexChanged.connect(lambda i: self.view3d.set_shade(["shaded", "wire", "ghost"][i]))
            top.addWidget(self.shade_combo)
        else:
            vbar = QtWidgets.QToolBar()
            vbar.setObjectName("ViewBar")
            vbar.setMovable(False)
            self.draft_view_buttons = {}
            vg = QtGui.QActionGroup(self)
            vg.setExclusive(True)
            for name in DRAFT_VIEWS:
                a = vbar.addAction(name)
                a.setCheckable(True)
                a.setChecked(name == "Top")
                a.setToolTip(f"Look at the model from the {name.lower()}; draw on that plane")
                a.triggered.connect(lambda _=False, n=name: self.view2d.set_view(n))
                vg.addAction(a)
                self.draft_view_buttons[name] = a
            vbar.addSeparator()
            a = vbar.addAction("Zoom all")
            a.triggered.connect(lambda: self.view_action("zoom_extents"))
            vbar.addSeparator()
            a = vbar.addAction("Export drawing...")
            a.setToolTip("Export this view and any section views to PDF, DWG, DXF, PNG or SVG (Ctrl+Alt+E)")
            a.triggered.connect(self.export_drawing)
        lay.addWidget(top)
        split = QtWidgets.QSplitter()
        if space == "2d":
            from .sectionview import SectionView
            holder = QtWidgets.QWidget()
            hl = QtWidgets.QVBoxLayout(holder)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(0)
            hl.addWidget(vbar)
            vsplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            vsplit.addWidget(view)
            # two section viewports (front and back of the section plane), shown while a section exists
            self.section_row = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            self.section_views = [SectionView(self.doc, self.settings, side=1), SectionView(self.doc, self.settings, side=-1)]
            for sv in self.section_views:
                self.section_row.addWidget(sv)
            vsplit.addWidget(self.section_row)
            vsplit.setStretchFactor(0, 3)
            vsplit.setStretchFactor(1, 2)
            self.section_row.setVisible(False)
            self._draft_vsplit = vsplit
            hl.addWidget(vsplit, 1)
            split.addWidget(holder)
            self.doc.subscribe(self._sections_changed)
        else:
            split.addWidget(view)
        side = QtWidgets.QWidget()
        side.setObjectName("Panel")
        side.setMinimumWidth(260)
        sl = QtWidgets.QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(0)
        split.addWidget(side)
        split.setStretchFactor(0, 1)
        split.setSizes([1160, 360])
        self.sidebar_host[space] = sl
        self.edit_banner = getattr(self, "edit_banner", {})
        banner = QtWidgets.QWidget()
        bl = QtWidgets.QHBoxLayout(banner)
        bl.setContentsMargins(10, 4, 10, 4)
        lab = QtWidgets.QLabel("Editing component")
        lab.setObjectName("Heading")
        close = QtWidgets.QPushButton("Close component")
        close.setObjectName("Primary")
        close.clicked.connect(lambda: self.run_command("CloseComponent"))
        cancel = QtWidgets.QPushButton("Discard changes")
        cancel.clicked.connect(self.discard_edit)
        bl.addWidget(lab)
        bl.addStretch()
        bl.addWidget(cancel)
        bl.addWidget(close)
        banner.hide()
        self.edit_banner[space] = (banner, lab)
        lay.addWidget(banner)
        lay.addWidget(split, 1)
        return page

    def _view_bar(self):
        tb = QtWidgets.QToolBar()
        tb.setMovable(False)
        for label, act, tip in (("Top", "view_top", "Top view"), ("Front", "view_front", "Front view"),
                                ("Right", "view_right", "Right view"), ("Persp", "view_persp", "Perspective"),
                                ("Iso", "view_iso", "Isometric (click again to turn to the next corner)"),
                                ("Eye level", "view_two_point", "Two-point perspective: vertical lines stay vertical"),
                                ("Zoom all", "zoom_extents", "Zoom to everything")):
            a = tb.addAction(label)
            a.setToolTip(tip)
            a.triggered.connect(lambda _=False, a_=act: self.view_action(a_))
        self.isolate_btn = QtWidgets.QToolButton()
        self.isolate_btn.setText("Isolate")
        self.isolate_btn.setCheckable(True)
        self.isolate_btn.setToolTip("Show only the selected objects (Ctrl+Shift+H). Click again to show everything.")
        self.isolate_btn.toggled.connect(self.set_isolate)
        tb.addWidget(self.isolate_btn)
        self.parallel_btn = QtWidgets.QToolButton()
        self.parallel_btn.setText("Parallel")
        self.parallel_btn.setCheckable(True)
        self.parallel_btn.setToolTip("Parallel (orthographic) projection on or off")
        self.parallel_btn.toggled.connect(self.view3d.set_parallel)
        tb.addWidget(self.parallel_btn)
        self.lens_combo = QtWidgets.QComboBox()
        self.lens_combo.addItems(["Lens 50°", "Lens 35°", "Lens 20°"])
        self.lens_combo.setCurrentIndex(1)
        self.lens_combo.setToolTip("Perspective field of view")
        self.lens_combo.currentIndexChanged.connect(lambda i: self.view3d.set_lens([50, 35, 20][i]))
        tb.addWidget(self.lens_combo)
        self.shade_combo = QtWidgets.QComboBox()
        self.shade_combo.addItems(["Shaded", "Wireframe", "Ghosted"])
        self.shade_combo.currentIndexChanged.connect(lambda i: self.view3d.set_shade(["shaded", "wire", "ghost"][i]))
        tb.addWidget(self.shade_combo)
        return tb

    def _scene_page(self):
        """One 3D scene for Model, Components and Nodes. Only the tools and side panels change."""
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        top = QtWidgets.QWidget()
        tl = QtWidgets.QHBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(0)
        self.tool_stack = QtWidgets.QStackedWidget()
        self.tool_stack.addWidget(self._toolbar(MODEL_TOOLS, "3d"))
        self.tool_stack.addWidget(self._toolbar(COMPONENT_TOOLS, "3d"))
        ntb = self._toolbar(NODE_TOOLS, "3d")
        ntb.addSeparator()
        ntb.addAction("Import Grasshopper...").triggered.connect(lambda: self.import_grasshopper())
        ntb.addAction("Save to Grasshopper...").triggered.connect(lambda: self.nodes.export_grasshopper())
        ntb.addAction("Example: tower").triggered.connect(lambda: self.nodes.load_demo())
        ex = QtWidgets.QComboBox()
        ex.addItem("Analysis examples...")
        ex.addItems(self.nodes.EXAMPLES)
        ex.setToolTip("Load a ready-made analysis graph (adds to the current graph)")

        def pick(i, box=ex):
            if i > 0:
                self.nodes.load_example(box.itemText(i))
                box.blockSignals(True)
                box.setCurrentIndex(0)
                box.blockSignals(False)
        ex.currentIndexChanged.connect(pick)
        ntb.addWidget(ex)
        self.tool_stack.addWidget(ntb)
        tl.addWidget(self.tool_stack, 1)
        lay.addWidget(top)

        # component-edit banner
        banner = QtWidgets.QWidget()
        bl = QtWidgets.QHBoxLayout(banner)
        bl.setContentsMargins(10, 4, 10, 4)
        lab = QtWidgets.QLabel("Editing component")
        lab.setObjectName("Heading")
        close = QtWidgets.QPushButton("Close component")
        close.setObjectName("Primary")
        close.clicked.connect(lambda: self.run_command("CloseComponent"))
        cancel = QtWidgets.QPushButton("Discard changes")
        cancel.clicked.connect(self.discard_edit)
        bl.addWidget(lab)
        bl.addStretch()
        bl.addWidget(cancel)
        bl.addWidget(close)
        banner.hide()
        self.edit_banner = getattr(self, "edit_banner", {})
        self.edit_banner["3d"] = (banner, lab)
        lay.addWidget(banner)

        split = QtWidgets.QSplitter()
        # left: component library or node library
        self.left_stack = QtWidgets.QStackedWidget()
        self.left_stack.addWidget(self.components.library_panel)
        nl = QtWidgets.QWidget()
        nl.setObjectName("Panel")
        nll = QtWidgets.QVBoxLayout(nl)
        nll.setContentsMargins(8, 8, 8, 8)
        nll.addWidget(heading("Nodes"))
        nll.addWidget(NodeLibraryTree(self.nodes), 1)
        nll.addWidget(muted("Drag a node onto the graph below, or double-click / Space on the graph to search. "
                            "Results show live in the 3D view. Geometry node: select objects, then 'Set from selection'."))
        self.left_stack.addWidget(nl)
        self.left_stack.setMinimumWidth(240)
        split.addWidget(self.left_stack)
        # middle: the shared view, with the node graph underneath in Nodes
        self.center_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        vholder = QtWidgets.QWidget()
        vhl = QtWidgets.QVBoxLayout(vholder)
        vhl.setContentsMargins(0, 0, 0, 0)
        vhl.setSpacing(0)
        vb = self._view_bar()
        vb.setObjectName("ViewBar")
        vhl.addWidget(vb)
        vhl.addWidget(self.view3d, 1)
        self.center_split.addWidget(vholder)
        self.center_split.addWidget(self.nodes)
        self.center_split.setStretchFactor(0, 3)
        self.center_split.setStretchFactor(1, 2)
        split.addWidget(self.center_split)
        # right: properties/layers, or component details
        self.right_stack = QtWidgets.QStackedWidget()
        side = QtWidgets.QWidget()
        side.setObjectName("Panel")
        sl = QtWidgets.QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(0)
        self.sidebar_host["3d"] = sl
        self.right_stack.addWidget(side)
        self.right_stack.setMinimumWidth(260)
        split.addWidget(self.right_stack)
        split.setStretchFactor(1, 1)
        split.setSizes([280, 1060, 360])
        lay.addWidget(split, 1)
        return page

    def _make_preview_view(self, source):
        return View3D(self.doc, self.ix, self.settings, None, preview_only=True, source=source)

    def _attach_sidebar(self, space):
        host = self.sidebar_host.get(space)
        if host is None:
            return
        tabs = getattr(self, "_side_tabs", None)
        if tabs is None:
            tabs = QtWidgets.QTabWidget()
            tabs.addTab(self.props, "Properties")
            tabs.addTab(self.layers, "Layers")
            tabs.addTab(self.components.details_panel, "Component")
            self._side_tabs = tabs
        host.addWidget(tabs)
        comp = getattr(self, "persona", 0) == 2
        tabs.setTabVisible(2, comp)
        if comp and tabs.currentIndex() == 0:
            tabs.setCurrentIndex(2)
        elif not comp and tabs.currentIndex() == 2:
            tabs.setCurrentIndex(0)

    # ================================================================ personas
    def set_persona(self, i):
        self.persona = i
        self.persona_buttons[i].setChecked(True)
        if self.ix.active:
            self.ix.cancel()
        self.stack.setCurrentIndex(0 if i == 0 else 1)
        space = "2d" if i == 0 else "3d"
        self.ix.space = space
        self._attach_sidebar(space)
        self.props.set_space(space)
        if i >= 1:
            self.tool_stack.setCurrentIndex(i - 1)
            self.left_stack.setVisible(i in (2, 3))
            if i in (2, 3):
                self.left_stack.setCurrentIndex(i - 2)
            self.right_stack.setCurrentIndex(0)
            self.nodes.setVisible(i == 3)
            if i == 3:
                h = max(self.center_split.height(), 600)
                self.center_split.setSizes([int(h * 0.55), int(h * 0.45)])
                self.nodes.emit_preview()
            else:
                self.view3d.set_extra([])
        (self.view2d if i == 0 else self.view3d.plotter.interactor).setFocus()
        if i == 2:
            sel = self.doc.selected()
            if len(sel) == 1 and sel[0].kind == "instance":
                self.components.show_definition(sel[0].def_id)
        self.cmdline.refresh()

    # ================================================================ commands
    def run_command(self, name, space=None, **kw):
        if space is None:
            space = "2d" if self.persona == 0 else "3d"
        if space == "2d" and self.persona != 0 and name not in ("Undo", "Redo"):
            space = "3d"   # every 3D workspace draws in the shared 3D view
        self.ix.start(name, space, **kw)
        self.cmdline.edit.setFocus()

    def message(self, text):
        self.cmdline.log(text)
        self.statusBar().showMessage(text, 5000)

    def view_action(self, action):
        v3 = self.view3d
        if action == "zoom_extents":
            (self.view2d.zoom_extents() if self.persona == 0 else v3.zoom_extents())
        elif action == "zoom_selected":
            (self.view2d.zoom_extents(True) if self.persona == 0 else v3.zoom_extents(True))
        elif action == "view_iso":
            corners = ["SE", "SW", "NW", "NE"]
            self._iso_i = (getattr(self, "_iso_i", -1) + 1) % 4
            v3.view_iso(corners[self._iso_i])
            self.message(f"Isometric from {corners[self._iso_i]}")
        elif action.startswith("view_"):
            getattr(v3, action)()
        if action.startswith("view_") and hasattr(self, "parallel_btn"):
            self.parallel_btn.blockSignals(True)
            self.parallel_btn.setChecked(bool(v3.ren.GetActiveCamera().GetParallelProjection()))
            self.parallel_btn.blockSignals(False)
        elif action.startswith("shade_"):
            idx = {"shade_shaded": 0, "shade_wire": 1, "shade_ghost": 2}[action]
            self.shade_combo.setCurrentIndex(idx)
        elif action.startswith("toggle_"):
            key = {"toggle_grid": "show_grid", "toggle_snap": "snap", "toggle_ortho": "ortho", "toggle_osnap": "osnap"}[action]
            self.toggle_buttons[key].setChecked(not getattr(self.settings, key))

    def _sections_changed(self, what):
        if what not in ("changed", "loaded", "layers", "selection"):
            return
        has = bool(self.doc.sections())
        row = getattr(self, "section_row", None)
        if row is None:
            return
        if has and not row.isVisible():
            row.setVisible(True)
            h = max(self._draft_vsplit.height(), 600)
            self._draft_vsplit.setSizes([int(h * 0.55), int(h * 0.45)])
        elif not has and row.isVisible():
            row.setVisible(False)
        if not has:
            return
        if what == "selection" and not any(g.data.get("section") for g in self.doc.selected()):
            return
        for sv in self.section_views:
            sv.schedule()

    def set_isolate(self, on):
        if on:
            ids = [g.id for g in self.doc.selected()]
            if not ids:
                self.message("Select the objects to isolate first.")
                self._sync_isolate()
                return
            self.doc.isolate(ids)
            self.view3d.zoom_extents()
            self.view2d.zoom_extents()
            self.message(f"Isolated {len(ids)} object(s). Click Isolate again (or Ctrl+Shift+H) to show everything.")
        else:
            if self.doc.unisolate():
                self.message("All objects shown.")
        self._sync_isolate()

    def toggle_points(self):
        """F10: control points on for the selection, or off if any are showing."""
        if self.doc.points_on:
            self.run_command("PointsOff")
        else:
            self.run_command("PointsOn")

    def toggle_isolate(self):
        self.set_isolate(self.doc.isolated is None)

    def _sync_isolate(self):
        b = getattr(self, "isolate_btn", None)
        if b is not None:
            b.blockSignals(True)
            b.setChecked(self.doc.isolated is not None)
            b.setText(f"Isolated ({len(self.doc.isolated)})" if self.doc.isolated is not None else "Isolate")
            b.blockSignals(False)

    def set_toggle(self, key, v):
        setattr(self.settings, key, v)
        self.message(f"{key.replace('show_', '').capitalize()} {'on' if v else 'off'}")
        if key == "show_grid":
            self.view3d.apply_appearance()
        if key == "gumball" and self.view3d.gumball is not None:
            self.view3d.gumball.refresh()
            self.view3d.plotter.render()
        self.view2d.update()

    def place_component(self, def_id, space):
        if space == "select":  # "Select placed copies" in the Components workspace
            self.view3d.zoom_extents(selected=True)
            self.message(f"{len(self.doc.selection)} copies selected.")
            return
        if self.persona == 0 and space != "2d":
            space = "2d"
        self.run_command("Insert", "2d" if self.persona == 0 else "3d", def_id=def_id)

    def drop_component(self, def_id, point, space):
        from . import components as C
        d = self.components.ensure_in_doc(def_id)
        if d is None:
            return
        self.doc.checkpoint("Place component")
        inst = C.instance(def_id, G.translation(point), layer=self.doc.current_layer, space=space)
        self.doc.add(inst, select=True)
        self.message(f"Placed '{d.name}'. Use Connect to snap it to another component.")

    def discard_edit(self):
        if self.doc.editing:
            self.doc.end_edit(apply=False)
            self.message("Component edit discarded.")

    # ================================================================ events
    def _cursor(self, p, snap):
        if p is None:
            return
        txt = f"x {p[0]:.3f}   y {p[1]:.3f}" + (f"   z {p[2]:.3f}" if self.persona == 1 else "")
        self.coord.setText(txt + (f"   [{snap}]" if snap else "") + f"   {self.doc.units}")

    def _doc_event(self, what):
        if what in ("changed", "layers", "loaded"):
            self._sync_isolate()
            self.layer_lbl.setText(f"Layer: {self.doc.current_layer}")
            self._update_title()
        if what in ("editing", "changed", "loaded"):
            d = self.doc.definitions.get(self.doc.editing[0]) if self.doc.editing else None
            for banner, lab in self.edit_banner.values():
                banner.setVisible(d is not None)
                if d is not None:
                    lab.setText(f"Editing component: {d.name}. All {len(self.doc.instances_of(d.id))} copies update when you close it.")

    def _node_preview(self, geoms):
        parts = []
        for g in geoms:
            parts += self.doc.expand(g) if g.kind == "instance" else [g]
        self.view3d.set_extra(parts)

    def _update_title(self):
        name = os.path.basename(self.doc.path) if self.doc.path else "Untitled"
        self.setWindowTitle(f"{name}{' *' if self.doc.modified else ''} — {APP_NAME}")

    # ================================================================ menus
    def _menus(self):
        mb = self.menuBar()
        m = mb.addMenu("File")
        self._act(m, "New", self.file_new, "Ctrl+N")
        self._act(m, "Open...", self.file_open, "Ctrl+O")
        self._act(m, "Save", self.file_save, "Ctrl+S")
        self._act(m, "Save As...", self.file_save_as, "Ctrl+Shift+S")
        m.addSeparator()
        self._act(m, "Import...", self.file_import, "Ctrl+I")
        self._act(m, "Import Grasshopper definition (.gh, .ghx)...", lambda: self.import_grasshopper())
        self._act(m, "Save nodes back to Grasshopper (.gh)...", lambda: self.nodes.export_grasshopper())
        self._act(m, "Export...", self.file_export, "Ctrl+E")
        self._act(m, "Export selected...", lambda: self.file_export(selected=True), "Ctrl+Shift+E")
        self._act(m, "Export 2D drawing (PDF, DWG, DXF, PNG, SVG)...", self.export_drawing, "Ctrl+Alt+E")
        m.addSeparator()
        self._act(m, "Supported formats...", self.show_formats)
        m.addSeparator()
        self._act(m, "Screenshot of 3D view...", self.screenshot)
        self._act(m, "Settings...", self.open_settings, "Ctrl+,")
        m.addSeparator()
        self._act(m, "Quit", self.close, "Ctrl+Q")

        m = mb.addMenu("Edit")
        self._act(m, "Undo", lambda: self.run_command("Undo"), "Ctrl+Z")
        self._act(m, "Redo", lambda: self.run_command("Redo"), "Ctrl+Shift+Z")
        self._act(m, "Redo ", lambda: self.run_command("Redo"), "Ctrl+Y")
        m.addSeparator()
        self._act(m, "Copy", self.copy, "Ctrl+C")
        self._act(m, "Paste", self.paste, "Ctrl+V")
        self._act(m, "Select all", lambda: self.run_command("SelAll"), "Ctrl+A")
        self._act(m, "Select similar", lambda: self.run_command("SelSimilar"))
        m.addSeparator()
        self._act(m, "Command palette...", self.palette, "Ctrl+K")
        self._act(m, "Repeat last command", self.ix.submit_empty)

        m = mb.addMenu("View")
        for label, act, sc in (("Zoom extents", "zoom_extents", "Ctrl+Shift+E"), ("Zoom selected", "zoom_selected", None),
                               ("Top", "view_top", None), ("Front", "view_front", None), ("Right", "view_right", None),
                               ("Perspective", "view_persp", None), ("Shaded", "shade_shaded", None),
                               ("Wireframe", "shade_wire", None), ("Ghosted", "shade_ghost", None)):
            self._act(m, label, lambda _=False, a=act: self.view_action(a), sc)

        m = mb.addMenu("Workspace")
        for i, name in enumerate(PERSONAS):
            self._act(m, name, lambda _=False, i=i: self.set_persona(i), f"Ctrl+{i + 1}")

        cm = mb.addMenu("Commands")
        groups = {}
        for c in K.COMMANDS.values():
            groups.setdefault(c.group, []).append(c)
        for gname in sorted(groups):
            sub = cm.addMenu(gname)
            for c in sorted(groups[gname], key=lambda c: c.name):
                a = sub.addAction(c.name)
                a.setToolTip(c.help)
                a.triggered.connect(lambda _=False, n=c.name: self.run_command(n))

        m = mb.addMenu("Help")
        self._act(m, "Commands and shortcuts", self.show_help, "F1")
        self._act(m, "About", lambda: QtWidgets.QMessageBox.about(self, APP_NAME, f"{APP_NAME} {__version__}\n"
                                                                    "2D drafting, 3D modeling, components and nodes.\n\n"
                                                                    "Created by catsyntax"))

    def _act(self, menu, text, fn, shortcut=None):
        a = menu.addAction(text)
        a.triggered.connect(fn)
        if shortcut:
            a.setShortcut(QtGui.QKeySequence(shortcut))
            a.setShortcutContext(QtCore.Qt.ApplicationShortcut)
        return a

    def shortcut_actions(self):
        """Everything that can have a keyboard shortcut: (id, label, group, default key, function)."""
        acts = [
            ("app:toggle_osnap", "Object snaps on/off", "Toggles", "F3", lambda: self.view_action("toggle_osnap")),
            ("app:toggle_grid", "Grid on/off", "Toggles", "F7", lambda: self.view_action("toggle_grid")),
            ("app:toggle_ortho", "Ortho on/off", "Toggles", "F8", lambda: self.view_action("toggle_ortho")),
            ("app:toggle_snap", "Grid snap on/off", "Toggles", "F9", lambda: self.view_action("toggle_snap")),
            ("app:points", "Control points on/off", "Editing", "F10", self.toggle_points),
            ("app:points_off", "Control points off", "Editing", "F11", lambda: self.run_command("PointsOff")),
            ("app:isolate", "Isolate selection on/off", "View", "Ctrl+Shift+H", self.toggle_isolate),
            ("app:palette", "Command palette", "General", "Ctrl+Shift+P", self.palette),
            ("app:export2d", "Export 2D drawing", "File", "", self.export_drawing),
            ("app:zoom_all", "Zoom all", "View", "", lambda: self.view_action("zoom_extents")),
            ("app:view_top", "3D: Top view", "View", "", lambda: self.view_action("view_top")),
            ("app:view_persp", "3D: Perspective view", "View", "", lambda: self.view_action("view_persp")),
            ("app:view_iso", "3D: Isometric view", "View", "", lambda: self.view_action("view_iso")),
        ]
        seen = set()
        for key in sorted(K.COMMANDS, key=lambda k: K.COMMANDS[k].name.lower()):
            cmd = K.COMMANDS[key]
            name = getattr(cmd, "name", key)
            if name in seen or name == "GumballValue":
                continue
            seen.add(name)
            acts.append(("cmd:" + name, name, getattr(cmd, "group", "Commands") or "Commands",
                         CMD_KEYS.get(name, ""), lambda n=name: self.run_command(n)))
        return acts

    def apply_shortcuts(self):
        for sc in getattr(self, "_shortcut_objs", []):
            sc.setEnabled(False)
            sc.setParent(None)
            sc.deleteLater()
        self._shortcut_objs = []
        custom = getattr(self.settings, "shortcuts", {}) or {}
        for aid, label, group, default, fn in self.shortcut_actions():
            key = custom.get(aid, default)
            if not key:
                continue
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            sc.setContext(QtCore.Qt.ApplicationShortcut)
            sc.activated.connect(fn)
            self._shortcut_objs.append(sc)

    def _shortcuts(self):
        self.apply_shortcuts()

    # ================================================================ actions
    def palette(self):
        space = "2d" if self.persona == 0 else "3d"
        dlg = CommandPalette(space, self)
        g = self.geometry()
        dlg.move(g.x() + (g.width() - dlg.width()) // 2, g.y() + 120)
        dlg.edit.setFocus()
        if dlg.exec() and dlg.choice:
            self.run_command(dlg.choice)

    def copy(self):
        if self.persona == 3:
            return
        space = "2d" if self.persona == 0 else "3d"
        self.clipboard = [g.copy(keep_id=False) for g in self.doc.selected(space)]
        self.message(f"Copied {len(self.clipboard)} objects.")

    def paste(self):
        if not self.clipboard or self.persona == 3:
            return
        space = "2d" if self.persona == 0 else "3d"
        self.doc.checkpoint("Paste")
        new = [g.copy(keep_id=False, space=space) for g in self.clipboard]
        self.doc.add(new, select=True)
        self.message(f"Pasted {len(new)} objects in place.")

    def open_settings(self):
        dlg = SettingsDialog(self.settings, self)
        dlg.applied.connect(self.apply_settings)
        dlg.exec()

    def apply_settings(self):
        self._apply_style()
        self.apply_shortcuts()
        self.doc.units = self.settings.units
        self.view3d.apply_appearance()
        self.components.preview.apply_appearance()
        self.view2d.update()
        self.nodes.apply_appearance()
        self.components.refresh_list()

    def confirm_discard(self):
        if not self.doc.modified:
            return True
        r = QtWidgets.QMessageBox.question(self, APP_NAME, "Save changes to the current document?",
                                           QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel)
        if r == QtWidgets.QMessageBox.Save:
            return self.file_save()
        return r == QtWidgets.QMessageBox.Discard

    def file_new(self):
        if not self.confirm_discard():
            return
        self.doc.clear()
        self.nodes.load_graph(None)
        self.doc.notify("layers")
        self.doc.notify("changed")
        self.doc.notify("definitions")
        self._update_title()

    def file_open(self, path=None):
        if not self.confirm_discard():
            return
        if not path:
            from .exchange import file_filter
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open", "", file_filter(True))
        if not path:
            return
        try:
            if path.lower().endswith(".forma"):
                self.doc.load(path)
            else:
                self.doc.clear()
                self.nodes.load_graph(None)
                self.doc.notify("changed")
                self._import(path)
                self.doc.undo_stack.clear()
                self.doc.modified = False
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open failed", str(e))
            return
        self.view3d.zoom_extents()
        self.view2d.zoom_extents()
        self._update_title()
        self.message(f"Opened {path}")

    def file_save(self):
        if not self.doc.path or not self.doc.path.endswith(".forma"):
            return self.file_save_as()
        self.doc.node_graph = self.nodes.to_dict()
        self.doc.save(self.doc.path)
        self._update_title()
        self.message(f"Saved {self.doc.path}")
        return True

    def file_save_as(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save As", "untitled.forma", "Forma project (*.forma)")
        if not path:
            return False
        if not path.endswith(".forma"):
            path += ".forma"
        self.doc.path = path
        return self.file_save()

    def import_grasshopper(self, path=None):
        """Bring a Grasshopper definition into the Nodes workspace."""
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import Grasshopper definition", "",
                                                            "Grasshopper (*.gh *.ghx)")
        if not path:
            return
        self.set_persona(PERSONAS.index("Nodes"))
        self.nodes.import_grasshopper(path, self.doc.units)

    def _import(self, path):
        from .exchange import import_file
        if path.lower().endswith((".gh", ".ghx")):
            return self.import_grasshopper(path)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            layers, geoms, warnings = import_file(path, self.doc.units)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if not geoms:
            QtWidgets.QMessageBox.information(self, "Import", "Nothing importable was found in that file." +
                                              ("\n" + "\n".join(warnings[:5]) if warnings else ""))
            return
        self.doc.checkpoint("Import " + os.path.basename(path))
        for l in layers:
            if not self.doc.layer(l.name):
                nl = self.doc.add_layer(l.name, l.color)   # also creates missing parent layers
                nl.visible, nl.locked, nl.weight = l.visible, l.locked, l.weight
        self.doc.add(geoms)
        self.doc.notify("layers")
        self.message(f"Imported {len(geoms)} objects from {os.path.basename(path)}." +
                     (f" {len(warnings)} warnings: " + "; ".join(warnings[:3]) if warnings else ""))

    def file_import(self):
        from .exchange import file_filter
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import", "", file_filter(True))
        if path:
            try:
                self._import(path)
                self.view3d.zoom_extents()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Import failed", str(e))

    def file_export(self, selected=False):
        from .exchange import file_filter, export_file
        if selected and not self.doc.selection:
            QtWidgets.QMessageBox.information(self, "Export selected", "Select some objects first.")
            return
        base = os.path.splitext(os.path.basename(self.doc.path))[0] if self.doc.path else "export"
        path, flt = QtWidgets.QFileDialog.getSaveFileName(self, "Export selected" if selected else "Export",
                                                          base + ".fbx", file_filter(False))
        if not path:
            return
        if not os.path.splitext(path)[1]:
            m = re.search(r"\*\.(\w+)", flt or "")
            path += "." + (m.group(1) if m and m.group(1) != "forma" else "fbx")
        objs = self.doc.selected() if selected else None
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            n, warnings = export_file(self.doc, path, objs)
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self.message(f"Exported {n} objects to {os.path.basename(path)}." + (" " + " ".join(warnings[:2]) if warnings else ""))

    def export_drawing(self):
        from .exportdialog import ExportDrawingDialog
        ExportDrawingDialog(self).exec()

    def show_formats(self):
        from .exchange import FORMATS
        rows = "".join(f"<tr><td><b>.{e}</b></td><td>{l}</td><td>{'yes' if i else ''}</td><td>{'yes' if x else ''}</td></tr>"
                       for e, l, i, x in FORMATS if e != "forma")
        QtWidgets.QMessageBox.information(self, "Supported formats",
            "<div style='font-family:Arial'><table cellpadding=3><tr><th>Ext</th><th>Format</th><th>Import</th><th>Export</th></tr>"
            f"{rows}</table><p>SketchUp: use .dae (Collada). 3ds Max: .fbx, .3ds or .obj. "
            "Exact solids to Rhino/SolidWorks/Fusion: .step. DWG uses the free ODA File Converter if installed.</p></div>")

    def _nurbs_objects(self, selected=False):
        src = self.doc.selected("3d") if selected else self.doc.in_space("3d")
        out = []
        for g in src:
            parts = self.doc.expand(g) if g.kind == "instance" else [g]
            out += [p for p in parts if p.kind == "brep"]
        return out

    def file_export_step(self, fmt="step"):
        from . import brep as B
        objs = self._nurbs_objects(selected=bool(self.doc.selected("3d")))
        if not objs:
            QtWidgets.QMessageBox.information(self, "Export", "No NURBS surfaces or solids to export.")
            return
        ext = "step" if fmt == "step" else "igs"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, f"Export {fmt.upper()}", f"export.{ext}",
                                                        "STEP (*.step *.stp)" if fmt == "step" else "IGES (*.igs *.iges)")
        if not path:
            return
        try:
            (B.export_step if fmt == "step" else B.export_iges)(objs, path)
            self.message(f"Exported {len(objs)} NURBS objects to {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))

    def file_import_step(self, path=None):
        from . import brep as B
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import STEP / IGES", "", "STEP or IGES (*.step *.stp *.igs *.iges)")
        if not path:
            return
        try:
            gs = B.import_step_iges(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Import failed", str(e))
            return
        self.doc.checkpoint("Import STEP/IGES")
        self.doc.add(gs)
        self.view3d.zoom_extents()
        self.message(f"Imported {len(gs)} exact NURBS objects from {os.path.basename(path)}.")

    def screenshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save screenshot", "view.png", "PNG (*.png)")
        if path:
            self.view3d.screenshot(path)

    def show_help(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Commands and shortcuts")
        dlg.resize(760, 640)
        lay = QtWidgets.QVBoxLayout(dlg)
        t = QtWidgets.QTextBrowser()
        rows = []
        for gname in sorted({c.group for c in K.COMMANDS.values()}):
            rows.append(f"<h3>{gname}</h3><table cellpadding=3>")
            for c in sorted([c for c in K.COMMANDS.values() if c.group == gname], key=lambda c: c.name):
                ws = "Draft + Model" if len(c.spaces) == 2 else ("Model" if c.spaces == ("3d",) else "Draft")
                rows.append(f"<tr><td><b>{c.name}</b></td><td>{', '.join(c.aliases)}</td><td>{c.help}</td><td>{ws}</td></tr>")
            rows.append("</table>")
        keys = """<h3>Shortcuts</h3><table cellpadding=3>
<tr><td><b>Enter / Space / Right-click</b></td><td>Finish input, or repeat the last command</td></tr>
<tr><td><b>Esc</b></td><td>Cancel command / clear selection</td></tr>
<tr><td><b>Ctrl+K</b></td><td>Command palette</td></tr>
<tr><td><b>Ctrl+1..4</b></td><td>Draft, Model, Components, Nodes workspaces</td></tr>
<tr><td><b>F3 / F7 / F8 / F9</b></td><td>Object snap / Grid / Ortho / Grid snap</td></tr>
<tr><td><b>Ctrl+Z, Ctrl+Shift+Z</b></td><td>Undo, Redo</td></tr>
<tr><td><b>Ctrl+C, Ctrl+V</b></td><td>Copy, paste in place</td></tr>
<tr><td><b>Right-drag / Shift+right-drag / wheel</b></td><td>Orbit / pan / zoom (3D). In 2D right-drag pans.</td></tr>
<tr><td><b>Typed input</b></td><td>x,y,z absolute · @dx,dy relative · @r&lt;angle polar · a single number = distance along the cursor direction</td></tr>
<tr><td><b>Nodes</b></td><td>Space: add node · F5: solve · Delete: remove · double-click input: set value</td></tr>
</table>"""
        t.setHtml(f"<div style='font-family:Arial'>{keys}{''.join(rows)}</div>")
        lay.addWidget(t)
        dlg.exec()

    def closeEvent(self, ev):
        print("[forma] main window closed", flush=True)
        if not self.confirm_discard():
            ev.ignore()
            return
        self.settings.save()
        self.nodes.shutdown()
        for v in (self.view3d, self.components.preview):
            try:
                v.plotter.close()
            except Exception:
                pass
        super().closeEvent(ev)


def main(argv=None, splash=None, file_filter=None):
    argv = argv if argv is not None else sys.argv
    _trace("starting Qt")
    app = create_qapp(argv)
    app.setWindowIcon(app_icon())
    if file_filter is None:
        file_filter = _FileOpenFilter()
        app.installEventFilter(file_filter)
    _trace(f"Qt {QtCore.qVersion()} ready - building main window")
    w = MainWindow()
    file_filter.window = w
    w.show()
    w.raise_()
    w.activateWindow()
    if splash is not None:
        splash.finish(w)
    print("[forma] window open - ready.", flush=True)
    try:
        import faulthandler
        faulthandler.cancel_dump_traceback_later()
    except Exception:
        pass
    files = [a for a in argv[1:] if os.path.exists(a) and not a.startswith("-psn")] + file_filter.pending
    if files:
        QtCore.QTimer.singleShot(200, lambda: w.file_open(files[0]))
    # let Python signal handlers run while Qt's loop is busy, and log a heartbeat
    beat = QtCore.QTimer(app)
    beat.setInterval(500)
    _n = [0]

    def _tick():
        _n[0] += 1
        if _n[0] in (20, 60, 240) or _n[0] % 1200 == 0:
            print(f"[forma] alive {_n[0] // 2}s, windows visible: {w.isVisible()}", flush=True)
    beat.timeout.connect(_tick)
    beat.start()
    app.aboutToQuit.connect(lambda: print("[forma] Qt aboutToQuit", flush=True))
    rc = app.exec()
    print(f"[forma] event loop ended (code {rc})", flush=True)
    return rc
