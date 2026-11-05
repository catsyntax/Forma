"""'Export 2D drawing' window: pick viewports (Draft view, section views), a format and a scale."""
from __future__ import annotations

import os

from PySide6 import QtCore, QtWidgets

from . import drawing2d as D

FORMATS = [("PDF (vector, one page per view)", "pdf"), ("PNG image", "png"), ("SVG (vector)", "svg"),
           ("DXF (AutoCAD, true scale)", "dxf"), ("DWG (AutoCAD, needs ODA File Converter)", "dwg")]


class ExportDrawingDialog(QtWidgets.QDialog):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.setWindowTitle("Export 2D drawing")
        self.setMinimumWidth(460)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel("<b>Viewports to export</b>"))
        self.views = []
        v2 = app.view2d
        cb = QtWidgets.QCheckBox(f"Draft view ({v2.view_name})")
        cb.setChecked(True)
        self.views.append((cb, "draft", None))
        lay.addWidget(cb)
        for sv in getattr(app, "section_views", []):
            if sv.data is None:
                sv.recompute()
            if sv.data is None:
                continue
            cb = QtWidgets.QCheckBox("Section A" if sv.side > 0 else "Section B")
            cb.setChecked(True)
            self.views.append((cb, "section", sv))
            lay.addWidget(cb)
        if len(self.views) == 1:
            lay.addWidget(QtWidgets.QLabel("<span style='color:gray'>Tip: add a SectionPlane to export sections too.</span>"))
        form = QtWidgets.QFormLayout()
        self.fmt = QtWidgets.QComboBox()
        for label, ext in FORMATS:
            self.fmt.addItem(label, ext)
        form.addRow("Format", self.fmt)
        self.scale = QtWidgets.QComboBox()
        self.scale.addItems(D.SCALES)
        self.scale.setCurrentText("Fit to page")
        form.addRow("Scale (PDF)", self.scale)
        self.paper = QtWidgets.QComboBox()
        self.paper.addItems(list(D.PAPERS))
        self.paper.setCurrentText("A3")
        form.addRow("Paper (PDF, landscape)", self.paper)
        self.px = QtWidgets.QComboBox()
        self.px.addItems(["2000", "4000", "8000"])
        self.px.setCurrentText("4000")
        form.addRow("Image width, px (PNG)", self.px)
        self.shaded = QtWidgets.QCheckBox("Shade surfaces seen beyond the section (PDF, PNG)")
        self.shaded.setChecked(True)
        form.addRow("", self.shaded)
        lay.addLayout(form)
        bb = QtWidgets.QDialogButtonBox()
        go = bb.addButton("Export...", QtWidgets.QDialogButtonBox.AcceptRole)
        go.setObjectName("Primary")
        bb.addButton(QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.do_export)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.fmt.currentIndexChanged.connect(self._fmt_changed)
        self._fmt_changed()

    def _fmt_changed(self):
        ext = self.fmt.currentData()
        self.scale.setEnabled(ext == "pdf")
        self.paper.setEnabled(ext == "pdf")
        self.px.setEnabled(ext == "png")
        self.shaded.setEnabled(ext in ("pdf", "png"))

    def sheets(self):
        out = []
        for cb, kind, sv in self.views:
            if not cb.isChecked():
                continue
            sh = D.sheet_from_draft(self.app.view2d) if kind == "draft" else D.sheet_from_section(sv, self.shaded.isChecked())
            if sh and sh["items"]:
                out.append(sh)
        return out

    def do_export(self):
        sheets = self.sheets()
        if not sheets:
            QtWidgets.QMessageBox.information(self, "Export", "Nothing to export: tick at least one viewport with content.")
            return
        ext = self.fmt.currentData()
        base = os.path.splitext(os.path.basename(self.app.doc.path or "drawing.forma"))[0]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export drawing", f"{base}.{ext}", f"*.{ext}")
        if not path:
            return
        if not path.lower().endswith("." + ext):
            path += "." + ext
        units = getattr(self.app.doc, "units", "mm")
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            written = write(sheets, path, ext, self.scale.currentText(), self.paper.currentText(),
                            int(self.px.currentText()), self.shaded.isChecked(), units)
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(self, "Export failed", str(e))
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self.app.message("Exported " + ", ".join(os.path.basename(p) for p in written))
        self.accept()


def write(sheets, path, ext, scale="Fit to page", paper="A3", px=4000, shaded=True, units="mm"):
    """Write sheets; returns the list of files written (PNG/SVG make one file per view)."""
    if ext == "pdf":
        D.export_pdf(sheets, path, paper, scale, shaded, units)
        return [path]
    if ext in ("dxf", "dwg"):
        D.export_dxf(sheets, path, units)
        return [path]
    out = []
    stem, e = os.path.splitext(path)
    for sh in sheets:
        p = path if len(sheets) == 1 else f"{stem}_{sh['title'].replace(' ', '_').replace('-', '')}{e}"
        if ext == "png":
            D.export_image(sh, p, px, shaded)
        else:
            D.export_svg(sh, p, False)
        out.append(p)
    return out
