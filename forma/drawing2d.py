"""2D drawing export: take what a Draft or section viewport shows and write PDF, PNG, SVG, DXF or DWG.

A "sheet" is a list of drawing items in model units (mm) as seen in that view:
    {"type": "lines", "segs": (N,2,2)}            loose segments (edges, linework beyond a section)
    {"type": "poly",  "pts": (N,2), "closed": b}  polylines (curves, cut lines)
    {"type": "fill",  "loops": [(N,2)...]}        filled regions (cut solids, odd-even)
    {"type": "tris",  "tris": (K,3,2), "shade": (K,)}  shaded faces beyond a section (raster/PDF only)
each with "layer", "color" (#rrggbb) and "width" (mm on paper).
"""
from __future__ import annotations

import datetime
import os

import numpy as np
from PySide6 import QtCore, QtGui

PAPERS = {"A4": (297, 210), "A3": (420, 297), "A2": (594, 420), "A1": (841, 594), "A0": (1189, 841)}
SCALES = ["Fit to page", "1:20", "1:50", "1:100", "1:200", "1:500", "1:1000", "1:2000", "1:5000"]


# ---------------------------------------------------------------- collecting linework
def sheet_from_draft(view2d):
    """Everything the Draft view shows, in its current orientation (Top, Front, ...)."""
    from . import geometry as G
    doc = view2d.doc
    items = []
    for owner, g, col, dim in doc.display("2d"):
        if g.data.get("section"):
            continue
        try:
            vg = view2d._vg(g)
            layer = doc.objects[owner].layer.replace("::", "$") if owner in doc.objects else "0"
            color = col if col.lower() not in ("#d0d0d0", "#ffffff", "#e6e6e6") else "#000000"
            wmm = doc.weight_of(owner)
            if vg.kind == "point":
                continue
            if vg.kind in ("mesh", "brep"):
                segs = G.feature_edges(vg)
                if segs:
                    S = np.array([[a[:2], b[:2]] for a, b in segs], float)
                    items.append({"type": "lines", "segs": S, "layer": layer, "color": color, "width": wmm or 0.25})
            else:
                s = np.asarray(vg.sample(), float)[:, :2]
                if len(s) > 1:
                    items.append({"type": "poly", "pts": s, "closed": bool(getattr(vg, "closed", False)),
                                  "layer": layer, "color": color, "width": wmm or 0.35})
        except Exception:
            continue
    return {"title": f"Draft - {view2d.view_name} view", "items": items}


def sheet_from_section(sv, shaded=True):
    d = sv.data
    if not d:
        return None
    items = []
    if shaded and len(d["faces"]):
        items.append({"type": "tris", "tris": d["faces"], "shade": d["shade"], "layer": "Elevation", "color": "#b8bcc2",
                      "width": 0})
    if len(d["beyond"]):
        items.append({"type": "lines", "segs": d["beyond"], "layer": "Beyond", "color": "#555a60", "width": 0.13})
    if d["fill"]:
        items.append({"type": "fill", "loops": d["fill"], "layer": "Cut fill", "color": "#202226", "width": 0})
    for pl in d["cut"]:
        items.append({"type": "poly", "pts": np.asarray(pl), "closed": False, "layer": "Cut", "color": "#c8202c",
                      "width": 0.5})
    n = d["normal"]
    side = "A" if sv.side > 0 else "B"
    return {"title": f"Section {side}", "items": items}


def bounds(sheet):
    pts = []
    for it in sheet["items"]:
        if it["type"] == "lines" and len(it["segs"]):
            pts.append(it["segs"].reshape(-1, 2))
        elif it["type"] == "poly":
            pts.append(it["pts"])
        elif it["type"] == "fill":
            pts += list(it["loops"])
        elif it["type"] == "tris" and len(it["tris"]):
            pts.append(it["tris"].reshape(-1, 2)[::5])
    if not pts:
        return np.zeros(2), np.ones(2)
    a = np.vstack(pts)
    return a.min(0), a.max(0)


# ---------------------------------------------------------------- painting (PDF / PNG / SVG)
def _paint(qp, sheet, rect, mm_to_dev, model_to_paper, shaded=True, title=None):
    """Draw a sheet into rect (device units). model_to_paper: paper mm per model unit."""
    lo, hi = bounds(sheet)
    cx, cy = (lo + hi) / 2
    k = model_to_paper * mm_to_dev                          # device units per model unit
    t = QtGui.QTransform()
    t.translate(rect.center().x(), rect.center().y())
    t.scale(k, -k)
    t.translate(-cx, -cy)
    qp.save()
    qp.setClipRect(rect)
    qp.setTransform(t, True)
    for it in sheet["items"]:
        col = QtGui.QColor(it["color"])
        w = max(it["width"] * mm_to_dev / k, 0) if it["width"] else 0
        if it["type"] == "tris" and shaded:
            qp.setPen(QtCore.Qt.NoPen)
            base = np.array([232, 233, 230])
            for tri, sh in zip(it["tris"], it["shade"]):
                c = (base * (0.82 + 0.18 * sh)).clip(0, 255).astype(int)
                qp.setBrush(QtGui.QColor(*c))
                qp.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(*p) for p in tri]))
        elif it["type"] == "fill":
            path = QtGui.QPainterPath()
            path.setFillRule(QtCore.Qt.OddEvenFill)
            for loop in it["loops"]:
                path.moveTo(*loop[0])
                for p in loop[1:]:
                    path.lineTo(*p)
                path.closeSubpath()
            qp.setPen(QtCore.Qt.NoPen)
            qp.setBrush(col)
            qp.drawPath(path)
        elif it["type"] in ("lines", "poly"):
            pen = QtGui.QPen(col, w)
            pen.setCapStyle(QtCore.Qt.RoundCap)
            pen.setJoinStyle(QtCore.Qt.RoundJoin)
            qp.setPen(pen)
            qp.setBrush(QtCore.Qt.NoBrush)
            path = QtGui.QPainterPath()
            if it["type"] == "lines":
                for a, b in it["segs"]:
                    path.moveTo(*a)
                    path.lineTo(*b)
            else:
                P = it["pts"]
                path.moveTo(*P[0])
                for p in P[1:]:
                    path.lineTo(*p)
                if it.get("closed"):
                    path.closeSubpath()
            qp.drawPath(path)
    qp.restore()


def _scale_value(scale_text, sheet, area_mm):
    lo, hi = bounds(sheet)
    size = np.maximum(hi - lo, 1e-9)
    if scale_text == "Fit to page" or not scale_text:
        f = min(area_mm[0] / size[0], area_mm[1] / size[1]) * 0.95   # paper mm per model unit
        return f, f"fit (about 1:{max(1, round(1 / f)):g})"
    n = float(scale_text.split(":")[1])
    return 1.0 / n, scale_text


def _title_block(qp, page_rect, mm, title, scale_label, units):
    qp.save()
    qp.setPen(QtGui.QPen(QtGui.QColor("#000000"), 0.3 * mm))
    qp.setBrush(QtCore.Qt.NoBrush)
    qp.drawRect(page_rect)
    band = QtCore.QRectF(page_rect.left(), page_rect.bottom() - 9 * mm, page_rect.width(), 9 * mm)
    qp.drawLine(band.topLeft(), band.topRight())
    f = QtGui.QFont("Arial")
    f.setPixelSize(max(1, int(3.2 * mm)))
    f.setBold(True)
    qp.setFont(f)
    qp.drawText(band.adjusted(3 * mm, 0, 0, 0), QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, title)
    f.setBold(False)
    qp.setFont(f)
    txt = f"Scale {scale_label}   ·   units {units}   ·   {datetime.date.today():%d %b %Y}   ·   Forma"
    qp.drawText(band.adjusted(0, 0, -3 * mm, 0), QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight, txt)
    qp.restore()
    return QtCore.QRectF(page_rect.left() + 4 * mm, page_rect.top() + 4 * mm,
                         page_rect.width() - 8 * mm, page_rect.height() - 17 * mm)


def export_pdf(sheets, path, paper="A3", scale="Fit to page", shaded=True, units="mm"):
    w_mm, h_mm = PAPERS.get(paper, PAPERS["A3"])
    pdf = QtGui.QPdfWriter(path)
    pdf.setPageSize(QtGui.QPageSize(QtCore.QSizeF(w_mm, h_mm), QtGui.QPageSize.Millimeter))
    pdf.setPageMargins(QtCore.QMarginsF(0, 0, 0, 0))
    pdf.setResolution(600)
    pdf.setTitle("Forma drawing")
    pdf.setCreator("Forma")
    mm = 600 / 25.4
    qp = QtGui.QPainter(pdf)
    qp.setRenderHint(QtGui.QPainter.Antialiasing)
    labels = []
    for i, sh in enumerate(sheets):
        if i:
            pdf.newPage()
        page = QtCore.QRectF(8 * mm, 8 * mm, (w_mm - 16) * mm, (h_mm - 16) * mm)
        f, label = _scale_value(scale, sh, ((w_mm - 24), (h_mm - 33)))
        area = _title_block(qp, page, mm, sh["title"], label, units)
        _paint(qp, sh, area, mm, f, shaded)
        labels.append(label)
    qp.end()
    return labels


def export_image(sheet, path, width_px=4000, shaded=True, background="#ffffff"):
    lo, hi = bounds(sheet)
    size = np.maximum(hi - lo, 1e-9)
    aspect = size[1] / size[0]
    W = int(width_px)
    H = max(200, int(W * min(max(aspect, 0.2), 3.0) + 0.12 * W))
    img = QtGui.QImage(W, H, QtGui.QImage.Format_ARGB32)
    img.fill(QtGui.QColor(background))
    qp = QtGui.QPainter(img)
    qp.setRenderHint(QtGui.QPainter.Antialiasing)
    margin = 0.04 * W
    rect = QtCore.QRectF(margin, margin, W - 2 * margin, H - 2 * margin)
    k = min(rect.width() / size[0], rect.height() / size[1])
    px_per_mm = W / 420.0                                   # line widths as if printed on A3
    _paint(qp, sheet, rect, px_per_mm, k / px_per_mm, shaded)
    qp.end()
    img.save(path)


def export_svg(sheet, path, shaded=False):
    from PySide6.QtSvg import QSvgGenerator
    lo, hi = bounds(sheet)
    size = np.maximum(hi - lo, 1e-9)
    W = 1600.0
    H = W * size[1] / size[0] + 80
    gen = QSvgGenerator()
    gen.setFileName(path)
    gen.setSize(QtCore.QSize(int(W), int(H)))
    gen.setViewBox(QtCore.QRectF(0, 0, W, H))
    gen.setTitle(sheet["title"])
    gen.setDescription("Exported from Forma")
    qp = QtGui.QPainter(gen)
    rect = QtCore.QRectF(40, 40, W - 80, H - 80)
    k = min(rect.width() / size[0], rect.height() / size[1])
    _paint(qp, sheet, rect, W / 420.0, k / (W / 420.0), shaded)
    qp.end()


_DXF_WEIGHTS = [5, 9, 13, 15, 18, 20, 25, 30, 35, 40, 50, 53, 60, 70, 80, 90, 100, 106, 120, 140, 158, 200, 211]


def _dxf_weight(mm):
    """Nearest standard DXF lineweight (hundredths of a mm)."""
    if not mm:
        return 25
    v = mm * 100
    return min(_DXF_WEIGHTS, key=lambda w: abs(w - v))


def export_dxf(sheets, path, units="mm"):
    """True-scale 2D linework in model space (one drawing per sheet, side by side, each on its own layers)."""
    import ezdxf
    from ezdxf import colors
    doc = ezdxf.new("R2018", setup=True)
    doc.units = {"mm": 4, "cm": 5, "m": 6, "in": 1, "ft": 2}.get(units, 4)
    msp = doc.modelspace()
    x_off = 0.0
    for sh in sheets:
        lo, hi = bounds(sh)
        dx = x_off - lo[0]
        prefix = sh["title"].replace(" ", "_").replace("-", "")[:20]
        for it in sh["items"]:
            lay = f"{prefix}-{it['layer']}"[:250]
            if lay not in doc.layers:
                rgb = QtGui.QColor(it["color"])
                doc.layers.add(lay, true_color=colors.rgb2int((rgb.red(), rgb.green(), rgb.blue())),
                               lineweight=int(round(max(it["width"], 0.13) * 100)))
            attrs = {"layer": lay, "lineweight": _dxf_weight(it["width"])}
            rgb = QtGui.QColor(it["color"])
            attrs["true_color"] = colors.rgb2int((rgb.red(), rgb.green(), rgb.blue()))
            if it["type"] == "lines":
                for a, b in it["segs"]:
                    msp.add_line((a[0] + dx, a[1]), (b[0] + dx, b[1]), dxfattribs=attrs)
            elif it["type"] == "poly":
                msp.add_lwpolyline([(p[0] + dx, p[1]) for p in it["pts"]], close=bool(it.get("closed")), dxfattribs=attrs)
            elif it["type"] == "fill":
                hatch = msp.add_hatch(dxfattribs=attrs)
                hatch.set_solid_fill()
                for loop in it["loops"]:
                    hatch.paths.add_polyline_path([(p[0] + dx, p[1]) for p in loop], is_closed=True)
        msp.add_text(sh["title"], height=max((hi - lo).max() * 0.02, 10),
                     dxfattribs={"layer": f"{prefix}-Title"}).set_placement((lo[0] + dx, lo[1] - (hi - lo).max() * 0.05))
        x_off += (hi[0] - lo[0]) * 1.15 + 1.0
    if path.lower().endswith(".dwg"):
        from ezdxf.addons import odafc
        if not odafc.is_installed():
            raise RuntimeError("DWG needs the free ODA File Converter (opendesign.com/guestfiles/oda_file_converter). "
                               "Install it, or export DXF, which AutoCAD, Rhino and others open directly.")
        odafc.export_dwg(doc, path, version="R2018", replace=True)
    else:
        doc.saveas(path)
