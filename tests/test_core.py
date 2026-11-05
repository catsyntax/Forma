"""Core checks that run without a screen (geometry kernel, editing, trimming, nodes, file exchange)."""
import os
import tempfile

import numpy as np
import pytest

from forma import geometry as G
from forma import brep as B
from forma import editpoints as E
from forma.document import Document
from forma.nodes.registry import NODE_TYPES


def test_box_volume_and_boolean():
    a = B.box((0, 0, 0), 1000, 1000, 1000)
    b = B.box((250, 250, -10), 500, 500, 1100)
    d = B.boolean([a], [b], "difference")
    assert d.volume() == pytest.approx(1e9 - 500 * 500 * 1000, rel=1e-6)


def test_exact_circle_is_nurbs():
    c = B.exact_nurbs(G.circle((0, 0, 0), 300))
    assert c.kind == "nurbs" and c.deg == 2
    assert c.length() == pytest.approx(2 * np.pi * 300, rel=1e-4)


def test_trim_line_and_surface_hole():
    r = B.trim(G.line((0, 0, 0), (1000, 0, 0)), [G.line((400, -100, 0), (400, 100, 0))], (100, 0, 0))
    assert r[0].length() == pytest.approx(600)
    srf = B.planar_surface(G.rectangle((0, 0, 0), (1000, 1000, 0)))
    r = B.trim(srf, [G.circle((500, 500, 0), 200)], (500, 500, 0))
    assert r[0].area() == pytest.approx(1e6 - np.pi * 200 ** 2, rel=1e-5)


def test_control_points_on_box_and_face_move():
    box = B.box((0, 0, 0), 1000, 1000, 1000)
    pts = E.grips(box)
    assert len(pts) == 8
    top = [i for i, p in enumerate(pts) if p[2] > 999]
    moved = E.apply(box, top, G.translation((0, 0, 500)))
    assert moved.volume() == pytest.approx(1.5e9, rel=1e-6)


def test_non_uniform_scale_and_mirror():
    box = B.box((0, 0, 0), 1000, 1000, 1000)
    M = np.diag([-2.0, 1, 1, 1])
    assert abs(box.transformed(M).volume()) == pytest.approx(2e9, rel=1e-6)
    with pytest.raises(ValueError):
        box.transformed(np.diag([0.0, 1, 1, 1]))


def test_document_subobject_delete_and_undo():
    d = Document()
    box = d.add(B.box((0, 0, 0), 1000, 1000, 1000))[0]
    d.select([box.id])
    d.face_sel = (box.id, 0)
    assert "deleted" in d.delete_subobjects().lower()
    assert len(B.faces(B.shape_of(list(d.objects.values())[0]))) == 5
    d.undo()
    assert len(B.faces(B.shape_of(list(d.objects.values())[0]))) == 6


def test_analysis_nodes():
    b = B.box((0, 0, 0), 10000, 8000, 6000)
    r = NODE_TYPES["Sun Exposure"].fn(Surfaces=[b], Latitude=-34.93, Longitude=138.6,
                                      **{"Time Zone": 9.5, "Month": 6, "Day": 21, "From Hour": 8,
                                         "To Hour": 16, "Step": 2, "Grid Size": 4000})
    assert max(r["Sun Hours"]) > 0
    r = NODE_TYPES["U-Value"].fn(Thicknesses=[110, 90], Conductivities=[0.77, 0.035])
    assert 0.3 < r["U"] < 0.4
    bd = G.polyline([(0, 0, 0), (10000, 0, 0), (10000, 6000, 0), (0, 6000, 0)], closed=True)
    r = NODE_TYPES["Visibility Graph"].fn(Boundary=bd, Obstacles=[G.line((5000, 0, 0), (5000, 4000, 0))],
                                          **{"Grid Size": 600})
    assert max(r["Integration"]) > min(r["Integration"])


@pytest.mark.parametrize("ext", [".forma", ".3dm", ".step", ".obj", ".fbx", ".stl", ".dxf", ".svg"])
def test_file_round_trip(ext):
    from forma import exchange as X
    d = Document()
    d.add(B.box((0, 0, 0), 1000, 600, 400))
    d.add(G.circle((0, 0, 0), 300))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t" + ext)
        if ext == ".forma":
            d.save(path)
            d2 = Document()
            d2.load(path)
            assert len(d2.objects) == 2
            return
        X.export_file(d, path, list(d.objects.values()))
        assert os.path.getsize(path) > 0
        objs = X.import_file(path)
        assert objs
