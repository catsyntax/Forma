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


def test_section_plane_cuts_box():
    from forma import section as S
    d = Document()
    d.add(B.box((0, 0, 0), 1000, 1000, 1000))
    sec = S.make_plane((-500, 500, 0), (1500, 500, 0), -100, 1100)
    d.add(sec)
    for side in (1, -1):
        r = S.compute(d, sec, side)
        assert r["cut"], "the plane through the box must produce cut lines"
        loop = max(r["fill"], key=len)
        w = np.ptp(loop[:, 0])
        h = np.ptp(loop[:, 1])
        assert w == pytest.approx(1000, rel=1e-3) and h == pytest.approx(1000, rel=1e-3)


def test_parametric_buildings():
    from forma import parametric as PA
    for fn in (PA.stadium, PA.diagrid_tower, PA.pavilion):
        parts = fn()
        assert len(parts) >= 3
        assert any(g.kind == "brep" for _, g in parts)


def test_modular_plan_generator():
    from forma.nodes.registry import NODE_TYPES
    t = NODE_TYPES["Modular Plan Generator"]
    kw = {p.name: p.default for p in t.inputs}
    kw["Links"] = "L1-B1, L1-C1, C1-B2"
    r = t.fn(**kw)
    assert len(r["Rooms"]) == 5 and all(len(p) == 7 for p in r["Rooms"])
    assert "unmet links" in r["Report"]


def test_space_syntax_voxel_vga():
    from forma import geometry as G
    from forma.nodes.registry import NODE_TYPES
    t = NODE_TYPES["Space Syntax 3D (voxel VGA)"]
    kw = {p.name: p.default for p in t.inputs}
    kw.update(Space=[G.box((0, 0, 0), 4000, 3000, 2700)], Furniture=[G.box((800, 800, 0), 1500, 800, 800)])
    kw["Voxel Size"] = 300.0
    kw["Eye Band"] = 5
    r = t.fn(**kw)
    assert r["Count"] > 100 and len(r["Walk Values"]) == len(r["Vis Values"]) > 10
    assert r["Walk Heatmap"].kind == "mesh"


def test_grasshopper_import(tmp_path):
    import base64
    from forma import ghimport
    code = base64.b64encode(b"a = x * 2").decode()
    ghx = f"""<?xml version="1.0" encoding="utf-8"?>
<Archive name="Root"><items count="0"/><chunks count="1">
 <chunk name="Definition"><items count="0"/><chunks count="1">
  <chunk name="DefinitionObjects"><items count="1"><item name="ObjectCount" type_name="gh_int32">2</item></items>
   <chunks count="2">
    <chunk name="Object" index="0"><items count="2">
      <item name="GUID" type_name="gh_guid">57da07bd-ecab-415d-9d86-af36d7073abc</item>
      <item name="Name" type_name="gh_string">Number Slider</item></items>
     <chunks count="1"><chunk name="Container"><items count="2">
       <item name="InstanceGuid" type_name="gh_guid">11111111-1111-1111-1111-111111111111</item>
       <item name="NickName" type_name="gh_string">count</item></items>
      <chunks count="1"><chunk name="Slider"><items count="4">
       <item name="Value" type_name="gh_double">4</item><item name="Min" type_name="gh_double">0</item>
       <item name="Max" type_name="gh_double">10</item><item name="Digits" type_name="gh_int32">0</item>
      </items></chunk></chunks></chunk></chunks></chunk>
    <chunk name="Object" index="1"><items count="2">
      <item name="GUID" type_name="gh_guid">c9b2d725-6f87-4b07-af90-bd9aefef68eb</item>
      <item name="Name" type_name="gh_string">Script</item></items>
     <chunks count="1"><chunk name="Container"><items count="1">
       <item name="NickName" type_name="gh_string">double it</item></items>
      <chunks count="2">
       <chunk name="ParameterData"><items count="0"/><chunks count="2">
        <chunk name="InputParam" index="0"><items count="3">
          <item name="NickName" type_name="gh_string">x</item>
          <item name="InstanceGuid" type_name="gh_guid">22222222-2222-2222-2222-222222222222</item>
          <item name="Source" index="0" type_name="gh_guid">11111111-1111-1111-1111-111111111111</item></items></chunk>
        <chunk name="OutputParam" index="0"><items count="2">
          <item name="NickName" type_name="gh_string">a</item>
          <item name="InstanceGuid" type_name="gh_guid">33333333-3333-3333-3333-333333333333</item></items></chunk>
       </chunks></chunk>
       <chunk name="Script"><items count="1"><item name="Text" type_name="gh_string">{code}</item></items></chunk>
      </chunks></chunk></chunks></chunk>
   </chunks></chunk></chunks></chunk></chunks></Archive>"""
    p = tmp_path / "t.ghx"
    p.write_text(ghx)
    g, report = ghimport.convert(str(p))
    types = sorted(n["type"] for n in g["nodes"].values())
    assert types == ["Number Slider", "Python"]
    py = next(n for n in g["nodes"].values() if n["type"] == "Python")
    assert py["props"]["code"] == "a = x * 2" and py["props"]["inputs"] == ["x"] and py["props"]["outputs"] == ["a"]
    assert len(g["wires"]) == 1 and g["wires"][0][3] == "x"
