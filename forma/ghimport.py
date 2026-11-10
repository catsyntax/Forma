"""Turn a Grasshopper definition (.gh / .ghx) into a Forma node graph.

What comes across:
  * Number Slider, Boolean Toggle, Button, Panel, Value List -> Forma parameter nodes with their values
  * Python script components (Rhino 8 'Script' / 'Python 3 Script' and GhPython) -> Forma Python nodes
    with the same code, inputs and outputs. Scripts recognised as one of Forma's ported tools
    (Space Syntax voxel VGA, Modular Plan Generator) become the native node instead, because those
    run here without Rhino.
  * Geometry parameters (Brep, Curve, Mesh, ...) -> Geometry nodes. Grasshopper keeps referenced
    geometry in the Rhino file, not the .gh, so pick the objects again in Forma.
  * Common components with a Forma equivalent (Solid Union, Move, Box, ...) -> that node.
  * Anything else -> a labelled placeholder, listed in the import report.
Wires are rebuilt between everything that came across.
"""
from __future__ import annotations

import base64
import os
import re
import uuid

from . import ghfile

SLIDER = "57da07bd-ecab-415d-9d86-af36d7073abc"
TOGGLE = "2e78987b-9dfb-42a2-8b76-3923ac8bd91a"
BUTTON = "a8b97322-2d53-47cd-905e-b932c3ccd74e"
PANEL = "59e0b89a-e487-49f8-bab8-b5bab16be14c"
VALUE_LIST = "00027467-0d24-4fa7-b178-8dc0ac5f42ec"
SCALE = 1.6   # Grasshopper canvas units -> Forma node editor units

GEOMETRY_PARAMS = {"Brep", "Curve", "Mesh", "Geometry", "Surface", "Point", "Line", "Circle", "Rectangle",
                   "Box", "Plane", "Vector", "Polyline", "Arc", "Extrusion", "SubD", "Group"}
DATA_PARAMS = {"Number", "Integer", "Boolean", "Text", "Data", "Generic Data", "Domain", "Colour", "Color"}

# Grasshopper component name -> (Forma node, {GH input nickname/name: Forma port}, {GH output: Forma output})
KNOWN = {
    "Solid Union": ("Union", {"B": "A", "Breps": "A"}, {"R": "Result", "Result": "Result"}),
    "Solid Difference": ("Difference", {"A": "A", "B": "B"}, {"R": "Result", "Result": "Result"}),
    "Solid Intersection": ("Intersection", {"A": "A", "B": "B"}, {"R": "Result", "Result": "Result"}),
    "Move": ("Move", {"G": "Geometry", "T": "Vector"}, {"G": "Geometry"}),
    "Rotate": ("Rotate", {"G": "Geometry", "A": "Angle"}, {"G": "Geometry"}),
    "Scale": ("Scale", {"G": "Geometry", "C": "Center", "F": "Factor"}, {"G": "Geometry"}),
    "Mirror": ("Mirror", {"G": "Geometry"}, {"G": "Geometry"}),
    "Construct Point": ("Point", {"X": "X", "Y": "Y", "Z": "Z"}, {"Pt": "Point"}),
    "Deconstruct": ("Deconstruct", {"P": "Point"}, {"X": "X", "Y": "Y", "Z": "Z"}),
    "Unit X": ("Vector", {"F": "X"}, {"V": "Vector"}),
    "Unit Y": ("Vector", {"F": "Y"}, {"V": "Vector"}),
    "Unit Z": ("Vector", {"F": "Z"}, {"V": "Vector"}),
    "Vector XYZ": ("Vector", {"X": "X", "Y": "Y", "Z": "Z"}, {"V": "Vector"}),
    "Line": ("Line", {"A": "Start", "B": "End"}, {"L": "Line"}),
    "Circle": ("Circle", {"P": "Center", "R": "Radius"}, {"C": "Circle"}),
    "Polyline": ("Polyline", {"V": "Points", "C": "Closed"}, {"Pl": "Polyline"}),
    "Interpolate": ("Interpolate", {"V": "Points", "D": "Degree"}, {"C": "Curve"}),
    "Divide Curve": ("Divide Curve", {"C": "Curve", "N": "Count"}, {"P": "Points"}),
    "Length": ("Curve Length", {"C": "Curve"}, {"L": "Length"}),
    "Offset Curve": ("Offset", {"C": "Curve", "D": "Distance"}, {"C": "Curve"}),
    "Center Box": ("Center Box", {"B": "Center", "X": "X", "Y": "Y", "Z": "Z"}, {"B": "Box"}),
    "Sphere": ("Sphere", {"B": "Center", "R": "Radius"}, {"S": "Sphere"}),
    "Extrude": ("Extrude", {"B": "Curve", "D": "Height"}, {"E": "Extrusion"}),
    "Loft": ("Loft", {"C": "Curves"}, {"L": "Surface"}),
    "Pipe": ("Pipe", {"C": "Curve", "R": "Radius"}, {"P": "Pipe"}),
    "Boundary Surfaces": ("Planar Surface", {"E": "Curve"}, {"S": "Surface"}),
    "Mesh Brep": ("Mesh", {"B": "Brep"}, {"M": "Mesh"}),
    "Area": ("Area", {"G": "Geometry"}, {"A": "Area", "C": "Centroid"}),
    "Volume": ("Volume", {"G": "Geometry"}, {"V": "Volume"}),
    "Bounding Box": ("Bounding Box", {"C": "Geometry"}, {"B": "Box"}),
    "Deconstruct Brep": ("Brep Faces", {"B": "Brep"}, {"F": "Faces"}),
    "Addition": ("Add", {"A": "A", "B": "B"}, {"R": "Result"}),
    "Subtraction": ("Subtract", {"A": "A", "B": "B"}, {"R": "Result"}),
    "Multiplication": ("Multiply", {"A": "A", "B": "B"}, {"R": "Result"}),
    "Division": ("Divide", {"A": "A", "B": "B"}, {"R": "Result"}),
    "Range": ("Range", {"D": "End", "N": "Steps"}, {"R": "Numbers"}),
    "Series": ("Series", {"S": "Start", "N": "Step", "C": "Count"}, {"S": "Numbers"}),
    "Random": ("Random", {"N": "Count", "S": "Seed"}, {"R": "Numbers"}),
    "List Item": ("List Item", {"L": "List", "i": "Index"}, {"i": "Item", "E": "Item"}),
    "List Length": ("List Length", {"L": "List"}, {"L": "Length"}),
    "Reverse List": ("Reverse", {"L": "List"}, {"L": "List"}),
    "Flatten Tree": ("Flatten", {"T": "List"}, {"T": "List"}),
    "Merge": ("Merge", {"D1": "A", "D2": "B"}, {"R": "List"}),
    "Sort List": ("Sort", {"K": "Numbers"}, {"K": "Sorted"}),
    "Cull Nth": ("Cull Nth", {"L": "List", "N": "N"}, {"L": "List"}),
}

# Scripts Forma has a native, Rhino-free version of. Port maps use the script's variable names.
NATIVE_SCRIPTS = [
    {"match": ("Space Syntax voxel VGA", "_compute_core"), "node": "Space Syntax 3D (voxel VGA)",
     "inputs": {"space": "Space", "furniture": "Furniture", "voxel_size": "Voxel Size", "eye_band": "Eye Band",
                "vis_radius": "Vis Radius", "metric": "Metric", "compute_choice": "Choice", "make_lines": "Lines",
                "vis_3d": "Vis 3D", "run": "Run"},
     "lengths": {"Voxel Size", "Vis Radius"}, "defaults": {"Metric": 0},
     "helpers": [("Number Slider", "metric (0-14)", "Metric", -420, 520, {"value": 0, "min": 0, "max": 14, "decimals": 0}),
                 ("Panel", "Progress", "Progress", 420, 0, {}), ("Panel", "Metric names", "Metric Names", 420, 200, {})],
     "outputs": {"progress": "Progress", "out": "Progress", "points": "Points", "count": "Count",
                 "walk_lines": "Walk Lines", "walk_heatmap": "Walk Heatmap", "walk_values": "Walk Values",
                 "walk_points": "Walk Points", "vis_lines": "Vis Lines", "vis_heatmap": "Vis Heatmap",
                 "vis_values": "Vis Values", "vis_points": "Vis Points", "walk_all": "Walk All",
                 "vis_all": "Vis All", "metric_names": "Metric Names"},
     "note": "runs natively in Forma (same NumPy engine). 'bake' and 'reset' inputs are not needed: use Bake "
             "selected in the toolbar; results re-compute when inputs change."},
    {"match": ("MODULAR PLAN GENERATOR", "def generate_plans"), "node": "Modular Plan Generator",
     "inputs": {}, "lengths": set(),
     "outputs": {"Plans": "Outlines", "out": "Outlines", "Report": "Report"},
     "helpers": [("Text", "Brief", "Brief", -420, -40, {"value": "@brief", "multiline": True}),
                 ("Text", "Links (optional)", "Links", -420, 230, {"value": "", "multiline": True}),
                 ("Number Slider", "Plans", "Plans", -420, 400, {"value": 5, "min": 1, "max": 12, "decimals": 0}),
                 ("Panel", "Report", "Report", 380, 0, {}), ("Panel", "Code check", "Code Check", 380, 180, {})],
     "note": "runs natively in Forma. The brief that the original typed into its web window is the Brief "
             "text input here (one space per line: code type modules [attached-to]); Links sets adjacencies."},
]

RHINO_ONLY = re.compile(r"^\s*(import|from)\s+(Rhino|rhinoscriptsyntax|scriptcontext|Grasshopper|ghpythonlib|"
                        r"System|Eto|clr)\b", re.M)


def _container(obj):
    return obj.chunk("Container") or obj


def _pos(c):
    att = c.chunk("Attributes")
    if att is None:
        return [0.0, 0.0]
    b = att.item("Bounds")
    p = att.item("Pivot")
    if b:
        return [float(b[0]) * SCALE, float(b[1]) * SCALE]
    if p:
        return [float(p[0]) * SCALE, float(p[1]) * SCALE]
    return [0.0, 0.0]


def _sources(param):
    return [v for i, v in sorted(param.items_named("Source"))]


def _script_parts(c):
    """(code, [input params], [output params]) of a script component, or None."""
    code = None
    sc = c.chunk("Script")
    if sc is not None and sc.item("Text"):
        txt = sc.item("Text")
        try:
            code = base64.b64decode(txt).decode("utf-8")
        except Exception:
            code = txt
    for key in ("CodeInput", "ScriptSource", "Code"):
        if code is None and c.item(key):
            code = c.item(key)
    if code is None:
        return None
    pd = c.chunk("ParameterData")
    if pd is not None:
        ins = sorted(pd.chunks_named("InputParam"), key=lambda x: x.index)
        outs = sorted(pd.chunks_named("OutputParam"), key=lambda x: x.index)
    else:
        ins = sorted(c.chunks_named("param_input"), key=lambda x: x.index)
        outs = sorted(c.chunks_named("param_output"), key=lambda x: x.index)
    return code, ins, outs


def _var(p):
    return (p.item("NickName") or p.item("Name") or "x").strip()


def convert(path, length_scale=1.0):
    """Returns (graph, report_lines). length_scale converts GH lengths to Forma model units
    (1000 when the Rhino file was in metres and Forma works in millimetres)."""
    raw = open(path, "rb").read()
    root = ghfile.read_bytes(raw)
    src_key = uuid.uuid4().hex[:8]
    defn = root.chunk("Definition")
    objs = defn.chunk("DefinitionObjects") if defn else None
    if objs is None:
        raise ValueError("No Grasshopper objects found in this file")
    nodes, wires, report = {}, [], []
    out_map = {}        # GH output param guid -> (nid, port)
    in_links = []       # (nid, forma port, [source guids], gh name)
    passthrough = {}    # GH floating-param guid -> [source guids]
    length_inputs = []  # (nid, port) that should be scaled
    dropped = []        # GH sources of script inputs the native node does not need
    counts = {"script": 0, "native": 0, "param": 0, "known": 0, "missing": 0}

    def add(type_, c, props=None, values=None, label=None):
        nid = uuid.uuid4().hex[:8]
        props = dict(props or {})
        if c.item("InstanceGuid"):
            props["gh_guid"] = c.item("InstanceGuid")
            props["gh_src"] = src_key
        nodes[nid] = {"type": type_, "props": props, "values": values or {}, "pos": _pos(c), "preview": True}
        if label:
            nodes[nid]["label"] = label
        return nid

    for obj in objs.chunks_named("Object"):
        guid = (obj.item("GUID") or "").lower()
        name = obj.item("Name") or ""
        c = _container(obj)
        nick = c.item("NickName") or name
        inst = c.item("InstanceGuid")
        if guid == SLIDER:
            s = c.chunk("Slider")
            val, lo, hi = s.item("Value", default=0.0), s.item("Min", default=0.0), s.item("Max", default=10.0)
            dec = int(s.item("Digits", default=1))
            if s.item("Interval") in (1, 2, 3):   # integer / even / odd sliders
                dec = 0
            nid = add("Number Slider", c, {"value": float(val), "min": float(lo), "max": float(hi), "decimals": dec},
                      label=nick)
            out_map[inst] = (nid, "Value")
            counts["param"] += 1
        elif guid in (TOGGLE, BUTTON):
            nid = add("Toggle", c, {"value": bool(c.item("ToggleValue", default=False))}, label=nick)
            out_map[inst] = (nid, "Value")
            counts["param"] += 1
        elif guid == PANEL:
            srcs = _sources(c)
            if srcs:
                nid = add("Panel", c, label=nick)
                in_links.append((nid, "Data", srcs, nick))
                out_map[inst] = (nid, "Data")
            else:
                nid = add("Text", c, {"value": c.item("UserText", default="")}, label=nick)
                out_map[inst] = (nid, "Text")
            counts["param"] += 1
        elif guid == VALUE_LIST:
            items = [ch for ch in c.chunks_named("ListItem")]
            sel = next((ch for ch in items if ch.item("Selected")), items[0] if items else None)
            val = sel.item("Expression", default="") if sel else ""
            try:
                nid = add("Number", c, {"value": float(val)}, label=nick)
                out_map[inst] = (nid, "Value")
            except ValueError:
                nid = add("Text", c, {"value": str(val).strip('"')}, label=nick)
                out_map[inst] = (nid, "Text")
            counts["param"] += 1
        elif (sp := _script_parts(c)) is not None:
            code, ins, outs = sp
            native = next((n for n in NATIVE_SCRIPTS if all(m in code for m in n["match"])), None)
            if native:
                nid = add(native["node"], c, values=dict(native.get("defaults", {})), label=nick)
                for p in ins:
                    port = native["inputs"].get(_var(p)) or native["inputs"].get(p.item("Name"))
                    if not port:
                        dropped += _sources(p)
                    elif _sources(p):
                        in_links.append((nid, port, _sources(p), _var(p)))
                        if port in native["lengths"]:
                            length_inputs.append((nid, port))
                for p in outs:
                    port = native["outputs"].get(_var(p)) or native["outputs"].get(p.item("Name"))
                    if port:
                        out_map[p.item("InstanceGuid")] = (nid, port)
                for htype, hlabel, hport, hx, hy, hprops in native.get("helpers", []):
                    hid = add(htype, c, label=hlabel)
                    nodes[hid]["pos"] = [nodes[nid]["pos"][0] + hx, nodes[nid]["pos"][1] + hy]
                    hprops = dict(hprops)
                    if hprops.get("value") == "@brief":
                        from .nodes.userlib import DEFAULT_BRIEF
                        hprops["value"] = DEFAULT_BRIEF
                    nodes[hid]["props"] = hprops
                    if htype == "Panel":
                        wires.append([nid, hport, hid, "Data"])
                    else:
                        wires.append([hid, "Text" if htype == "Text" else "Value", nid, hport])
                report.append(f"'{nick}': {native['note']}")
                counts["native"] += 1
            else:
                in_names = [_var(p) for p in ins] or ["x"]
                out_names = [_var(p) for p in outs if _var(p) not in ("out",) and p.item("Name") != "out"] or ["a"]
                nid = add("Python", c, {"code": code, "inputs": in_names, "outputs": out_names}, label=nick)
                for p in ins:
                    if _sources(p):
                        in_links.append((nid, _var(p), _sources(p), _var(p)))
                for p in outs:
                    v = _var(p)
                    out_map[p.item("InstanceGuid")] = (nid, "out" if (p.item("Name") == "out") else v)
                if RHINO_ONLY.search(code):
                    mods = sorted(set(m.group(2) for m in RHINO_ONLY.finditer(code)))
                    report.append(f"'{nick}': Python script imported with its code, but it uses {', '.join(mods)} "
                                  f"which only exist inside Rhino. Edit the node (double-click) to use Forma's "
                                  f"geometry functions instead, or ask for it to be ported like the built-in tools.")
                else:
                    report.append(f"'{nick}': Python script imported (plain Python, should run as is).")
                counts["script"] += 1
        elif name in KNOWN:
            ftype, imap, omap = KNOWN[name]
            nid = add(ftype, c, label=nick if nick != name else None)
            for p in sorted(c.chunks_named("param_input"), key=lambda x: x.index):
                port = imap.get(p.item("NickName")) or imap.get(p.item("Name"))
                if port and _sources(p):
                    in_links.append((nid, port, _sources(p), p.item("Name")))
            for p in sorted(c.chunks_named("param_output"), key=lambda x: x.index):
                port = omap.get(p.item("NickName")) or omap.get(p.item("Name"))
                if port:
                    out_map[p.item("InstanceGuid")] = (nid, port)
            counts["known"] += 1
        elif name in GEOMETRY_PARAMS or name in DATA_PARAMS:
            srcs = _sources(c)
            if srcs:
                passthrough[inst] = srcs
            else:
                pdata = c.chunk("PersistentData")
                n_ref = sum(len(b.chunks_named("Item")) for b in pdata.chunks_named("Branch")) if pdata else 0
                nid = add("Geometry", c, {"ref_ids": []}, label=f"{nick} (pick in Forma)")
                out_map[inst] = (nid, "Geometry")
                if n_ref:
                    report.append(f"'{nick}': held {n_ref} Rhino object(s). Grasshopper stores those in the .3dm, "
                                  f"not the .gh: import or draw them in Forma, select them and click "
                                  f"'Set from selection' on this node.")
                counts["param"] += 1
        else:
            nid = add("Text", c, {"value": f"Grasshopper '{name}' has no Forma equivalent yet"},
                      label=f"Missing: {nick}")
            for p in c.chunks_named("param_output"):
                out_map[p.item("InstanceGuid")] = (nid, "Text")
            report.append(f"'{nick}' ({name}): no Forma equivalent, left as a placeholder.")
            counts["missing"] += 1

    def resolve(g, depth=0):
        if g in out_map:
            return [out_map[g]]
        if g in passthrough and depth < 20:
            out = []
            for s in passthrough[g]:
                out += resolve(s, depth + 1)
            return out
        return []

    for nid, port, srcs, ghname in in_links:
        for s in srcs:
            for (snid, sport) in resolve(s):
                w = [snid, sport, nid, port]
                if w not in wires:
                    wires.append(w)
    # the air volume and furniture would hide the heat maps inside them: preview them off
    for s_, o, d, i in list(wires):
        if nodes.get(d, {}).get("type") == "Space Syntax 3D (voxel VGA)" and i in ("Space", "Furniture"):
            stack = [s_]
            while stack:
                n_ = stack.pop()
                if n_ in nodes and nodes[n_]["preview"]:
                    nodes[n_]["preview"] = False
                    stack += [w[0] for w in wires if w[2] == n_]
    # parameters that only fed inputs the native tools do not need (bake, reset, launch, ...)
    used = {w[0] for w in wires}
    for g in dropped:
        for snid, _ in resolve(g):
            if snid in nodes and snid not in used and nodes[snid]["type"] in ("Toggle", "Number Slider", "Text"):
                del nodes[snid]
                counts["param"] -= 1
    # length sliders that feed metre-based inputs
    if length_scale != 1.0:
        scaled = set()
        for nid, port in length_inputs:
            for s, o, d, i in wires:
                if d == nid and i == port and nodes[s]["type"] == "Number Slider" and s not in scaled:
                    pr = nodes[s]["props"]
                    for k in ("value", "min", "max"):
                        pr[k] = pr[k] * length_scale
                    pr["gh_scale"] = length_scale   # undone when saving back to Grasshopper
                    pr["decimals"] = 0 if length_scale >= 100 else pr["decimals"]
                    scaled.add(s)
        if scaled:
            report.append(f"{len(scaled)} length slider(s) converted to Forma's units (x{length_scale:g}).")
    summary = (f"Imported {len(nodes)} nodes and {len(wires)} wires: {counts['native']} native tool(s), "
               f"{counts['script']} Python script(s), {counts['known']} component(s), {counts['param']} parameter(s), "
               f"{counts['missing']} placeholder(s).")
    for n in nodes.values():   # helper nodes added by Forma are not Grasshopper components
        if n.get("label") in ("Brief", "Links (optional)", "Plans", "Report", "Code check", "metric (0-14)",
                              "Progress", "Metric names") and n["type"] in ("Text", "Number Slider", "Panel"):
            n["props"].pop("gh_guid", None)
            n["props"]["gh_src"] = src_key
            n["props"]["forma_helper"] = True
    source = {"name": os.path.basename(path), "data": base64.b64encode(raw).decode("ascii")}
    return {"nodes": nodes, "wires": wires, "gh_sources": {src_key: source}}, [summary] + report


# ===================================================================== saving back
def _patch_text(text, old, new):
    """Apply the line edits that turn `old` into `new` to `text`, which contains old's blocks verbatim.
    Returns (patched text, number of edits that could not be placed)."""
    import difflib
    o, n = old.splitlines(keepends=True), new.splitlines(keepends=True)
    failed = 0
    for tag, i1, i2, j1, j2 in reversed(difflib.SequenceMatcher(None, o, n, autojunk=False).get_opcodes()):
        if tag == "equal":
            continue
        for ctx in (3, 2, 1, 0):
            before = "".join(o[max(0, i1 - ctx):i1])
            after = "".join(o[i2:i2 + ctx])
            anchor = before + "".join(o[i1:i2]) + after
            if not anchor:
                break
            if text.count(anchor) == 1:
                k = text.index(anchor)
                text = text[:k] + before + "".join(n[j1:j2]) + after + text[k + len(anchor):]
                break
        else:
            failed += 1
            continue
        if not anchor:
            failed += 1
    return text, failed


def export(graph, path, source_key=None):
    """Write the imported Grasshopper definition back to `path` with the edits made in Forma.
    Returns report lines."""
    from .nodes.userlib import ENGINES, engine_source
    sources = graph.get("gh_sources") or {}
    if not sources:
        raise ValueError("This graph was not imported from Grasshopper, so there is no definition to save back to.")
    key = source_key or next(iter(sources))
    root = ghfile.read_bytes(base64.b64decode(sources[key]["data"]))
    objs = root.chunk("Definition").chunk("DefinitionObjects")
    by_guid = {}
    for obj in objs.chunks_named("Object"):
        c = _container(obj)
        if c.item("InstanceGuid"):
            by_guid[c.item("InstanceGuid")] = (obj, c)
    report, changed = [], 0
    for nid, n in graph["nodes"].items():
        pr = n.get("props", {})
        if pr.get("forma_helper"):
            continue
        if pr.get("gh_src") != key:
            if n["type"] not in ("Panel",) and not pr.get("gh_src"):
                report.append(f"'{n.get('label') or n['type']}' was added in Forma and has no Grasshopper equivalent "
                              f"here, so it is not saved.")
            continue
        hit = by_guid.get(pr.get("gh_guid"))
        if hit is None:
            continue
        obj, c = hit
        t = n["type"]
        label = n.get("label") or t
        if t == "Number Slider":
            sl = c.chunk("Slider")
            sc_ = float(pr.get("gh_scale", 1.0)) or 1.0
            for k, item in (("value", "Value"), ("min", "Min"), ("max", "Max")):
                if k in pr and sl is not None and sl.item(item) is not None:
                    v = float(pr[k]) / sc_
                    if abs(float(sl.item(item)) - v) > 1e-9 * max(1.0, abs(v)):
                        sl.set_item(item, v)
                        changed += 1
        elif t == "Toggle":
            if bool(c.item("ToggleValue", default=False)) != bool(pr.get("value")):
                c.set_item("ToggleValue", bool(pr.get("value")), 1)
                changed += 1
        elif t == "Text":
            if c.item("UserText") is not None and c.item("UserText") != pr.get("value", ""):
                c.set_item("UserText", str(pr.get("value", "")), 10)
                changed += 1
        elif t == "Python":
            sp = _script_parts(c)
            if sp and sp[0] != pr.get("code"):
                _set_script(c, pr.get("code", ""))
                changed += 1
                ins = [_var(p) for p in sp[1]]
                outs = [_var(p) for p in sp[2] if p.item("Name") != "out"]
                if ins != list(pr.get("inputs", ins)) or [o for o in outs if o != "out"] != list(pr.get("outputs", outs)):
                    report.append(f"'{label}': code saved; its inputs/outputs were changed in Forma, add the same "
                                  f"ones on the component in Grasshopper (zoom in and use + / -).")
        elif t in ENGINES:
            sp = _script_parts(c)
            if pr.get("engine_code") and sp:
                patched, failed = _patch_text(sp[0], engine_source(ENGINES[t]), pr["engine_code"])
                if patched != sp[0]:
                    _set_script(c, patched)
                    changed += 1
                if failed:
                    report.append(f"'{label}': {failed} of your engine edits are outside the code the Grasshopper "
                                  f"script shares with Forma and could not be placed; they stay in Forma only.")
            if t == "Modular Plan Generator" and pr.get("design"):
                report.append(f"'{label}': the brief made in the Plan Designer lives in Forma; in Grasshopper the "
                              f"script still opens its own window to set the brief.")
    ghfile.write(root, path)
    return [f"Saved {os.path.basename(path)} with {changed} change(s) from Forma."] + report


def _set_script(c, code):
    sc = c.chunk("Script")
    if sc is not None and sc.item("Text") is not None:
        sc.set_item("Text", base64.b64encode(code.encode("utf-8")).decode("ascii"), 10)
        return
    for key in ("CodeInput", "ScriptSource", "Code"):
        if c.item(key) is not None:
            c.set_item(key, code, 10)
            return
