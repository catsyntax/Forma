"""The Forma document: objects, layers, selection, undo/redo and project files."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np

from . import geometry as G
from .geometry import Geom
from . import components as C

LAYER_COLORS = ["#d0d0d0", "#e05a47", "#4f9de0", "#58b368", "#e0b347", "#a66ee0", "#47c9c2"]


@dataclass
class Layer:
    name: str
    color: str = "#d0d0d0"
    visible: bool = True
    locked: bool = False


def _mesh_preview(g, M):
    from . import brep as B
    V, _, T = B.tessellate(g)[:3]
    V2 = G.xform_pts(M, np.asarray(V, float))
    T = np.asarray(T, int)
    if np.linalg.det(np.asarray(M)[:3, :3]) < 0:
        T = T[:, ::-1]
    return G.mesh(V2, T.tolist(), source="Preview", layer=g.layer, space=g.space)


class Document:
    def __init__(self):
        self.objects: dict[str, Geom] = {}
        self.layers: list[Layer] = [Layer("Default", "#d0d0d0")]
        self.current_layer = "Default"
        self.selection: set[str] = set()
        self.listeners = []
        self.undo_stack: list = []
        self.redo_stack: list = []
        self.path: str | None = None
        self.modified = False
        self.node_graph: dict | None = None  # filled in by node editor when saving
        self.units = "mm"
        self.definitions: dict[str, C.ComponentDef] = {}
        self.editing = None  # (def_id, instance_id) while editing a component in place
        self.isolated = None  # set of object ids shown in isolation mode, or None
        self.face_sel = None  # (object id, face index) after Cmd/Ctrl+Shift-clicking a face
        self.edge_sel = None  # (object id, (corner i, corner j)) after Cmd/Ctrl+Option+Shift-clicking an edge
        self.points_on: set[str] = set()       # objects showing control points
        self.grip_sel: set = set()             # selected control points: (object id, index)
        self._grip_cache = (None, [])
        self._expand_cache = {}

    # ----------------------------------------------------------- events
    def subscribe(self, fn):
        self.listeners.append(fn)

    def notify(self, what="changed"):
        if what != "selection":
            self.version = getattr(self, "version", 0) + 1
        for fn in list(self.listeners):
            fn(what)

    # ----------------------------------------------------------- undo
    def _snapshot(self):
        return (dict(self.objects), [Layer(**asdict(l)) for l in self.layers], self.current_layer,
                dict(self.definitions))

    def checkpoint(self, label=""):
        self.undo_stack.append((label, self._snapshot()))
        if len(self.undo_stack) > 200:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.modified = True

    def _restore(self, snap):
        self.objects, self.layers, self.current_layer, defs = snap
        self.objects = dict(self.objects)
        self.definitions = dict(defs)
        self.selection &= set(self.objects)
        self.notify("changed")
        self.notify("layers")

    def undo(self):
        if not self.undo_stack:
            return None
        label, snap = self.undo_stack.pop()
        self.redo_stack.append((label, self._snapshot()))
        self._restore(snap)
        return label

    def redo(self):
        if not self.redo_stack:
            return None
        label, snap = self.redo_stack.pop()
        self.undo_stack.append((label, self._snapshot()))
        self._restore(snap)
        return label

    # ----------------------------------------------------------- objects
    def add(self, g: Geom | list, notify=True, select=False):
        gs = g if isinstance(g, list) else [g]
        for x in gs:
            if not self.layer(x.layer):
                x.layer = self.current_layer
            if self.editing:
                x.data["_edit"] = True
            self.objects[x.id] = x
            if self.isolated is not None:
                self.isolated.add(x.id)   # new work stays visible while isolated
        if select:
            self.selection = {x.id for x in gs}
        if notify:
            self.notify("changed")
        return gs

    def replace(self, g: Geom, notify=True):
        self.objects[g.id] = g
        if notify:
            self.notify("changed")

    def remove(self, ids, notify=True):
        for i in list(ids):
            self.objects.pop(i, None)
            self.selection.discard(i)
        if notify:
            self.notify("changed")

    def get(self, oid):
        return self.objects.get(oid)

    # One shared model: Draft (top view) and Model (3D) show and edit the same objects.
    # `space` only records where an object was drawn; set shared_model = False to separate them again.
    shared_model = True

    def _in(self, g, space):
        return self.shared_model or space is None or g.space == space

    def in_space(self, space):
        return [g for g in self.objects.values() if self._in(g, space)]

    def sections(self):
        """Section planes in the model (newest last)."""
        return [g for g in self.objects.values() if g.data.get("section")]

    def _iso(self, g):
        return self.isolated is None or g.id in self.isolated

    def visible(self, space):
        vis = {l.name for l in self.layers if l.visible}
        return [g for g in self.objects.values() if self._in(g, space) and g.layer in vis and self._iso(g)]

    def selectable(self, space):
        ok = {l.name for l in self.layers if l.visible and not l.locked}
        return [g for g in self.objects.values() if self._in(g, space) and g.layer in ok and self._iso(g)
                and (not self.editing or g.data.get("_edit"))]

    # ----------------------------------------------------------- isolation
    def isolate(self, ids):
        """Show only these objects (everything else is hidden until unisolate)."""
        ids = {i for i in ids if i in self.objects}
        if not ids:
            return False
        self.isolated = ids
        self.selection &= ids
        self.notify("layers")
        return True

    def unisolate(self):
        if self.isolated is None:
            return False
        self.isolated = None
        self.notify("layers")
        return True

    # ----------------------------------------------------------- components
    def expand(self, g: Geom):
        """Displayable geometry for an object: instances become their world-space parts."""
        if g.kind != "instance":
            return [g]
        d = self.definitions.get(g.def_id)
        if d is None:
            return []
        key = (g.id, d.id, d.version, g.M.tobytes(), repr(sorted(g.params.items())), g.layer)
        hit = self._expand_cache.get(g.id)
        if hit and hit[0] == key:
            return hit[1]
        parts = C.expand(g, self.definitions)
        self._expand_cache[g.id] = (key, parts)
        return parts

    def display(self, space):
        """List of (owner_id, geom, color, dim) for drawing."""
        out = []
        hidden_inst = self.editing[1] if self.editing else None
        for g in self.visible(space):
            if g.id == hidden_inst:
                continue
            dim = (bool(self.editing) and not g.data.get("_edit")) or bool(g.data.get("section"))
            col = self.layer_color(g.layer)
            for part in self.expand(g):
                out.append((g.id, part, col, dim))
        return out

    def snap_points(self, space):
        pts = []
        for g in self.visible(space):
            if g.kind == "instance":
                for part in self.expand(g):
                    pts += part.snap_points()[:200]
                for c in C.world_connectors(g, self.definitions):
                    pts.append((c["p"], "Connector"))
                pts.append((g.M[:3, 3], "Insert"))
            else:
                pts += g.snap_points()
        return pts

    def add_definition(self, d: C.ComponentDef):
        self.definitions[d.id] = d
        self.notify("definitions")
        return d

    def instances_of(self, def_id):
        return [g for g in self.objects.values() if g.kind == "instance" and g.def_id == def_id]

    def describe(self, g: Geom):
        if g.kind == "instance":
            d = self.definitions.get(g.def_id)
            nm = d.name if d else "?"
            extra = " (parametric)" if d and d.parametric else ""
            return f"Component '{nm}'{extra} at {tuple(round(float(x), 3) for x in g.M[:3, 3])}"
        return g.describe()

    def make_component(self, ids, name, base_point=(0, 0, 0), category="My Components"):
        geoms = [self.objects[i] for i in ids if i in self.objects]
        if not geoms:
            return None
        bp = np.asarray(base_point, float)
        T = np.eye(4)
        T[:3, 3] = -bp
        local = []
        for g in geoms:
            lg = g.transformed(T, keep_id=False)
            lg.layer = "Default" if g.layer == self.current_layer else g.layer
            local.append(lg)
        d = C.ComponentDef(name, local, category=category, connectors=C.bbox_connectors(local))
        self.definitions[d.id] = d
        self.remove(ids, notify=False)
        M = np.eye(4)
        M[:3, 3] = bp
        inst = C.instance(d.id, M, layer=self.current_layer, space=geoms[0].space)
        self.add(inst, notify=False)
        self.selection = {inst.id}
        self.notify("definitions")
        self.notify("changed")
        return inst

    def begin_edit(self, inst_id):
        inst = self.objects.get(inst_id)
        if inst is None or inst.kind != "instance" or self.editing:
            return False
        d = self.definitions[inst.def_id]
        if d.parametric:
            return False
        self.editing = (d.id, inst.id)
        for g in d.geoms:
            w = g.transformed(inst.M, keep_id=False)
            if not self.layer(w.layer):
                w.layer = inst.layer
            w.space = inst.space
            w.data["_edit"] = True
            self.objects[w.id] = w
        self.selection = set()
        self.notify("changed")
        self.notify("editing")
        return True

    def end_edit(self, apply=True):
        if not self.editing:
            return
        def_id, inst_id = self.editing
        inst = self.objects.get(inst_id)
        edited = [g for g in self.objects.values() if g.data.get("_edit")]
        for g in edited:
            self.objects.pop(g.id, None)
        self.editing = None
        if apply and inst is not None:
            Minv = np.linalg.inv(inst.M)
            local = []
            for g in edited:
                lg = g.transformed(Minv, keep_id=False)
                lg.data.pop("_edit", None)
                local.append(lg)
            d = self.definitions[def_id]
            self.definitions[def_id] = d.clone(geoms=local)
        self.selection = {inst_id} if inst is not None else set()
        self.notify("definitions")
        self.notify("changed")
        self.notify("editing")

    def make_unique(self, inst_id):
        inst = self.objects.get(inst_id)
        if inst is None or inst.kind != "instance":
            return None
        d = self.definitions[inst.def_id]
        nd = d.clone(id=C.new_id(), name=d.name + " (unique)", version=0)
        self.definitions[nd.id] = nd
        ni = inst.copy(def_id=nd.id)
        self.objects[ni.id] = ni
        self.notify("definitions")
        self.notify("changed")
        return ni

    def explode_instance(self, inst_id):
        inst = self.objects.get(inst_id)
        if inst is None or inst.kind != "instance":
            return []
        parts = [p.copy(keep_id=False) for p in C.expand(inst, self.definitions)]
        self.remove([inst_id], notify=False)
        self.add(parts, notify=False)
        return parts

    def selected(self, space=None):
        return [self.objects[i] for i in self.selection
                if i in self.objects and self._in(self.objects[i], space)]

    def select(self, ids, add=False):
        self.face_sel = None
        self.edge_sel = None
        if not add:
            self.selection = set()
        self.selection |= set(ids)
        self.notify("selection")

    def toggle_select(self, oid):
        if oid in self.selection:
            self.selection.discard(oid)
        else:
            self.selection.add(oid)
        self.notify("selection")

    def clear_selection(self):
        self.face_sel = None
        self.edge_sel = None
        if self.selection:
            self.selection = set()
            self.notify("selection")

    # ----------------------------------------------------------- layers
    def layer(self, name) -> Layer | None:
        for l in self.layers:
            if l.name == name:
                return l
        return None

    def layer_color(self, name):
        l = self.layer(name)
        return l.color if l else "#d0d0d0"

    def add_layer(self, name=None, color=None):
        if name is None:
            i = len(self.layers)
            name = f"Layer {i:02d}"
            while self.layer(name):
                i += 1
                name = f"Layer {i:02d}"
        if not self.layer(name):
            self.layers.append(Layer(name, color or LAYER_COLORS[len(self.layers) % len(LAYER_COLORS)]))
            self.notify("layers")
        return self.layer(name)

    # ----------------------------------------------------------- files
    def to_json(self):
        return {
            "format": "forma-project", "version": 1, "units": self.units,
            "layers": [asdict(l) for l in self.layers],
            "current_layer": self.current_layer,
            "objects": [g.to_dict() for g in self.objects.values()],
            "nodes": self.node_graph,
            "definitions": [d.to_dict() for d in self.definitions.values()],
        }

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_json(), f)
        self.path = path
        self.modified = False

    def load(self, path):
        with open(path) as f:
            d = json.load(f)
        self.clear()
        self.layers = [Layer(**l) for l in d.get("layers", [])] or [Layer("Default")]
        self.current_layer = d.get("current_layer", self.layers[0].name)
        self.units = d.get("units", "mm")
        for od in d.get("objects", []):
            g = Geom.from_dict(od)
            self.objects[g.id] = g
        self.node_graph = d.get("nodes")
        for dd in d.get("definitions", []):
            cd = C.ComponentDef.from_dict(dd)
            self.definitions[cd.id] = cd
        self.path = path
        self.modified = False
        self.notify("layers")
        self.notify("changed")
        self.notify("loaded")

    # ----------------------------------------------------------- control points
    def grip_points(self):
        """[(object id, index, point)] for every object with points on (cached per document version)."""
        from . import editpoints as E
        key = (getattr(self, "version", 0), frozenset(self.points_on))
        if self._grip_cache[0] == key:
            return self._grip_cache[1]
        out = []
        vis = {l.name for l in self.layers if l.visible}
        for oid in list(self.points_on):
            g = self.objects.get(oid)
            if g is None or g.layer not in vis:
                self.points_on.discard(oid)
                continue
            pts = E.grips(g)
            if pts is None:
                continue
            out += [(oid, i, np.asarray(p, float)) for i, p in enumerate(pts)]
        valid = {(o, i) for o, i, _ in out}
        self.grip_sel &= valid
        self._grip_cache = (key, out)
        return out

    def set_points_on(self, ids, on=True):
        if on:
            self.points_on |= set(ids)
        else:
            self.points_on -= set(ids)
            self.grip_sel = {gs for gs in self.grip_sel if gs[0] not in set(ids)}
        self.notify("grips")

    def select_grips(self, items, add=False):
        if not add:
            self.grip_sel = set()
        self.grip_sel |= set(items)
        self.notify("grips")

    def toggle_grip(self, item):
        self.grip_sel ^= {item}
        self.notify("grips")

    def clear_grips(self):
        self.face_sel = None
        self.edge_sel = None
        if self.grip_sel:
            self.grip_sel = set()
            self.notify("grips")

    def transform_targets(self, M, label="Transform", copy=False):
        """Apply M to the selected control points if any, otherwise to the selected objects.
        copy=True leaves the originals and adds transformed copies (they become the selection)."""
        if copy and not self.grip_sel:
            sel = self.selected()
            if not sel:
                return 0
            self.checkpoint(label)
            new = []
            for g in sel:
                try:
                    new.append(g.transformed(M, keep_id=False))
                except Exception:
                    pass
            self.add(new, notify=False, select=True)
            self.notify("changed")
            self.notify("selection")
            return len(new)
        from . import editpoints as E
        if self.grip_sel:
            by = {}
            for oid, i in self.grip_sel:
                by.setdefault(oid, []).append(i)
            self.checkpoint(label)
            for oid, idx in by.items():
                g = self.objects.get(oid)
                if g is not None:
                    self.objects[oid] = E.apply(g, idx, M)
            self.notify("changed")
            return len(self.grip_sel)
        sel = self.selected()
        if not sel:
            return 0
        self.checkpoint(label)
        done = 0
        for g in sel:
            try:
                self.objects[g.id] = g.transformed(M)
                done += 1
            except Exception:
                pass
        self.notify("changed")
        return done

    def select_face(self, oid, face_index):
        """Sub-object selection of one face. Flat faces of flat-faced solids get their corners selected
        as control points, so the gumball (and Move) moves the face; Cmd/Ctrl-drag push/pulls it."""
        from . import editpoints as E
        g = self.objects.get(oid)
        if g is None or face_index is None:
            return False
        self.selection = {oid}
        self.face_sel = (oid, face_index)
        idx = E.face_grips(g, face_index)
        if idx:
            self.points_on = {oid}
            self.grip_sel = {(oid, i) for i in idx}
        else:
            self.grip_sel = set()
        self.notify("selection")
        self.notify("grips")
        return True

    def delete_subobjects(self):
        """Delete the selected face, edge or control points (Delete/Backspace).
        Returns a message, or None if nothing sub-object was selected (then whole objects are deleted)."""
        from . import brep as B, editpoints as E
        if self.face_sel:
            oid, fi = self.face_sel
            g = self.objects.get(oid)
            if g is None or g.kind != "brep":
                return None
            parts = B.remove_faces(g, [fi])
            self._replace_with(oid, parts, "Delete face")
            return f"Face deleted; {len(parts)} object(s) remain."
        if self.edge_sel:
            oid, (i, j) = self.edge_sel
            g = self.objects.get(oid)
            if g is None:
                return None
            if g.kind == "polyline":
                pts = list(g.pts)
                n = len(pts)
                a, b = sorted((i, j))
                if g.closed and {a, b} == {0, n - 1}:
                    parts = [G.polyline(pts, closed=False, layer=g.layer, space=g.space)]
                elif g.closed:
                    parts = [G.polyline(pts[b:] + pts[:a + 1], closed=False, layer=g.layer, space=g.space)]
                else:
                    parts = [G.polyline(seg, layer=g.layer, space=g.space) for seg in (pts[:a + 1], pts[b:]) if len(seg) >= 2]
                self._replace_with(oid, parts, "Delete segment")
                return f"Segment deleted; {len(parts)} curve(s) remain."
            if g.kind == "brep":
                fs = E.faces_with_vertices(g, [i, j], need_all=True)
                if not fs:
                    return None
                parts = B.remove_faces(g, fs)
                self._replace_with(oid, parts, "Delete edge")
                return f"Edge deleted with its {len(fs)} face(s); {len(parts)} object(s) remain."
            return None
        if self.grip_sel:
            by = {}
            for oid, i in self.grip_sel:
                by.setdefault(oid, []).append(i)
            changed, total = False, 0
            self.checkpoint("Delete points")
            for oid, idx in by.items():
                g = self.objects.get(oid)
                if g is None:
                    continue
                try:
                    parts = E.delete_points(g, idx)
                except Exception:
                    parts = None
                if parts is None:
                    continue
                self._replace_with(oid, parts, None)
                changed, total = True, total + len(idx)
            if not changed:
                return None
            return f"Deleted {total} point(s)."
        return None

    def _replace_with(self, oid, parts, label):
        if label:
            self.checkpoint(label)
        old = self.objects.pop(oid, None)
        self.selection.discard(oid)
        self.points_on.discard(oid)
        self.grip_sel = {(o, i) for o, i in self.grip_sel if o != oid}
        self.face_sel = None
        self.edge_sel = None
        for k, p in enumerate(parts):
            if old is not None:
                p.layer = old.layer
                p.space = old.space
                if k == 0:
                    p.id = old.id
        if parts:
            self.add(parts, notify=False)
            self.selection = {p.id for p in parts}
        self.notify("changed")
        self.notify("grips")
        self.notify("selection")

    def push_pull_face(self, vector, label="Push/pull face"):
        from . import brep as B
        if not self.face_sel:
            return False
        oid, fi = self.face_sel
        g = self.objects.get(oid)
        if g is None or g.kind != "brep":
            return False
        new = B.push_pull(g, fi, vector)
        self.checkpoint(label)
        self.objects[oid] = new
        self.face_sel = None
        self.grip_sel = set()
        self.points_on.discard(oid)
        self.notify("changed")
        self.notify("grips")
        return True

    def face_center(self):
        from . import brep as B
        if not self.face_sel:
            return None
        g = self.objects.get(self.face_sel[0])
        try:
            return B.face_info(g, self.face_sel[1])[2]
        except Exception:
            return None

    def extrude_targets(self, vector, label="Extrude"):
        """Extrude the selected curves and surfaces along a vector (gumball Cmd/Ctrl-drag).
        Closed curves and surfaces become solids. Returns (made, skipped)."""
        from . import brep as B
        sel = [g for g in self.selected() if g.is_curve or g.kind == "brep"]
        if not sel or float(np.linalg.norm(vector)) < 1e-12:
            return 0, len(self.selected())
        made, skipped, pushed = [], 0, []
        self.checkpoint(label)
        for g in sel:
            try:
                if g.kind == "brep" and len(B.faces(B.shape_of(g))) != 1:
                    # solid or polysurface: push/pull the face that points most along the drag
                    v = np.asarray(vector, float)
                    vn = v / np.linalg.norm(v)
                    best, bd = None, 0.5
                    for i in range(len(B.faces(B.shape_of(g)))):
                        try:
                            planar, n, _ = B.face_info(g, i)
                        except Exception:
                            continue
                        dd = abs(float(np.dot(n, vn))) + (0.05 if planar else 0.0)
                        if dd > bd and np.dot(n, vn) > 0:
                            best, bd = i, dd
                    if best is None:
                        skipped += 1
                        continue
                    self.objects[g.id] = B.push_pull(g, best, v)
                    pushed.append(g.id)
                    continue
                made.append(B.extrude(g, tuple(float(v) for v in vector), cap=True, layer=g.layer))
            except Exception:
                skipped += 1
        if made:
            self.add(made, notify=False, select=True)
        self.notify("changed")
        self.notify("selection")
        return len(made) + len(pushed), skipped

    def preview_targets(self, M):
        """Transformed copies of what transform_targets would change (for live preview)."""
        from . import editpoints as E
        out = []
        if self.grip_sel:
            by = {}
            for oid, i in self.grip_sel:
                by.setdefault(oid, []).append(i)
            for oid, idx in by.items():
                g = self.objects.get(oid)
                if g is not None:
                    try:
                        out.append(E.apply(g, idx, M))
                    except Exception:
                        pass
            return out
        R = np.asarray(M)[:3, :3]
        RRt = R @ R.T
        similar = np.allclose(RRt, np.eye(3) * RRt[0, 0], atol=1e-9 * max(1.0, RRt[0, 0]))
        for g in self.selected():
            for part in self.expand(g):
                try:
                    if part.kind == "brep" and not similar:
                        # live preview of a stretch: transform the display mesh, not the exact solid
                        out.append(_mesh_preview(part, M))
                    else:
                        out.append(part.transformed(M, keep_id=False))
                except Exception:
                    pass   # e.g. a scale factor of zero while dragging through the centre
        return out

    def target_center(self):
        if self.face_sel and not self.grip_sel:
            c = self.face_center()
            if c is not None:
                return c
        if self.grip_sel:
            pts = [p for o, i, p in self.grip_points() if (o, i) in self.grip_sel]
            return np.mean(pts, axis=0) if pts else None
        sel = self.selected()
        if not sel:
            return None
        pts = []
        for g in sel:
            for part in self.expand(g):
                lo, hi = part.bbox()
                pts += [lo, hi]
        a = np.array(pts)
        return (a.min(0) + a.max(0)) / 2

    def clear(self):
        self.objects.clear()
        self.points_on = set()
        self.grip_sel = set()
        self.layers = [Layer("Default")]
        self.current_layer = "Default"
        self.selection.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.node_graph = None
        self.definitions = {}
        self.editing = None
        self.isolated = None
        self._expand_cache = {}
        self.path = None
        self.modified = False
