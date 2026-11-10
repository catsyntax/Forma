"""The Forma document: objects, layers, selection, undo/redo and project files."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np

from . import geometry as G
from .geometry import Geom
from . import components as C

LAYER_COLORS = ["#d0d0d0", "#e05a47", "#4f9de0", "#58b368", "#e0b347", "#a66ee0", "#47c9c2"]


SEP = "::"   # sublayers use full paths, like Rhino: "Plans::PLAN_01::Rooms"
WEIGHTS = [0.0, 0.09, 0.13, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0]   # mm; 0 = default


@dataclass
class Layer:
    name: str
    color: str = "#d0d0d0"
    visible: bool = True
    locked: bool = False
    weight: float = 0.0      # line weight in mm (0 = default); used on screen and in PDF/DWG
    expanded: bool = True    # tree state in the Layers panel

    @property
    def short(self):
        return self.name.split(SEP)[-1]

    @property
    def parent(self):
        return SEP.join(self.name.split(SEP)[:-1]) or None


def display_px(weight_mm, default):
    """Screen line width for a print weight (0.25 mm ~ 1.4 px)."""
    if not weight_mm:
        return default
    return max(0.8, min(10.0, weight_mm * 5.6))


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

    # ----------------------------------------------------------- display state
    def layers_on(self):
        """Names of layers that are visible, counting their parents."""
        by = {l.name: l for l in self.layers}
        out = set()
        for l in self.layers:
            n, ok = l.name, True
            while n:
                L = by.get(n)
                if L is not None and not L.visible:
                    ok = False
                    break
                n = SEP.join(n.split(SEP)[:-1])
            if ok:
                out.add(l.name)
        return out

    def layers_locked(self):
        by = {l.name: l for l in self.layers}
        out = set()
        for l in self.layers:
            n = l.name
            while n:
                L = by.get(n)
                if L is not None and L.locked:
                    out.add(l.name)
                    break
                n = SEP.join(n.split(SEP)[:-1])
        return out

    def _def_flag(self, g, flag):
        if g.kind != "instance":
            return False
        d = self.definitions.get(g.def_id)
        return bool(d is not None and getattr(d, flag, False))

    def is_shown(self, g, on=None):
        on = self.layers_on() if on is None else on
        return g.layer in on and not g.data.get("hidden") and not self._def_flag(g, "hidden") and self._iso(g)

    def is_locked(self, g, locked=None):
        locked = self.layers_locked() if locked is None else locked
        return g.layer in locked or bool(g.data.get("locked")) or self._def_flag(g, "locked")

    def visible(self, space):
        on = self.layers_on()
        return [g for g in self.objects.values() if self._in(g, space) and self.is_shown(g, on)]

    def selectable(self, space):
        on, locked = self.layers_on(), self.layers_locked()
        return [g for g in self.objects.values() if self._in(g, space) and self.is_shown(g, on)
                and not self.is_locked(g, locked) and (not self.editing or g.data.get("_edit"))]

    def color_of(self, g):
        return g.data.get("attr_color") or self.layer_color(g.layer)

    def weight_of(self, oid):
        """Line weight in mm for an object id (object override, else its layer; 0 = default)."""
        g = self.objects.get(oid)
        if g is None:
            return 0.0
        w = g.data.get("attr_weight")
        if w:
            return float(w)
        l = self.layer(g.layer)
        return float(l.weight) if l else 0.0

    def set_attr(self, ids, key, value, label="Object properties"):
        """Set (value) or clear (None) a display attribute on objects: attr_color, attr_weight, hidden, locked."""
        ids = [i for i in ids if i in self.objects]
        if not ids:
            return
        self.checkpoint(label)
        for i in ids:
            g = self.objects[i].copy()
            if value is None or value is False:
                g.data.pop(key, None)
            else:
                g.data[key] = value
            self.objects[i] = g
        if key in ("hidden", "locked"):
            self.selection -= set(ids)
            self.notify("selection")
        self.notify("changed")
        self.notify("layers")

    def hidden_objects(self):
        return [g for g in self.objects.values() if g.data.get("hidden")]

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
            col = self.color_of(g)
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

    def add_layer(self, name=None, color=None, parent=None):
        """Create a layer (and any missing parents of a 'A::B::C' path). Returns the layer."""
        if name is None:
            i = len(self.layers)
            base = (parent + SEP) if parent else ""
            name = f"{base}Layer {i:02d}"
            while self.layer(name):
                i += 1
                name = f"{base}Layer {i:02d}"
        elif parent:
            name = parent + SEP + name
        parts = name.split(SEP)
        for k in range(1, len(parts) + 1):
            n = SEP.join(parts[:k])
            if not self.layer(n):
                l = Layer(n, color if k == len(parts) and color else LAYER_COLORS[len(self.layers) % len(LAYER_COLORS)])
                # keep children right after their parent in the list
                par = SEP.join(parts[:k - 1])
                idx = len(self.layers)
                if par:
                    kids = [j for j, x in enumerate(self.layers) if x.name == par or x.name.startswith(par + SEP)]
                    idx = (max(kids) + 1) if kids else idx
                self.layers.insert(idx, l)
        self.notify("layers")
        return self.layer(name)

    def sublayers(self, name, deep=True):
        return [l for l in self.layers if l.name.startswith(name + SEP) and (deep or l.parent == name)]

    def rename_layer(self, old, new_short):
        """Rename a layer (its last path part); sublayers and objects follow."""
        l = self.layer(old)
        if l is None or not new_short.strip() or SEP in new_short:
            return False
        new = (l.parent + SEP if l.parent else "") + new_short.strip()
        if new == old or self.layer(new):
            return False
        self.checkpoint("Rename layer")
        mapping = {}
        for x in [l] + self.sublayers(old):
            nn = new + x.name[len(old):]
            mapping[x.name] = nn
            x.name = nn
        for g in list(self.objects.values()):
            if g.layer in mapping:
                self.objects[g.id] = g.copy(layer=mapping[g.layer])
        if self.current_layer in mapping:
            self.current_layer = mapping[self.current_layer]
        self.notify("layers")
        self.notify("changed")
        return True

    def move_layer(self, name, new_parent):
        """Re-parent a layer (None = top level), keeping its sublayers and objects."""
        l = self.layer(name)
        if l is None or (new_parent and (new_parent == name or new_parent.startswith(name + SEP))):
            return False
        new = (new_parent + SEP if new_parent else "") + l.short
        if new == name or self.layer(new):
            return False
        self.checkpoint("Move layer")
        group = [l] + self.sublayers(name)
        for x in group:
            self.layers.remove(x)
        mapping = {}
        for x in group:
            mapping[x.name] = new + x.name[len(name):]
            x.name = mapping[x.name]
        idx = len(self.layers)
        if new_parent:
            kids = [j for j, x in enumerate(self.layers) if x.name == new_parent or x.name.startswith(new_parent + SEP)]
            idx = max(kids) + 1 if kids else idx
        self.layers[idx:idx] = group
        for g in list(self.objects.values()):
            if g.layer in mapping:
                self.objects[g.id] = g.copy(layer=mapping[g.layer])
        if self.current_layer in mapping:
            self.current_layer = mapping[self.current_layer]
        self.notify("layers")
        self.notify("changed")
        return True

    def delete_layer(self, name, delete_objects=False, checkpoint=True):
        """Delete a layer and its sublayers. Their objects are deleted or moved to the parent/first layer."""
        l = self.layer(name)
        group = [l] + self.sublayers(name) if l else []
        if not group or len(group) >= len(self.layers):
            return False
        if checkpoint:
            self.checkpoint("Delete layer")
        names = {x.name for x in group}
        for x in group:
            self.layers.remove(x)
        target = l.parent if l.parent and self.layer(l.parent) else self.layers[0].name
        for g in list(self.objects.values()):
            if g.layer in names:
                if delete_objects:
                    self.objects.pop(g.id)
                    self.selection.discard(g.id)
                else:
                    self.objects[g.id] = g.copy(layer=target)
        if self.current_layer in names:
            self.current_layer = target
        self.notify("layers")
        self.notify("changed")
        return True

    def move_to_layer(self, ids, layer):
        ids = [i for i in ids if i in self.objects and self.objects[i].layer != layer]
        if not ids or not self.layer(layer):
            return
        self.checkpoint("Move to layer")
        for i in ids:
            self.objects[i] = self.objects[i].copy(layer=layer)
        self.notify("changed")
        self.notify("layers")

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
        fields = set(Layer.__dataclass_fields__)
        self.layers = [Layer(**{k: v for k, v in l.items() if k in fields}) for l in d.get("layers", [])] or [Layer("Default")]
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
        on = self.layers_on()
        for oid in list(self.points_on):
            g = self.objects.get(oid)
            if g is None or not self.is_shown(g, on):
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
