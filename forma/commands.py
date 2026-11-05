"""Command engine and command library.

Commands are generator functions. They ``yield`` input requests (GetPoint, GetNumber, ...)
and receive the user's answer back. Views (2D/3D) and the command line feed the answers.
This keeps commands independent of the UI and lets the same command work in 2D and 3D.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import geometry as G
from . import components as C
from .geometry import P


# ======================================================================= requests
@dataclass
class GetPoint:
    prompt: str
    base: np.ndarray | None = None
    preview: Callable | None = None   # fn(point) -> list[Geom]
    allow_none: bool = False          # Enter returns None
    options: list = field(default_factory=list)  # keywords; typing one returns that string
    window: bool = False              # a drag in a view returns ("window", inside_fn, crossing) instead


@dataclass
class GetNumber:
    prompt: str
    default: float | None = None
    base: np.ndarray | None = None    # clicking a point gives distance from base
    preview: Callable | None = None   # fn(number) -> list[Geom]
    integer: bool = False


@dataclass
class GetObjects:
    prompt: str
    filter: Callable | None = None
    single: bool = False
    preselect: bool = True


@dataclass
class GetString:
    prompt: str
    default: str = ""


@dataclass
class GetChoice:
    prompt: str
    choices: list
    default: str | None = None


class Cancel(Exception):
    pass


# ======================================================================= registry
@dataclass
class Command:
    name: str
    fn: Callable
    aliases: tuple = ()
    spaces: tuple = ("2d", "3d")
    group: str = "General"
    help: str = ""


COMMANDS: dict[str, Command] = {}


def command(name, aliases=(), spaces=("2d", "3d"), group="General", help=""):
    def deco(fn):
        COMMANDS[name.lower()] = Command(name, fn, tuple(a.lower() for a in aliases), spaces, group, help or (fn.__doc__ or "").strip())
        return fn
    return deco


def find_command(text):
    t = text.strip().lower()
    if not t:
        return None
    if t in COMMANDS:
        return COMMANDS[t]
    for c in COMMANDS.values():
        if t in c.aliases:
            return c
    return None


def complete(prefix, space=None):
    p = prefix.strip().lower()
    out = []
    for c in COMMANDS.values():
        if space and space not in c.spaces:
            continue
        if c.name.lower().startswith(p) or any(a.startswith(p) for a in c.aliases):
            out.append(c.name)
    for c in COMMANDS.values():  # fuzzy: substring matches after prefix matches
        if c.name not in out and p and p in c.name.lower() and (not space or space in c.spaces):
            out.append(c.name)
    return out


# ======================================================================= context
class Context:
    def __init__(self, doc, space, app=None):
        self.doc = doc
        self.space = space
        self.app = app
        self.last_point = None
        self.messages = []

    def msg(self, text):
        self.messages.append(text)
        if self.app is not None:
            self.app.message(text)

    def commit(self, label):
        self.doc.checkpoint(label)

    def new(self, geoms, select=False):
        gs = geoms if isinstance(geoms, list) else [geoms]
        for g in gs:
            g.space = self.space
            if g.layer == "Default" and self.doc.current_layer != "Default":
                g.layer = self.doc.current_layer
        self.doc.add(gs, select=select)
        return gs


class Runner:
    """Drives one command generator."""

    def __init__(self, ctx: Context, cmd: Command, **kw):
        self.ctx = ctx
        self.cmd = cmd
        self.gen = cmd.fn(ctx, **kw)
        self.request = None
        self.done = False
        self.error = None

    def start(self):
        return self._advance(None, first=True)

    def feed(self, value):
        return self._advance(value)

    def _advance(self, value, first=False):
        try:
            self.request = next(self.gen) if first else self.gen.send(value)
            # skip GetObjects if preselection satisfies it
            while isinstance(self.request, GetObjects) and self.request.preselect:
                pre = [g for g in self.ctx.doc.selected(self.ctx.space)
                       if self.request.filter is None or self.request.filter(g)]
                if not pre or (self.request.single and len(pre) != 1):
                    break
                self.request = self.gen.send(pre[0] if self.request.single else pre)
        except StopIteration:
            self.done, self.request = True, None
        except Cancel:
            self.done, self.request = True, None
        except Exception as e:  # report and end the command
            self.done, self.request, self.error = True, None, e
            self.ctx.msg(f"{self.cmd.name} failed: {e}")
        return self.request

    def cancel(self):
        try:
            self.gen.close()
        except Exception:
            pass
        self.done, self.request = True, None


# ======================================================================= parsing
SAFE_MATH = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}


def eval_number(text):
    return float(eval(text, {"__builtins__": {}}, SAFE_MATH))


def parse_point(text, base=None, last=None, space="3d"):
    """'x,y[,z]', '@dx,dy[,dz]' relative, 'r<angle' / '@r<angle' polar."""
    t = text.strip().replace(" ", "")
    rel = t.startswith("@")
    if rel:
        t = t[1:]
    ref = base if base is not None else (last if last is not None else np.zeros(3))
    if "<" in t:
        r, a = t.split("<", 1)
        r, a = eval_number(r), math.radians(eval_number(a))
        off = np.array([r * math.cos(a), r * math.sin(a), 0.0])
        return (ref if (rel or base is not None) else np.zeros(3)) + off
    parts = [eval_number(x) for x in t.split(",")]
    if len(parts) not in (2, 3):
        raise ValueError("Point must be x,y or x,y,z")
    p = P(*parts)
    if space == "2d":
        p[2] = 0
    return (ref + p) if rel else p


def first_point_of(geoms):
    return geoms[0].bbox()[0] if geoms else np.zeros(3)


def _bbox_center(doc, geoms):
    pts = []
    for g in geoms:
        for part in doc.expand(g):
            lo, hi = part.bbox()
            pts += [lo, hi]
    if not pts:
        return np.zeros(3)
    a = np.array(pts)
    return (a.min(0) + a.max(0)) / 2


def _preview_xform(doc, geoms, M):
    out = []
    for g in geoms:
        for part in doc.expand(g):
            out.append(part.transformed(M, keep_id=False))
    return out


# ======================================================================= drawing
@command("Point", ("po",), group="Draw", help="Place point objects. Enter to finish.")
def cmd_point(ctx):
    first = True
    while True:
        p = yield GetPoint("Location of point" + (" (Enter to finish)" if not first else ""), allow_none=not first)
        if p is None:
            return
        if first:
            ctx.commit("Point")
            first = False
        ctx.new(G.point(p))


@command("Line", ("l",), group="Draw", help="Line from two points.")
def cmd_line(ctx):
    a = yield GetPoint("Start of line")
    b = yield GetPoint("End of line", base=a, preview=lambda p: [G.line(a, p)])
    ctx.commit("Line")
    ctx.new(G.line(a, b))
    ctx.last_point = b


@command("Polyline", ("pl", "pline"), group="Draw", help="Connected line segments. Enter to finish, C to close.")
def cmd_polyline(ctx):
    pts = [(yield GetPoint("Start of polyline"))]
    closed = False
    while True:
        p = yield GetPoint("Next point (Enter to finish, C to close, U to undo)", base=pts[-1], allow_none=True,
                           options=["C", "U"],
                           preview=lambda q: [G.polyline(pts + [q])])
        if p is None:
            break
        if isinstance(p, str):
            if p.upper() == "C" and len(pts) > 2:
                closed = True
                break
            if p.upper() == "U" and len(pts) > 1:
                pts.pop()
            continue
        if len(pts) > 2 and np.linalg.norm(p - pts[0]) < 1e-9:
            closed = True
            break
        pts.append(p)
    if len(pts) >= 2:
        ctx.commit("Polyline")
        ctx.new(G.polyline(pts, closed=closed))


def _inplane(ctx, make, *pts):
    """Build a flat shape in the current Draft view's plane (Front, Right...) instead of world XY."""
    R = getattr(ctx, "view_R", None)
    if R is None or np.allclose(R, np.eye(3)):
        return make(*pts)
    M = np.eye(4)
    M[:3, :3] = R.T
    local = [None if p is None else (R @ np.asarray(p, float) if np.ndim(p) else p) for p in pts]
    out = make(*local)
    return [g.transformed(M) for g in out] if isinstance(out, list) else out.transformed(M)


@command("Rectangle", ("rec", "rect"), group="Draw", help="Rectangle from two corners.")
def cmd_rect(ctx):
    a = yield GetPoint("First corner")
    b = yield GetPoint("Other corner or length,width", base=a, preview=lambda p: [_inplane(ctx, G.rectangle, a, p)])
    ctx.commit("Rectangle")
    ctx.new(_inplane(ctx, G.rectangle, a, b))


@command("Circle", ("c", "ci"), group="Draw", help="Circle from center and radius.")
def cmd_circle(ctx):
    c = yield GetPoint("Center of circle")
    r = yield GetNumber("Radius", base=c, preview=lambda r: [_inplane(ctx, G.circle, c, max(r, 1e-6))])
    ctx.commit("Circle")
    ctx.new(_inplane(ctx, G.circle, c, abs(r)))


@command("Circle3Pt", ("c3",), group="Draw", help="Circle through three points.")
def cmd_circle3(ctx):
    a = yield GetPoint("First point")
    b = yield GetPoint("Second point", base=a, preview=lambda p: [G.line(a, p)])
    c = yield GetPoint("Third point", base=b, preview=lambda p: _circ3(a, b, p))
    ctx.commit("Circle")
    ctx.new(_circ3(a, b, c))


def _circ3(a, b, c):
    cen, r, n = G.circle_from_3pts(a, b, c)
    return [G.line(a, b)] if cen is None else [G.circle(cen, r, normal=n)]


@command("Arc", ("a",), group="Draw", help="Arc through start, point on arc, end.")
def cmd_arc(ctx):
    a = yield GetPoint("Start of arc")
    b = yield GetPoint("Point on arc", base=a, preview=lambda p: [G.line(a, p)])
    c = yield GetPoint("End of arc", base=b, preview=lambda p: [G.arc_3pt(a, b, p)])
    ctx.commit("Arc")
    ctx.new(G.arc_3pt(a, b, c))


@command("Ellipse", ("el",), group="Draw", help="Ellipse from center and two radii.")
def cmd_ellipse(ctx):
    c = yield GetPoint("Center of ellipse")
    rx = yield GetNumber("First radius (X)", base=c, preview=lambda r: [_inplane(ctx, G.circle, c, max(abs(r), 1e-6))])
    ry = yield GetNumber("Second radius (Y)", base=c, preview=lambda r: [_inplane(ctx, G.ellipse, c, abs(rx), max(abs(r), 1e-6))])
    ctx.commit("Ellipse")
    ctx.new(_inplane(ctx, G.ellipse, c, abs(rx), abs(ry)))


@command("Polygon", ("pol", "ngon"), group="Draw", help="Regular polygon.")
def cmd_polygon(ctx):
    n = yield GetNumber("Number of sides", default=6, integer=True)
    c = yield GetPoint("Center of polygon")
    r = yield GetNumber("Radius", base=c, preview=lambda r: [_inplane(ctx, G.polygon, c, max(abs(r), 1e-6), int(n))])
    ctx.commit("Polygon")
    ctx.new(_inplane(ctx, G.polygon, c, abs(r), int(max(3, n))))


@command("Curve", ("crv",), group="Draw", help="NURBS curve from control points. Enter to finish.")
def cmd_curve(ctx):
    pts = [(yield GetPoint("Start of curve"))]
    while True:
        p = yield GetPoint("Next control point (Enter to finish, C to close)", base=pts[-1], allow_none=True,
                           options=["C"],
                           preview=lambda q: [G.nurbs_curve(pts + [q], 3)] if len(pts) >= 1 else [])
        if p is None:
            break
        if isinstance(p, str):
            if p.upper() == "C" and len(pts) > 2:
                pts.append(pts[0])
                break
            continue
        pts.append(p)
    if len(pts) >= 2:
        ctx.commit("Curve")
        ctx.new(G.nurbs_curve(pts, 3))


@command("InterpCrv", ("ic", "spline"), group="Draw", help="Smooth curve through points. Enter to finish.")
def cmd_interp(ctx):
    pts = [(yield GetPoint("Start of curve"))]
    while True:
        p = yield GetPoint("Next point (Enter to finish)", base=pts[-1], allow_none=True,
                           preview=lambda q: [G.interp_curve(pts + [q])])
        if p is None:
            break
        pts.append(p)
    if len(pts) >= 2:
        ctx.commit("InterpCrv")
        ctx.new(G.interp_curve(pts))


# ======================================================================= edit
@command("Move", ("m",), group="Transform", help="Move objects (or selected control points) from a base point to a target.")
def cmd_move(ctx):
    if ctx.doc.grip_sel:
        a = yield GetPoint(f"Point to move {len(ctx.doc.grip_sel)} control point(s) from")
        b = yield GetPoint("Point to move to", base=a, preview=lambda p: ctx.doc.preview_targets(G.translation(p - a)))
        ctx.doc.transform_targets(G.translation(b - a), "Move points")
        return
    objs = yield GetObjects("Select objects to move")
    a = yield GetPoint("Point to move from")
    b = yield GetPoint("Point to move to", base=a, preview=lambda p: _preview_xform(ctx.doc, objs, G.translation(p - a)))
    ctx.commit("Move")
    M = G.translation(b - a)
    for g in objs:
        ctx.doc.replace(g.transformed(M), notify=False)
    ctx.doc.notify("changed")


@command("Copy", ("co", "cp"), group="Transform", help="Copy objects. Keeps placing copies until Enter.")
def cmd_copy(ctx):
    objs = yield GetObjects("Select objects to copy")
    a = yield GetPoint("Point to copy from")
    first = True
    while True:
        b = yield GetPoint("Point to copy to (Enter to finish)", base=a, allow_none=True,
                           preview=lambda p: _preview_xform(ctx.doc, objs, G.translation(p - a)))
        if b is None:
            return
        if first:
            ctx.commit("Copy")
            first = False
        M = G.translation(b - a)
        ctx.doc.add([g.transformed(M, keep_id=False) for g in objs])


@command("Rotate", ("ro",), group="Transform", help="Rotate objects around a center point.")
def cmd_rotate(ctx):
    objs = yield GetObjects("Select objects to rotate")
    c = yield GetPoint("Center of rotation")
    ang = yield GetNumber("Angle (degrees) or pick reference point", default=90, base=None,
                          preview=lambda a: _preview_xform(ctx.doc, objs, G.rotation(a, (0, 0, 1), c)))
    ctx.commit("Rotate")
    M = G.rotation(ang, (0, 0, 1), c)
    for g in objs:
        ctx.doc.replace(g.transformed(M), notify=False)
    ctx.doc.notify("changed")


@command("Rotate3D", ("ro3",), spaces=("3d",), group="Transform", help="Rotate around an axis defined by two points.")
def cmd_rotate3d(ctx):
    objs = yield GetObjects("Select objects to rotate")
    a = yield GetPoint("Start of rotation axis")
    b = yield GetPoint("End of rotation axis", base=a, preview=lambda p: [G.line(a, p)])
    ang = yield GetNumber("Angle (degrees)", default=90,
                          preview=lambda x: _preview_xform(ctx.doc, objs, G.rotation(x, b - a, a)))
    ctx.commit("Rotate3D")
    M = G.rotation(ang, b - a, a)
    for g in objs:
        ctx.doc.replace(g.transformed(M), notify=False)
    ctx.doc.notify("changed")


@command("Scale", ("sc",), group="Transform", help="Uniform scale about a base point.")
def cmd_scale(ctx):
    objs = yield GetObjects("Select objects to scale")
    c = yield GetPoint("Origin point")
    f = yield GetNumber("Scale factor", default=2.0,
                        preview=lambda s: _preview_xform(ctx.doc, objs, G.scaling(s if s else 1, c)))
    if not f:
        return
    ctx.commit("Scale")
    M = G.scaling(f, c)
    for g in objs:
        ctx.doc.replace(g.transformed(M), notify=False)
    ctx.doc.notify("changed")


@command("Scale1D", ("sc1",), group="Transform", help="Scale in one direction.")
def cmd_scale1d(ctx):
    objs = yield GetObjects("Select objects to scale")
    c = yield GetPoint("Origin point")
    a = yield GetPoint("Reference direction point", base=c, preview=lambda p: [G.line(c, p)])
    f = yield GetNumber("Scale factor", default=2.0)
    d = a - c
    d /= np.linalg.norm(d) or 1
    S = np.eye(3) + (f - 1) * np.outer(d, d)
    M = np.eye(4)
    M[:3, :3] = S
    M[:3, 3] = c - S @ c
    ctx.commit("Scale1D")
    for g in objs:
        ctx.doc.replace(g.transformed(M), notify=False)
    ctx.doc.notify("changed")


@command("Mirror", ("mi",), group="Transform", help="Mirror across a line (XY plane). Keeps originals.")
def cmd_mirror(ctx):
    objs = yield GetObjects("Select objects to mirror")
    a = yield GetPoint("Start of mirror line")

    def mm(p):
        d = p - a
        n = np.array([-d[1], d[0], 0.0])
        return G.mirror(a, n if np.linalg.norm(n) > 1e-12 else (1, 0, 0))
    b = yield GetPoint("End of mirror line", base=a, preview=lambda p: _preview_xform(ctx.doc, objs, mm(p)) + [G.line(a, p)])
    ch = yield GetChoice("Delete original objects?", ["No", "Yes"], default="No")
    ctx.commit("Mirror")
    M = mm(b)
    if ch == "Yes":
        for g in objs:
            ctx.doc.replace(g.transformed(M), notify=False)
        ctx.doc.notify("changed")
    else:
        ctx.doc.add([g.transformed(M, keep_id=False) for g in objs])


@command("Delete", ("del", "erase"), group="Edit", help="Delete selected objects.")
def cmd_delete(ctx):
    objs = yield GetObjects("Select objects to delete")
    ctx.commit("Delete")
    ctx.doc.remove([g.id for g in objs])


def _nearest_object(ctx, p, exclude=()):
    best, bd = None, float("inf")
    for g in ctx.doc.selectable(ctx.space):
        if g.id in exclude or not (g.is_curve or g.kind == "brep"):
            continue
        try:
            d = _B().distance_to(g, p)
        except Exception:
            continue
        if d < bd:
            best, bd = g, d
    return best


@command("Trim", ("tr",), group="Edit",
         help="Cut away parts of curves, surfaces or solids: pick the cutting objects, press Enter, "
              "then click each part to remove.")
def cmd_trim(ctx):
    cutters = yield GetObjects("Select cutting objects (curves, surfaces or solids), then Enter",
                               filter=lambda g: g.is_curve or g.kind == "brep")
    ids = {c.id for c in cutters}
    n = 0
    while True:
        p = yield GetPoint("Click the part to trim away, or drag a window across parts (Enter to finish)",
                           allow_none=True, window=True)
        if p is None:
            break
        if isinstance(p, tuple) and p and p[0] == "window":
            _, inside, crossing = p
            done = 0
            for tgt in list(ctx.doc.selectable(ctx.space)):
                if not (tgt.is_curve or tgt.kind == "brep"):
                    continue
                others = [c for c in ctx.doc.objects.values() if c.id in ids and c.id != tgt.id]
                if not others:
                    continue
                try:
                    pieces = _B().trim_pieces(tgt, others)
                except Exception:
                    continue
                if len(pieces) < 2:
                    continue
                hit = []
                for k, sh in enumerate(pieces):
                    pts = _B().piece_points(sh)
                    if not len(pts):
                        continue
                    ins = np.array([inside(q) for q in pts])
                    if (crossing and ins.any()) or (not crossing and ins.all()):
                        hit.append(k)
                if not hit or len(hit) == len(pieces):
                    continue
                rest = [sh for k, sh in enumerate(pieces) if k not in hit]
                if not done:
                    ctx.commit("Trim")
                res = _B().assemble(tgt, rest)
                ctx.doc.remove([tgt.id], notify=False)
                if res:
                    res[0].id = tgt.id
                ctx.new(res)
                done += len(hit)
            if done:
                n += done
            else:
                ctx.msg("No trimmable parts " + ("touched by" if crossing else "fully inside") + " that window.")
            continue
        tgt = _nearest_object(ctx, p)
        if tgt is None:
            ctx.msg("Nothing there to trim.")
            continue
        others = [c for c in ctx.doc.objects.values() if c.id in ids and c.id != tgt.id]
        if not others:
            ctx.msg("Pick a part of another object (the cutting object cannot trim itself).")
            continue
        try:
            res = _B().trim(tgt, others, p)
        except Exception as e:
            ctx.msg(f"Could not trim: {e}")
            continue
        ctx.commit("Trim")
        keep_id = tgt.id
        ctx.doc.remove([tgt.id], notify=False)
        if res:
            res[0].id = keep_id      # the trimmed object keeps its identity (and stays a cutter if it was one)
        ctx.new(res)
        n += 1
    if n:
        ctx.msg(f"Trimmed {n} part(s).")


@command("Split", ("sp",), group="Edit",
         help="Split curves, surfaces or solids into pieces where other objects cross them.")
def cmd_split(ctx):
    objs = yield GetObjects("Select objects to split", filter=lambda g: g.is_curve or g.kind == "brep")
    cutters = yield GetObjects("Select cutting objects, then Enter", filter=lambda g: g.is_curve or g.kind == "brep")
    ctx.commit("Split")
    total = 0
    for g in objs:
        others = [c for c in cutters if c.id != g.id]
        try:
            pieces = _B().split(g, others)
        except Exception as e:
            ctx.msg(f"Could not split: {e}")
            continue
        if len(pieces) > 1:
            ctx.doc.remove([g.id], notify=False)
            ctx.new(pieces)
            total += len(pieces)
    ctx.msg(f"Split into {total} pieces." if total else "Nothing was crossed by the cutting objects.")


@command("ExtractSrf", ("extractsrf", "dupface"), spaces=("3d",), group="Surface",
         help="Copy the face picked with Cmd/Ctrl+Shift-click out of its solid as a separate surface.")
def cmd_extract_srf(ctx):
    fs = ctx.doc.face_sel
    if not fs or fs[0] not in ctx.doc.objects:
        ctx.msg("First Cmd/Ctrl+Shift-click a face of a solid, then run ExtractSrf.")
        return
    g = ctx.doc.objects[fs[0]]
    ctx.commit("Extract surface")
    ctx.new(_B().face_geom(g, fs[1], layer=g.layer))
    return
    yield


@command("SectionPlane", ("sec", "section", "clippingplane"), group="View",
         help="Place a vertical section plane (a rectangle you can move and rotate). The Draft tab then shows "
              "the cut from both sides.")
def cmd_section_plane(ctx):
    from . import section as S
    a = yield GetPoint("Start of the section line")
    b = yield GetPoint("End of the section line", base=a, preview=lambda p: [G.line(a, p)])
    zs = [0.0, 3000.0]
    for g in ctx.doc.visible("3d"):
        try:
            lo, hi = g.bbox()
            zs += [lo[2], hi[2]]
        except Exception:
            pass
    z0, z1 = min(zs), max(zs)
    pad = 0.05 * (z1 - z0)
    if not ctx.doc.layer("Sections"):
        ctx.doc.add_layer("Sections", "#e0525b")
    ctx.commit("Section plane")
    k = len(ctx.doc.sections())
    g = S.make_plane(a, b, z0 - pad, z1 + pad, name=f"Section {chr(65 + k % 26)}")
    ctx.doc.add(g, select=True)
    ctx.msg(f"{g.name} added. Open the Draft tab to see the cut from both sides; move or rotate the plane "
            "with the gumball and the section views follow.")
    return
    yield


@command("Isolate", ("iso", "hideothers"), group="View",
         help="Show only the selected objects; everything else is hidden until Unisolate.")
def cmd_isolate(ctx):
    objs = yield GetObjects("Select objects to isolate")
    if ctx.doc.isolate([g.id for g in objs]):
        ctx.msg(f"Isolated {len(objs)} object(s). Run Unisolate (or click Isolate again) to show everything.")


@command("Unisolate", ("uniso", "showall"), group="View", help="Leave isolation mode and show all objects again.")
def cmd_unisolate(ctx):
    if ctx.doc.unisolate():
        ctx.msg("All objects shown.")
    else:
        ctx.msg("Nothing is isolated.")
    return
    yield


@command("Offset", ("o",), group="Edit", help="Offset a curve (in the XY plane).")
def cmd_offset(ctx):
    crv = yield GetObjects("Select curve to offset", filter=lambda g: g.is_curve, single=True)
    d = yield GetNumber("Offset distance (negative = other side)", default=10.0,
                        preview=lambda x: [G.offset_polyline_2d(crv, x)])
    ctx.commit("Offset")
    ctx.new(G.offset_polyline_2d(crv, d))


@command("Join", ("j",), group="Edit", help="Join touching curves into polylines.")
def cmd_join(ctx):
    objs = yield GetObjects("Select curves to join", filter=lambda g: g.is_curve)
    res = G.join_curves(objs, tol=1e-4)
    ctx.commit("Join")
    ctx.doc.remove([g.id for g in objs], notify=False)
    for r in res:
        r.layer = objs[0].layer
    ctx.new(res)
    ctx.msg(f"{len(objs)} curves joined into {len(res)}.")


@command("Explode", ("x",), group="Edit", help="Explode polylines, meshes and components into parts.")
def cmd_explode(ctx):
    objs = yield GetObjects("Select objects to explode")
    ctx.commit("Explode")
    n = 0
    for g in objs:
        if g.kind == "instance":
            n += len(ctx.doc.explode_instance(g.id))
            continue
        parts = G.explode(g)
        if len(parts) > 1:
            ctx.doc.remove([g.id], notify=False)
            ctx.doc.add(parts, notify=False)
            n += len(parts)
    ctx.doc.notify("changed")
    ctx.msg(f"Exploded into {n} objects.")


@command("Array", ("ar",), group="Transform", help="Rectangular array: counts and spacing in X, Y, Z.")
def cmd_array(ctx):
    objs = yield GetObjects("Select objects to array")
    nx = yield GetNumber("Number in X", default=3, integer=True)
    ny = yield GetNumber("Number in Y", default=1, integer=True)
    nz = 1
    if ctx.space == "3d":
        nz = yield GetNumber("Number in Z", default=1, integer=True)
    lo, hi = _bbox_of(ctx.doc, objs)
    size = hi - lo
    dx = yield GetNumber("Spacing X", default=round(float(size[0] * 1.5) or 10, 4))
    dy = (yield GetNumber("Spacing Y", default=round(float(size[1] * 1.5) or 10, 4))) if ny > 1 else 0
    dz = (yield GetNumber("Spacing Z", default=round(float(size[2] * 1.5) or 10, 4))) if nz > 1 else 0
    ctx.commit("Array")
    new = []
    for i in range(int(nx)):
        for j in range(int(ny)):
            for k in range(int(nz)):
                if i == j == k == 0:
                    continue
                M = G.translation((i * dx, j * dy, k * dz))
                new += [g.transformed(M, keep_id=False) for g in objs]
    ctx.doc.add(new)
    ctx.msg(f"Created {len(new)} copies.")


@command("ArrayPolar", ("ap",), group="Transform", help="Polar array around a center.")
def cmd_arraypolar(ctx):
    objs = yield GetObjects("Select objects to array")
    c = yield GetPoint("Center of polar array")
    n = yield GetNumber("Number of items", default=6, integer=True)
    tot = yield GetNumber("Angle to fill", default=360)
    ctx.commit("ArrayPolar")
    n = int(max(2, n))
    step = tot / n if abs(tot) >= 360 else tot / (n - 1)
    new = []
    for i in range(1, n):
        M = G.rotation(step * i, (0, 0, 1), c)
        new += [g.transformed(M, keep_id=False) for g in objs]
    ctx.doc.add(new)


@command("ArrayCrv", ("ac",), group="Transform", help="Array objects along a curve.")
def cmd_arraycrv(ctx):
    objs = yield GetObjects("Select objects to array")
    crv = yield GetObjects("Select path curve", filter=lambda g: g.is_curve, single=True, preselect=False)
    n = yield GetNumber("Number of items", default=10, integer=True)
    base = _bbox_center(ctx.doc, objs)
    pts = crv.divide(int(n) - (0 if crv.closed else 1))
    ctx.commit("ArrayCrv")
    new = []
    for p in pts:
        new += [g.transformed(G.translation(p - base), keep_id=False) for g in objs]
    ctx.doc.add(new)


def _bbox_of(doc, geoms):
    pts = []
    for g in geoms:
        for part in doc.expand(g):
            lo, hi = part.bbox()
            pts += [lo, hi]
    a = np.array(pts) if pts else np.zeros((1, 3))
    return a.min(0), a.max(0)


@command("SelAll", ("sa",), group="Select", help="Select all objects in this workspace.")
def cmd_selall(ctx):
    ctx.doc.select([g.id for g in ctx.doc.selectable(ctx.space)])
    return
    yield


@command("SelNone", ("sn",), group="Select", help="Clear the selection.")
def cmd_selnone(ctx):
    ctx.doc.clear_selection()
    return
    yield


@command("Invert", ("inv",), group="Select", help="Invert the selection.")
def cmd_invert(ctx):
    cur = ctx.doc.selection
    ctx.doc.select([g.id for g in ctx.doc.selectable(ctx.space) if g.id not in cur])
    return
    yield


@command("SelSimilar", ("ss",), group="Select", help="Select objects of the same type (or same component).")
def cmd_selsimilar(ctx):
    ref = yield GetObjects("Select reference object", single=True)
    same = [g.id for g in ctx.doc.selectable(ctx.space)
            if g.kind == ref.kind and (g.kind != "instance" or g.def_id == ref.def_id)]
    ctx.doc.select(same)
    ctx.msg(f"{len(same)} similar objects selected.")


@command("ChangeLayer", ("cl",), group="Edit", help="Move selected objects to the current layer.")
def cmd_changelayer(ctx):
    objs = yield GetObjects("Select objects to move to the current layer")
    ctx.commit("ChangeLayer")
    for g in objs:
        ctx.doc.replace(g.copy(layer=ctx.doc.current_layer), notify=False)
    ctx.doc.notify("changed")


@command("Undo", ("u",), group="Edit")
def cmd_undo(ctx):
    lbl = ctx.doc.undo()
    ctx.msg(f"Undo {lbl}" if lbl is not None else "Nothing to undo.")
    return
    yield


@command("Redo", (), group="Edit")
def cmd_redo(ctx):
    lbl = ctx.doc.redo()
    ctx.msg(f"Redo {lbl}" if lbl is not None else "Nothing to redo.")
    return
    yield


# ======================================================================= analyze
@command("Length", ("len",), group="Analyze", help="Report curve length.")
def cmd_length(ctx):
    objs = yield GetObjects("Select curves", filter=lambda g: g.is_curve)
    tot = sum(g.length() for g in objs)
    ctx.msg(f"Length = {tot:.6g} {ctx.doc.units}" + (f" ({len(objs)} curves)" if len(objs) > 1 else ""))


@command("Area", (), group="Analyze", help="Report area of closed curves, surfaces and meshes.")
def cmd_area(ctx):
    objs = yield GetObjects("Select closed curves or surfaces")
    tot = sum(sum(p.area() for p in ctx.doc.expand(g)) for g in objs)
    ctx.msg(f"Area = {tot:.6g} {ctx.doc.units}²")


@command("Volume", ("vol",), spaces=("3d",), group="Analyze", help="Report volume of closed solids.")
def cmd_volume(ctx):
    objs = yield GetObjects("Select closed solids")
    tot = sum(sum(p.volume() for p in ctx.doc.expand(g)) for g in objs)
    ctx.msg(f"Volume = {tot:.6g} {ctx.doc.units}³")


@command("Distance", ("dist", "di"), group="Analyze", help="Distance between two points.")
def cmd_distance(ctx):
    a = yield GetPoint("First point")
    b = yield GetPoint("Second point", base=a, preview=lambda p: [G.line(a, p)])
    d = b - a
    ctx.msg(f"Distance = {np.linalg.norm(d):.6g}   dx={d[0]:.4g} dy={d[1]:.4g} dz={d[2]:.4g}")


@command("What", ("props", "info"), group="Analyze", help="Describe selected objects.")
def cmd_what(ctx):
    objs = yield GetObjects("Select objects")
    for g in objs[:20]:
        ctx.msg(f"[{g.layer}] {ctx.doc.describe(g)}")


# ======================================================================= 3D (exact NURBS via OpenCascade)
# Previews use fast mesh primitives; the committed objects are exact NURBS breps.
class _MeshKernel:
    """Fallback when OpenCascade isn't installed: mesh primitives, no exact operations."""
    box, sphere, cylinder = staticmethod(G.box), staticmethod(G.sphere), staticmethod(G.cylinder)
    cone, torus, plane_surface = staticmethod(G.cone), staticmethod(G.torus), staticmethod(G.plane_surface)
    extrude, revolve, pipe = staticmethod(G.extrude), staticmethod(G.revolve), staticmethod(G.pipe)

    @staticmethod
    def loft(curves, ruled=False, **kw):
        return G.loft(curves)

    @staticmethod
    def planar_surface(curves, **kw):
        return G.planar_surface(curves[0] if isinstance(curves, list) else curves)

    def __getattr__(self, name):
        raise RuntimeError("This needs the NURBS kernel. Install it with: pip install cadquery-ocp")


def _B():
    try:
        from . import brep
        return brep
    except ImportError:
        return _MeshKernel()


def _srf(g):
    return g.kind == "brep"


def _solidish(g):
    return g.kind in ("brep", "mesh")


@command("Box", ("b",), spaces=("3d",), group="Solid", help="NURBS box from base rectangle and height.")
def cmd_box(ctx):
    a = yield GetPoint("First corner of base")
    b = yield GetPoint("Other corner of base", base=a, preview=lambda p: [G.rectangle(a, p)])
    h = yield GetNumber("Height", default=round(float(np.linalg.norm(b - a) / 2), 4) or 1, base=b,
                        preview=lambda z: [G.box(a, b[0] - a[0], b[1] - a[1], z or 1e-6)])
    ctx.commit("Box")
    ctx.new(_B().box(a, b[0] - a[0], b[1] - a[1], h))


@command("Sphere", ("sph",), spaces=("3d",), group="Solid", help="NURBS sphere from center and radius.")
def cmd_sphere(ctx):
    c = yield GetPoint("Center of sphere")
    r = yield GetNumber("Radius", base=c, preview=lambda r: [G.circle(c, max(abs(r), 1e-6))])
    ctx.commit("Sphere")
    ctx.new(_B().sphere(c, abs(r)))


@command("Cylinder", ("cyl",), spaces=("3d",), group="Solid", help="NURBS cylinder from base center, radius and height.")
def cmd_cylinder(ctx):
    c = yield GetPoint("Center of base")
    r = yield GetNumber("Radius", base=c, preview=lambda r: [G.circle(c, max(abs(r), 1e-6))])
    h = yield GetNumber("Height", default=round(abs(r) * 2, 4), base=c,
                        preview=lambda h: [G.cylinder(c, abs(r), h or 1e-6)])
    ctx.commit("Cylinder")
    ctx.new(_B().cylinder(c, abs(r), h))


@command("Cone", (), spaces=("3d",), group="Solid", help="NURBS cone from base center, radius and height.")
def cmd_cone(ctx):
    c = yield GetPoint("Center of base")
    r = yield GetNumber("Radius", base=c, preview=lambda r: [G.circle(c, max(abs(r), 1e-6))])
    h = yield GetNumber("Height", default=round(abs(r) * 2, 4), base=c, preview=lambda h: [G.cone(c, abs(r), h or 1e-6)])
    ctx.commit("Cone")
    ctx.new(_B().cone(c, abs(r), h))


@command("TruncatedCone", ("tcone",), spaces=("3d",), group="Solid", help="Cone with a flat top.")
def cmd_tcone(ctx):
    c = yield GetPoint("Center of base")
    r = yield GetNumber("Base radius", base=c, preview=lambda r: [G.circle(c, max(abs(r), 1e-6))])
    h = yield GetNumber("Height", default=round(abs(r) * 2, 4), base=c)
    r2 = yield GetNumber("Top radius", default=round(abs(r) / 2, 4),
                         preview=lambda t: [G.cylinder(c, abs(r), h, top_radius=abs(t))])
    ctx.commit("TruncatedCone")
    ctx.new(_B().cone(c, abs(r), h, top_radius=abs(r2)))


@command("Torus", (), spaces=("3d",), group="Solid", help="NURBS torus from center and two radii.")
def cmd_torus(ctx):
    c = yield GetPoint("Center of torus")
    R = yield GetNumber("Major radius", base=c, preview=lambda r: [G.circle(c, max(abs(r), 1e-6))])
    r = yield GetNumber("Minor radius", default=round(abs(R) / 4, 4), preview=lambda x: [G.torus(c, abs(R), abs(x) or 1e-6)])
    ctx.commit("Torus")
    ctx.new(_B().torus(c, abs(R), abs(r)))


@command("Plane", ("plane", "srf"), spaces=("3d",), group="Surface", help="Rectangular planar NURBS surface.")
def cmd_plane(ctx):
    a = yield GetPoint("First corner")
    b = yield GetPoint("Other corner", base=a, preview=lambda p: [G.rectangle(a, p)])
    ctx.commit("Plane")
    ctx.new(_B().plane_surface(a, b))


@command("PlanarSrf", ("ps",), spaces=("3d",), group="Surface", help="Trimmed planar surface from closed curves (inner curves become holes).")
def cmd_planarsrf(ctx):
    objs = yield GetObjects("Select closed planar curves", filter=lambda g: g.is_curve and g.closed)
    ctx.commit("PlanarSrf")
    ctx.new(_B().planar_surface(objs))


@command("Extrude", ("e", "ext", "extrudecrv", "extrudesrf"), spaces=("3d",), group="Surface",
         help="Extrude curves or surfaces straight up. Closed curves become solids.")
def cmd_extrude(ctx):
    objs = yield GetObjects("Select curves or surfaces to extrude", filter=lambda g: g.is_curve or g.kind == "brep")
    h = yield GetNumber("Extrusion distance (Z)", default=10.0,
                        preview=lambda z: [G.extrude(c, (0, 0, z or 1e-6), cap=True) for c in objs[:20] if c.is_curve])
    cap = yield GetChoice("Cap closed curves (solid)?", ["Yes", "No"], default="Yes")
    ctx.commit("Extrude")
    ctx.new([_B().extrude(c, (0, 0, h), cap=(cap == "Yes")) for c in objs])


@command("ExtrudeAlong", ("ea",), spaces=("3d",), group="Surface", help="Extrude along a direction picked with two points.")
def cmd_extrude_vec(ctx):
    objs = yield GetObjects("Select curves or surfaces", filter=lambda g: g.is_curve or g.kind == "brep")
    a = yield GetPoint("Start of direction")
    b = yield GetPoint("End of direction", base=a, preview=lambda p: [G.extrude(c, p - a) for c in objs[:20] if c.is_curve])
    ctx.commit("Extrude")
    ctx.new([_B().extrude(c, b - a) for c in objs])


@command("Loft", ("lo",), spaces=("3d",), group="Surface", help="NURBS surface through curves (closed curves make a solid).")
def cmd_loft(ctx):
    objs = yield GetObjects("Select curves to loft", filter=lambda g: g.is_curve)
    if len(objs) < 2:
        ctx.msg("Loft needs at least 2 curves.")
        return
    style = yield GetChoice("Style", ["Normal", "Straight"], default="Normal")
    objs = sorted(objs, key=lambda g: tuple(np.round(g.centroid()[::-1], 6)))
    ctx.commit("Loft")
    ctx.new(_B().loft(objs, ruled=(style == "Straight")))


@command("Revolve", ("rev",), spaces=("3d",), group="Surface", help="Revolve a curve around an axis.")
def cmd_revolve(ctx):
    crv = yield GetObjects("Select curve to revolve", filter=lambda g: g.is_curve, single=True)
    a = yield GetPoint("Start of revolve axis")
    b = yield GetPoint("End of revolve axis", base=a, preview=lambda p: [G.line(a, p)])
    ang = yield GetNumber("Revolve angle", default=360)
    ctx.commit("Revolve")
    ctx.new(_B().revolve(crv, a, b - a, ang))


@command("Pipe", (), spaces=("3d",), group="Surface", help="Round solid pipe along curves.")
def cmd_pipe(ctx):
    objs = yield GetObjects("Select curves", filter=lambda g: g.is_curve)
    r = yield GetNumber("Pipe radius", default=1.0, preview=lambda r: [G.pipe(c, abs(r) or 1e-3) for c in objs[:10]])
    ctx.commit("Pipe")
    ctx.new([_B().pipe(c, abs(r)) for c in objs])


@command("Sweep1", ("sw",), spaces=("3d",), group="Surface", help="Sweep a profile curve along a rail curve.")
def cmd_sweep1(ctx):
    rail = yield GetObjects("Select rail curve", filter=lambda g: g.is_curve, single=True)
    prof = yield GetObjects("Select profile curve", filter=lambda g: g.is_curve and g is not rail, single=True, preselect=False)
    ctx.commit("Sweep1")
    ctx.new(_B().sweep1(rail, prof))


def _bool(ctx, op):
    objs = yield GetObjects(f"Select first set of solids for {op}", filter=_solidish)
    others = yield GetObjects("Select second set", filter=lambda g: _solidish(g) and g not in objs, preselect=False)
    if all(g.kind == "brep" for g in objs + others):
        res = _B().boolean(objs, others, op, layer=objs[0].layer)
        if _B().n_faces(_B().shape_of(res)) == 0:
            ctx.msg("Boolean produced an empty result (do the solids overlap?).")
            return
        ctx.commit("Boolean " + op)
        ctx.doc.remove([g.id for g in objs + others], notify=False)
        ctx.new(res)
        return
    import pyvista as pv  # mesh fallback for imported meshes

    def pd(g):
        F = np.hstack([[3, *t] for t in g.triangles()])
        return pv.PolyData(np.asarray(g.sample(), float) if g.kind == "brep" else g.V, F).triangulate().clean()
    A = pd(objs[0])
    for g in objs[1:]:
        A = A.boolean_union(pd(g))
    for g in others:
        A = {"union": A.boolean_union, "difference": A.boolean_difference, "intersection": A.boolean_intersection}[op](pd(g))
    if A.n_points == 0:
        ctx.msg("Boolean produced an empty result.")
        return
    ctx.commit("Boolean " + op)
    ctx.doc.remove([g.id for g in objs + others], notify=False)
    ctx.new(G.mesh(np.asarray(A.points), A.faces.reshape(-1, 4)[:, 1:], source=f"Mesh boolean {op}", layer=objs[0].layer))
    ctx.msg("Mesh inputs: used a mesh boolean. Convert to NURBS solids for exact results.")


@command("BooleanUnion", ("bu",), spaces=("3d",), group="Solid", help="Exact union of solids.")
def cmd_bu(ctx):
    yield from _bool(ctx, "union")


@command("BooleanDifference", ("bd",), spaces=("3d",), group="Solid", help="Exact subtraction of solids.")
def cmd_bd(ctx):
    yield from _bool(ctx, "difference")


@command("BooleanIntersection", ("bi",), spaces=("3d",), group="Solid", help="Exact intersection of solids.")
def cmd_bi(ctx):
    yield from _bool(ctx, "intersection")


@command("FilletEdge", ("fillet", "fe"), spaces=("3d",), group="Solid", help="Round all edges of a NURBS solid.")
def cmd_fillet(ctx):
    objs = yield GetObjects("Select solids to fillet", filter=_srf)
    r = yield GetNumber("Fillet radius", default=1.0)
    ctx.commit("FilletEdge")
    for g in objs:
        res = _B().fillet(g, abs(r), id=g.id, layer=g.layer, space=g.space)
        if res.data.get("skipped_edges"):
            ctx.msg(f"{res.data['skipped_edges']} edge(s) could not take that radius and were left sharp.")
        ctx.doc.replace(res, notify=False)
    ctx.doc.notify("changed")


@command("ChamferEdge", ("chamfer", "ce"), spaces=("3d",), group="Solid", help="Bevel all edges of a NURBS solid.")
def cmd_chamfer(ctx):
    objs = yield GetObjects("Select solids to chamfer", filter=_srf)
    d = yield GetNumber("Chamfer distance", default=1.0)
    ctx.commit("ChamferEdge")
    for g in objs:
        ctx.doc.replace(_B().chamfer(g, abs(d), id=g.id, layer=g.layer, space=g.space), notify=False)
    ctx.doc.notify("changed")


@command("Intersect", ("int",), spaces=("3d",), group="Curve", help="Exact intersection curves between surfaces/solids/curves.")
def cmd_intersect(ctx):
    a = yield GetObjects("Select first object", filter=lambda g: _srf(g) or g.is_curve, single=True)
    b = yield GetObjects("Select second object", filter=lambda g: (_srf(g) or g.is_curve) and g is not a, single=True, preselect=False)
    crvs = _B().intersect(a, b)
    if not crvs:
        ctx.msg("No intersection.")
        return
    ctx.commit("Intersect")
    ctx.new(crvs)
    ctx.msg(f"{len(crvs)} intersection curve(s).")


@command("DupEdge", ("dupedge", "de"), spaces=("3d",), group="Curve", help="Extract the exact edge curves of surfaces/solids.")
def cmd_dupedge(ctx):
    objs = yield GetObjects("Select surfaces or solids", filter=_srf)
    out = []
    for g in objs:
        out += _B().edge_curves(g)
    ctx.commit("DupEdge")
    ctx.new(out)
    ctx.msg(f"{len(out)} edge curves.")


@command("JoinSrf", ("joinsrf", "js"), spaces=("3d",), group="Surface", help="Join (sew) surfaces into a polysurface or closed solid.")
def cmd_joinsrf(ctx):
    objs = yield GetObjects("Select surfaces to join", filter=_srf)
    res = _B().join(objs, layer=objs[0].layer)
    ctx.commit("JoinSrf")
    ctx.doc.remove([g.id for g in objs], notify=False)
    ctx.new(res)
    ctx.msg(res.describe())


@command("Cap", (), spaces=("3d",), group="Surface", help="Close planar openings to make a solid.")
def cmd_cap(ctx):
    objs = yield GetObjects("Select open polysurfaces", filter=_srf)
    ctx.commit("Cap")
    for g in objs:
        ctx.doc.replace(_B().cap(g, id=g.id, layer=g.layer, space=g.space), notify=False)
    ctx.doc.notify("changed")


@command("MeshFromNurbs", ("mesh",), spaces=("3d",), group="Mesh", help="Make a mesh copy of NURBS surfaces/solids.")
def cmd_mesh(ctx):
    objs = yield GetObjects("Select surfaces or solids", filter=_srf)
    ctx.commit("Mesh")
    ctx.new([_B().to_mesh(g) for g in objs])


@command("Rebuild", ("rb",), group="Curve", help="Rebuild curves as smooth NURBS with a chosen point count and degree.")
def cmd_rebuild(ctx):
    objs = yield GetObjects("Select curves", filter=lambda g: g.is_curve)
    n = yield GetNumber("Point count", default=10, integer=True)
    d = yield GetNumber("Degree", default=3, integer=True)
    ctx.commit("Rebuild")
    for g in objs:
        pts = g.resample(max(int(n), int(d) + 1))
        if g.closed:
            pts[-1] = pts[0]
        ctx.doc.replace(G.interp_curve(pts, int(d), id=g.id, layer=g.layer, space=g.space), notify=False)
    ctx.doc.notify("changed")


@command("ConvertToNurbs", ("tonurbs",), group="Curve", help="Convert lines, polylines, arcs and circles to exact NURBS curves.")
def cmd_tonurbs(ctx):
    objs = yield GetObjects("Select curves", filter=lambda g: g.is_curve and g.kind != "nurbs")
    ctx.commit("ConvertToNurbs")
    for g in objs:
        n = _B().exact_nurbs(g)
        ctx.doc.replace(G.Geom("nurbs", g.space, g.layer, g.name, g.id, **n.data), notify=False)
    ctx.doc.notify("changed")


@command("Make2D", ("m2d",), spaces=("3d",), group="Drafting", help="Project 3D objects into the Draft (2D) workspace.")
def cmd_make2d(ctx):
    objs = yield GetObjects("Select objects to project")
    view = yield GetChoice("View", ["Top", "Front", "Right"], default="Top")
    flat = []
    for g in objs:
        flat += ctx.doc.expand(g)
    res = G.project_to_plane_2d(flat, view.lower())
    ctx.commit("Make2D")
    lay = ctx.doc.add_layer(f"Make2D {view}", "#e0b347")
    lo, hi = _bbox_of(ctx.doc, objs)
    off = G.translation((0, -(hi[1] - lo[1]) * 1.3 - 1, 0)) if view.lower() == "top" else np.eye(4)
    res = [r.transformed(off) for r in res]
    for r in res:
        r.layer = lay.name
    ctx.doc.add(res)
    ctx.msg(f"Make2D: {len(res)} curves on layer '{lay.name}', placed beside the model (no hidden-line removal yet).")


@command("SendTo3D", ("s3",), spaces=("2d",), group="Drafting", help="Copy 2D drawing into the 3D model at Z=0.")
def cmd_send3d(ctx):
    objs = yield GetObjects("Select 2D objects to copy into 3D")
    ctx.commit("SendTo3D")
    new = [g.copy(keep_id=False, space="3d") for g in objs]
    ctx.doc.add(new)
    ctx.msg(f"{len(new)} objects copied to the Model workspace.")


@command("SendTo2D", ("s2",), spaces=("3d",), group="Drafting", help="Copy 3D curves flat into the Draft workspace.")
def cmd_send2d(ctx):
    objs = yield GetObjects("Select curves", filter=lambda g: g.is_curve or g.kind == "point")
    ctx.commit("SendTo2D")
    new = G.project_to_plane_2d(objs, "top")
    ctx.doc.add(new)


# ======================================================================= control points
def _editable(g):
    from . import editpoints as E
    return E.grips(g) is not None


@command("PointsOn", ("pon", "pointson", "editpoints"), group="Points",
         help="Show control points of curves, meshes and single untrimmed surfaces (F10).")
def cmd_pointson(ctx):
    objs = yield GetObjects("Select objects to show control points", filter=_editable)
    ctx.doc.set_points_on([g.id for g in objs], True)
    ctx.doc.clear_selection()
    n = len(ctx.doc.grip_points())
    ctx.msg(f"{n} control points shown. Click or drag a box to select points (Shift adds), then drag them "
            f"or use the gumball. SelAllPoints selects all; PointsOff (F11) hides them.")


@command("PointsOff", ("poff", "pointsoff"), group="Points", help="Hide all control points (F11).")
def cmd_pointsoff(ctx):
    ctx.doc.set_points_on(list(ctx.doc.points_on), False)
    ctx.doc.grip_sel = set()
    ctx.doc.notify("grips")
    return
    yield


@command("SelAllPoints", ("selpts", "selallpoints"), group="Points", help="Select every visible control point.")
def cmd_selallpoints(ctx):
    ctx.doc.select_grips([(o, i) for o, i, _ in ctx.doc.grip_points()])
    ctx.msg(f"{len(ctx.doc.grip_sel)} control points selected.")
    return
    yield


@command("GumballValue", (), group="Transform", help="Exact move/rotate/scale along a gumball axis (click a gumball handle).")
def cmd_gumball_value(ctx, kind="move", axis=(1, 0, 0), center=(0, 0, 0)):
    ax = np.asarray(axis, float)
    c = np.asarray(center, float)
    name = "XYZ"[int(np.argmax(np.abs(ax)))]
    if kind == "move":
        d = yield GetNumber(f"Move distance along {name}", default=0.0,
                            preview=lambda v: ctx.doc.preview_targets(G.translation(ax * v)))
        M = G.translation(ax * d)
    elif kind == "rotate":
        d = yield GetNumber(f"Rotation angle around {name} (degrees)", default=90.0,
                            preview=lambda v: ctx.doc.preview_targets(G.rotation(v, ax, c)))
        M = G.rotation(d, ax, c)
    else:
        d = yield GetNumber(f"Scale factor along {name}", default=2.0)
        S = np.eye(3) + (d - 1) * np.outer(ax, ax)
        M = np.eye(4)
        M[:3, :3] = S
        M[:3, 3] = c - S @ c
    if ctx.doc.transform_targets(M, f"Gumball {kind}") == 0:
        ctx.msg("Nothing selected.")


# ======================================================================= components
def _is_inst(g):
    return g.kind == "instance"


@command("MakeComponent", ("mc", "block"), group="Components", help="Turn selected objects into a reusable component.")
def cmd_makecomponent(ctx):
    objs = yield GetObjects("Select objects for the component", filter=lambda g: not g.data.get("_edit"))
    name = yield GetString("Component name", default=f"Component {len(ctx.doc.definitions) + 1}")
    bp = yield GetPoint("Base (insertion) point (Enter = bottom-left corner)", allow_none=True)
    if bp is None:
        lo, _ = _bbox_of(ctx.doc, objs)
        bp = lo
    ctx.commit("MakeComponent")
    inst = ctx.doc.make_component([g.id for g in objs], name or "Component", bp)
    ctx.msg(f"Component '{name}' created. Edit it with EditComponent; every copy updates.")


@command("Insert", ("i", "place"), group="Components", help="Place component instances. Keeps placing until Enter.")
def cmd_insert(ctx, def_id=None, params=None):
    if def_id is None:
        names = {d.name: d.id for d in ctx.doc.definitions.values()}
        if not names:
            ctx.msg("No components in this document yet. Use the Components workspace library.")
            return
        nm = yield GetChoice("Component to insert", sorted(names), default=sorted(names)[0])
        def_id = names[nm]
    d = ctx.doc.definitions[def_id]
    rot = [0.0]

    def make(p):
        M = G.translation(p) @ G.rotation(rot[0])
        return C.instance(def_id, M, params=params, layer=ctx.doc.current_layer)

    first = True
    while True:
        p = yield GetPoint(f"Insertion point for '{d.name}' (R = rotate 90°, Enter to finish)", allow_none=True,
                           options=["R"], preview=lambda p: ctx.doc.expand(make(p)) if True else [])
        if p is None:
            return
        if isinstance(p, str):
            if p.upper() == "R":
                rot[0] = (rot[0] + 90) % 360
            continue
        if first:
            ctx.commit("Insert")
            first = False
        ctx.new(make(p))


@command("EditComponent", ("ec", "edit"), group="Components", help="Edit a component in place; all copies update.")
def cmd_editcomponent(ctx):
    inst = yield GetObjects("Select component to edit", filter=_is_inst, single=True)
    d = ctx.doc.definitions.get(inst.def_id)
    if d is not None and d.parametric:
        ctx.msg(f"'{d.name}' is parametric: change its parameters in Properties, or edit its code in the Components workspace.")
        return
    ctx.commit("EditComponent")
    ctx.doc.begin_edit(inst.id)
    ctx.msg(f"Editing '{d.name}'. Other objects are locked. Run CloseComponent (or press the Close button) when done.")


@command("CloseComponent", ("cc",), group="Components", help="Finish editing a component and update all copies.")
def cmd_closecomponent(ctx):
    if not ctx.doc.editing:
        ctx.msg("Not editing a component.")
        return
    ctx.doc.end_edit(True)
    ctx.msg("Component updated everywhere.")
    return
    yield


@command("MakeUnique", ("mu",), group="Components", help="Give selected copies their own separate definition.")
def cmd_makeunique(ctx):
    objs = yield GetObjects("Select component copies to make unique", filter=_is_inst)
    ctx.commit("MakeUnique")
    for g in objs:
        ctx.doc.make_unique(g.id)


@command("Connect", ("cn", "snapto"), group="Components", help="Snap a component's connector onto another component's connector.")
def cmd_connect(ctx):
    inst = yield GetObjects("Select component to move", filter=_is_inst, single=True)
    defs = ctx.doc.definitions
    mine = C.world_connectors(inst, defs)
    if not mine:
        ctx.msg("That component has no connectors.")
        return
    p = yield GetPoint("Pick connector on the component to move (snaps to ◆ connectors)")
    ca = min(mine, key=lambda c: np.linalg.norm(c["p"] - p))
    others = []
    for g in ctx.doc.objects.values():
        if _is_inst(g) and g.id != inst.id and g.space == inst.space:
            others += [(g, c) for c in C.world_connectors(g, defs)]
    if not others:
        ctx.msg("No other components with connectors to connect to.")
        return
    spin = [0.0]

    def target_of(q):
        return min(others, key=lambda gc: np.linalg.norm(gc[1]["p"] - q))[1]

    def moved(q):
        M = C.connect_matrix(ca, target_of(q), spin[0])
        return _preview_xform(ctx.doc, [inst], M)

    while True:
        q = yield GetPoint("Pick target connector (R = spin 90°)", options=["R"], preview=moved)
        if isinstance(q, str):
            spin[0] = (spin[0] + 90) % 360
            continue
        break
    cb = target_of(q)
    ctx.commit("Connect")
    ctx.doc.replace(inst.transformed(C.connect_matrix(ca, cb, spin[0])))
    ctx.msg(f"Connected '{ca['name']}' → '{cb['name']}'.")


@command("Chain", ("ch",), group="Components", help="Keep adding copies of a component, each snapped to the last one's connector.")
def cmd_chain(ctx):
    inst = yield GetObjects("Select starting component", filter=_is_inst, single=True)
    defs = ctx.doc.definitions
    conns = C.world_connectors(inst, defs)
    if len(conns) < 2:
        ctx.msg("Chain needs a component with at least 2 connectors.")
        return
    names = [c["name"] for c in conns]
    out_name = yield GetChoice("Grow from which connector?", names, default=names[-1])
    n = yield GetNumber("How many copies?", default=5, integer=True)
    in_conn = {c["name"]: c for c in C.world_connectors(inst, defs)}
    # the opposite connector: whichever faces most opposite the outgoing one
    out_c = in_conn[out_name]
    back = min(conns, key=lambda c: np.dot(c["n"], out_c["n"]))
    ctx.commit("Chain")
    cur = inst
    for _ in range(int(n)):
        tgt = {c["name"]: c for c in C.world_connectors(cur, defs)}[out_name]
        new = cur.copy(keep_id=False)
        mov = {c["name"]: c for c in C.world_connectors(new, defs)}[back["name"]]
        new = new.transformed(C.connect_matrix(mov, tgt), keep_id=True)
        ctx.doc.add(new, notify=False)
        cur = new
    ctx.doc.notify("changed")
    ctx.msg(f"Chained {int(n)} copies.")


# ======================================================================= view (handled by app)
def _view_cmd(name, action, aliases=(), spaces=("2d", "3d"), help=""):
    def fn(ctx):
        if ctx.app is not None:
            ctx.app.view_action(action)
        return
        yield
    command(name, aliases, spaces, "View", help)(fn)


_view_cmd("ZoomExtents", "zoom_extents", ("ze", "zea"), help="Zoom to show everything.")
_view_cmd("ZoomSelected", "zoom_selected", ("zs",), help="Zoom to the selection.")
_view_cmd("Top", "view_top", (), ("3d",))
_view_cmd("Front", "view_front", (), ("3d",))
_view_cmd("Right", "view_right", (), ("3d",))
_view_cmd("Perspective", "view_persp", ("persp", "iso"), ("3d",))
_view_cmd("Shaded", "shade_shaded", ("sh",), ("3d",))
_view_cmd("Wireframe", "shade_wire", ("wf",), ("3d",))
_view_cmd("Ghosted", "shade_ghost", ("gh", "xray"), ("3d",))
_view_cmd("Grid", "toggle_grid", (), help="Toggle the grid.")
_view_cmd("Snap", "toggle_snap", (), help="Toggle grid snap (F9).")
_view_cmd("Ortho", "toggle_ortho", (), help="Toggle ortho (F8).")
_view_cmd("Osnap", "toggle_osnap", ("os",), help="Toggle object snaps (F3).")
