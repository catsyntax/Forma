"""Read and write Rhino .3dm files through McNeel's open-source rhino3dm library."""
from __future__ import annotations

import math
import numpy as np

from . import geometry as G
from .document import Layer

try:
    import rhino3dm as r3
except ImportError:  # pragma: no cover
    r3 = None

PREFIX_2D = "2D::"


def _hex(c):
    return "#%02x%02x%02x" % tuple(int(x) for x in c[:3])


def _rgb(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)


def _p3(p):
    return r3.Point3d(float(p[0]), float(p[1]), float(p[2]))


def export_3dm(doc, path, objects=None):
    if r3 is None:
        raise RuntimeError("rhino3dm is not installed (pip install rhino3dm)")
    f = r3.File3dm()
    layer_index = {}

    def get_layer(name, space):
        key = (PREFIX_2D + name) if space == "2d" else name
        if key not in layer_index:
            src = doc.layer(name)
            L = r3.Layer()
            L.Name = key
            L.Color = _rgb(src.color if src else "#d0d0d0")
            if src is not None:
                L.Visible = src.visible
            layer_index[key] = f.Layers.Add(L)
        return layer_index[key]

    for l in doc.layers:
        get_layer(l.name, "3d")
    count = 0
    fallback_faces = [0]
    for g in (objects if objects is not None else doc.objects.values()):
        a = r3.ObjectAttributes()
        a.LayerIndex = get_layer(g.layer, g.space)
        if g.name:
            a.Name = g.name
        if g.kind == "brep":
            from . import brep as B
            objs, nfall = B.rhino_objects(g, r3)
            fallback_faces[0] += nfall
            if len(objs) > 1:  # faces of one solid go in a group; Join in Rhino (or re-import here) reunites them
                grp = r3.Group()
                grp.Name = f"Forma {g.name or g.data.get('source', 'solid')} {g.id}-{count}"
                f.Groups.Add(grp)
                a.AddToGroup(len(f.Groups) - 1)
            for o in objs:
                (f.Objects.AddBrep if isinstance(o, r3.Brep) else f.Objects.AddMesh)(o, a)
            count += 1
            continue
        geo = to_rhino(g)
        if geo is None:
            continue
        if isinstance(geo, r3.Point):
            f.Objects.AddPoint(geo.Location, a)
        elif isinstance(geo, r3.Mesh):
            f.Objects.AddMesh(geo, a)
        elif isinstance(geo, r3.Circle):
            f.Objects.AddCircle(geo, a)
        elif isinstance(geo, r3.Arc):
            f.Objects.AddArc(geo, a)
        else:
            f.Objects.AddCurve(geo, a)
        count += 1
    f.Write(path, 8)
    export_3dm.last_fallback_faces = fallback_faces[0]
    return count


def to_rhino(g):
    k = g.kind
    if k == "point":
        return r3.Point(_p3(g.p))
    if k == "polyline":
        pl = r3.Point3dList()
        for p in g.pts:
            pl.Add(*map(float, p))
        if g.closed:
            pl.Add(*map(float, g.pts[0]))
        if len(g.pts) == 2 and not g.closed:
            return r3.LineCurve(_p3(g.pts[0]), _p3(g.pts[1]))
        return r3.PolylineCurve(pl)
    if k == "ellipse":
        ru, rv = np.linalg.norm(g.u), np.linalg.norm(g.v)
        ortho = abs(np.dot(g.u, g.v)) < 1e-9 * max(ru * rv, 1)
        if ortho and abs(ru - rv) < 1e-9:
            try:
                if g.closed:
                    c = r3.Circle(_p3(g.c), float(ru))
                    xa, ya = g.u / ru, g.v / rv
                    c.Plane = r3.Plane(_p3(g.c), r3.Vector3d(*map(float, xa)), r3.Vector3d(*map(float, ya)))
                    return c
                s = g.sample()
                return r3.Arc(_p3(s[0]), _p3(g.point_at(0.5)), _p3(s[-1]))
            except Exception:
                pass
        # general ellipse: write as interpolated nurbs
        return _nurbs_to_rhino(G.to_nurbs_cps(g))
    if k == "nurbs":
        return _nurbs_to_rhino(g)
    if k == "mesh":
        m = r3.Mesh()
        for v in g.V:
            m.Vertices.Add(*map(float, v))
        for face in g.mesh_faces():
            if len(face) == 3:
                m.Faces.AddFace(*face)
            elif len(face) == 4:
                m.Faces.AddFace(*face)
            else:
                for j in range(1, len(face) - 1):
                    m.Faces.AddFace(face[0], face[j], face[j + 1])
        m.Normals.ComputeNormals()
        m.Compact()
        return m
    return None


def _nurbs_to_rhino(g):
    n = len(g.cps)
    rational = not np.allclose(g.w, 1)
    c = r3.NurbsCurve(3, rational, g.deg + 1, n)
    for i, (p, w) in enumerate(zip(g.cps, g.w)):
        c.Points[i] = r3.Point4d(float(p[0] * w), float(p[1] * w), float(p[2] * w), float(w))
    U = g.knots[1:-1]  # Rhino omits the superfluous end knots
    for i in range(len(c.Knots)):
        c.Knots[i] = float(U[i])
    return c


def import_3dm(path):
    """Return (layers, geoms, warnings)."""
    if r3 is None:
        raise RuntimeError("rhino3dm is not installed (pip install rhino3dm)")
    f = r3.File3dm.Read(path)
    if f is None:
        raise IOError(f"Could not read {path}")
    layers, names = [], []
    for L in f.Layers:
        name = L.FullPath if hasattr(L, "FullPath") and L.FullPath else L.Name
        names.append(name)
    out, warnings = [], []
    seen_layers = {}
    for o in f.Objects:
        idx = o.Attributes.LayerIndex
        lname = names[idx] if 0 <= idx < len(names) else "Default"
        space = "3d"
        if lname.startswith(PREFIX_2D):
            space, lname = "2d", lname[len(PREFIX_2D):]
        if lname not in seen_layers:
            src = f.Layers[idx] if 0 <= idx < len(names) else None
            seen_layers[lname] = Layer(lname, _hex(src.Color) if src else "#d0d0d0",
                                       bool(src.Visible) if src else True)
        name = o.Attributes.Name or ""
        try:
            gs = from_rhino(o.Geometry)
        except Exception as e:  # keep going on bad objects
            warnings.append(f"Skipped {type(o.Geometry).__name__}: {e}")
            continue
        if not gs:
            warnings.append(f"Unsupported object type: {type(o.Geometry).__name__}")
        groups = list(o.Attributes.GetGroupList() or []) if o.Attributes.GroupCount else []
        for g in gs:
            g.layer, g.space, g.name = lname, space, name
            if groups and g.kind == "brep":
                g.data["_group"] = groups[0]
            out.append(g)
    out = _rejoin_groups(out)
    layers = list(seen_layers.values())
    return layers, out, warnings


def from_rhino(geo):
    if isinstance(geo, r3.Point):
        p = geo.Location
        return [G.point((p.X, p.Y, p.Z))]
    if isinstance(geo, r3.PointCloud):
        return [G.point((p.X, p.Y, p.Z)) for p in geo.GetPoints()] if hasattr(geo, "GetPoints") else []
    if isinstance(geo, r3.Curve):
        if isinstance(geo, r3.LineCurve):
            a, b = geo.PointAtStart, geo.PointAtEnd
            return [G.line((a.X, a.Y, a.Z), (b.X, b.Y, b.Z))]
        if isinstance(geo, r3.PolylineCurve):
            pts = [geo.Point(i) for i in range(geo.PointCount)]
            pts = [(p.X, p.Y, p.Z) for p in pts]
            closed = len(pts) > 2 and np.allclose(pts[0], pts[-1])
            return [G.polyline(pts[:-1] if closed else pts, closed=closed)]
        if isinstance(geo, r3.ArcCurve):
            arc = geo.Arc
            pl = arc.Plane
            c = (pl.Origin.X, pl.Origin.Y, pl.Origin.Z)
            u = np.array([pl.XAxis.X, pl.XAxis.Y, pl.XAxis.Z]) * arc.Radius
            v = np.array([pl.YAxis.X, pl.YAxis.Y, pl.YAxis.Z]) * arc.Radius
            dom = arc.AngleDomain
            return [G.Geom("ellipse", c=G.P(c), u=u, v=v, a0=float(dom.T0), a1=float(dom.T1))]
        nc = geo.ToNurbsCurve()
        return [_nurbs_from_rhino(nc)]
    if isinstance(geo, r3.Mesh):
        return [_mesh_from_rhino(geo)]
    if isinstance(geo, r3.Extrusion):
        b = geo.ToBrep(False) if hasattr(geo, "ToBrep") else None
        if b is not None:
            return from_rhino(b)
        m = geo.GetMesh(r3.MeshType.Any)
        return [_mesh_from_rhino(m, "Extrusion")] if m is not None else []
    if isinstance(geo, r3.Brep):
        exact = _brep_exact(geo)
        if exact is not None:
            return exact
        meshes = []
        for i in range(len(geo.Faces)):
            face = geo.Faces[i]
            m = face.GetMesh(r3.MeshType.Any)
            if m is not None:
                meshes.append(_mesh_from_rhino(m, "Brep"))
            else:
                meshes.append(_sample_surface(face))
        if not meshes:
            return []
        return [_merge_meshes(meshes, "Polysurface")]
    if isinstance(geo, r3.Surface):
        return [_sample_surface(geo)]
    return []


def _rejoin_groups(geoms):
    """Faces exported as a group (one solid) are sewn back into one polysurface/solid."""
    groups, rest = {}, []
    for g in geoms:
        k = g.data.pop("_group", None) if g.kind == "brep" else None
        (groups.setdefault(k, []) if k is not None else rest).append(g)
    if not groups:
        return geoms
    from . import brep as B
    for k, gs in groups.items():
        if len(gs) == 1:
            rest.append(gs[0])
            continue
        try:
            j = B.join(gs, source="Imported")
            j.layer, j.space, j.name = gs[0].layer, gs[0].space, gs[0].name
            rest.append(j)
        except Exception:
            rest += gs
    return rest


def _brep_exact(geo):
    """Exact NURBS import for breps whose faces are untrimmed (boxes, extrusions, revolves, lofts...).
    Returns None if any face is trimmed, so the caller falls back to render meshes."""
    try:
        from . import brep as B
    except ImportError:
        return None
    faces = []
    for i in range(len(geo.Faces)):
        face = geo.Faces[i]
        dup = face.DuplicateFace(False)
        if dup is not None and dup.IsSurface:
            faces.append(B.from_rhino_surface(face))
            continue
        if face.IsPlanar() and len(face.Loops) >= 1:
            loops = []
            for li in range(len(face.Loops)):
                loop = face.Loops[li]
                curves = []
                for ti in range(loop.TrimCount):
                    ei = loop.Trims[ti].EdgeIndex
                    if ei < 0:
                        continue
                    curves.append(_nurbs_from_rhino(geo.Edges[ei].ToNurbsCurve()))
                if not curves:
                    return None
                j = G.join_curves(curves, tol=1e-4) if len(curves) > 1 else curves
                if len(j) != 1 or not j[0].closed:
                    if len(curves) == 1 and curves[0].closed:
                        j = curves
                    else:
                        return None
                loops.append(j[0])
            faces.append(B.planar_surface(loops))
            continue
        return None
    if not faces:
        return None
    try:
        return [B.join(faces, source="Imported")]
    except Exception:
        return faces


def _nurbs_from_rhino(nc):
    n = len(nc.Points)
    deg = nc.Degree
    cps, w = [], []
    for i in range(n):
        p = nc.Points[i]
        W = p.W if nc.IsRational else 1.0
        cps.append((p.X / W, p.Y / W, p.Z / W))
        w.append(W)
    K = [nc.Knots[i] for i in range(len(nc.Knots))]
    U = np.array([K[0]] + K + [K[-1]], float)
    U = (U - U[0]) / ((U[-1] - U[0]) or 1)
    return G.Geom("nurbs", cps=np.array(cps), w=np.array(w), knots=U, deg=deg)


def _mesh_from_rhino(m, source="Mesh"):
    V = [(v.X, v.Y, v.Z) for v in m.Vertices]
    F = []
    for i in range(len(m.Faces)):
        f = m.Faces[i]
        f = list(f)
        F.append(f[:3] if f[2] == f[3] else f)
    return G.mesh(V, F, source=source)


def _sample_surface(s, n=16):
    du, dv = s.Domain(0), s.Domain(1)
    grid = np.zeros((n + 1, n + 1, 3))
    for i, u in enumerate(np.linspace(du.T0, du.T1, n + 1)):
        for j, v in enumerate(np.linspace(dv.T0, dv.T1, n + 1)):
            p = s.PointAt(u, v)
            grid[i, j] = (p.X, p.Y, p.Z)
    return G.grid_mesh(grid, source="Surface (untrimmed)")


def _merge_meshes(ms, source):
    V, F, off = [], [], 0
    for m in ms:
        V.extend(m.V)
        F.extend([[i + off for i in f] for f in m.mesh_faces()])
        off += len(m.V)
    return G.mesh(V, F, source=source)
