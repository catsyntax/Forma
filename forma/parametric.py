"""Parametric building generators: a stadium, a twisting diagrid tower and a free-form pavilion.

Each generator returns a list of (role, Geom) pairs. Surfaces are exact NURBS (lofts through interpolated
curves, via OpenCascade); structural members are light merged meshes so hundreds of them stay fast.
All dimensions are millimetres. Everything here is an original design driven by a few parameters.
"""
from __future__ import annotations

import math

import numpy as np

from . import geometry as G

TAU = 2 * math.pi


def _B():
    from . import brep
    return brep


# ---------------------------------------------------------------- helpers
def superellipse(a, b, n=2.6, count=64, z=0.0, phase=0.0, zfun=None):
    """Closed rounded-rectangle-like plan curve (exponent n: 2 = ellipse, larger = squarer)."""
    pts = []
    for t in np.linspace(0, TAU, count, endpoint=False) + phase:
        c, s = math.cos(t), math.sin(t)
        x = a * math.copysign(abs(c) ** (2 / n), c)
        y = b * math.copysign(abs(s) ** (2 / n), s)
        pts.append((x, y, z if zfun is None else zfun(t)))
    return pts


def closed_curve(pts):
    """Smooth closed NURBS through points (periodic-looking interpolation)."""
    pts = list(pts) + [pts[0]]
    return G.interp_curve(pts, degree=3)


def tubes(segments, radius, sides=6):
    """Many straight members as ONE mesh (fast to build and draw)."""
    V, F = [], []
    for a, b in segments:
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        L = np.linalg.norm(d)
        if L < 1e-9:
            continue
        t = d / L
        u, v = G.plane_axes(t)
        ring = [radius * (math.cos(k * TAU / sides) * u + math.sin(k * TAU / sides) * v) for k in range(sides)]
        base = len(V)
        V += [a + r for r in ring] + [b + r for r in ring]
        for k in range(sides):
            k2 = (k + 1) % sides
            F.append([base + k, base + k2, base + sides + k2, base + sides + k])
    if not V:
        return None
    m = G.mesh(np.array(V), F, source="Members")
    m.data["no_edges"] = True                       # draw as smooth tubes, no wireframe clutter
    m.data["axes"] = np.array([[np.asarray(a, float), np.asarray(b, float)] for a, b in segments])
    return m


def slabs(outlines, thickness):
    """Floor plates (closed point loops) as one mesh: top, bottom and edges."""
    V, F = [], []
    for loop in outlines:
        loop = np.asarray(loop, float)
        n = len(loop)
        base = len(V)
        V += list(loop) + list(loop - np.array([0, 0, thickness]))
        c = len(V)
        V.append(loop.mean(0))
        V.append(loop.mean(0) - np.array([0, 0, thickness]))
        for k in range(n):
            k2 = (k + 1) % n
            F.append([base + k, base + k2, c])                       # top fan
            F.append([base + n + k2, base + n + k, c + 1])           # bottom fan
            F.append([base + k2, base + k, base + n + k, base + n + k2])
    return G.mesh(np.array(V), F, source="Slabs")


# ---------------------------------------------------------------- stadium
def stadium(length=230000.0, width=180000.0, height=48000.0, tiers=3, ribs=56, roof_depth=52000.0,
            saddle=9000.0, squareness=2.6, bulge=9000.0):
    """Bowl of raked seating tiers, a saddle-shaped cantilever roof and a ribbed facade."""
    out = []
    a0, b0 = 52000.0, 34000.0                      # pitch half-size
    out.append(("pitch", _B().planar_surface(closed_curve(superellipse(a0, b0, 2.2, 48, z=0.0)))))
    # seating tiers: each a NURBS ruled band between two rings, raking upwards
    z, off = 1500.0, 6000.0
    tier_depth = (min(length, width) / 2 - b0 - 18000.0) / max(tiers, 1)
    for k in range(int(tiers)):
        inner = closed_curve(superellipse(a0 + off, b0 + off, squareness, 72, z=z))
        off2 = off + tier_depth
        z2 = z + tier_depth * 0.62
        outer = closed_curve(superellipse(a0 + off2, b0 + off2, squareness, 72, z=z2))
        out.append(("seating", _B().loft([inner, outer], ruled=True, solid=False)))
        # vomitory gap / concourse step
        off, z = off2 + 2500.0, z2 + 3200.0
    A, Bw = length / 2, width / 2
    # roof: saddle-shaped ring between the facade top and the inner roof edge
    def roof_z(t, base):
        return base + saddle * math.cos(2 * t)
    ring_out = superellipse(A, Bw, squareness, 96, zfun=lambda t: roof_z(t, height))
    ring_in = superellipse(A - roof_depth, Bw - roof_depth, squareness, 96, zfun=lambda t: roof_z(t, height + 6000.0))
    out.append(("roof", _B().loft([closed_curve(ring_out), closed_curve(ring_in)], solid=False)))
    # facade skin: ground ring -> bulging mid ring -> roof ring
    ground = superellipse(A - 6000.0, Bw - 6000.0, squareness, 96, z=0.0)
    mid = superellipse(A + bulge, Bw + bulge, squareness, 96, zfun=lambda t: roof_z(t, height) * 0.55)
    out.append(("facade", _B().loft([closed_curve(ground), closed_curve(mid), closed_curve(ring_out)], solid=False)))
    # ribs over the facade and radial roof trusses
    segs, roof_segs = [], []
    for i in range(int(ribs)):
        t = TAU * i / ribs
        g = np.array(superellipse(A - 6000.0, Bw - 6000.0, squareness, 1, z=0.0, phase=t)[0])
        m = np.array(superellipse(A + bulge + 1500.0, Bw + bulge + 1500.0, squareness, 1, phase=t,
                                  zfun=lambda tt: roof_z(tt, height) * 0.55)[0])
        r = np.array(superellipse(A + 1500.0, Bw + 1500.0, squareness, 1, phase=t, zfun=lambda tt: roof_z(tt, height) + 900)[0])
        tt_ = np.linspace(0, 1, 11)[:, None]
        rib = (1 - tt_) ** 2 * g + 2 * (1 - tt_) * tt_ * (2 * m - 0.5 * (g + r)) + tt_ ** 2 * r   # curve through g, m, r
        segs += list(zip(rib[:-1], rib[1:]))
        ri = np.array(superellipse(A - roof_depth, Bw - roof_depth, squareness, 1, phase=t,
                                   zfun=lambda tt: roof_z(tt, height + 6000.0) + 900)[0])
        roof_segs.append((r, ri))
    out.append(("ribs", tubes(segs, 650.0)))
    out.append(("trusses", tubes(roof_segs, 450.0)))
    return [(r, g) for r, g in out if g is not None]


# ---------------------------------------------------------------- tower
def diagrid_tower(height=240000.0, floors=60, radius=24000.0, twist=110.0, taper=0.62, lobes=3,
                  modules=10, columns=24, floor_step=4):
    """Twisting, tapering tower on a rounded-lobed plan, with a glass skin, floor plates and a diagrid."""
    def section(z, count=96, grow=0.0):
        f = z / height
        r = radius * (1 + 0.10 * math.sin(math.pi * f)) * (1 - (1 - taper) * f ** 1.4) + grow
        rot = math.radians(twist) * f
        pts = []
        for t in np.linspace(0, TAU, count, endpoint=False):
            rr = r * (1 + 0.12 * math.cos(lobes * t))
            pts.append((rr * math.cos(t + rot), rr * math.sin(t + rot), z))
        return pts
    out = []
    zs = np.linspace(0, height, 9)
    out.append(("glass", _B().loft([closed_curve(section(z, 48)) for z in zs])))
    fh = height / floors
    out.append(("slabs", slabs([section(k * fh, 48, grow=-600.0) for k in range(1, int(floors)) if k % floor_step == 0], 450.0)))
    out.append(("core", _B().cylinder((0, 0, 0), radius * 0.28, height * 1.03)))
    # diagrid: nodes on the skin, crossing diagonals between modules
    nz = int(modules)
    nodes = []
    for k in range(nz + 1):
        z = height * k / nz
        ring = section(z, int(columns), grow=900.0)
        if k % 2:
            ring = section(z, int(columns) * 2, grow=900.0)[1::2]   # offset half a bay on odd rows
        nodes.append([np.array(p) for p in ring])
    segs = []
    n = int(columns)
    for k in range(nz):
        lo, hi = nodes[k], nodes[k + 1]
        for i in range(n):
            if k % 2 == 0:
                segs += [(lo[i], hi[i]), (lo[(i + 1) % n], hi[i])]
            else:
                segs += [(lo[i], hi[(i + 1) % n]), (lo[i], hi[i])]
        segs += [(lo[i], lo[(i + 1) % n]) for i in range(n)]
    segs += [(nodes[-1][i], nodes[-1][(i + 1) % n]) for i in range(n)]
    out.append(("diagrid", tubes(segs, 700.0)))
    # crown: lofted ring beyond the top
    out.append(("crown", _B().loft([closed_curve(section(height, 48, grow=900.0)),
                                    closed_curve([(x * 0.8, y * 0.8, z + 9000.0) for x, y, z in section(height, 48)])],
                                   solid=False)))
    return [(r, g) for r, g in out if g is not None]


# ---------------------------------------------------------------- pavilion
def pavilion(length=110000.0, width=56000.0, height=19000.0, waves=3, sections=7, columns=5):
    """Free-form undulating roof shell lofted through arched sections, on funnel-shaped columns."""
    out = []
    nu, nv = max(int(sections), 3) * 2 + 1, 13
    grid = np.zeros((nu, nv, 3))
    for i in range(nu):
        f = i / (nu - 1)
        x = -length / 2 + f * length
        w = width / 2 * (0.75 + 0.25 * math.sin(math.pi * f))
        h = height * (0.65 + 0.35 * math.sin(TAU * waves * f / 2 + 0.6))
        dy = 0.18 * width * math.sin(TAU * f)
        for j, s_ in enumerate(np.linspace(-1, 1, nv)):
            z = h * (1 - 0.75 * s_ * s_) + 0.12 * height * math.sin(3 * s_ + TAU * f)
            grid[i, j] = (x + 0.06 * length * s_ * math.sin(math.pi * f), dy + s_ * w, z)
    out.append(("shell", _B().surface_through_grid(grid)))
    # funnel columns: lofted circles that flare into the roof
    for j in range(int(columns)):
        f = (j + 0.5) / columns
        x = -length / 2 + f * length
        y = 0.18 * width * math.sin(TAU * f) * 0.6
        h = height * (0.65 + 0.35 * math.sin(TAU * waves * f / 2 + 0.6)) * 0.97
        rings = [G.circle((x, y, 0.0), 700.0), G.circle((x, y, h * 0.55), 1400.0), G.circle((x, y, h * 0.92), 5200.0)]
        out.append(("columns", _B().loft(rings)))
    # plinth
    out.append(("ground", _B().planar_surface(closed_curve(superellipse(length / 2 + 15000, width / 2 + 15000, 2.4, 48, z=0.0)))))
    return [(r, g) for r, g in out if g is not None]
