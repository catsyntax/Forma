"""Environmental and spatial analysis nodes: sun position, sun hours and radiation, a simple
energy balance, and space syntax (visibility graph, isovists).

Everything here is plain numpy so it runs in the compute worker and can be cancelled.
Results come back as coloured meshes (per-face colours in data["fcolors"]) that the 3D view
draws as a heat map, plus the raw numbers for further nodes.
"""
from __future__ import annotations

import math

import numpy as np

from .. import geometry as G
from ..geometry import Geom
from .registry import node, Port, flatten, vec, num

UNIT_TO_M = {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254, "ft": 0.3048}


# ===================================================================== colour helpers
_RAMP = np.array([[0.19, 0.30, 0.85], [0.10, 0.65, 0.90], [0.20, 0.80, 0.45],
                  [0.98, 0.85, 0.20], [0.95, 0.45, 0.15], [0.85, 0.12, 0.12]])


def ramp(values, lo=None, hi=None):
    v = np.asarray(values, float)
    lo = float(np.nanmin(v)) if lo is None else lo
    hi = float(np.nanmax(v)) if hi is None else hi
    t = np.clip((v - lo) / (hi - lo if hi > lo else 1.0), 0, 1) * (len(_RAMP) - 1)
    i = np.minimum(t.astype(int), len(_RAMP) - 2)
    f = (t - i)[:, None]
    return _RAMP[i] * (1 - f) + _RAMP[i + 1] * f


def colored_mesh(V, T, values, lo=None, hi=None, label="", offset=None):
    """A mesh whose faces carry heat-map colours. Vertices are unshared so every face has its own colour."""
    V = np.asarray(V, float)
    T = np.asarray(T, int).reshape(-1, 3)
    P = V[T].reshape(-1, 3)
    if offset is not None:
        P = P + np.repeat(offset, 3, axis=0)
    F = np.arange(len(P)).reshape(-1, 3)
    vals = np.asarray(values, float)
    g = G.mesh(P, F.tolist(), source="Analysis")
    g.data["fcolors"] = ramp(vals, lo, hi)
    g.data["values"] = vals
    lo_ = float(np.nanmin(vals)) if lo is None else lo
    hi_ = float(np.nanmax(vals)) if hi is None else hi
    g.data["legend"] = f"{label}: {lo_:.4g} to {hi_:.4g}"
    return g


# ===================================================================== geometry helpers
def triangles_of(geoms):
    """All triangles of the given surfaces / solids / meshes as (V, T)."""
    Vs, Ts, n = [], [], 0
    for g in flatten(geoms if geoms is not None else []):
        if not isinstance(g, Geom) or g.kind not in ("brep", "mesh"):
            continue
        if g.kind == "brep":
            from .. import brep as B
            V, _, T = B.tessellate(g)[:3]
        else:
            V, T = g.V, g.triangles()
        V = np.asarray(V, float)
        T = np.asarray(T, int).reshape(-1, 3)
        if not len(T):
            continue
        Vs.append(V)
        Ts.append(T + n)
        n += len(V)
    if not Vs:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    return np.vstack(Vs), np.vstack(Ts)


def subdivide(V, T, max_edge, max_faces=12000):
    """Split triangles until every edge is shorter than max_edge (analysis grid size)."""
    V = [tuple(p) for p in np.asarray(V, float)]
    index = {p: i for i, p in enumerate(V)}
    T = [tuple(t) for t in np.asarray(T, int)]

    def mid(a, b):
        p = tuple((np.array(V[a]) + np.array(V[b])) / 2)
        i = index.get(p)
        if i is None:
            i = len(V)
            V.append(p)
            index[p] = i
        return i

    for _ in range(12):
        if len(T) * 4 > max_faces:
            break
        A = np.array(V)
        tt = np.array(T)
        e = np.max(np.stack([np.linalg.norm(A[tt[:, 0]] - A[tt[:, 1]], axis=1),
                             np.linalg.norm(A[tt[:, 1]] - A[tt[:, 2]], axis=1),
                             np.linalg.norm(A[tt[:, 2]] - A[tt[:, 0]], axis=1)]), axis=0)
        if e.max() <= max_edge:
            break
        out = []
        for (a, b, c), el in zip(T, e):
            if el <= max_edge:
                out.append((a, b, c))
                continue
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            out += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        T = out
    return np.array(V), np.array(T, int)


def face_frames(V, T):
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    n = np.cross(b - a, c - a)
    area2 = np.linalg.norm(n, axis=1)
    n = n / np.maximum(area2, 1e-18)[:, None]
    return (a + b + c) / 3, n, area2 / 2


def occluded(origins, direction, V, T, chunk=256):
    """True where a ray from origin along direction hits any triangle (Moller-Trumbore, vectorised)."""
    origins = np.asarray(origins, float)
    hit = np.zeros(len(origins), bool)
    if not len(T):
        return hit
    d = np.asarray(direction, float)
    v0, v1, v2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    e1, e2 = v1 - v0, v2 - v0
    p = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    v0, e1, e2, p, det = v0[ok], e1[ok], e2[ok], p[ok], det[ok]
    inv = 1.0 / det
    for s in range(0, len(origins), chunk):
        o = origins[s:s + chunk]
        tv = o[:, None, :] - v0[None, :, :]
        u = np.einsum("ijk,jk->ij", tv, p) * inv
        q = np.cross(tv, e1[None, :, :])
        v = np.einsum("ijk,k->ij", q, d) * inv
        t = np.einsum("ijk,jk->ij", q, e2) * inv
        h = (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
        hit[s:s + chunk] = h.any(axis=1)
    return hit


# ===================================================================== sun
def sun_position(lat, lon, month, day, hour, tz):
    """Solar altitude and azimuth in degrees (NOAA approximation, good to about 0.5 degree).
    Azimuth is clockwise from north. hour is local clock time; tz is the UTC offset in hours."""
    import datetime
    month = min(12, max(1, int(month)))
    day = min(28 if month == 2 else 30 if month in (4, 6, 9, 11) else 31, max(1, int(day)))
    doy = datetime.date(2025, month, day).timetuple().tm_yday
    g = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                    - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    tst = hour * 60 + eqt + 4 * lon - 60 * tz
    ha = math.radians(tst / 4 - 180)
    phi = math.radians(lat)
    cos_z = math.sin(phi) * math.sin(decl) + math.cos(phi) * math.cos(decl) * math.cos(ha)
    zen = math.acos(max(-1, min(1, cos_z)))
    alt = 90 - math.degrees(zen)
    az = math.degrees(math.atan2(math.sin(ha), math.cos(ha) * math.sin(phi) - math.tan(decl) * math.cos(phi))) + 180
    return alt, az % 360


def sun_vector(alt, az, north=0.0):
    """Unit vector pointing to the sun. +Y is north unless 'north' (degrees, clockwise) rotates it."""
    a, z = math.radians(alt), math.radians(az - north)
    return np.array([math.cos(a) * math.sin(z), math.cos(a) * math.cos(z), math.sin(a)])


def _hours(start, end, step):
    step = max(0.1, float(step))
    return list(np.arange(float(start), float(end) + 1e-9, step))


def clear_sky_dni(alt):
    """Direct normal irradiance (W/m2) from a simple clear-sky model."""
    if alt <= 0:
        return 0.0
    am = 1 / (math.sin(math.radians(alt)) + 0.50572 * (alt + 6.07995) ** -1.6364)
    return 1353 * 0.7 ** (am ** 0.678)


@node("Sun Vector", "Climate", [Port("Latitude", -34.93), Port("Longitude", 138.6), Port("Month", 12),
      Port("Day", 21), Port("Hour", 12.0), Port("Time Zone", 9.5, hint="standard time, no daylight saving"), Port("North", 0.0, hint="degrees clockwise")],
      outputs=("Vector", "Altitude", "Azimuth"),
      desc="Direction to the sun for a place and time. Default is Adelaide at midsummer noon.")
def n_sun_vector(Latitude=0, Longitude=0, Month=6, Day=21, Hour=12, **kw):
    alt, az = sun_position(num(Latitude), num(Longitude), int(num(Month)), int(num(Day)), num(Hour), num(kw.get("Time Zone"), 0))
    return {"Vector": sun_vector(alt, az, num(kw.get("North"), 0)), "Altitude": alt, "Azimuth": az}


@node("Sun Path", "Climate", [Port("Latitude", -34.93), Port("Longitude", 138.6), Port("Time Zone", 9.5, hint="standard time, no daylight saving"),
      Port("Center", (0, 0, 0)), Port("Radius", 20000.0), Port("North", 0.0)], outputs=("Paths", "Hour Points"),
      desc="Sun path diagram: daily arcs for the 21st of each solstice/equinox month, and hour points.")
def n_sun_path(Latitude=0, Longitude=0, Center=None, Radius=20000, **kw):
    c, R = vec(Center), num(Radius, 20000)
    tz, north = num(kw.get("Time Zone"), 0), num(kw.get("North"), 0)
    paths, pts = [], []
    for m in (3, 6, 9, 12):
        line = []
        for h in np.arange(0, 24.01, 0.25):
            alt, az = sun_position(num(Latitude), num(Longitude), m, 21, h, tz)
            if alt > 0:
                line.append(c + sun_vector(alt, az, north) * R)
        if len(line) > 1:
            paths.append(G.polyline(line))
    for h in range(4, 21):
        for m in (6, 12):
            alt, az = sun_position(num(Latitude), num(Longitude), m, 21, h, tz)
            if alt > 0:
                pts.append(G.point(c + sun_vector(alt, az, north) * R))
    paths.append(G.circle(c, R))
    return {"Paths": paths, "Hour Points": pts}


@node("Sun Exposure", "Daylight",
      [Port("Surfaces", None, "list"), Port("Context", None, "list"), Port("Latitude", -34.93), Port("Longitude", 138.6),
       Port("Time Zone", 9.5, hint="standard time, no daylight saving"), Port("Month", 6), Port("Day", 21), Port("From Hour", 8.0), Port("To Hour", 17.0),
       Port("Step", 1.0, hint="hours"), Port("Grid Size", 1000.0, hint="model units"), Port("North", 0.0)],
      outputs=("Mesh", "Sun Hours", "Radiation", "Legend"),
      desc="Direct sun hours and clear-sky radiation (kWh/m2) on surfaces for one day, with shading from "
           "the surfaces themselves and any context geometry.")
def n_sun_exposure(Surfaces=None, Context=None, Latitude=0, Longitude=0, Month=6, Day=21, _progress=None, **kw):
    V, T = triangles_of(Surfaces)
    if not len(T):
        raise ValueError("Connect surfaces or solids to analyse")
    V, T = subdivide(V, T, num(kw.get("Grid Size"), 1000))
    cen, nrm, area = face_frames(V, T)
    oV, oT = triangles_of(flatten(Surfaces) + flatten(Context or []))
    diag = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
    origins = cen + nrm * diag * 1e-4
    hours = _hours(num(kw.get("From Hour"), 8), num(kw.get("To Hour"), 17), num(kw.get("Step"), 1))
    step = max(0.1, num(kw.get("Step"), 1))
    sun_h = np.zeros(len(T))
    rad = np.zeros(len(T))
    for k, h in enumerate(hours):
        alt, az = sun_position(num(Latitude), num(Longitude), int(num(Month)), int(num(Day)), h, num(kw.get("Time Zone"), 0))
        if alt > 0:
            s = sun_vector(alt, az, num(kw.get("North"), 0))
            facing = nrm @ s
            lit = facing > 0
            idx = np.where(lit)[0]
            if len(idx):
                blocked = occluded(origins[idx], s, oV, oT)
                ok = idx[~blocked]
                sun_h[ok] += step
                rad[ok] += clear_sky_dni(alt) * facing[ok] * step / 1000.0
        if _progress:
            _progress((k + 1) / len(hours))
    off = nrm * diag * 2e-4
    return {"Mesh": colored_mesh(V, T, sun_h, 0, max(sun_h.max(), 1e-9), "Sun hours", off),
            "Sun Hours": sun_h.tolist(), "Radiation": rad.tolist(),
            "Legend": f"Sun hours 0 to {sun_h.max():.1f} h; radiation up to {rad.max():.2f} kWh/m2 (clear sky, direct)"}


@node("Radiation Map", "Daylight",
      [Port("Surfaces", None, "list"), Port("Context", None, "list"), Port("Latitude", -34.93), Port("Longitude", 138.6),
       Port("Time Zone", 9.5, hint="standard time, no daylight saving"), Port("Grid Size", 1000.0), Port("North", 0.0)],
      outputs=("Mesh", "kWh per m2", "Legend"),
      desc="Approximate annual direct radiation: 12 representative days, hourly, clear sky, with shading.")
def n_radiation(Surfaces=None, Context=None, Latitude=0, Longitude=0, _progress=None, **kw):
    V, T = triangles_of(Surfaces)
    if not len(T):
        raise ValueError("Connect surfaces or solids to analyse")
    V, T = subdivide(V, T, num(kw.get("Grid Size"), 1000), max_faces=6000)
    cen, nrm, area = face_frames(V, T)
    oV, oT = triangles_of(flatten(Surfaces) + flatten(Context or []))
    diag = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
    origins = cen + nrm * diag * 1e-4
    tot = np.zeros(len(T))
    steps = [(m, h) for m in range(1, 13) for h in range(5, 21)]
    for k, (m, h) in enumerate(steps):
        alt, az = sun_position(num(Latitude), num(Longitude), m, 15, h + 0.5, num(kw.get("Time Zone"), 0))
        if alt > 2:
            s = sun_vector(alt, az, num(kw.get("North"), 0))
            facing = nrm @ s
            idx = np.where(facing > 0)[0]
            if len(idx):
                ok = idx[~occluded(origins[idx], s, oV, oT)]
                tot[ok] += clear_sky_dni(alt) * facing[ok] * (365 / 12) / 1000.0
        if _progress:
            _progress((k + 1) / len(steps))
    off = nrm * diag * 2e-4
    return {"Mesh": colored_mesh(V, T, tot, 0, None, "kWh/m2 per year", off), "kWh per m2": tot.tolist(),
            "Legend": f"Annual direct radiation 0 to {tot.max():.0f} kWh/m2 (clear-sky estimate)"}


# ===================================================================== energy
@node("Energy Balance", "Energy",
      [Port("Building", None, "list"), Port("Units", "mm"), Port("Window Ratio", 0.3), Port("U Wall", 0.5),
       Port("U Roof", 0.3), Port("U Floor", 0.4), Port("U Window", 2.0), Port("Air Changes", 0.5, hint="per hour"),
       Port("Heating Degree Days", 1100.0), Port("Cooling Degree Days", 350.0), Port("Design Delta T", 20.0)],
      outputs=("Mesh", "Heat Loss W/K", "Heating kWh", "Cooling kWh", "Report"),
      desc="Steady-state envelope heat balance of closed solids. Faces are sorted into roof, floor and walls by "
           "orientation. A quick early-design check, not a full simulation.")
def n_energy(Building=None, Units="mm", **kw):
    V, T = triangles_of(Building)
    if not len(T):
        raise ValueError("Connect closed solids (buildings)")
    s = UNIT_TO_M.get(str(Units).strip().lower(), 0.001)
    cen, nrm, area = face_frames(V, T)
    area_m2 = area * s * s
    roof, floor = nrm[:, 2] > 0.7, nrm[:, 2] < -0.7
    wall = ~(roof | floor)
    wwr = min(max(num(kw.get("Window Ratio"), 0.3), 0), 0.95)
    uw, ur, uf, uwin = (num(kw.get(k), d) for k, d in (("U Wall", 0.5), ("U Roof", 0.3), ("U Floor", 0.4), ("U Window", 2.0)))
    u_face = np.where(roof, ur, np.where(floor, uf, uw * (1 - wwr) + uwin * wwr))
    A_wall, A_roof, A_floor = area_m2[wall].sum(), area_m2[roof].sum(), area_m2[floor].sum()
    vol = 0.0
    for g in flatten(Building):
        if isinstance(g, Geom):
            try:
                vol += abs(g.volume()) * s ** 3
            except Exception:
                pass
    ua = float((u_face * area_m2).sum())
    vent = 0.33 * num(kw.get("Air Changes"), 0.5) * vol
    H = ua + vent
    hdd, cdd, dT = num(kw.get("Heating Degree Days"), 1100), num(kw.get("Cooling Degree Days"), 350), num(kw.get("Design Delta T"), 20)
    heat = H * hdd * 24 / 1000
    cool = H * cdd * 24 / 1000
    flux = u_face * dT
    env = A_wall + A_roof + A_floor
    rep = (f"Volume {vol:.1f} m3, floor {A_floor:.1f} m2, envelope {env:.1f} m2, compactness A/V {env / max(vol, 1e-9):.2f}\n"
           f"Walls {A_wall:.1f} m2 (windows {A_wall * wwr:.1f}), roof {A_roof:.1f} m2\n"
           f"Fabric UA {ua:.1f} W/K + ventilation {vent:.1f} W/K = {H:.1f} W/K\n"
           f"Peak heat loss at dT {dT:.0f} K: {H * dT / 1000:.2f} kW\n"
           f"Heating about {heat:.0f} kWh/yr ({heat / max(A_floor, 1e-9):.0f} kWh/m2 floor), cooling about {cool:.0f} kWh/yr")
    diag = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
    return {"Mesh": colored_mesh(V, T, flux, 0, None, "Heat flow W/m2", nrm * diag * 2e-4),
            "Heat Loss W/K": H, "Heating kWh": heat, "Cooling kWh": cool, "Report": rep}


# ===================================================================== space syntax
def _segments(geoms):
    segs = []
    for g in flatten(geoms if geoms is not None else []):
        if not isinstance(g, Geom):
            continue
        if g.kind in ("brep", "mesh"):
            segs += [(a, b) for a, b in G.feature_edges(g)]
            continue
        if not g.is_curve:
            continue
        s = g.sample()
        segs += list(zip(s[:-1], s[1:]))
    if not segs:
        return np.zeros((0, 2)), np.zeros((0, 2))
    A = np.array([a[:2] for a, b in segs], float)
    B = np.array([b[:2] for a, b in segs], float)
    keep = np.linalg.norm(B - A, axis=1) > 1e-9
    return A[keep], B[keep]


def _inside(pts, poly):
    """Even-odd point-in-polygon for many points (poly: Nx2)."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), bool)
    px, py = poly[:, 0], poly[:, 1]
    for i in range(len(poly)):
        j = i - 1
        if py[i] == py[j]:
            continue
        cond = ((py[i] > y) != (py[j] > y)) & (x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i]) + px[i])
        inside ^= cond
    return inside


def _blocked(P, Q, A, B, chunk=2000):
    """For segment pairs P[i]->Q[i], True where any wall segment A->B crosses it."""
    out = np.zeros(len(P), bool)
    if not len(A):
        return out
    r = B - A
    for s in range(0, len(P), chunk):
        p, q = P[s:s + chunk, None, :], Q[s:s + chunk, None, :]
        d = q - p
        den = d[..., 0] * r[None, :, 1] - d[..., 1] * r[None, :, 0]
        ap = A[None] - p
        t = (ap[..., 0] * r[None, :, 1] - ap[..., 1] * r[None, :, 0]) / np.where(np.abs(den) < 1e-12, np.inf, den)
        u = (ap[..., 0] * d[..., 1] - ap[..., 1] * d[..., 0]) / np.where(np.abs(den) < 1e-12, np.inf, den)
        hit = (t > 1e-9) & (t < 1 - 1e-9) & (u >= -1e-9) & (u <= 1 + 1e-9)
        out[s:s + chunk] = hit.any(axis=1)
    return out


def _grid(boundary, obstacles, size):
    poly = boundary.sample()[:, :2] if isinstance(boundary, Geom) else None
    if poly is None or len(poly) < 3:
        raise ValueError("Boundary must be a closed curve (the floor plan outline)")
    lo, hi = poly.min(0), poly.max(0)
    size = max(float(size), (hi - lo).max() / 60)  # at most about 60 x 60 cells
    xs = np.arange(lo[0] + size / 2, hi[0], size)
    ys = np.arange(lo[1] + size / 2, hi[1], size)
    P = np.array([(x, y) for y in ys for x in xs])
    ok = _inside(P, poly)
    for g in flatten(obstacles or []):
        if isinstance(g, Geom) and g.is_curve and g.closed:
            ok &= ~_inside(P, g.sample()[:, :2])
    return P[ok], size, float(boundary.sample()[:, 2].mean())


@node("Visibility Graph", "Space Syntax",
      [Port("Boundary"), Port("Obstacles", None, "list"), Port("Grid Size", 600.0, hint="model units"),
       Port("Metric", "Integration", hint="Connectivity, Integration, Mean Depth or Clustering")],
      outputs=("Mesh", "Connectivity", "Integration", "Mean Depth", "Clustering", "Legend"),
      desc="Visibility graph analysis (VGA) of a plan: who can see whom across a grid inside the boundary, "
           "blocked by walls and other obstacle curves.")
def n_vga(Boundary=None, Obstacles=None, _progress=None, **kw):
    P, size, z = _grid(Boundary, Obstacles, num(kw.get("Grid Size"), 600))
    n = len(P)
    if n < 3:
        raise ValueError("Grid has too few cells inside the boundary; make Grid Size smaller")
    if n > 2500:
        raise ValueError(f"{n} cells is too many; increase Grid Size")
    A, B = _segments(flatten(Obstacles or []) + [Boundary])
    I, J = np.triu_indices(n, 1)
    vis = np.zeros(len(I), bool)
    chunk = 20000
    for s in range(0, len(I), chunk):
        vis[s:s + chunk] = ~_blocked(P[I[s:s + chunk]], P[J[s:s + chunk]], A, B)
        if _progress:
            _progress(0.8 * min(1, (s + chunk) / len(I)))
    adj = np.zeros((n, n), bool)
    adj[I[vis], J[vis]] = True
    adj |= adj.T
    conn = adj.sum(1).astype(float)
    # step depth by breadth-first search from every cell
    depth_sum = np.zeros(n)
    reach = np.zeros(n)
    for i in range(n):
        dist = np.full(n, -1)
        dist[i] = 0
        frontier = np.zeros(n, bool)
        frontier[i] = True
        d = 0
        while frontier.any():
            d += 1
            nxt = adj[frontier].any(0) & (dist < 0)
            dist[nxt] = d
            frontier = nxt
        r = dist > 0
        depth_sum[i] = dist[r].sum()
        reach[i] = r.sum()
        if _progress and i % 25 == 0:
            _progress(0.8 + 0.2 * i / n)
    md = np.where(reach > 0, depth_sum / np.maximum(reach, 1), 0)
    k = np.maximum(reach + 1, 3)
    ra = 2 * (md - 1) / (k - 2)
    dk = 2 * (k * (np.log2((k + 2) / 3) - 1) + 1) / ((k - 1) * (k - 2))
    integ = np.where(reach > 0, dk / np.maximum(ra, 1e-6), 0)
    clus = np.zeros(n)
    for i in range(n):
        nb = np.where(adj[i])[0]
        if len(nb) > 1:
            sub = adj[np.ix_(nb, nb)]
            clus[i] = sub.sum() / (len(nb) * (len(nb) - 1))
    metric = str(kw.get("Metric") or "Integration").strip().lower()
    vals = {"connectivity": conn, "mean depth": md, "clustering": clus}.get(metric, integ)
    label = {"connectivity": "Connectivity", "mean depth": "Mean depth", "clustering": "Clustering"}.get(metric, "Integration (HH)")
    h = size / 2
    V, T = [], []
    for (x, y) in P:
        b = len(V)
        V += [(x - h, y - h, z), (x + h, y - h, z), (x + h, y + h, z), (x - h, y + h, z)]
        T += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
    face_vals = np.repeat(vals, 2)
    return {"Mesh": colored_mesh(np.array(V), np.array(T), face_vals, None, None, label),
            "Connectivity": conn.tolist(),
            "Integration": integ.tolist(), "Mean Depth": md.tolist(), "Clustering": clus.tolist(),
            "Legend": f"{label}: {vals.min():.3g} to {vals.max():.3g} over {n} cells"}


@node("Isovist", "Space Syntax", [Port("Point", (0, 0, 0)), Port("Obstacles", None, "list"), Port("Radius", 20000.0),
      Port("Rays", 180)], outputs=("Isovist", "Area", "Perimeter", "Compactness"),
      desc="The area visible from a point, cut by walls and obstacle curves.")
def n_isovist(Point=None, Obstacles=None, Radius=20000.0, Rays=180, **kw):
    c = vec(Point)
    A, B = _segments(Obstacles)
    R = num(Radius, 20000)
    n = max(8, int(num(Rays, 180)))
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    D = np.stack([np.cos(ang), np.sin(ang)], 1)
    t_hit = np.full(n, R)
    if len(A):
        r = B - A
        den = D[:, None, 0] * r[None, :, 1] - D[:, None, 1] * r[None, :, 0]
        ap = A[None] - c[None, None, :2]
        safe = np.where(np.abs(den) < 1e-12, np.inf, den)
        t = (ap[..., 0] * r[None, :, 1] - ap[..., 1] * r[None, :, 0]) / safe
        u = (ap[..., 0] * D[:, None, 1] - ap[..., 1] * D[:, None, 0]) / safe
        t = np.where((t > 1e-9) & (u >= 0) & (u <= 1), t, np.inf)
        t_hit = np.minimum(t.min(1), R)
    pts = np.column_stack([c[0] + D[:, 0] * t_hit, c[1] + D[:, 1] * t_hit, np.full(n, c[2])])
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    per = float(np.linalg.norm(np.roll(pts, -1, 0) - pts, axis=1).sum())
    return {"Isovist": G.polyline(list(pts), closed=True), "Area": area, "Perimeter": per,
            "Compactness": 4 * math.pi * area / max(per * per, 1e-9)}


@node("Color by Values", "Analysis", [Port("Mesh"), Port("Values", None, "list"), Port("Min", None), Port("Max", None)],
      outputs=("Mesh", "Legend"), desc="Paint a mesh's faces with a heat-map from one value per face (e.g. from a Python node).")
def n_color(Mesh=None, Values=None, Min=None, Max=None, **kw):
    V, T = triangles_of([Mesh])
    vals = np.asarray([num(v) for v in flatten(Values or [])], float)
    if len(vals) != len(T):
        if len(T) and len(vals) and len(T) % len(vals) == 0:
            vals = np.repeat(vals, len(T) // len(vals))
        else:
            raise ValueError(f"Need one value per face ({len(T)} faces, got {len(vals)} values)")
    lo = None if Min in (None, "") else num(Min)
    hi = None if Max in (None, "") else num(Max)
    g = colored_mesh(V, T, vals, lo, hi, "Value")
    return {"Mesh": g, "Legend": g.data["legend"]}


from . import envlib  # noqa: E402,F401  (climate, daylight, energy and movement libraries)

from . import archlib  # noqa: E402,F401  (parametric architecture)
from . import userlib  # noqa: E402,F401  (nodes ported from Grasshopper scripts)
