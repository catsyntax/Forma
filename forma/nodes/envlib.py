"""Ready-made analysis libraries for the node editor:

Climate   Location presets, shadow study
Daylight  daylight factor (CIE overcast sky component), sky view factor
Energy    envelope by orientation, U-value from layers, material presets, solar gains through glazing
Space     walking distance and through-movement (betweenness) on a walkable grid

All values are early-design estimates. Material figures are typical textbook values; check product data.
"""
from __future__ import annotations

import math

import numpy as np

from .. import geometry as G
from ..geometry import Geom
from .registry import node, Port, flatten, vec, num
from .analysis import (triangles_of, subdivide, face_frames, occluded, colored_mesh, sun_position, sun_vector,
                       clear_sky_dni, _segments, _inside, _blocked, _grid, UNIT_TO_M)

# ===================================================================== climate
LOCATIONS = {
    # name: (latitude, longitude, UTC offset in standard time)
    "Adelaide": (-34.93, 138.60, 9.5),
    "Sydney": (-33.87, 151.21, 10.0),
    "Melbourne": (-37.81, 144.96, 10.0),
    "Brisbane": (-27.47, 153.03, 10.0),
    "Perth": (-31.95, 115.86, 8.0),
    "Darwin": (-12.46, 130.84, 9.5),
    "Hobart": (-42.88, 147.33, 10.0),
    "Auckland": (-36.85, 174.76, 12.0),
    "Singapore": (1.35, 103.82, 8.0),
    "Tokyo": (35.68, 139.69, 9.0),
    "Dubai": (25.20, 55.27, 4.0),
    "Tehran": (35.69, 51.39, 3.5),
    "Istanbul": (41.01, 28.98, 3.0),
    "London": (51.51, -0.13, 0.0),
    "Paris": (48.86, 2.35, 1.0),
    "Berlin": (52.52, 13.40, 1.0),
    "New York": (40.71, -74.01, -5.0),
    "Los Angeles": (34.05, -118.24, -8.0),
    "Sao Paulo": (-23.55, -46.63, -3.0),
    "Cape Town": (-33.92, 18.42, 2.0),
}


@node("Location", "Climate", [Port("City", "Adelaide", hint=", ".join(LOCATIONS))],
      outputs=("Latitude", "Longitude", "Time Zone", "Cities"),
      desc="Latitude, longitude and standard time zone of a city. Connect to the sun and daylight nodes.")
def n_location(City="Adelaide", **kw):
    key = str(City or "").strip().lower()
    for name, (la, lo, tz) in LOCATIONS.items():
        if name.lower() == key:
            return {"Latitude": la, "Longitude": lo, "Time Zone": tz, "Cities": list(LOCATIONS)}
    raise ValueError(f"Unknown city. Try one of: {', '.join(LOCATIONS)}; or type the latitude/longitude directly")


@node("Shadow Study", "Climate",
      [Port("Geometry", None, "list"), Port("Latitude", -34.93), Port("Longitude", 138.6), Port("Time Zone", 9.5),
       Port("Month", 6), Port("Day", 21), Port("Hour", 12.0), Port("Ground Z", 0.0), Port("North", 0.0)],
      outputs=("Shadows", "Sun Vector", "Altitude"),
      desc="Shadows cast on the ground plane at one moment. Plug a Number Slider into Hour to animate a day.")
def n_shadow(Geometry=None, Latitude=0, Longitude=0, Month=6, Day=21, Hour=12, **kw):
    alt, az = sun_position(num(Latitude), num(Longitude), int(num(Month)), int(num(Day)), num(Hour), num(kw.get("Time Zone"), 0))
    s = sun_vector(alt, az, num(kw.get("North"), 0))
    if alt <= 0.5:
        return {"Shadows": [], "Sun Vector": s, "Altitude": alt}
    V, T = triangles_of(Geometry)
    if not len(T):
        raise ValueError("Connect solids or surfaces")
    z0 = num(kw.get("Ground Z"), 0)
    above = V[:, 2] >= z0 - 1e-9
    P = V - s[None, :] * ((V[:, 2] - z0) / s[2])[:, None]
    P[:, 2] = z0 + 1e-3 * max(1.0, float(np.ptp(V[:, 0])))   # just above the ground to avoid flicker
    P[~above] = V[~above]
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    keep = np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1e-9
    T = T[keep]
    g = colored_mesh(P, T, np.zeros(len(T)), 0, 1, f"Shadows at {num(Hour):.2f} h (sun {alt:.0f} deg high)")
    g.data["fcolors"] = np.tile([0.12, 0.12, 0.16], (len(T), 1))
    return {"Shadows": g, "Sun Vector": s, "Altitude": alt}


# ===================================================================== daylight
def _hemisphere(n=240):
    """Directions evenly spread over the upper hemisphere (equal solid angle each)."""
    i = np.arange(n) + 0.5
    z = 1 - i / n            # uniform in z -> uniform solid angle on the hemisphere
    r = np.sqrt(1 - z * z)
    phi = i * math.pi * (3 - math.sqrt(5))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _analysis_points(floor, grid, height):
    """Grid of points (and cells for display) on a floor: a closed curve or flat surfaces, lifted to the work plane."""
    if isinstance(floor, Geom) and floor.is_curve:
        P2, size, z = _grid(floor, [], grid)
        h = size / 2
        V, T = [], []
        for x, y in P2:
            b = len(V)
            V += [(x - h, y - h, z + height), (x + h, y - h, z + height), (x + h, y + h, z + height), (x - h, y + h, z + height)]
            T += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
        pts = np.column_stack([P2, np.full(len(P2), z + height)])
        return pts, np.array(V, float), np.array(T, int), 2
    V, T = triangles_of(floor)
    if not len(T):
        raise ValueError("Floor must be a closed curve or a flat surface")
    V, T = subdivide(V, T, grid, max_faces=3000)
    cen, nrm, _ = face_frames(V, T)
    up = np.where(nrm[:, 2:3] < 0, -nrm, nrm)
    V2 = V[T].reshape(-1, 3) + np.repeat(up * height, 3, axis=0)
    return cen + up * height, V2, np.arange(len(V2)).reshape(-1, 3), 1


def _sky_fraction(pts, obstacles, weights_fn, _progress=None, n=240):
    D = _hemisphere(n)
    oV, oT = triangles_of(obstacles)
    w = weights_fn(D)
    tot = w.sum()
    vis = np.zeros(len(pts))
    for k, d in enumerate(D):
        if w[k] <= 0:
            continue
        blocked = occluded(pts, d, oV, oT) if len(oT) else np.zeros(len(pts), bool)
        vis[~blocked] += w[k]
        if _progress and k % 8 == 0:
            _progress(k / len(D))
    return vis / tot


@node("Daylight Factor", "Daylight",
      [Port("Floor"), Port("Context", None, "list"), Port("Work Plane Height", 750.0), Port("Grid Size", 500.0),
       Port("Sky Samples", 240)],
      outputs=("Mesh", "DF %", "Average DF", "Legend"),
      desc="Daylight factor under a CIE overcast sky: the sky component seen through openings in the walls and "
           "roof (model windows as openings). Internal reflections are not included, so real values are a bit higher.")
def n_daylight_factor(Floor=None, Context=None, _progress=None, **kw):
    pts, V, T, per = _analysis_points(Floor, num(kw.get("Grid Size"), 500), num(kw.get("Work Plane Height"), 750))
    if len(pts) > 4000:
        raise ValueError(f"{len(pts)} points is a lot; increase Grid Size")
    # CIE overcast luminance (1 + 2 sin(altitude)) / 3, times cos of incidence on a horizontal work plane
    df = 100 * _sky_fraction(pts + np.array([0, 0, 1e-3]), Context, lambda D: (1 + 2 * D[:, 2]) / 3 * D[:, 2], _progress,
                             int(num(kw.get("Sky Samples"), 240)))
    vals = np.repeat(df, per)
    avg = float(df.mean()) if len(df) else 0.0
    return {"Mesh": colored_mesh(V, T, vals, 0, max(5.0, float(df.max())), "Daylight factor %"), "DF %": df.tolist(),
            "Average DF": avg,
            "Legend": f"Daylight factor: average {avg:.1f}%, min {df.min():.1f}%, max {df.max():.1f}% "
                      f"({(df >= 2).mean() * 100:.0f}% of the floor at 2% or more)"}


@node("Sky View Factor", "Daylight",
      [Port("Floor"), Port("Context", None, "list"), Port("Height", 1500.0), Port("Grid Size", 1000.0)],
      outputs=("Mesh", "SVF", "Legend"),
      desc="How much of the sky is visible from each point (0 = none, 1 = open field). Useful for streets, "
           "courtyards and urban heat.")
def n_svf(Floor=None, Context=None, _progress=None, **kw):
    pts, V, T, per = _analysis_points(Floor, num(kw.get("Grid Size"), 1000), num(kw.get("Height"), 1500))
    if len(pts) > 5000:
        raise ValueError(f"{len(pts)} points is a lot; increase Grid Size")
    svf = _sky_fraction(pts + np.array([0, 0, 1e-3]), Context, lambda D: D[:, 2], _progress, 200)
    return {"Mesh": colored_mesh(V, T, np.repeat(svf, per), 0, 1, "Sky view factor"), "SVF": svf.tolist(),
            "Legend": f"Sky view factor: average {svf.mean():.2f} (0 to 1)"}


# ===================================================================== energy
MATERIALS = {
    # name: (conductivity W/mK, density kg/m3, specific heat J/kgK) - typical values
    "Concrete": (1.40, 2300, 1000),
    "Lightweight concrete": (0.40, 1200, 1000),
    "Brick": (0.77, 1700, 800),
    "Rammed earth": (1.00, 1900, 900),
    "Timber": (0.13, 500, 1600),
    "CLT": (0.12, 480, 1600),
    "Plasterboard": (0.21, 700, 1000),
    "Mineral wool": (0.035, 30, 1030),
    "Glass wool": (0.040, 20, 1030),
    "EPS": (0.035, 20, 1450),
    "XPS": (0.033, 35, 1450),
    "PIR": (0.022, 32, 1400),
    "Wood fibre": (0.040, 160, 2100),
    "Straw bale": (0.065, 100, 2000),
    "Glass": (1.00, 2500, 750),
    "Steel": (50.0, 7800, 450),
    "Aluminium": (160.0, 2700, 880),
    "Stone": (2.30, 2600, 1000),
}


@node("Material", "Energy", [Port("Name", "Mineral wool", hint=", ".join(MATERIALS))],
      outputs=("Conductivity", "Density", "Specific Heat", "Materials"),
      desc="Typical thermal properties of common building materials (check product data for real projects).")
def n_material(Name="Mineral wool", **kw):
    key = str(Name or "").strip().lower()
    for name, (k, rho, c) in MATERIALS.items():
        if name.lower() == key:
            return {"Conductivity": k, "Density": rho, "Specific Heat": c, "Materials": list(MATERIALS)}
    raise ValueError(f"Unknown material. Try one of: {', '.join(MATERIALS)}")


@node("U-Value", "Energy",
      [Port("Thicknesses", None, "list", hint="mm, outside to inside"), Port("Conductivities", None, "list", hint="W/mK"),
       Port("Rsi", 0.13), Port("Rse", 0.04)],
      outputs=("U", "R", "Report"),
      desc="U-value of a wall, roof or floor build-up from its layers: U = 1 / (Rsi + sum(d / k) + Rse).")
def n_uvalue(Thicknesses=None, Conductivities=None, Rsi=0.13, Rse=0.04, **kw):
    d = [num(x) / 1000 for x in flatten(Thicknesses or [])]
    k = [num(x) for x in flatten(Conductivities or [])]
    if not d or len(d) != len(k):
        raise ValueError("Give one conductivity per layer thickness")
    layers = [di / max(ki, 1e-9) for di, ki in zip(d, k)]
    R = num(Rsi, 0.13) + sum(layers) + num(Rse, 0.04)
    rep = "\n".join(f"layer {i + 1}: {di * 1000:.0f} mm, k {ki:g} -> R {r:.3f}" for i, (di, ki, r) in enumerate(zip(d, k, layers)))
    return {"U": 1 / R, "R": R, "Report": rep + f"\nTotal R {R:.3f} m2K/W, U {1 / R:.3f} W/m2K"}


def _orient(nrm, north):
    """Compass name of a wall normal (N, E, S, W)."""
    az = (np.degrees(np.arctan2(nrm[:, 0], nrm[:, 1])) - north) % 360
    return np.array(["N", "E", "S", "W"])[(((az + 45) % 360) // 90).astype(int)]


@node("Envelope by Orientation", "Energy", [Port("Building", None, "list"), Port("Units", "mm"), Port("North", 0.0)],
      outputs=("Mesh", "North m2", "East m2", "South m2", "West m2", "Roof m2", "Floor m2", "Report"),
      desc="Splits the outside of a building into walls facing N/E/S/W, roof and floor, with areas.")
def n_envelope(Building=None, Units="mm", **kw):
    V, T = triangles_of(Building)
    if not len(T):
        raise ValueError("Connect closed solids")
    s = UNIT_TO_M.get(str(Units).strip().lower(), 0.001)
    cen, nrm, area = face_frames(V, T)
    a = area * s * s
    roof, floor = nrm[:, 2] > 0.7, nrm[:, 2] < -0.7
    wall = ~(roof | floor)
    o = _orient(nrm, num(kw.get("North"), 0))
    out = {d: float(a[wall & (o == d)].sum()) for d in "NESW"}
    wall_code = np.array(["NESW".index(x) for x in o], int)
    code = np.where(roof, 4, np.where(floor, 5, wall_code))
    rep = ", ".join(f"{d} {out[d]:.1f} m2" for d in "NESW") + f", roof {a[roof].sum():.1f} m2, floor {a[floor].sum():.1f} m2"
    diag = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
    g = colored_mesh(V, T, code.astype(float), 0, 5, "N, E, S, W, roof, floor", nrm * diag * 2e-4)
    pal = np.array([[0.25, 0.45, 0.95], [0.95, 0.75, 0.2], [0.9, 0.3, 0.25], [0.3, 0.75, 0.45], [0.7, 0.7, 0.75], [0.4, 0.4, 0.45]])
    g.data["fcolors"] = pal[code]
    g.data["legend"] = "Blue N, yellow E, red S, green W, grey roof"
    return {"Mesh": g, "North m2": out["N"], "East m2": out["E"], "South m2": out["S"], "West m2": out["W"],
            "Roof m2": float(a[roof].sum()), "Floor m2": float(a[floor].sum()), "Report": rep}


@node("Solar Gains", "Energy",
      [Port("Building", None, "list"), Port("Context", None, "list"), Port("Units", "mm"), Port("Window Ratio", 0.3),
       Port("SHGC", 0.5), Port("Latitude", -34.93), Port("Longitude", 138.6), Port("Time Zone", 9.5), Port("North", 0.0)],
      outputs=("Mesh", "Annual kWh", "By Month kWh", "Report"),
      desc="Direct solar gain through windows (window ratio of each wall x SHGC), month by month, with shading "
           "from context. Clear-sky estimate; useful to compare orientations and shading.")
def n_solar_gains(Building=None, Context=None, Units="mm", _progress=None, **kw):
    V, T = triangles_of(Building)
    if not len(T):
        raise ValueError("Connect closed solids")
    s = UNIT_TO_M.get(str(Units).strip().lower(), 0.001)
    lo, hi = V.min(0), V.max(0)
    V, T = subdivide(V, T, float(np.linalg.norm(hi - lo)) / 12, max_faces=3000)
    cen, nrm, area = face_frames(V, T)
    wall = np.abs(nrm[:, 2]) <= 0.7
    a = area * s * s * num(kw.get("Window Ratio"), 0.3) * num(kw.get("SHGC"), 0.5)
    oV, oT = triangles_of(flatten(Building) + flatten(Context or []))
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    origins = cen + nrm * diag * 1e-4
    per_face = np.zeros(len(T))
    months = np.zeros(12)
    lat, lon, tz, north = (num(kw.get(k), d) for k, d in (("Latitude", 0), ("Longitude", 0), ("Time Zone", 0), ("North", 0)))
    for m in range(1, 13):
        for h in range(5, 21):
            alt, az = sun_position(lat, lon, m, 15, h + 0.5, tz)
            if alt <= 2:
                continue
            sv = sun_vector(alt, az, north)
            facing = nrm @ sv
            idx = np.where(wall & (facing > 0))[0]
            if not len(idx):
                continue
            ok = idx[~occluded(origins[idx], sv, oV, oT)]
            e = clear_sky_dni(alt) * facing[ok] * a[ok] / 1000 * (365 / 12)
            per_face[ok] += e
            months[m - 1] += e.sum()
        if _progress:
            _progress(m / 12)
    dens = np.where(a > 0, per_face / np.maximum(area * s * s, 1e-12), 0)
    o = _orient(nrm, north)
    by = {d: float(per_face[wall & (o == d)].sum()) for d in "NESW"}
    rep = (f"Annual direct solar gain through glazing about {months.sum():.0f} kWh "
           f"(N {by['N']:.0f}, E {by['E']:.0f}, S {by['S']:.0f}, W {by['W']:.0f}). "
           f"Peak month {int(np.argmax(months)) + 1}, lowest month {int(np.argmin(months)) + 1}.")
    return {"Mesh": colored_mesh(V, T, dens, 0, None, "Solar gain kWh per m2 of wall", nrm * diag * 2e-4),
            "Annual kWh": float(months.sum()), "By Month kWh": months.round(1).tolist(), "Report": rep}


# ===================================================================== space syntax: movement
def _walk_graph(Boundary, Obstacles, size):
    P, size, z = _grid(Boundary, Obstacles, size)
    if len(P) > 3000:
        raise ValueError(f"{len(P)} cells is too many; increase Grid Size")
    A, B = _segments(flatten(Obstacles or []) + [Boundary])
    edges, w = [], []
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        off = np.array([dx, dy]) * size
        Q = P + off
        # neighbour exists?
        key = {(round(x / size * 2), round(y / size * 2)): i for i, (x, y) in enumerate(P)}
        j = np.array([key.get((round(x / size * 2), round(y / size * 2)), -1) for x, y in Q])
        ok = j >= 0
        i_idx, j_idx = np.where(ok)[0], j[ok]
        if not len(i_idx):
            continue
        free = ~_blocked(P[i_idx], P[j_idx], A, B)
        edges += list(zip(i_idx[free], j_idx[free]))
        w += [float(np.hypot(*off))] * int(free.sum())
    return P, size, z, edges, w


def _cells(P, size, z):
    h = size / 2
    V, T = [], []
    for x, y in P:
        b = len(V)
        V += [(x - h, y - h, z), (x + h, y - h, z), (x + h, y + h, z), (x - h, y + h, z)]
        T += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
    return np.array(V, float), np.array(T, int)


@node("Walking Distance", "Space Syntax",
      [Port("Boundary"), Port("Obstacles", None, "list"), Port("Origins", None, "list"), Port("Grid Size", 500.0)],
      outputs=("Mesh", "Distance", "Max", "Legend"),
      desc="Shortest walking distance from the origin point(s) (e.g. entrances or stairs) to every spot, "
           "going around walls.")
def n_walk(Boundary=None, Obstacles=None, Origins=None, **kw):
    import networkx as nx
    P, size, z, edges, w = _walk_graph(Boundary, Obstacles, num(kw.get("Grid Size"), 500))
    Gr = nx.Graph()
    Gr.add_nodes_from(range(len(P)))
    Gr.add_weighted_edges_from([(a, b, ww) for (a, b), ww in zip(edges, w)])
    org = [vec(o) for o in flatten(Origins or [])]
    if not org:
        raise ValueError("Connect one or more origin points (entrances)")
    src = {int(np.argmin(np.linalg.norm(P - o[:2], axis=1))) for o in org}
    dist = nx.multi_source_dijkstra_path_length(Gr, src)
    d = np.array([dist.get(i, np.nan) for i in range(len(P))])
    reach = np.nan_to_num(d, nan=np.nanmax(d) if np.isfinite(d).any() else 0)
    V, T = _cells(P, size, z)
    return {"Mesh": colored_mesh(V, T, np.repeat(reach, 2), 0, None, "Walking distance"), "Distance": d.tolist(),
            "Max": float(np.nanmax(d)) if np.isfinite(d).any() else 0.0,
            "Legend": f"Walking distance 0 to {np.nanmax(d):.0f} (unreachable cells shown as farthest)"}


@node("Through Movement", "Space Syntax",
      [Port("Boundary"), Port("Obstacles", None, "list"), Port("Grid Size", 600.0), Port("Samples", 200)],
      outputs=("Mesh", "Choice", "Legend"),
      desc="How often each spot lies on shortest routes between all other spots (betweenness / 'choice'). "
           "Shows the likely busy corridors of a plan.")
def n_choice(Boundary=None, Obstacles=None, **kw):
    import networkx as nx
    P, size, z, edges, w = _walk_graph(Boundary, Obstacles, num(kw.get("Grid Size"), 600))
    Gr = nx.Graph()
    Gr.add_nodes_from(range(len(P)))
    Gr.add_weighted_edges_from([(a, b, ww) for (a, b), ww in zip(edges, w)])
    k = min(len(P), max(10, int(num(kw.get("Samples"), 200))))
    bc = nx.betweenness_centrality(Gr, k=k if k < len(P) else None, weight="weight", seed=1)
    v = np.array([bc[i] for i in range(len(P))])
    v = np.log1p(v * 1000)
    V, T = _cells(P, size, z)
    return {"Mesh": colored_mesh(V, T, np.repeat(v, 2), None, None, "Through movement (log)"), "Choice": v.tolist(),
            "Legend": "Through movement: red = on many shortest routes"}
