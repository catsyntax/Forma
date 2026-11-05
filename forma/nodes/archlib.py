"""Parametric architecture nodes: stadium, twisting diagrid tower, free-form pavilion."""
from __future__ import annotations

from .. import parametric as PA
from .registry import node, Port, num


def _split(parts):
    out = {}
    for role, g in parts:
        g.data["role"] = role
        out.setdefault(role, []).append(g)
    return out


@node("Stadium", "Architecture",
      [Port("Length", 230000.0), Port("Width", 180000.0), Port("Height", 48000.0), Port("Tiers", 3),
       Port("Ribs", 56), Port("Roof Depth", 52000.0), Port("Saddle", 9000.0), Port("Squareness", 2.6, hint="2 = oval"),
       Port("Bulge", 9000.0)],
      outputs=("Roof", "Facade", "Seating", "Pitch", "Structure"),
      desc="Parametric stadium: raked seating bowl, saddle-shaped cantilever roof (NURBS) and a ribbed facade.")
def n_stadium(**kw):
    p = _split(PA.stadium(num(kw.get("Length"), 230000), num(kw.get("Width"), 180000), num(kw.get("Height"), 48000),
                          int(num(kw.get("Tiers"), 3)), int(num(kw.get("Ribs"), 56)), num(kw.get("Roof Depth"), 52000),
                          num(kw.get("Saddle"), 9000), max(1.5, num(kw.get("Squareness"), 2.6)), num(kw.get("Bulge"), 9000)))
    return {"Roof": p.get("roof", []), "Facade": p.get("facade", []), "Seating": p.get("seating", []),
            "Pitch": p.get("pitch", []), "Structure": p.get("ribs", []) + p.get("trusses", [])}


@node("Diagrid Tower", "Architecture",
      [Port("Height", 240000.0), Port("Floors", 60), Port("Radius", 24000.0), Port("Twist", 110.0, hint="degrees"),
       Port("Taper", 0.62), Port("Lobes", 3), Port("Modules", 10), Port("Columns", 24)],
      outputs=("Skin", "Floors", "Diagrid", "Core"),
      desc="Twisting, tapering tower on a lobed plan with a NURBS glass skin, floor plates and a diagrid.")
def n_tower(**kw):
    p = _split(PA.diagrid_tower(num(kw.get("Height"), 240000), int(num(kw.get("Floors"), 60)), num(kw.get("Radius"), 24000),
                                num(kw.get("Twist"), 110), num(kw.get("Taper"), 0.62), int(num(kw.get("Lobes"), 3)),
                                int(num(kw.get("Modules"), 10)), int(num(kw.get("Columns"), 24))))
    return {"Skin": p.get("glass", []) + p.get("crown", []), "Floors": p.get("slabs", []),
            "Diagrid": p.get("diagrid", []), "Core": p.get("core", [])}


@node("Free-form Pavilion", "Architecture",
      [Port("Length", 110000.0), Port("Width", 56000.0), Port("Height", 19000.0), Port("Waves", 3.0),
       Port("Sections", 9), Port("Columns", 5)],
      outputs=("Shell", "Columns", "Ground"),
      desc="Undulating NURBS roof shell lofted through arched sections, on funnel-shaped columns.")
def n_pavilion(**kw):
    p = _split(PA.pavilion(num(kw.get("Length"), 110000), num(kw.get("Width"), 56000), num(kw.get("Height"), 19000),
                           num(kw.get("Waves"), 3), max(3, int(num(kw.get("Sections"), 9))), int(num(kw.get("Columns"), 5))))
    return {"Shell": p.get("shell", []), "Columns": p.get("columns", []), "Ground": p.get("ground", [])}
