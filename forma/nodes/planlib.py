"""Modular plan generation: rooms sized in modules, packed by an adjacency graph into plan options.

The packing engine below (Room ... generate_plans) and the STANDARDS table are from the
Grasshopper script "Modular Plan Generator" (Rhino 8 Python 3 component), kept as they were.
Lengths inside the engine are metres; ``forma.nodes.userlib`` converts to and from model units.
"""
import math
import random
import time

STANDARDS = {
    "living": {
        "label": "Living room",
        "min_area": 12.0,     # combined living area, indicative
        "min_dim": 3.6,       # min clear width, NSW ADG guidance for 1-2 bed
        "min_height": 2.4,    # habitable room ceiling height
        "habitable": True,
    },
    "bedroom": {
        "label": "Bedroom",
        "min_area": 9.0,      # secondary bedroom, 10.0 for a main bedroom
        "min_dim": 3.0,
        "min_height": 2.4,
        "habitable": True,
    },
    "kitchen": {
        "label": "Kitchen",
        "min_area": 5.0,
        "min_dim": 1.8,       # allows a 1.0 m+ clear working aisle
        "min_height": 2.4,
        "habitable": True,
    },
    "bathroom": {
        "label": "Bathroom",
        "min_area": 3.0,
        "min_dim": 1.5,
        "min_height": 2.1,    # non-habitable ceiling height
        "habitable": False,
    },
    "storage": {
        "label": "Storage",
        "min_area": 1.5,      # ~4 m3 at 2.4 m, studio-scale ADG storage
        "min_dim": 0.6,
        "min_height": 2.1,
        "habitable": False,
    },
    "corridor": {
        "label": "Corridor",
        "min_area": 0.0,      # governed by width, not area
        "min_dim": 1.0,       # clear circulation width
        "min_height": 2.1,
        "habitable": False,
    },
}

# Rhino layer colours by space type (R, G, B)
TYPE_COLOUR = {
    "living":   (196, 132,  34),
    "bedroom":  ( 46, 116, 181),
    "kitchen":  ( 40, 148, 106),
    "storage":  (122, 112, 140),
    "corridor": (188,  82,  82),
    "bathroom": ( 56, 152, 168),
}

ROOT_LAYER = "GH_PLANS"
TIME_BUDGET = 12.0            # seconds, hard ceiling on the search


class Room(object):
    __slots__ = ("nid", "code", "type", "modules", "w", "d", "x", "y")

    def __init__(self, nid, code, type_, modules, w, d):
        self.nid, self.code, self.type = nid, code, type_
        self.modules, self.w, self.d = modules, w, d
        self.x = self.y = 0.0

    @property
    def area(self):
        return self.w * self.d

    def copy(self):
        return Room(self.nid, self.code, self.type, self.modules, self.w, self.d)


def footprint(type_, modules, short, long_):
    """Rooms grow on the short side, corridors grow on the long side."""
    if type_ == "corridor":
        return short, modules * long_
    return modules * short, long_


def rooms_from_payload(payload):
    m = payload["module"]
    short = min(float(m["a"]), float(m["b"]))
    long_ = max(float(m["a"]), float(m["b"]))
    rooms = []
    for n in payload["nodes"]:
        w, d = footprint(n["type"], float(n["modules"]), short, long_)
        rooms.append(Room(n["id"], n["code"], n["type"], float(n["modules"]), w, d))
    return rooms, short, long_


def overlaps(a, b, eps=1e-7):
    return (a.x < b.x + b.w - eps and b.x < a.x + a.w - eps and
            a.y < b.y + b.d - eps and b.y < a.y + a.d - eps)


def touching(a, b, eps=1e-6, min_share=0.30):
    """True when the two rectangles share a usable length of wall."""
    if abs(a.x + a.w - b.x) < eps or abs(b.x + b.w - a.x) < eps:
        return min(a.y + a.d, b.y + b.d) - max(a.y, b.y) > min_share
    if abs(a.y + a.d - b.y) < eps or abs(b.y + b.d - a.y) < eps:
        return min(a.x + a.w, b.x + b.w) - max(a.x, b.x) > min_share
    return False


def bbox_of(rooms):
    return (min(r.x for r in rooms), min(r.y for r in rooms),
            max(r.x + r.w for r in rooms), max(r.y + r.d for r in rooms))


def adjacency_map(edges):
    adj = {}
    for e in edges:
        a, b = e[0], e[1]
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def visit_order(rooms, adj, rng):
    by_id = dict((r.nid, r) for r in rooms)
    roots = [r for r in rooms if r.type == "living"] or rooms
    start = max(roots, key=lambda r: r.area).nid

    order, seen, queue = [], set([start]), [start]
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        nbrs = [n for n in adj.get(cur, ()) if n not in seen and n in by_id]
        nbrs.sort(key=lambda n: -by_id[n].area)
        if rng.random() < 0.5:
            rng.shuffle(nbrs)
        for n in nbrs:
            seen.add(n)
            queue.append(n)
    for r in rooms:
        if r.nid not in seen:
            seen.add(r.nid)
            order.append(r.nid)
    return [by_id[i] for i in order]


def candidate_slots(anchor, room, grid, cap=14):
    """Positions where `room` sits flush against one face of `anchor`."""
    ax0, ay0 = anchor.x, anchor.y
    ax1, ay1 = anchor.x + anchor.w, anchor.y + anchor.d
    out = []

    def span(lo, hi, size):
        vals = [lo, hi - size, lo + (hi - lo - size) / 2.0]
        n, v = 0, lo - size + grid
        while v < hi - grid * 0.5 and n < cap:
            vals.append(v)
            v += grid
            n += 1
        return vals

    for yy in span(ay0, ay1, room.d):
        out.append((ax1, yy))
        out.append((ax0 - room.w, yy))
    for xx in span(ax0, ax1, room.w):
        out.append((xx, ay1))
        out.append((xx, ay0 - room.d))
    return out


def pack(rooms, adj, seed, grid):
    """Greedy adjacency-driven placement. Scoring is incremental, so the
    cost per candidate is O(placed) for the clash test only."""
    rng = random.Random(seed)
    order = visit_order(rooms, adj, rng)
    placed = []
    x0 = y0 = x1 = y1 = 0.0
    used = 0.0

    for room in order:
        if not placed:
            room.x = room.y = 0.0
            placed.append(room)
            x1, y1 = room.w, room.d
            used = room.area
            continue

        want = adj.get(room.nid, set())
        anchors = [p for p in placed if p.nid in want] or placed
        if len(anchors) > 6:
            anchors = sorted(anchors, key=lambda p: -p.area)[:6]

        best = None
        for anchor in anchors:
            for (cx, cy) in candidate_slots(anchor, room, grid):
                room.x, room.y = cx, cy
                clash = False
                for p in placed:
                    if overlaps(room, p):
                        clash = True
                        break
                if clash:
                    continue

                nx0, ny0 = min(x0, cx), min(y0, cy)
                nx1, ny1 = max(x1, cx + room.w), max(y1, cy + room.d)
                bw, bh = nx1 - nx0, ny1 - ny0
                box = bw * bh
                met = 0
                for p in placed:
                    if p.nid in want and touching(room, p):
                        met += 1
                miss = max(0, len([p for p in placed if p.nid in want]) - met)
                aspect = max(bw, bh) / max(1e-6, min(bw, bh))
                s = (box
                     + (box - used - room.area) * 0.45
                     + miss * (box * 0.35 + 30.0)
                     + (aspect - 1.0) * box * 0.08
                     + rng.random() * grid * grid * 0.5)
                if best is None or s < best[0]:
                    best = (s, cx, cy, nx0, ny0, nx1, ny1)

        if best is None:
            room.x, room.y = x1 + grid, y0
            x1 = room.x + room.w
            y1 = max(y1, room.y + room.d)
        else:
            room.x, room.y = best[1], best[2]
            x0, y0, x1, y1 = best[3], best[4], best[5], best[6]
        placed.append(room)
        used += room.area

    compact(placed, adj, grid)
    return placed


def kept_contacts(room, placed, adj):
    want = adj.get(room.nid, set())
    if not want:
        return 0
    return sum(1 for p in placed
               if p is not room and p.nid in want and touching(room, p))


def compact(placed, adj, grid, rounds=4):
    """Slide rooms back toward the origin while keeping required contacts."""
    for _ in range(rounds):
        moved = False
        for axis in (0, 1):
            order = sorted(placed, key=lambda q: q.x if axis == 0 else q.y)
            for r in order:
                base = kept_contacts(r, placed, adj)
                steps = 0
                while steps < 200:
                    ox, oy = r.x, r.y
                    if axis == 0:
                        r.x -= grid
                    else:
                        r.y -= grid
                    bad = False
                    for p in placed:
                        if p is not r and overlaps(r, p):
                            bad = True
                            break
                    if bad or kept_contacts(r, placed, adj) < base:
                        r.x, r.y = ox, oy
                        break
                    moved = True
                    steps += 1
        if not moved:
            break

    x0, y0, _, _ = bbox_of(placed)
    for r in placed:
        r.x -= x0
        r.y -= y0


def plan_score(placed, adj):
    x0, y0, x1, y1 = bbox_of(placed)
    bw, bh = x1 - x0, y1 - y0
    box = bw * bh
    used = sum(r.area for r in placed)
    miss = 0
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            a, b = placed[i], placed[j]
            if b.nid in adj.get(a.nid, ()) and not touching(a, b):
                miss += 1
    aspect = max(bw, bh) / max(1e-6, min(bw, bh))
    return box + (box - used) * 0.6 + miss * box * 0.25 + (aspect - 1.0) * box * 0.08, miss


def signature(placed):
    return tuple(sorted((r.code, round(r.x, 3), round(r.y, 3)) for r in placed))


def generate_plans(payload):
    rooms, short, long_ = rooms_from_payload(payload)
    adj = adjacency_map(payload.get("edges", []))
    grid = max(0.25, short / 2.0)
    want = int(payload.get("plans", 5))

    plans, seen, seed = [], set(), 0
    t0 = time.time()
    while len(plans) < want and seed < want * 8 and (time.time() - t0) < TIME_BUDGET:
        placed = pack([r.copy() for r in rooms], adj, seed, grid)
        sig = signature(placed)
        if sig not in seen:
            seen.add(sig)
            plans.append(placed)
        seed += 1

    plans.sort(key=lambda p: plan_score(p, adj)[0])
    return plans, adj


# ============================================================================
#  4.  BAKING TO RHINO
