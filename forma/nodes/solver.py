"""Graph evaluation. Runs inside a separate worker process so it can be cancelled at any time."""
from __future__ import annotations

import time
import traceback

from .registry import NODE_TYPES, Port, flatten, is_list


def ports(node):
    """(inputs, outputs) for a node dict, honouring dynamic (script) ports."""
    t = NODE_TYPES[node["type"]]
    if t.dynamic:
        props = node.get("props", {})
        ins = [Port(n, None, "list" if n in props.get("list_inputs", []) else "item") for n in props.get("inputs", ["x", "y"])]
        outs = list(props.get("outputs", ["a"])) + ["out"]
        return ins, outs
    return t.inputs, t.outputs


def topo_order(graph):
    nodes = graph["nodes"]
    deps = {nid: set() for nid in nodes}
    for s, _, d, _ in graph["wires"]:
        if s in nodes and d in nodes:
            deps[d].add(s)
    order, seen, temp = [], set(), set()

    def visit(n):
        if n in seen:
            return
        if n in temp:
            raise ValueError("The graph has a loop")
        temp.add(n)
        for m in deps[n]:
            visit(m)
        temp.discard(n)
        seen.add(n)
        order.append(n)

    for n in nodes:
        visit(n)
    return order


def downstream(graph, nids):
    out = set(nids)
    changed = True
    while changed:
        changed = False
        for s, _, d, _ in graph["wires"]:
            if s in out and d not in out:
                out.add(d)
                changed = True
    return out


def gather_inputs(graph, nid, results):
    node = graph["nodes"][nid]
    ins, _ = ports(node)
    vals = {}
    for p in ins:
        srcs = [(s, o) for s, o, d, i in graph["wires"] if d == nid and i == p.name]
        if srcs:
            got = []
            for s, o in srcs:
                r = results.get(s)
                if r is None or o not in r:
                    raise RuntimeError(f"Input '{p.name}' is waiting on a node that failed")
                got.append(r[o])
            vals[p.name] = got[0] if len(got) == 1 else flatten(got)
        else:
            v = node.get("values", {}).get(p.name, p.default)
            vals[p.name] = v
    return ins, vals


def evaluate(graph, nid, results, refs, defs, progress):
    node = graph["nodes"][nid]
    t = NODE_TYPES[node["type"]]
    ins, vals = gather_inputs(graph, nid, results)
    _, outs = ports(node)
    extra = {"_props": node.get("props", {}), "_progress": progress, "_defs": defs}
    if t.widget == "ref":
        extra["_refs"] = refs.get(nid, [])
    # longest-list matching over item-access inputs that received lists
    mapped = [p.name for p in ins if p.access == "item" and is_list(vals.get(p.name))]
    if not mapped:
        return _norm(t.fn(**vals, **extra), outs)
    n = max(len(vals[m]) for m in mapped)
    if any(len(vals[m]) == 0 for m in mapped):
        return {o: [] for o in outs}
    collected = {o: [] for o in outs}
    for i in range(n):
        call = dict(vals)
        for m in mapped:
            L = vals[m]
            call[m] = L[min(i, len(L) - 1)]
        sub = (lambda f, i=i: progress((i + f) / n)) if progress else None
        r = _norm(t.fn(**call, **{**extra, "_progress": sub}), outs)
        for o in outs:
            collected[o].append(r.get(o))
        if progress:
            progress((i + 1) / n)
    return collected


def _norm(r, outs):
    if isinstance(r, dict) and set(r) <= set(outs) and r:
        return r
    return {outs[0]: r}


def solve(graph, cache, dirty, refs, defs, emit):
    """Evaluate dirty nodes in order. cache: {nid: outputs} for clean nodes."""
    results = dict(cache)
    order = topo_order(graph)
    for nid in order:
        if nid not in dirty:
            continue
        node = graph["nodes"][nid]
        if node.get("disabled"):
            results.pop(nid, None)
            emit("error", nid, "Disabled")
            continue
        emit("start", nid)
        t0 = time.time()
        last = [0.0]

        def progress(f, nid=nid):
            now = time.time()
            if now - last[0] > 0.05:
                last[0] = now
                emit("progress", nid, float(max(0.0, min(1.0, f))))

        try:
            out = evaluate(graph, nid, results, refs, defs, progress)
            results[nid] = out
            emit("result", nid, out, time.time() - t0)
        except Exception as e:
            results.pop(nid, None)
            tb = traceback.format_exc(limit=3)
            emit("error", nid, f"{type(e).__name__}: {e}", tb)


def worker_main(inq, outq):
    """Persistent worker process loop."""
    while True:
        msg = inq.get()
        if msg is None:
            break
        job, graph, cache, dirty, refs, defs = msg
        try:
            solve(graph, cache, set(dirty), refs, defs, lambda *a: outq.put((job,) + a))
        except Exception as e:
            outq.put((job, "fatal", str(e)))
        outq.put((job, "done"))
