"""Tiny helpers for the build pipeline's smoke test (must live in an importable module for spawn)."""


def worker_check(q):
    from forma import brep as B
    q.put(B.box((0, 0, 0), 1000, 1000, 1000).volume())
