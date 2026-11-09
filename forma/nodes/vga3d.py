"""Voxel visibility graph analysis (VGA) for 3D interiors.

The computation core below is the NumPy/SciPy engine from the Grasshopper script
"Space Syntax Master v03" (Rhino 8 Python 3 component), kept as it was; only the Rhino
input/output layer is replaced by Forma geometry in ``forma.nodes.userlib``.

Model: ``space`` is one closed solid of the AIR volume; ``furniture`` solids remove the cells
they touch. Observer positions are cells a person can stand in; the walk band is the floor
layer and the visibility band is the same cells raised by ``eye_band`` cells.
"""
import math
import time

import numpy as np
import scipy.sparse as sp
from scipy import ndimage
from scipy.sparse.csgraph import dijkstra
from scipy.special import gammaln

CODE_VERSION = 3
STICKY_KEY = "ss_numpy_vga_cache"
SAMPLE_FRAC = 0.25       # sightline sample spacing = voxel_size * this
CHUNK = 256              # BFS sources per SciPy batch (memory control)
MAX_CELLS = 8_000_000    # safety cap on total grid cells
STANDABLE_COLUMNS = True # observer positions = floor cell free AND the
                         # whole column clear up to eye height.
                         # False = only floor + eye cell tested.

METRIC_NAMES = ["Connectivity", "Integration_HH", "Mean_Depth", "Node_Count",
                "Point_First_Moment", "Point_Second_Moment",
                "Harmonic_Mean_Depth", "Entropy", "Relativised_Entropy",
                "Intensity", "Integration_Pvalue", "Integration_Tekl",
                "Isovist_Area", "Choice", "Choice_Norm"]


# ======================================================================
#  PURE NUMPY CORE  (no Rhino calls below this banner)
# ======================================================================

def _tri_box_overlap(tri, centers, h):
    """Vectorised Akenine-Moller triangle / axis-aligned-box SAT test.
    tri: (3,3) triangle vertices. centers: (M,3) box centers.
    h: box half-size (scalar). Returns bool mask (M,)."""
    h = h + 1e-9          # tiny inflation: geometry exactly on a cell
    #                       boundary must register in BOTH cells
    v0 = tri[0] - centers
    v1 = tri[1] - centers
    v2 = tri[2] - centers
    ok = np.ones(len(centers), dtype=bool)

    # box-AABB vs triangle-AABB
    mn = np.minimum(np.minimum(v0, v1), v2)
    mx = np.maximum(np.maximum(v0, v1), v2)
    ok &= np.all(mn <= h, axis=1) & np.all(mx >= -h, axis=1)
    if not ok.any():
        return ok

    e0 = tri[1] - tri[0]
    e1 = tri[2] - tri[1]
    e2 = tri[0] - tri[2]

    # triangle plane vs box
    n = np.cross(e0, e1)
    r = h * np.abs(n).sum() + 1e-12
    ok &= np.abs(v0 @ n) <= r
    if not ok.any():
        return ok

    # 9 cross-axis tests
    eye = np.eye(3)
    for e in (e0, e1, e2):
        for u in eye:
            a = np.cross(u, e)
            s_ = np.abs(a).sum()
            if s_ < 1e-14:
                continue
            ra = h * s_ + 1e-12
            p0 = v0 @ a
            p1 = v1 @ a
            p2 = v2 @ a
            pmin = np.minimum(np.minimum(p0, p1), p2)
            pmax = np.maximum(np.maximum(p0, p1), p2)
            ok &= (pmin <= ra) & (pmax >= -ra)
            if not ok.any():
                return ok
    return ok


def _rasterize(tris, dims, origin, s, out=None):
    """Mark every voxel touched by any triangle as solid (conservative,
    exact SAT per candidate cell). tris: (N,3,3)."""
    if out is None:
        out = np.zeros(dims, dtype=bool)
    if len(tris) == 0:
        return out
    dmax = np.asarray(dims) - 1
    half = 0.5 * s
    for tri in tris:
        tmin = tri.min(axis=0)
        tmax = tri.max(axis=0)
        # +-1 cell so triangles lying exactly on a cell boundary are
        # tested against BOTH neighbouring cells (SAT decides).
        i0 = np.clip(np.floor((tmin - origin) / s).astype(int) - 1, 0, dmax)
        i1 = np.clip(np.floor((tmax - origin) / s).astype(int) + 1, 0, dmax)
        ii = np.arange(i0[0], i1[0] + 1)
        jj = np.arange(i0[1], i1[1] + 1)
        kk = np.arange(i0[2], i1[2] + 1)
        I, J, K = np.meshgrid(ii, jj, kk, indexing="ij")
        centers = np.stack(
            [origin[0] + (I + 0.5) * s,
             origin[1] + (J + 0.5) * s,
             origin[2] + (K + 0.5) * s], axis=-1).reshape(-1, 3)
        hit = _tri_box_overlap(tri, centers, half)
        if hit.any():
            idx = np.stack([I, J, K], axis=-1).reshape(-1, 3)[hit]
            out[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return out


def _flood_from_border(open_mask):
    """Cells of open_mask reachable (6-connected) from the grid border."""
    seed = np.zeros_like(open_mask)
    seed[0, :, :] = seed[-1, :, :] = True
    seed[:, 0, :] = seed[:, -1, :] = True
    seed[:, :, 0] = seed[:, :, -1] = True
    seed &= open_mask
    return ndimage.binary_propagation(seed, mask=open_mask)


def _free_grid(space_tris, furn_tris, dims, origin, s):
    """Boolean grid of free interior air cells + diagnostics."""
    space_shell = _rasterize(space_tris, dims, origin, s)
    outside = _flood_from_border(~space_shell)
    inside_air = ~space_shell & ~outside

    furn_solid = np.zeros(dims, dtype=bool)
    if len(furn_tris):
        furn_shell = _rasterize(furn_tris, dims, origin, s)
        furn_solid = ~_flood_from_border(~furn_shell)  # shell + interior

    free = inside_air & ~furn_solid
    removed = int((inside_air & furn_solid).sum())
    return free, removed


def _build_edges(pts, free, origin, s, R2, log=None):
    """Symmetric visibility adjacency among pts (n,3): a pair is linked
    when every sample (spacing s*SAMPLE_FRAC) along the segment falls
    in a free cell. Returns CSR adjacency + (m,2) edge index pairs."""
    n = len(pts)
    h = s * SAMPLE_FRAC
    dims = free.shape
    rows, cols = [], []
    t_last = time.time()
    for ai in range(n - 1):
        p0 = pts[ai]
        d = pts[ai + 1:] - p0
        d2 = np.einsum("ij,ij->i", d, d)
        cand = np.nonzero(d2 <= R2)[0] if R2 is not None \
            else np.arange(len(d))
        if len(cand) == 0:
            continue
        dc = d[cand]
        L = np.sqrt(d2[cand])
        K = np.maximum((L / h).astype(int) + 1, 1)
        Kmax = int(K.max())
        if Kmax > 1:
            steps = np.arange(1, Kmax)
            t = steps[None, :] / K[:, None]              # (m, Kmax-1)
            interior = steps[None, :] < K[:, None]
            smp = p0[None, None, :] + t[:, :, None] * dc[:, None, :]
            ci = np.floor((smp - origin) / s).astype(int)
            np.clip(ci[..., 0], 0, dims[0] - 1, out=ci[..., 0])
            np.clip(ci[..., 1], 0, dims[1] - 1, out=ci[..., 1])
            np.clip(ci[..., 2], 0, dims[2] - 1, out=ci[..., 2])
            f = free[ci[..., 0], ci[..., 1], ci[..., 2]]
            okpair = np.all(f | ~interior, axis=1)
        else:
            okpair = np.ones(len(cand), dtype=bool)
        hit = cand[okpair] + ai + 1
        if len(hit):
            rows.append(np.full(len(hit), ai, dtype=np.int32))
            cols.append(hit.astype(np.int32))
        if log and time.time() - t_last > 3.0:
            log("    ... sightlines %d / %d cells" % (ai + 1, n))
            t_last = time.time()
    if rows:
        r = np.concatenate(rows)
        c = np.concatenate(cols)
    else:
        r = np.zeros(0, dtype=np.int32)
        c = np.zeros(0, dtype=np.int32)
    data = np.ones(len(r), dtype=np.int8)
    A = sp.csr_matrix((data, (r, c)), shape=(n, n))
    A = (A + A.T).tocsr()
    A.data[:] = 1
    edges = np.stack([r, c], axis=1) if len(r) else np.zeros((0, 2), int)
    return A, edges


def _build_edges_3d(src_glob, pts, free, origin, s, R2, log=None):
    """2.5D visibility (old-script semantics): every source node
    (global index into pts = ALL free cells) is linked to every free
    3D cell it can see. Returns symmetric CSR over all n cells +
    (m,2) global edge index pairs."""
    n = len(pts)
    h = s * SAMPLE_FRAC
    dims = free.shape
    is_src = np.zeros(n, dtype=bool)
    is_src[src_glob] = True
    rows, cols = [], []
    t_last = time.time()
    BLK = 4096                       # candidate block (memory control)
    for si in range(len(src_glob)):
        gi = int(src_glob[si])
        p0 = pts[gi]
        d = pts - p0
        d2 = np.einsum("ij,ij->i", d, d)
        m_ = d2 > 1e-12
        if R2 is not None:
            m_ &= d2 <= R2
        # don't duplicate hub-hub pairs
        m_ &= ~(is_src & (np.arange(n) < gi))
        cand = np.nonzero(m_)[0]
        for b0 in range(0, len(cand), BLK):
            cb = cand[b0:b0 + BLK]
            dc = d[cb]
            L = np.sqrt(d2[cb])
            K = np.maximum((L / h).astype(int) + 1, 1)
            Kmax = int(K.max())
            if Kmax > 1:
                steps = np.arange(1, Kmax)
                t = steps[None, :] / K[:, None]
                interior = steps[None, :] < K[:, None]
                smp = p0[None, None, :] + t[:, :, None] * dc[:, None, :]
                ci = np.floor((smp - origin) / s).astype(int)
                np.clip(ci[..., 0], 0, dims[0] - 1, out=ci[..., 0])
                np.clip(ci[..., 1], 0, dims[1] - 1, out=ci[..., 1])
                np.clip(ci[..., 2], 0, dims[2] - 1, out=ci[..., 2])
                f = free[ci[..., 0], ci[..., 1], ci[..., 2]]
                okpair = np.all(f | ~interior, axis=1)
            else:
                okpair = np.ones(len(cb), dtype=bool)
            hit = cb[okpair]
            if len(hit):
                rows.append(np.full(len(hit), gi, dtype=np.int32))
                cols.append(hit.astype(np.int32))
        if log and time.time() - t_last > 3.0:
            log("    ... 3D sightlines %d / %d hubs" % (si + 1, len(src_glob)))
            t_last = time.time()
    if rows:
        r = np.concatenate(rows)
        c = np.concatenate(cols)
    else:
        r = np.zeros(0, dtype=np.int32)
        c = np.zeros(0, dtype=np.int32)
    data = np.ones(len(r), dtype=np.int8)
    A = sp.csr_matrix((data, (r, c)), shape=(n, n))
    A = (A + A.T).tocsr()
    A.data[:] = 1
    edges = np.stack([r, c], axis=1) if len(r) else np.zeros((0, 2), int)
    return A, edges


def _dvalue(k):
    if k <= 2:
        return 1.0
    return 2.0 * (k * (math.log((k + 2.0) / 3.0, 2) - 1.0) + 1.0) \
        / ((k - 1.0) * (k - 2.0))


def _pvalue(k):
    if k <= 2:
        return 1.0
    return 2.0 * (k - 5.9544 * math.sqrt(k) + 2.6062) \
        / ((k - 1.0) * (k - 2.0))


def _tekl_integration(k, td):
    if k <= 2:
        return 0.0
    base = td - k + 1.0
    if base <= 1.0 + 1e-9:
        return 0.0
    num = 0.5 * (k - 2.0)
    if num <= 0.0:
        return 0.0
    return math.log(num) / math.log(base)


def _graph_metrics(A, cell_measure, log=None, sources=None, ntotal=None):
    """All Group-1 measures, chunked SciPy BFS.
    sources: node subset to compute FOR (default: all nodes).
    ntotal:  system size n used in the formulas (default: A.shape[0]).
    Result arrays are indexed in `sources` order."""
    nA = A.shape[0]
    if sources is None:
        sources = np.arange(nA)
    sources = np.asarray(sources)
    nsrc = len(sources)
    n = int(ntotal) if ntotal is not None else nA
    res = dict((nm, np.zeros(nsrc)) for nm in METRIC_NAMES)
    deg = np.diff(A.indptr).astype(float)[sources]
    res["Connectivity"] = deg
    res["Isovist_Area"] = (deg + 1.0) * cell_measure
    Dk, Pk = _dvalue(n), _pvalue(n)
    LOG300 = math.log(1e-300)
    chunk = max(1, min(CHUNK, int(2.0e7 / max(nA, 1))))  # ~160MB cap

    for c0 in range(0, nsrc, chunk):
        pos = np.arange(c0, min(c0 + chunk, nsrc))
        idxs = sources[pos]
        D = dijkstra(A, unweighted=True, indices=idxs)
        fin = np.isfinite(D) & (D > 0)
        Dm = np.where(fin, D, 0.0)
        reach = fin.sum(axis=1)
        td_all = Dm.sum(axis=1)
        sm_all = (Dm * Dm).sum(axis=1)
        inv_all = np.where(fin, 1.0 / np.where(fin, D, 1.0), 0.0).sum(axis=1)

        for ri, v in enumerate(pos):
            k = int(reach[ri])
            res["Node_Count"][v] = k + 1
            if k <= 0:
                continue
            td = td_all[ri]
            mdv = td / k
            res["Mean_Depth"][v] = mdv
            res["Point_First_Moment"][v] = td
            res["Point_Second_Moment"][v] = sm_all[ri]
            res["Harmonic_Mean_Depth"][v] = \
                k / inv_all[ri] if inv_all[ri] > 0 else 0.0

            dd = D[ri][fin[ri]].astype(int)
            counts = np.bincount(dd)
            depths = np.nonzero(counts)[0]
            depths = depths[depths > 0]
            p = counts[depths] / float(k)
            ent = float(-(p * np.log2(p)).sum())
            res["Entropy"][v] = ent
            lam = mdv - 1.0
            if lam > 1e-9:
                logq = (-lam + (depths - 1) * math.log(lam)
                        - gammaln(depths))          # log((d-1)!) = gammaln(d)
                keep = logq > LOG300
                if keep.any():
                    rel = (p[keep] * (np.log2(p[keep])
                                      - logq[keep] / math.log(2.0))).sum()
                    res["Relativised_Entropy"][v] = float(rel)
            res["Intensity"][v] = ent / mdv if mdv > 1e-9 else 0.0

            ra = 2.0 * (mdv - 1.0) / (n - 2.0) if n > 2 else 0.0
            rra_d = ra / Dk if Dk > 0 else 0.0
            res["Integration_HH"][v] = (1.0 / rra_d) if rra_d > 1e-9 else 0.0
            rra_p = ra / Pk if abs(Pk) > 1e-12 else 0.0
            res["Integration_Pvalue"][v] = \
                (1.0 / rra_p) if rra_p > 1e-9 else 0.0
            res["Integration_Tekl"][v] = _tekl_integration(n, td)
        if log:
            log("    ... measures %d / %d" % (min(c0 + chunk, nsrc), nsrc))
    return res


def _add_choice(A, res, log=None):
    """Betweenness (Brandes, level-synchronous, sparse-matvec form)."""
    n = A.shape[0]
    Af = A.astype(np.float64).tocsr()
    cb = np.zeros(n)
    for c0 in range(0, n, CHUNK):
        idxs = np.arange(c0, min(c0 + CHUNK, n))
        D = dijkstra(A, unweighted=True, indices=idxs)
        for ri, s0 in enumerate(idxs):
            d = D[ri]
            fin = np.isfinite(d)
            if fin.sum() <= 1:
                continue
            maxd = int(d[fin].max())
            sigma = np.zeros(n)
            sigma[s0] = 1.0
            for l in range(1, maxd + 1):
                x = np.where(d == l - 1, sigma, 0.0)
                y = Af.dot(x)
                cur = d == l
                sigma[cur] = y[cur]
            delta = np.zeros(n)
            for l in range(maxd, 0, -1):
                cur = (d == l) & (sigma > 0)
                w = np.zeros(n)
                w[cur] = (1.0 + delta[cur]) / sigma[cur]
                y = Af.dot(w)
                prv = d == l - 1
                delta[prv] += sigma[prv] * y[prv]
            delta[s0] = 0.0
            cb += delta
        if log:
            log("    ... Choice %d / %d sources" % (min(c0 + CHUNK, n), n))
    c = 0.5 * cb
    td = res["Point_First_Moment"]
    with np.errstate(divide="ignore", invalid="ignore"):
        cn = np.where(td + 3.0 > 1.0,
                      np.log10(c + 1.0) / np.log10(td + 3.0), 0.0)
    res["Choice"] = c
    res["Choice_Norm"] = np.nan_to_num(cn)


def _compute_core(space_tris, furn_tris, s, eb, R2, want_choice, log,
                  vis3d=False):
    """Full pipeline on plain arrays. Returns a result dict."""
    allpts = space_tris.reshape(-1, 3)
    bbmin = allpts.min(axis=0)
    bbmax = allpts.max(axis=0)
    origin = bbmin - 2.0 * s
    dims = tuple(np.ceil((bbmax - origin) / s).astype(int) + 3)
    ncells = dims[0] * dims[1] * dims[2]
    if ncells > MAX_CELLS:
        raise Exception("Grid would need %d cells (>%d). Increase "
                        "voxel_size." % (ncells, MAX_CELLS))
    log("Grid %d x %d x %d (%d cells), origin %s"
        % (dims[0], dims[1], dims[2], ncells, np.round(origin, 3)))

    t0 = time.time()
    free, removed = _free_grid(space_tris, furn_tris, dims, origin, s)
    nfree = int(free.sum())
    if nfree == 0:
        raise Exception("No free cells. Is 'space' a closed solid of the "
                        "AIR volume (voxel_size smaller than the room)?")
    log("Voxelisation: %d free cells (%d removed by furniture)  [%.1fs]"
        % (nfree, removed, time.time() - t0))

    V = np.argwhere(free)                       # (nV,3) int
    kmin = int(V[:, 2].min())
    ke = kmin + eb
    if ke >= free.shape[2]:
        raise Exception("eye_band=%d puts the eye level above the grid "
                        "(k=%d, max %d). Lower eye_band."
                        % (eb, ke, free.shape[2] - 1))

    # An observer position must be a place a PERSON CAN STAND: the floor
    # cell is free AND the whole column up to eye height is free. Both
    # bands are derived from this one set, so walk cell i and eye cell i
    # are the same plan position (heatmaps are directly comparable).
    n_floor = int(free[:, :, kmin].sum())
    if STANDABLE_COLUMNS:
        col = free[:, :, kmin:ke + 1].all(axis=2)
    else:
        col = free[:, :, kmin] & free[:, :, ke]
    ij = np.argwhere(col)
    if len(ij) < 2:
        raise Exception("Fewer than 2 standable positions (floor cell free "
                        "and clear up to eye level). Check eye_band=%d and "
                        "that furniture is not sealing the room." % eb)
    Wc = np.column_stack([ij, np.full(len(ij), kmin, dtype=ij.dtype)])
    Hc = np.column_stack([ij, np.full(len(ij), ke, dtype=ij.dtype)])
    log("Standable positions: %d  (%d floor cells dropped: furniture, "
        "low soffits or blocked head height)" % (len(ij), n_floor - len(ij)))

    Wpts = origin + (Wc + 0.5) * s
    Hpts = origin + (Hc + 0.5) * s

    t0 = time.time()
    Aw, Ew = _build_edges(Wpts, free, origin, s, R2, log)
    log("Walk graph: %d edges  [%.1fs]" % (len(Ew), time.time() - t0))

    t0 = time.time()
    if vis3d:
        if R2 is None:
            log("WARNING: vis_3d without vis_radius can be very slow "
                "and memory-heavy - a radius is strongly recommended.")
        Vpts = origin + (V + 0.5) * s
        lut = np.full(free.shape, -1, dtype=np.int64)
        lut[V[:, 0], V[:, 1], V[:, 2]] = np.arange(len(V))
        Hglob = lut[Hc[:, 0], Hc[:, 1], Hc[:, 2]]   # eye cells, global idx
        assert (Hglob >= 0).all()
        Av, Ev = _build_edges_3d(Hglob, Vpts, free, origin, s, R2, log)
        log("Vis graph (2.5D, eye band -> all 3D cells): %d edges  "
            "[%.1fs]" % (len(Ev), time.time() - t0))
    else:
        Av, Ev = _build_edges(Hpts, free, origin, s, R2, log)
        log("Vis graph:  %d edges  [%.1fs]" % (len(Ev), time.time() - t0))

    t0 = time.time()
    wmet = _graph_metrics(Aw, s * s, log)
    if vis3d:
        vmet = _graph_metrics(Av, s * s * s, log,
                              sources=Hglob, ntotal=len(V))
    else:
        vmet = _graph_metrics(Av, s * s, log)
    log("Measures done  [%.1fs]" % (time.time() - t0))

    if want_choice:
        t0 = time.time()
        _add_choice(Aw, wmet, log)
        log("Choice (walk band) done  [%.1fs]" % (time.time() - t0))

    return {"origin": origin, "dims": dims, "s": s, "vis3d": bool(vis3d),
            "V": V, "W": Wc, "H": Hc, "kmin": kmin, "ke": ke,
            "Wpts": Wpts, "Hpts": Hpts,
            "w_edges": Ew, "v_edges": Ev,
            "w_metrics": wmet, "v_metrics": vmet}


# ======================================================================
#  RHINO / GRASSHOPPER SIDE
