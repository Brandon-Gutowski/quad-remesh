"""4-PoSy position field: per-vertex lattice origin aligned with the cross field."""

import numpy as np

from .geom import dot, normalize, rotate_between, tangent_frames
from .hierarchy import restrict_pick


def round4(o, q, t, p, h):
    """Lattice point (origin o, axes q,t, spacing h) nearest to p."""
    d = p - o
    return o + q * (np.round(dot(q, d) / h) * h)[:, None] + t * (np.round(dot(t, d) / h) * h)[:, None]


def floor4(o, q, t, p, h):
    d = p - o
    return o + q * (np.floor(dot(q, d) / h) * h)[:, None] + t * (np.floor(dot(t, d) / h) * h)[:, None]


def middle_point(p0, n0, p1, n1):
    n0p0 = dot(n0, p0)
    n0p1 = dot(n0, p1)
    n1p0 = dot(n1, p0)
    n1p1 = dot(n1, p1)
    n0n1 = dot(n0, n1)
    denom = 1.0 / (1.0 - n0n1 * n0n1 + 1e-4)
    l0 = 2.0 * (n0p1 - n0p0 - n0n1 * (n1p0 - n1p1)) * denom
    l1 = 2.0 * (n1p0 - n1p1 - n0n1 * (n0p1 - n0p0)) * denom
    return 0.5 * (p0 + p1) - 0.25 * (n0 * l0[:, None] + n1 * l1[:, None])


_CORNERS = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)


def compat_pair(Pi, Ni, Qi, Ti, Oi, Pj, Nj, Qj, Tj, Oj, h):
    """Closest pair of lattice points (one from each vertex's lattice) near the
    edge midpoint. Returns (point of i's lattice, point of j's lattice)."""
    return compat_pair_mid(middle_point(Pi, Ni, Pj, Nj), Qi, Ti, Oi, Qj, Tj, Oj, h)


def compat_pair_mid(mid, Qi, Ti, Oi, Qj, Tj, Oj, h):
    base = floor4(Oi, Qi, Ti, mid, h)
    hh = np.asarray(h, dtype=np.float64)
    hc = hh[:, None] if hh.ndim else hh
    best_cost = np.full(len(mid), np.inf)
    bi = base.copy()
    bj = base.copy()
    for cx, cy in _CORNERS:
        c = base + Qi * (cx * hc) + Ti * (cy * hc)
        cj = round4(Oj, Qj, Tj, c, h)
        d = cj - c
        cost = dot(d, d)
        better = cost < best_cost
        best_cost = np.where(better, cost, best_cost)
        bi[better] = c[better]
        bj[better] = cj[better]
    return bi, bj


class _Slots:
    """Per-level neighbour slots with everything that stays constant during
    the sweeps gathered once (positions, frames, edge midpoints, spacing)."""

    def __init__(self, P, N, Q, T, ej, indptr, hv):
        deg = np.diff(indptr)
        self.items = []
        for k in range(int(deg.max()) if len(deg) else 0):
            i = np.nonzero(deg > k)[0]
            j = ej[indptr[i] + k]
            mid = middle_point(P[i], N[i], P[j], N[j])
            self.items.append((i, j, mid, Q[i], T[i], Q[j], T[j], N[i], P[i], 0.5 * (hv[i] + hv[j])))


def _gs_sweep(O, slots):
    """One smoothing sweep: every vertex folds in its neighbours one at a time
    (running average, as in Instant Meshes), vectorized across vertices."""
    S = O.copy()
    for k, (i, j, mid, Qi, Ti, Qj, Tj, Ni, Pi, he) in enumerate(slots.items):
        a, b = compat_pair_mid(mid, Qi, Ti, S[i], Qj, Tj, O[j], he)
        s = (a * k + b) / (k + 1.0)
        S[i] = s - Ni * np.einsum("ij,ij->i", Ni, s - Pi)[:, None]
    return S


def _restrict_q(levels, Q):
    """Average the fine cross field onto every level (4-RoSy aware)."""
    out = [Q]
    for li, L in enumerate(levels[:-1]):
        C = levels[li + 1]
        e1, e2 = tangent_frames(C.N)
        qc = rotate_between(out[-1], L.N, C.N[L.parent])
        ang = np.arctan2(dot(qc, e2[L.parent]), dot(qc, e1[L.parent]))
        z = np.exp(4j * ang) * L.A
        zs = np.bincount(L.parent, weights=z.real, minlength=C.n) + 1j * np.bincount(L.parent, weights=z.imag, minlength=C.n)
        th = np.angle(zs) / 4.0
        out.append(np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
    return out


def _apply_constraints(O, P, N, line_mask, line_pt, line_dir, corner_mask, corner_pt):
    if line_mask.any():
        perp = normalize(np.cross(N[line_mask], line_dir[line_mask]))
        d = O[line_mask] - line_pt[line_mask]
        O[line_mask] = O[line_mask] - perp * dot(perp, d)[:, None]
    if corner_mask.any():
        O[corner_mask] = corner_pt[corner_mask]
    return O


def solve_position(levels, Q, h, line_mask, line_dir, corner_mask, iters_coarse=None, iters_fine=None):
    import os as _os
    iters_coarse = iters_coarse or int(_os.environ.get("QR_PIC", "30"))
    iters_fine = iters_fine or int(_os.environ.get("QR_PIF", "40"))
    """Generator; yields progress and returns (lattice origins, field) at the fine level."""
    nlev = len(levels)
    Qs = _restrict_q(levels, Q)
    P0 = levels[0].P
    hs = [np.broadcast_to(np.asarray(h, dtype=np.float64), (levels[0].n,)).copy()]
    for L in levels[:-1]:
        nc = int(L.parent.max()) + 1
        w = np.bincount(L.parent, weights=L.A * hs[-1], minlength=nc)
        hs.append(w / np.bincount(L.parent, weights=L.A, minlength=nc))
    lm, lv = restrict_pick(levels, line_mask, np.concatenate([P0, line_dir], 1))
    cm, cv = restrict_pick(levels, corner_mask, P0)
    O = None
    for li in range(nlev - 1, -1, -1):
        L = levels[li]
        P, N = L.P, L.N
        hv = hs[li]
        Ql = normalize(Qs[li] - N * dot(Qs[li], N)[:, None])
        Tl = np.cross(N, Ql)
        line_m = lm[li] & ~cm[li]
        line_pt, line_d = lv[li][:, :3], lv[li][:, 3:]
        if O is None:
            O = P.copy()
        else:
            Oc = O[L.parent]
            O = Oc - N * dot(N, Oc - P)[:, None]
            O = round4(O, Ql, Tl, P, hv)
        O = _apply_constraints(O, P, N, line_m, line_pt, line_d, cm[li], cv[li])
        indptr = np.concatenate([[0], np.cumsum(np.bincount(L.ei, minlength=L.n))])
        slots = _Slots(P, N, Ql, Tl, L.ej, indptr, hv)
        iters = iters_fine if li == 0 else iters_coarse
        for it in range(iters):
            O = _gs_sweep(O, slots)
            O = round4(O, Ql, Tl, P, hv)
            O = _apply_constraints(O, P, N, line_m, line_pt, line_d, cm[li], cv[li])
            if it % 5 == 0:
                yield (nlev - 1 - li + it / iters) / nlev
    return O, Ql
