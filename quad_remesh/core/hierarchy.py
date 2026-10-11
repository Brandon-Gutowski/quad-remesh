# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

"""Multiresolution vertex hierarchy (Instant Meshes style) for fast field smoothing."""

import numpy as np

from .geom import dot, normalize


class Level:
    def __init__(self, P, N, A, ei, ej):
        self.P = P
        self.N = N
        self.A = A
        self.ei = ei
        self.ej = ej
        self.parent = None  # index into the next coarser level
        n = len(P)
        self.deg = np.bincount(ei, minlength=n).astype(np.float64)

    @property
    def n(self):
        return len(self.P)


def _match(level, rounds=12):
    P, N, A, ei, ej = level.P, level.N, level.A, level.ei, level.ej
    n = len(P)
    score = dot(N[ei], N[ej]) * -np.maximum(A[ei] / A[ej], A[ej] / A[ei])
    mate = np.full(n, -1, dtype=np.int64)
    for _ in range(rounds):
        free = mate < 0
        valid = free[ei] & free[ej] & (ei != ej)
        if not valid.any():
            break
        a, b, s = ei[valid], ej[valid], score[valid]
        order = np.lexsort((-s, a))
        a, b = a[order], b[order]
        first = np.ones(len(a), bool)
        first[1:] = a[1:] != a[:-1]
        bi, bj = a[first], b[first]
        best = np.full(n, -1, dtype=np.int64)
        best[bi] = bj
        mutual = best[bj] == bi
        mate[bi[mutual]] = bj[mutual]
    idx = np.arange(n)
    rep = np.where(mate < 0, idx, np.minimum(idx, mate))
    # Leftover singletons join the best-scoring matched neighbour's cluster.
    lone = mate < 0
    valid = lone[ei] & ~lone[ej]
    if valid.any():
        a, b, s = ei[valid], ej[valid], score[valid]
        order = np.lexsort((-s, a))
        a, b = a[order], b[order]
        first = np.ones(len(a), bool)
        first[1:] = a[1:] != a[:-1]
        rep[a[first]] = rep[b[first]]
    _, parent = np.unique(rep, return_inverse=True)
    return parent.reshape(-1)


def build_hierarchy(P, N, A, ei, ej, min_verts=48, max_levels=24):
    levels = [Level(P, N, A, ei, ej)]
    while levels[-1].n > min_verts and len(levels) < max_levels:
        L = levels[-1]
        parent = _match(L)
        nc = int(parent.max()) + 1
        if nc > 0.92 * L.n:
            break
        Ac = np.bincount(parent, weights=L.A, minlength=nc)
        Pc = np.stack([np.bincount(parent, weights=L.P[:, k] * L.A, minlength=nc) for k in range(3)], 1) / Ac[:, None]
        Nc = normalize(np.stack([np.bincount(parent, weights=L.N[:, k] * L.A, minlength=nc) for k in range(3)], 1))
        ci, cj = parent[L.ei], parent[L.ej]
        keep = ci != cj
        keys = np.unique(ci[keep].astype(np.int64) * nc + cj[keep])
        L.parent = parent
        levels.append(Level(Pc, Nc, Ac, keys // nc, keys % nc))
    return levels


def restrict_mask(levels, mask_fine):
    """Propagate a boolean per-vertex mask to all levels (any child)."""
    out = [mask_fine]
    for L in levels[:-1]:
        nc = int(L.parent.max()) + 1
        m = np.zeros(nc, bool)
        m[L.parent[out[-1]]] = True
        out.append(m)
    return out


def restrict_pick(levels, mask_fine, values_fine):
    """Propagate per-vertex values of masked vertices (one child wins)."""
    masks = [mask_fine]
    vals = [values_fine]
    for L in levels[:-1]:
        nc = int(L.parent.max()) + 1
        m = np.zeros(nc, bool)
        v = np.zeros((nc,) + vals[-1].shape[1:], dtype=vals[-1].dtype)
        src = np.nonzero(masks[-1])[0]
        m[L.parent[src]] = True
        v[L.parent[src]] = vals[-1][src]
        masks.append(m)
        vals.append(v)
    return masks, vals
