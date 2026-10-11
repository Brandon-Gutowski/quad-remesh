# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

"""Triangle mesh preparation: cleaning, refinement and connectivity."""

import numpy as np

from .geom import norm, normalize


def edge_keys(a, b, n):
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    return np.minimum(a, b) * n + np.maximum(a, b)


def clean(V, F, flag_pairs=None, tol=None):
    """Merge coincident vertices and drop degenerate / duplicate triangles.

    flag_pairs is an optional (k,2) array of vertex pairs (e.g. marked sharp
    edges) that is remapped alongside the vertices.
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    if tol is None:
        diag = np.linalg.norm(V.max(0) - V.min(0)) if len(V) else 1.0
        tol = max(diag * 1e-6, 1e-12)
    q = np.round(V / tol).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    V2 = V[first]
    F2 = inv[F]
    ok = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 2] != F2[:, 0])
    F2 = F2[ok]
    if len(F2):
        _, uidx = np.unique(np.sort(F2, axis=1), axis=0, return_index=True)
        F2 = F2[np.sort(uidx)]
    # Drop zero-area triangles.
    if len(F2):
        area = norm(np.cross(V2[F2[:, 1]] - V2[F2[:, 0]], V2[F2[:, 2]] - V2[F2[:, 0]]))
        F2 = F2[area > tol * tol]
    # Drop unreferenced vertices.
    used = np.zeros(len(V2), bool)
    used[F2.reshape(-1)] = True
    remap = np.cumsum(used) - 1
    V3 = V2[used]
    F3 = remap[F2]
    pairs = None
    if flag_pairs is not None and len(flag_pairs):
        p = inv[np.asarray(flag_pairs, dtype=np.int64)]
        p = p[(p[:, 0] != p[:, 1]) & used[p[:, 0]] & used[p[:, 1]]]
        pairs = remap[p]
    return V3, F3, pairs


def split_long_edges(V, F, max_len, flag_pairs=None, max_iter=12, max_faces=4_000_000):
    """Conforming refinement: bisect every edge longer than max_len.

    Child edges of flagged edges stay flagged.
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    pairs = None if flag_pairs is None else np.asarray(flag_pairs, dtype=np.int64).reshape(-1, 2)
    for _ in range(max_iter):
        n = len(V)
        a = F
        b = F[:, [1, 2, 0]]
        keys = edge_keys(a, b, n)
        L = np.linalg.norm(V[a] - V[b], axis=2)
        long = L > max_len
        if not long.any() or len(F) > max_faces:
            break
        lk = np.unique(keys[long])
        ea, eb = lk // n, lk % n
        V = np.vstack([V, 0.5 * (V[ea] + V[eb])])
        M = np.full(F.shape, -1, dtype=np.int64)
        M[long] = n + np.searchsorted(lk, keys[long])
        cnt = long.sum(1)
        out = [F[cnt == 0]]

        sel = np.nonzero(cnt == 1)[0]
        if len(sel):
            k = np.argmax(long[sel], axis=1)
            idx = (k[:, None] + np.arange(3)) % 3
            f = F[sel[:, None], idx]
            m0 = M[sel, k]
            out.append(np.stack([f[:, 0], m0, f[:, 2]], 1))
            out.append(np.stack([m0, f[:, 1], f[:, 2]], 1))

        sel = np.nonzero(cnt == 2)[0]
        if len(sel):
            s = np.argmin(long[sel], axis=1)       # the short edge -> becomes edge 2
            start = (s + 1) % 3
            idx = (start[:, None] + np.arange(3)) % 3
            f = F[sel[:, None], idx]
            mm = M[sel[:, None], idx]
            m0, m1 = mm[:, 0], mm[:, 1]
            out.append(np.stack([m0, f[:, 1], m1], 1))
            # Split the remaining quad (v0, m0, m1, v2) along its shorter diagonal.
            d1 = np.linalg.norm(V[f[:, 0]] - V[m1], axis=1)
            d2 = np.linalg.norm(V[m0] - V[f[:, 2]], axis=1)
            use1 = d1 <= d2
            t1 = np.where(use1[:, None], np.stack([f[:, 0], m0, m1], 1), np.stack([f[:, 0], m0, f[:, 2]], 1))
            t2 = np.where(use1[:, None], np.stack([f[:, 0], m1, f[:, 2]], 1), np.stack([m0, m1, f[:, 2]], 1))
            out.append(t1)
            out.append(t2)

        sel = np.nonzero(cnt == 3)[0]
        if len(sel):
            f = F[sel]
            m = M[sel]
            out.append(np.stack([f[:, 0], m[:, 0], m[:, 2]], 1))
            out.append(np.stack([m[:, 0], f[:, 1], m[:, 1]], 1))
            out.append(np.stack([m[:, 2], m[:, 1], f[:, 2]], 1))
            out.append(np.stack([m[:, 0], m[:, 1], m[:, 2]], 1))
        F = np.vstack(out)

        if pairs is not None and len(pairs):
            pk = edge_keys(pairs[:, 0], pairs[:, 1], n)
            hit = np.isin(pk, lk)
            if hit.any():
                mid = n + np.searchsorted(lk, pk[hit])
                ph = pairs[hit]
                pairs = np.vstack([pairs[~hit],
                                   np.stack([ph[:, 0], mid], 1),
                                   np.stack([mid, ph[:, 1]], 1)])
    return V, F, pairs


class TriMesh:
    """Triangle mesh with the derived quantities the solvers need."""

    def __init__(self, V, F):
        self.V = np.asarray(V, dtype=np.float64)
        self.F = np.asarray(F, dtype=np.int64)
        n = len(self.V)
        V, F = self.V, self.F
        cr = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        dbl = norm(cr)
        self.face_area = 0.5 * dbl
        self.face_normal = cr / np.maximum(dbl, 1e-30)[:, None]
        vn = np.zeros((n, 3))
        for k in range(3):
            np.add.at(vn, F[:, k], cr)
        bad = np.linalg.norm(vn, axis=1) < 1e-30
        vn[bad] = [0.0, 0.0, 1.0]
        self.vertex_normal = normalize(vn)
        va = np.zeros(n)
        for k in range(3):
            np.add.at(va, F[:, k], self.face_area / 3.0)
        self.vertex_area = np.maximum(va, 1e-30)
        self.area = float(self.face_area.sum())

        # Undirected edges and their incident faces.
        a = F.reshape(-1)
        b = F[:, [1, 2, 0]].reshape(-1)
        keys = edge_keys(a, b, n)
        face_of = np.repeat(np.arange(len(F)), 3)
        order = np.argsort(keys, kind="stable")
        ks = keys[order]
        uniq, start, counts = np.unique(ks, return_index=True, return_counts=True)
        self.edges = np.stack([uniq // n, uniq % n], 1)
        self.edge_keys = uniq
        self.edge_face_count = counts
        f0 = face_of[order[start]]
        f1 = np.where(counts >= 2, face_of[order[np.minimum(start + 1, len(ks) - 1)]], -1)
        self.edge_faces = np.stack([f0, f1], 1)
        self.edge_length = norm(V[self.edges[:, 0]] - V[self.edges[:, 1]])

        # Directed adjacency (both directions), sorted by source -> CSR.
        ei = np.concatenate([self.edges[:, 0], self.edges[:, 1]])
        ej = np.concatenate([self.edges[:, 1], self.edges[:, 0]])
        order = np.lexsort((ej, ei))
        self.ei = ei[order]
        self.ej = ej[order]
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(self.ei, minlength=n))])

    def edge_index(self, pairs):
        """Index into self.edges for each (a,b) pair, -1 if absent."""
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if not len(pairs):
            return np.zeros(0, dtype=np.int64)
        k = edge_keys(pairs[:, 0], pairs[:, 1], len(self.V))
        pos = np.searchsorted(self.edge_keys, k)
        pos = np.minimum(pos, len(self.edge_keys) - 1)
        return np.where(self.edge_keys[pos] == k, pos, -1)


def smooth_normals(mesh, blocked_edges, radius):
    """Smooth vertex normals over roughly `radius`, never across blocked
    (feature) edges, so noisy surfaces still give a calm cross field."""
    N = mesh.vertex_normal.copy()
    mean_edge = float(mesh.edge_length.mean()) if len(mesh.edge_length) else 1.0
    iters = int(min(40, (radius / max(mean_edge, 1e-12)) ** 2))
    if iters < 1:
        return N
    ok = ~blocked_edges
    a = mesh.edges[ok, 0]
    b = mesh.edges[ok, 1]
    n = len(N)
    for _ in range(iters):
        acc = N.copy()
        for k in range(3):
            acc[:, k] += np.bincount(a, weights=N[b, k], minlength=n) + np.bincount(b, weights=N[a, k], minlength=n)
        N = normalize(acc)
    return N
