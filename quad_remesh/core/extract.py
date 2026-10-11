# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

"""Extract a polygon mesh from the position field (Instant Meshes style)."""

import numpy as np

from .geom import dot, newell_normal, normalize, tangent_frames
from .orientation import match4
from .position import compat_pair


def _components(n, a, b):
    """Connected components of an undirected graph via label propagation."""
    lab = np.arange(n)
    if not len(a):
        return lab
    while True:
        m = np.minimum(lab[a], lab[b])
        new = lab.copy()
        np.minimum.at(new, a, m)
        np.minimum.at(new, b, m)
        new = new[new]  # pointer jumping
        if np.array_equal(new, lab):
            break
        lab = new
    return lab


def _bounded_clusters(n, a, b, err, O, radius):
    """Union-find over zero-offset edges in order of increasing error; a merge
    is refused when it would put the two cluster centroids more than `radius`
    apart, so clusters never drift across lattice cells."""
    parent = list(range(n))
    cnt = [1] * n
    sx = O[:, 0].tolist()
    sy = O[:, 1].tolist()
    sz = O[:, 2].tolist()
    rad = np.broadcast_to(np.asarray(radius, dtype=np.float64), (n,)).tolist()

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    order = np.argsort(err, kind="stable")
    al = a[order].tolist()
    bl = b[order].tolist()
    for x, y in zip(al, bl):
        rx, ry = find(x), find(y)
        if rx == ry:
            continue
        cx, cy = cnt[rx], cnt[ry]
        dx = sx[rx] / cx - sx[ry] / cy
        dy = sy[rx] / cx - sy[ry] / cy
        dz = sz[rx] / cx - sz[ry] / cy
        rr = 0.5 * (rad[rx] + rad[ry])
        if dx * dx + dy * dy + dz * dz > rr * rr:
            continue
        if cx < cy:
            rx, ry = ry, rx
        parent[ry] = rx
        cnt[rx] += cnt[ry]
        sx[rx] += sx[ry]
        sy[rx] += sy[ry]
        sz[rx] += sz[ry]
    return np.array([find(x) for x in range(n)])


def _prune_low_degree(n, ea, eb):
    while True:
        deg = np.bincount(np.concatenate([ea, eb]), minlength=n)
        bad = deg == 1
        keep = ~(bad[ea] | bad[eb])
        if keep.all():
            return ea, eb
        ea, eb = ea[keep], eb[keep]


def trace_faces(VP, VN, ea, eb, max_len=12):
    """Faces of the embedded graph: cycles of the 'next half-edge' permutation."""
    n = len(VP)
    if not len(ea):
        return []
    src = np.concatenate([ea, eb])
    dst = np.concatenate([eb, ea])
    e1, e2 = tangent_frames(VN)
    d = VP[dst] - VP[src]
    ang = np.arctan2(dot(d, e2[src]), dot(d, e1[src]))
    order = np.lexsort((ang, src))
    src, dst = src[order], dst[order]
    H = len(src)
    deg = np.bincount(src, minlength=n)
    indptr = np.concatenate([[0], np.cumsum(deg)])
    keys = src.astype(np.int64) * n + dst
    korder = np.argsort(keys)
    tk = dst.astype(np.int64) * n + src
    twin = korder[np.searchsorted(keys[korder], tk)]
    kpos = twin - indptr[dst]
    nxt = indptr[dst] + (kpos - 1) % deg[dst]

    visited = np.zeros(H, bool)
    faces = []
    nxt_l = nxt.tolist()
    src_l = src.tolist()
    for h0 in range(H):
        if visited[h0]:
            continue
        loop = []
        h = h0
        ok = True
        while True:
            visited[h] = True
            loop.append(src_l[h])
            h = nxt_l[h]
            if h == h0:
                break
            if visited[h] or len(loop) > max_len:
                ok = False
                # consume the rest of the cycle
                while not visited[h]:
                    visited[h] = True
                    h = nxt_l[h]
                break
        if ok and len(loop) >= 3 and len(set(loop)) == len(loop):
            faces.append(loop)
    return faces


def extract_mesh(P, N, Q, O, edges, h, is_feature, is_corner, is_boundary, vcomp=None, max_face=12):
    i, j = edges[:, 0], edges[:, 1]
    hv = np.broadcast_to(np.asarray(h, dtype=np.float64), (len(P),))
    h = 0.5 * (hv[i] + hv[j])
    T = np.cross(N, Q)
    # Lattice points of i and j closest to each other near the edge midpoint.
    a, b = compat_pair(P[i], N[i], Q[i], T[i], O[i], P[j], N[j], Q[j], T[j], O[j], h)
    # Express both integer indices in i's frame (j's cross matched to i's).
    qj = match4(Q[i], Q[j], N[j])
    tj = np.cross(N[j], qj)
    ia = np.stack([np.round(dot(a - O[i], Q[i]) / h), np.round(dot(a - O[i], T[i]) / h)], 1)
    ib = np.stack([np.round(dot(b - O[j], qj) / h), np.round(dot(b - O[j], tj) / h)], 1)
    off = ia - ib
    err = np.linalg.norm(a - b, axis=1)
    l1 = np.abs(off).sum(1)
    zero = (l1 == 0) & (err < 0.5 * h)
    unit = (l1 == 1) & (err < 0.5 * h)

    n = len(P)
    lab = _bounded_clusters(n, i[zero], j[zero], err[zero], O, 0.5 * hv)
    _, cl = np.unique(lab, return_inverse=True)
    cl = cl.reshape(-1)
    nc = int(cl.max()) + 1
    cnt = np.bincount(cl, minlength=nc).astype(np.float64)
    VP = np.stack([np.bincount(cl, weights=O[:, k], minlength=nc) for k in range(3)], 1) / cnt[:, None]
    VN = normalize(np.stack([np.bincount(cl, weights=N[:, k], minlength=nc) for k in range(3)], 1))

    def any_flag(mask):
        f = np.zeros(nc, bool)
        f[cl[mask]] = True
        return f

    c_feat = any_flag(is_feature)
    c_corner = any_flag(is_corner)
    c_bnd = any_flag(is_boundary)
    c_comp = np.zeros(nc, dtype=np.int64)
    if vcomp is not None:
        c_comp[cl] = vcomp
    VQ = np.zeros((nc, 3))
    VQ[cl] = Q
    VQ = normalize(VQ - VN * dot(VQ, VN)[:, None])
    hc = np.bincount(cl, weights=hv, minlength=nc) / cnt
    corner_pos = np.zeros((nc, 3))
    corner_pos[cl[is_corner]] = P[is_corner]
    VP[c_corner] = corner_pos[c_corner]

    # Every fine edge between two different clusters votes on whether the
    # clusters are lattice neighbours; keep pairs where "unit" wins.
    ca, cb = cl[i], cl[j]
    keep = ca != cb
    lo, hi = np.minimum(ca, cb)[keep], np.maximum(ca, cb)[keep]
    pk = lo.astype(np.int64) * nc + hi
    uk, inv = np.unique(pk, return_inverse=True)
    votes_unit = np.bincount(inv, weights=unit[keep].astype(np.float64), minlength=len(uk))
    votes_all = np.bincount(inv, minlength=len(uk))
    k = uk[votes_unit > 0.5 * votes_all]
    ea, eb = k // nc, k % nc
    L = np.linalg.norm(VP[ea] - VP[eb], axis=1)
    good = L < 0.95 * (hc[ea] + hc[eb])
    ea, eb = _prune_low_degree(nc, ea[good], eb[good])

    faces = trace_faces(VP, VN, ea, eb, max_len=max_face)
    # Drop "outer" faces of open surfaces: inverted orientation, or long loops
    # made only of boundary vertices.
    kept = []
    for f in faces:
        if len(f) > 4 and c_bnd[f].all():
            continue
        nn = newell_normal(VP[f])
        if np.dot(nn, VN[f].sum(0)) < 0:
            continue
        kept.append(f)
    return VP, VN, kept, c_feat, c_corner, c_bnd, c_comp, VQ, hc
