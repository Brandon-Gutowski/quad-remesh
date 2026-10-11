"""Shape-matching relaxation: pull every face toward a regular polygon of the
target size, then reproject onto the source surface / feature lines."""

import numpy as np

from .geom import normalize


def _group_faces(faces):
    groups = {}
    for f in faces:
        groups.setdefault(len(f), []).append(f)
    return {m: np.array(fl, dtype=np.int64) for m, fl in groups.items()}


def _regular_targets(V, F, size):
    """Best-fit regular m-gon (edge length `size`) for each face in F (k,m)."""
    m = F.shape[1]
    P = V[F]                                   # (k,m,3)
    C = P.mean(1)
    nxt = np.roll(P, -1, axis=1)
    nrm = normalize(np.cross(P, nxt).sum(1))
    e1 = P[:, 0] - C
    e1 = normalize(e1 - nrm * np.einsum("ij,ij->i", e1, nrm)[:, None])
    e2 = np.cross(nrm, e1)
    R = P - C[:, None]
    w = np.einsum("kmj,kj->km", R, e1) + 1j * np.einsum("kmj,kj->km", R, e2)
    ang = 2.0 * np.pi * np.arange(m) / m
    rot = (w * np.exp(-1j * ang)[None]).sum(1)
    alpha = np.angle(rot)
    radius = size / (2.0 * np.sin(np.pi / m))
    if np.ndim(radius):
        radius = radius[:, None]
    tz = radius * np.exp(1j * (alpha[:, None] + ang[None]))
    return C[:, None] + tz.real[..., None] * e1[:, None] + tz.imag[..., None] * e2[:, None]


def _vertex_normals(V, groups, n):
    acc = np.zeros((n, 3))
    for m, F in groups.items():
        P = V[F]
        C = P.mean(1)
        fn = np.cross(P - C[:, None], np.roll(P, -1, axis=1) - C[:, None]).sum(1)
        for k in range(m):
            np.add.at(acc, F[:, k], fn)
    return acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-30)


def relax(V, faces, target, surf_proj, seg_proj, feat, corner, iterations=10, step=0.6,
          lap_weight=0.25, local_mix=0.25, comp=None):
    V = np.array(V, dtype=np.float64)
    groups = _group_faces(faces)
    n = len(V)
    # Uniform Laplacian over polygon edges.
    ea, eb = [], []
    for F in groups.values():
        m = F.shape[1]
        for k in range(m):
            ea.append(F[:, k])
            eb.append(F[:, (k + 1) % m])
    ea = np.concatenate(ea)
    eb = np.concatenate(eb)
    src = np.concatenate([ea, eb])
    dst = np.concatenate([eb, ea])
    deg = np.maximum(np.bincount(src, minlength=n), 1)
    free = ~corner
    on_feat = feat & ~corner
    surf = ~feat

    for _ in range(iterations):
        acc = np.zeros_like(V)
        cnt = np.zeros(n)
        for m, F in groups.items():
            P = V[F]
            edge = np.linalg.norm(P - np.roll(P, -1, axis=1), axis=2).mean(1)
            size = (1.0 - local_mix) * target + local_mix * edge
            T = _regular_targets(V, F, size)
            for k in range(m):
                np.add.at(acc, F[:, k], T[:, k])
                np.add.at(cnt, F[:, k], 1.0)
        shape_t = acc / np.maximum(cnt, 1.0)[:, None]
        lap = np.stack([np.bincount(src, weights=V[dst, k], minlength=n) for k in range(3)], 1) / deg[:, None]
        tgt = (1.0 - lap_weight) * shape_t + lap_weight * lap
        Vn = V + step * (tgt - V)
        Vn[~free] = V[~free]
        if surf.any():
            nrm = _vertex_normals(V, groups, n)
            Vn[surf], _ = surf_proj.project(Vn[surf], None if comp is None else comp[surf], nrm[surf])
        if on_feat.any() and seg_proj is not None and not seg_proj.empty:
            Vn[on_feat], _, _ = seg_proj.project(Vn[on_feat], comps=None if comp is None else comp[on_feat])
        V = Vn
    return V


def untangle(V, faces, surf_proj, seg_proj, feat, corner, comp=None, rounds=12):
    """Locally smooth the neighbourhood of folded / flipped faces until none remain."""
    from .metrics import flipped_faces
    V = np.array(V, dtype=np.float64)
    n = len(V)
    nbrs = [set() for _ in range(n)]
    vfaces = [[] for _ in range(n)]
    for fi, f in enumerate(faces):
        m = len(f)
        for k in range(m):
            nbrs[f[k]].add(f[(k + 1) % m])
            nbrs[f[k]].add(f[k - 1])
            vfaces[f[k]].append(fi)
    nbrs = [np.array(sorted(s), dtype=np.int64) for s in nbrs]
    bad = flipped_faces(V, faces, surf_proj, comp)
    best_V, best_n = V.copy(), len(bad)
    grow = 0
    for _ in range(rounds):
        if not bad:
            break
        region = set()
        for fi in bad:
            region.update(faces[fi])
        for _g in range(grow):
            region |= {u for v in list(region) for u in nbrs[v]}
        idx = np.array([v for v in region if not corner[v]], dtype=np.int64)
        if not len(idx):
            break
        for _s in range(3):
            tgt = np.array([V[nbrs[v]].mean(0) if len(nbrs[v]) else V[v] for v in idx])
            V[idx] = tgt
            s_idx = idx[~feat[idx]]
            f_idx = idx[feat[idx]]
            if len(s_idx):
                V[s_idx], _ = surf_proj.project(V[s_idx], None if comp is None else comp[s_idx])
            if len(f_idx) and seg_proj is not None and not seg_proj.empty:
                V[f_idx], _, _ = seg_proj.project(V[f_idx], comps=None if comp is None else comp[f_idx])
        new_bad = flipped_faces(V, faces, surf_proj, comp)
        if len(new_bad) < best_n:
            best_V, best_n = V.copy(), len(new_bad)
        if len(new_bad) >= len(bad):
            grow += 1
        bad = new_bad
    return best_V
