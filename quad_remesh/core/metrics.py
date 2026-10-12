# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

"""Quality metrics for the output mesh."""

import numpy as np

from .polymesh import ekey


def compute(V, faces, surf_proj=None, sample=4000, comp=None):
    V = np.asarray(V)
    sizes = np.array([len(f) for f in faces])
    ef = {}
    for fi, f in enumerate(faces):
        m = len(f)
        for k in range(m):
            ef.setdefault(ekey(f[k], f[(k + 1) % m]), []).append(fi)
    val = np.zeros(len(V), int)
    open_v = np.zeros(len(V), bool)
    for (a, b), fl in ef.items():
        val[a] += 1
        val[b] += 1
        if len(fl) != 2:
            open_v[a] = open_v[b] = True
    interior = ~open_v & (val > 0)
    E = np.array(list(ef.keys())).reshape(-1, 2)
    L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1) if len(E) else np.zeros(1)
    angles = []
    for m in (4,):
        F = np.array([f for f in faces if len(f) == m]).reshape(-1, 4)
        if len(F):
            P = V[F]
            a = np.roll(P, 1, axis=1) - P
            b = np.roll(P, -1, axis=1) - P
            c = np.einsum("kmj,kmj->km", a, b) / np.maximum(
                np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2), 1e-30)
            angles = np.degrees(np.arccos(np.clip(c, -1, 1))).reshape(-1)
    angles = np.asarray(angles)
    out = {
        "faces": int(len(faces)),
        "verts": int((val > 0).sum()),
        "quads": int((sizes == 4).sum()),
        "tris": int((sizes == 3).sum()),
        "ngons": int((sizes > 4).sum()),
        "quad_ratio": float((sizes == 4).mean()) if len(sizes) else 0.0,
        "v3": int((interior & (val == 3)).sum()),
        "v5": int((interior & (val == 5)).sum()),
        "v6plus": int((interior & (val >= 6)).sum()),
        "irregular": int((interior & (val != 4)).sum()),
        "edge_mean": float(L.mean()),
        "edge_cv": float(L.std() / max(L.mean(), 1e-30)),
        "angle_dev": float(np.abs(angles - 90.0).mean()) if len(angles) else 0.0,
        "boundary_edges": int(sum(1 for fl in ef.values() if len(fl) == 1)),
        "nonmanifold_edges": int(sum(1 for fl in ef.values() if len(fl) > 2)),
    }
    if surf_proj is not None and len(V):
        idx = np.linspace(0, len(V) - 1, min(sample, len(V))).astype(int)
        P, _ = surf_proj.project(V[idx], None if comp is None else comp[idx])
        out["max_dev"] = float(np.linalg.norm(P - V[idx], axis=1).max())
        out["flipped"] = int(len(flipped_faces(V, faces, surf_proj, comp)))
    return out


def summary(m):
    return (f"{m['faces']} faces, {100 * m['quad_ratio']:.1f}% quads "
            f"(tris {m['tris']}, ngons {m['ngons']}) | irregular verts: "
            f"v3={m['v3']} v5={m['v5']} v6+={m['v6plus']} | edge CV {m['edge_cv']:.2f}, "
            f"angle dev {m['angle_dev']:.1f}°" + (f" | flipped {m['flipped']}" if m.get('flipped') else "")
            + (f" | holes {m['holes']}" if m.get('holes') else ""))


def flipped_faces(V, faces, surf_proj, comp=None):
    """Indices of faces whose normal disagrees with the source surface, or
    that are folded onto themselves (a corner turning the wrong way)."""
    V = np.asarray(V)
    if not len(faces):
        return []
    groups = {}
    for fi, f in enumerate(faces):
        groups.setdefault(len(f), []).append(fi)
    C = np.zeros((len(faces), 3))
    for m, idx in groups.items():
        C[idx] = V[np.array([faces[i] for i in idx])].mean(1)
    fc = None if comp is None else np.array([comp[f[0]] for f in faces])
    _, SN = surf_proj.project(C, fc)
    bad = []
    for m, idx in groups.items():
        idx = np.array(idx)
        Fm = np.array([faces[i] for i in idx])
        P = V[Fm]
        c = C[idx][:, None]
        fn = np.cross(P - c, np.roll(P, -1, axis=1) - c).sum(1)
        corner_n = np.cross(np.roll(P, -1, axis=1) - P, np.roll(P, 1, axis=1) - P)
        neg = np.einsum("ij,ij->i", fn, SN[idx]) <= 0
        fold = (np.einsum("kmj,kj->km", corner_n, fn) <= 0).any(1)
        bad.extend(idx[neg | fold].tolist())
    return sorted(bad)


def unexpected_boundary(V, faces, bnd_proj, size_fn, far=None):
    """Open edges of the result that do not run along an opening of the input
    (i.e. holes). `size_fn(points)` gives the local quad size. With `far`,
    only edges more than far*size away from any opening are counted."""
    V = np.asarray(V)
    count = {}
    for f in faces:
        for k in range(len(f)):
            e = ekey(f[k], f[(k + 1) % len(f)])
            count[e] = count.get(e, 0) + 1
    open_e = np.array([e for e, c in count.items() if c == 1], dtype=np.int64).reshape(-1, 2)
    if not len(open_e):
        return 0
    mid = 0.5 * (V[open_e[:, 0]] + V[open_e[:, 1]])
    if bnd_proj is None or bnd_proj.empty:
        return int(len(open_e))
    _, dist, _ = bnd_proj.project(mid)
    return int((dist > (0.5 if far is None else far) * np.asarray(size_fn(mid))).sum())


def guide_coverage(V, faces, gproj, size_fn, tol=0.15, min_cos=0.85):
    """How much of the guide strokes' length has a result edge lying on it
    (both ends within tol*size, running along it). 1.0 = complete loops."""
    V = np.asarray(V)
    total = float(np.linalg.norm(gproj.B - gproj.A, axis=1).sum())
    if total <= 0:
        return 1.0
    E = set()
    for f in faces:
        for k in range(len(f)):
            E.add(ekey(f[k], f[(k + 1) % len(f)]))
    E = np.array(sorted(E), dtype=np.int64).reshape(-1, 2)
    near = np.array([gproj.kd.find(p)[2] for p in V]) < 2.0 * np.asarray(size_fn(V))
    E = E[near[E[:, 0]] & near[E[:, 1]]]
    if not len(E):
        return 0.0
    vi = np.unique(E)
    dv = np.full(len(V), np.inf)
    dv[vi] = gproj.project(V[vi])[1]
    mid = 0.5 * (V[E[:, 0]] + V[E[:, 1]])
    _, _, dirs = gproj.project(mid)
    d = V[E[:, 1]] - V[E[:, 0]]
    L = np.linalg.norm(d, axis=1)
    cos = np.abs(np.einsum("ij,ij->i", d, dirs)) / np.maximum(L * np.linalg.norm(dirs, axis=1), 1e-30)
    tol_v = tol * np.asarray(size_fn(mid))
    on = (dv[E[:, 0]] < tol_v) & (dv[E[:, 1]] < tol_v) & (cos > min_cos)
    return float(min(1.0, L[on].sum() / total))
