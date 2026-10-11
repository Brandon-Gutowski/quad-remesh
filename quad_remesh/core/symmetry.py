# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

"""Mirror symmetry: remesh one half of the input, then mirror and weld it.

The input is cut at the mirror plane and only the positive side is kept. The
cut becomes an open boundary, which the pipeline treats as a feature curve, so
an edge loop lands exactly on the plane. After remeshing, the seam vertices are
snapped onto the plane, the half is reflected, and both halves are welded along
the seam, giving an exactly symmetric result.
"""

import numpy as np

from .polymesh import ekey


def signed_distance(V, p0, n):
    return (np.asarray(V, dtype=np.float64) - p0) @ n


def cut_half(V, F, p0, n, pairs=None, tol=None):
    """Keep the part of the triangle mesh on the positive side of the plane.

    Returns (V, F, pairs) for the half; marked pairs are clipped too.
    """
    V = np.array(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    if tol is None:
        diag = np.linalg.norm(V.max(0) - V.min(0)) if len(V) else 1.0
        tol = 1e-5 * diag
    s = signed_distance(V, p0, n)
    on = np.abs(s) <= tol
    V[on] -= np.outer(s[on], n)          # snap near-plane vertices onto it
    s[on] = 0.0

    sf = s[F]
    keep = (sf >= 0).all(1) & (sf > 0).any(1)
    cross = (sf > 0).any(1) & (sf < 0).any(1)
    out_faces = [F[keep]]
    newV = []
    cut_key = {}

    def cut_vertex(a, b):
        k = ekey(int(a), int(b))
        if k not in cut_key:
            t = s[a] / (s[a] - s[b])
            p = V[a] + t * (V[b] - V[a])
            p -= n * float((p - p0) @ n)  # exactly on the plane
            cut_key[k] = len(V) + len(newV)
            newV.append(p)
        return cut_key[k]

    extra = []
    for f in F[cross]:
        poly = []
        for k in range(3):
            a, b = f[k], f[(k + 1) % 3]
            if s[a] >= 0:
                poly.append(int(a))
            if (s[a] > 0 and s[b] < 0) or (s[a] < 0 and s[b] > 0):
                poly.append(cut_vertex(a, b))
        for k in range(1, len(poly) - 1):
            extra.append([poly[0], poly[k], poly[k + 1]])
    if extra:
        out_faces.append(np.array(extra, dtype=np.int64))
    if newV:
        V = np.vstack([V, np.array(newV)])
    F2 = np.vstack(out_faces) if out_faces else np.zeros((0, 3), dtype=np.int64)

    P2 = None
    if pairs is not None and len(pairs):
        pairs = np.asarray(pairs, dtype=np.int64)
        keep_p = []
        for a, b in pairs:
            if s[a] >= 0 and s[b] >= 0:
                keep_p.append((a, b))
            elif s[a] > 0 > s[b]:
                keep_p.append((a, cut_key.get(ekey(a, b), a)))
            elif s[b] > 0 > s[a]:
                keep_p.append((b, cut_key.get(ekey(a, b), b)))
        P2 = np.array([p for p in keep_p if p[0] != p[1]], dtype=np.int64).reshape(-1, 2)
    return V, F2, P2


def fold_guides(guides, p0, n):
    """Reflect guide points from the negative side onto the kept half."""
    out = []
    for g in guides:
        g = np.array(g, dtype=np.float64)
        s = signed_distance(g, p0, n)
        neg = s < 0
        g[neg] -= 2.0 * np.outer(s[neg], n)
        out.append(g)
    return out


def mirror_weld(V, faces, feature_edges, p0, n, seam_tol):
    """Snap seam vertices onto the plane, reflect the half, and weld."""
    V = np.array(V, dtype=np.float64)
    count = {}
    for f in faces:
        for k in range(len(f)):
            e = ekey(f[k], f[(k + 1) % len(f)])
            count[e] = count.get(e, 0) + 1
    boundary_v = set()
    for (a, b), c in count.items():
        if c == 1:
            boundary_v.update((a, b))
    s = signed_distance(V, p0, n)
    seam = np.zeros(len(V), bool)
    for v in boundary_v:
        if abs(s[v]) <= seam_tol:
            seam[v] = True
    V[seam] -= np.outer(s[seam], n)
    s[seam] = 0.0
    faces = fix_seam_valence(V, faces, seam)
    # Interior vertices that strayed across the plane are nudged back just
    # inside it (not onto it, or they would coincide with their own mirror).
    neg = (s < 0.02 * seam_tol) & ~seam
    V[neg] += np.outer(0.02 * seam_tol - s[neg], n)
    s[neg] = 0.02 * seam_tol

    n_half = len(V)
    mirror_idx = np.where(seam, np.arange(n_half), -1)
    others = np.nonzero(~seam)[0]
    mirror_idx[others] = n_half + np.arange(len(others))
    Vm = V[others] - 2.0 * np.outer(s[others], n)
    V_all = np.vstack([V, Vm])
    faces_all = [list(f) for f in faces]
    for f in faces:
        faces_all.append([int(mirror_idx[v]) for v in reversed(f)])

    seam_edges = {e for e in count if seam[e[0]] and seam[e[1]] and count[e] == 1}
    fe = set()
    for a, b in feature_edges:
        e = ekey(a, b)
        if e in seam_edges:
            continue  # the cut is not a real crease
        fe.add(e)
        fe.add(ekey(int(mirror_idx[a]), int(mirror_idx[b])))
    return V_all, faces_all, fe


def _interior_edges(faces):
    count = {}
    for fi, f in enumerate(faces):
        for k in range(len(f)):
            count.setdefault(ekey(f[k], f[(k + 1) % len(f)]), []).append(fi)
    return count


def fix_seam_valence(V, faces, seam, rounds=4):
    """After welding, a seam vertex with k interior edges gets valence 2+2k, so
    every seam vertex must have exactly one interior edge. Extra edges are
    moved off the seam by merging the two faces across one of them and
    re-splitting the hexagon with a diagonal that avoids the seam vertex."""
    from .quadify import merge_polys
    faces = [list(f) for f in faces]
    for _ in range(rounds):
        ef = _interior_edges(faces)
        nbr_int = {}
        for (a, b), fl in ef.items():
            if len(fl) == 2:
                nbr_int.setdefault(a, []).append((b, fl))
                nbr_int.setdefault(b, []).append((a, fl))
        changed = False
        dead = set()
        new = []
        existing = set(ef.keys())
        for v in np.nonzero(seam)[0]:
            ints = [(w, fl) for w, fl in nbr_int.get(v, []) if not seam[w]]
            if len(ints) < 2:
                continue
            best = None
            for w, (fa, fb) in ints:
                if fa in dead or fb in dead:
                    continue
                Fa, Fb = faces[fa], faces[fb]
                if len(Fa) != 4 or len(Fb) != 4:
                    continue
                hexa = merge_polys(Fa, Fb)
                if hexa is None or len(hexa) != 6:
                    continue
                for k in range(3):
                    p, q = hexa[k], hexa[k + 3]
                    if v in (p, q) or ekey(p, q) in existing:
                        continue
                    d = np.linalg.norm(V[p] - V[q])
                    if best is None or d < best[0]:
                        i0 = hexa.index(p)
                        r = hexa[i0:] + hexa[:i0]
                        best = (d, fa, fb, [r[0], r[1], r[2], r[3]], [r[3], r[4], r[5], r[0]], ekey(p, q))
            if best is None:
                continue
            _d, fa, fb, q1, q2, e = best
            dead.update((fa, fb))
            new += [q1, q2]
            existing.add(e)
            changed = True
        faces = [f for i, f in enumerate(faces) if i not in dead] + new
        if not changed:
            break
    return faces
