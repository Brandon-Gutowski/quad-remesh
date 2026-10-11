# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

"""Feature (sharp / boundary) edge detection on the input triangle mesh."""

import math

import numpy as np

from .geom import dot, normalize


class Features:
    def __init__(self, mesh, sharp_angle=None, marked_pairs=None, corner_angle=math.radians(35.0),
                 scale=None):
        n = len(mesh.V)
        E = mesh.edges
        cnt = mesh.edge_face_count
        mask = cnt != 2  # boundaries and non-manifold edges are always features
        self.boundary_edge = cnt == 1
        if sharp_angle is not None:
            two = cnt == 2
            f0, f1 = mesh.edge_faces[two, 0], mesh.edge_faces[two, 1]
            cosang = dot(mesh.face_normal[f0], mesh.face_normal[f1])
            sharp = np.zeros(len(E), bool)
            sharp[two] = cosang < math.cos(sharp_angle)
            if scale is not None and sharp.any():
                sharp = _scale_dihedral_filter(mesh, sharp, max(0.35 * scale, 3.0 * float(mesh.edge_length.mean())),
                                               sharp_angle)
            mask |= sharp
        if marked_pairs is not None and len(marked_pairs):
            idx = mesh.edge_index(marked_pairs)
            mask[idx[idx >= 0]] = True
        if scale is not None:
            mask = _drop_short_chains(E, mesh.edge_length, mask, self.boundary_edge, 1.0 * scale)
        self.edge_mask = mask

        fe = E[mask]
        self.feature_edges = fe
        vc = np.bincount(fe.reshape(-1), minlength=n)
        self.vertex_count = vc
        self.is_feature = vc > 0
        self.is_boundary = np.zeros(n, bool)
        self.is_boundary[E[self.boundary_edge].reshape(-1)] = True
        V = mesh.V

        # Feature tangent: principal direction of incident feature edges.
        d = normalize(V[fe[:, 1]] - V[fe[:, 0]]) if len(fe) else np.zeros((0, 3))
        T = np.zeros((n, 3, 3))
        outer = d[:, :, None] * d[:, None, :]
        np.add.at(T, fe[:, 0], outer)
        np.add.at(T, fe[:, 1], outer)
        tangent = np.zeros((n, 3))
        fv = np.nonzero(self.is_feature)[0]
        if len(fv):
            w, vecs = np.linalg.eigh(T[fv])
            tangent[fv] = vecs[:, :, -1]
        self.tangent = tangent

        # Corners: feature valence != 2, or a sharp turn along the feature line.
        corner = self.is_feature & (vc != 2)
        two = np.nonzero(vc == 2)[0]
        if len(two):
            hv = np.concatenate([fe[:, 0], fe[:, 1]])
            ho = np.concatenate([fe[:, 1], fe[:, 0]])
            order = np.argsort(hv, kind="stable")
            hv, ho = hv[order], ho[order]
            first = np.searchsorted(hv, two)
            a = ho[first]
            b = ho[first + 1]
            d1 = normalize(V[two] - V[a])
            d2 = normalize(V[b] - V[two])
            turn = dot(d1, d2) < math.cos(corner_angle)
            corner[two[turn]] = True
        if scale is not None and corner.any():
            corner = _thin_corners(V, corner, vc, 0.5 * scale)
        self.is_corner = corner
        self.segments = np.stack([V[fe[:, 0]], V[fe[:, 1]]], 1) if len(fe) else np.zeros((0, 2, 3))

    @property
    def any(self):
        return bool(len(self.feature_edges))


def _edge_chains(E, mask):
    """Label feature edges by chain (chains break at vertices of feature valence != 2)."""
    idx = np.nonzero(mask)[0]
    fe = E[idx]
    n = int(E.max()) + 1 if len(E) else 0
    vc = np.bincount(fe.reshape(-1), minlength=n)
    # half-edges sorted by vertex: consecutive pairs at valence-2 vertices link edges
    hv = np.concatenate([fe[:, 0], fe[:, 1]])
    he = np.concatenate([np.arange(len(fe)), np.arange(len(fe))])
    order = np.argsort(hv, kind="stable")
    hv, he = hv[order], he[order]
    v2 = np.nonzero(vc == 2)[0]
    first = np.searchsorted(hv, v2)
    a, b = he[first], he[first + 1]
    lab = np.arange(len(fe))
    while len(a):
        m = np.minimum(lab[a], lab[b])
        new = lab.copy()
        np.minimum.at(new, a, m)
        np.minimum.at(new, b, m)
        new = new[new]
        if np.array_equal(new, lab):
            break
        lab = new
    return idx, lab


def _drop_short_chains(E, length, mask, protected, min_len, rounds=4):
    mask = mask.copy()
    for _ in range(rounds):
        idx, lab = _edge_chains(E, mask)
        if not len(idx):
            break
        tot = np.bincount(lab, weights=length[idx], minlength=len(idx))
        prot = np.bincount(lab, weights=protected[idx].astype(float), minlength=len(idx)) > 0
        short = (tot[lab] < min_len) & ~prot[lab]
        if not short.any():
            break
        mask[idx[short]] = False
    return mask


def _thin_corners(V, corner, vc, radius):
    """Keep corners at least `radius` apart, preferring higher feature valence."""
    cand = np.nonzero(corner)[0]
    order = cand[np.lexsort((cand, -vc[cand]))]
    kept = []
    out = np.zeros_like(corner)
    for c in order:
        if kept:
            d = np.linalg.norm(V[kept] - V[c], axis=1)
            if d.min() < radius:
                continue
        kept.append(c)
        out[c] = True
    return out


def _scale_dihedral_filter(mesh, cand, radius, angle):
    """Keep candidate sharp edges only if the surface is still creased at
    `radius`: fit a plane to the points on each side of the edge and compare."""
    from .spatial import PointIndex
    V = mesh.V
    index = PointIndex(V)
    out = cand.copy()
    cos_thr = math.cos(0.75 * angle)
    for e in np.nonzero(cand)[0]:
        u, v = mesh.edges[e]
        f0, f1 = mesh.edge_faces[e]
        mid = 0.5 * (V[u] + V[v])
        d = V[v] - V[u]
        d /= max(np.linalg.norm(d), 1e-30)
        n0, n1 = mesh.face_normal[f0], mesh.face_normal[f1]
        navg = n0 + n1
        side = np.cross(d, navg)
        # orient `side` toward face f0
        c0 = V[mesh.F[f0]].mean(0)
        if np.dot(c0 - mid, side) < 0:
            side = -side
        pts = V[index.within(mid, radius)]
        rel = pts - mid
        rel = rel - np.outer(rel @ d, d) * 0.0
        s_ = rel @ side
        A = rel[s_ > 1e-12 * radius]
        B = rel[s_ < -1e-12 * radius]
        if len(A) < 3 or len(B) < 3:
            continue
        na = _plane_normal(A, d, n0)
        nb = _plane_normal(B, d, n1)
        if np.dot(na, nb) > cos_thr:
            out[e] = False
    return out


def _plane_normal(rel, d, ref):
    """Normal of the plane through the edge line (origin, direction d) that best
    fits the points: the direction perpendicular to d with least variance."""
    q = rel - np.outer(rel @ d, d)
    C = q.T @ q
    w, vecs = np.linalg.eigh(C)
    # among eigenvectors, the smallest-variance one orthogonal-ish to d
    for k in range(3):
        cand = vecs[:, k]
        if abs(np.dot(cand, d)) < 0.5:
            break
    if np.dot(cand, ref) < 0:
        cand = -cand
    return cand
