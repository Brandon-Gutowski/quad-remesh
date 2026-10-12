# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spatial queries built on mathutils (available in Blender's Python, no bpy needed)."""

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree


class SurfaceProjector:
    """Closest point on the source triangle surface.

    With per-face component labels, points are only projected onto their own
    connected component (so e.g. eyeballs and head never trade vertices). With
    normals, hits on the far side of thin parts are rejected in favour of a ray
    cast along the normal.
    """

    def __init__(self, V, F, face_comp=None):
        V = np.asarray(V, dtype=np.float64)
        F = np.asarray(F, dtype=np.int64)
        if face_comp is None:
            face_comp = np.zeros(len(F), dtype=np.int64)
        self.face_comp = np.asarray(face_comp, dtype=np.int64)
        self.trees = {}
        comps = np.unique(self.face_comp)
        for c in comps:
            Fc = F[self.face_comp == c]
            used, local = np.unique(Fc, return_inverse=True)
            self.trees[int(c)] = BVHTree.FromPolygons(V[used].tolist(), local.reshape(-1, 3).tolist(),
                                                       all_triangles=True)
        if len(comps) == 1:
            self.global_tree = self.trees[int(comps[0])]
            self._global_faces = np.nonzero(self.face_comp == comps[0])[0]
        else:
            self.global_tree = BVHTree.FromPolygons(V.tolist(), F.tolist(), all_triangles=True)
            self._global_faces = np.arange(len(F))
        self.default = int(np.bincount(self.face_comp).argmax()) if len(self.face_comp) else 0

    def component_of(self, P):
        """Component label of the source surface nearest to each point."""
        out = np.full(len(P), self.default, dtype=np.int64)
        for i, p in enumerate(P):
            hit = self.global_tree.find_nearest(Vector(p))
            if hit[2] is not None:
                out[i] = self.face_comp[self._global_faces[hit[2]]]
        return out

    def ray_thickness(self, P, N, comps=None, eps=1e-7):
        """Distance from each point to the far side of its part along -normal
        (inf where nothing is hit)."""
        out = np.full(len(P), np.inf)
        for i in range(len(P)):
            tree = self.trees.get(int(comps[i]) if comps is not None else self.default, self.global_tree)
            nv = Vector(N[i])
            hit = tree.ray_cast(Vector(P[i]) - nv * eps, -nv)
            if hit[0] is not None and hit[1].dot(nv) < 0.0:   # the far side faces away
                out[i] = hit[3]
        return out

    def project(self, P, comps=None, normals=None):
        out = np.array(P, dtype=np.float64, copy=True)
        nrm = np.zeros_like(out)
        for i, p in enumerate(P):
            tree = self.trees.get(int(comps[i]) if comps is not None else self.default)
            if tree is None:
                tree = self.trees[self.default]
            pv = Vector(p)
            loc, n, _idx, _d = tree.find_nearest(pv)
            if loc is None:
                continue
            if normals is not None:
                nv = Vector(normals[i])
                if nv.length_squared > 0 and n.dot(nv) < 0.0:
                    best = None
                    for direction in (-nv, nv):
                        hit = tree.ray_cast(pv, direction)
                        if hit[0] is not None and hit[1].dot(nv) > 0.0:
                            if best is None or hit[3] < best[3]:
                                best = hit
                    if best is not None:
                        loc, n = best[0], best[1]
            out[i] = loc
            nrm[i] = n
        return out, nrm


class SegmentProjector:
    """Closest point on a set of 3D line segments (the feature network)."""

    def __init__(self, segments, spacing, seg_comp=None):
        self.seg_comp = np.zeros(len(segments), dtype=np.int64) if seg_comp is None else np.asarray(seg_comp)
        self.A = np.asarray(segments[:, 0], dtype=np.float64)
        self.B = np.asarray(segments[:, 1], dtype=np.float64)
        L = np.linalg.norm(self.B - self.A, axis=1)
        nsamp = np.maximum(1, np.ceil(L / max(spacing, 1e-12)).astype(int)) + 1
        seg_id = np.repeat(np.arange(len(L)), nsamp)
        t = np.concatenate([np.linspace(0.0, 1.0, k) for k in nsamp]) if len(L) else np.zeros(0)
        pts = self.A[seg_id] + (self.B - self.A)[seg_id] * t[:, None]
        self.seg_id = seg_id
        self.kd = KDTree(len(pts))
        for i, p in enumerate(pts):
            self.kd.insert(p, i)
        self.kd.balance()
        self.empty = len(pts) == 0

    def _closest_on(self, p, sids):
        A, B = self.A[sids], self.B[sids]
        AB = B - A
        t = np.clip(np.einsum("ij,ij->i", p - A, AB) / np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-30), 0.0, 1.0)
        C = A + AB * t[:, None]
        d = np.linalg.norm(C - p, axis=1)
        k = int(np.argmin(d))
        return C[k], d[k], AB[k]

    def project(self, P, k=8, comps=None):
        """Returns (points, distances, segment directions)."""
        P = np.asarray(P, dtype=np.float64)
        out = P.copy()
        dist = np.full(len(P), np.inf)
        dirs = np.zeros_like(P)
        if self.empty:
            return out, dist, dirs
        for i, p in enumerate(P):
            hits = self.kd.find_n(p, k)
            sids = np.unique(self.seg_id[[h[1] for h in hits]])
            if comps is not None:
                same = sids[self.seg_comp[sids] == comps[i]]
                if len(same):
                    sids = same
            c, d, ab = self._closest_on(p, sids)
            out[i], dist[i], dirs[i] = c, d, ab
        return out, dist, dirs


class PointIndex:
    """KD-tree over points for nearest / radius queries."""

    def __init__(self, P):
        self.kd = KDTree(len(P))
        for i, p in enumerate(P):
            self.kd.insert(p, i)
        self.kd.balance()

    def nearest(self, p):
        return self.kd.find(p)[1]

    def within(self, p, r):
        return [h[1] for h in self.kd.find_range(p, r)]
