"""Feature (sharp / boundary) edge detection on the input triangle mesh."""

import math

import numpy as np

from .geom import dot, normalize


class Features:
    def __init__(self, mesh, sharp_angle=None, marked_pairs=None, corner_angle=math.radians(35.0)):
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
            mask |= sharp
        if marked_pairs is not None and len(marked_pairs):
            idx = mesh.edge_index(marked_pairs)
            mask[idx[idx >= 0]] = True
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
        self.is_corner = corner
        self.segments = np.stack([V[fe[:, 0]], V[fe[:, 1]]], 1) if len(fe) else np.zeros((0, 2, 3))

    @property
    def any(self):
        return bool(len(self.feature_edges))
