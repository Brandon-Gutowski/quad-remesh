"""Lightweight editable polygon mesh used for topology clean-up."""

import numpy as np


def ekey(a, b):
    return (a, b) if a < b else (b, a)


class PolyMesh:
    def __init__(self, V, faces, feat, corner, bnd, feature_edges=None, comp=None):
        self.V = np.asarray(V, dtype=np.float64)
        self.faces = [list(f) for f in faces]
        self.feat = np.asarray(feat, bool).copy()
        self.corner = np.asarray(corner, bool).copy()
        self.bnd = np.asarray(bnd, bool).copy()
        self.feature_edges = set(feature_edges or ())
        self.comp = np.zeros(len(self.V), dtype=np.int64) if comp is None else np.asarray(comp, dtype=np.int64).copy()

    # -- connectivity ---------------------------------------------------------
    def edge_faces(self):
        ef = {}
        for fi, f in enumerate(self.faces):
            if f is None:
                continue
            m = len(f)
            for k in range(m):
                ef.setdefault(ekey(f[k], f[(k + 1) % m]), []).append(fi)
        return ef

    def vertex_faces(self):
        vf = [[] for _ in range(len(self.V))]
        for fi, f in enumerate(self.faces):
            if f is None:
                continue
            for v in f:
                vf[v].append(fi)
        return vf

    def valence(self, ef=None):
        ef = ef if ef is not None else self.edge_faces()
        val = np.zeros(len(self.V), dtype=np.int64)
        open_v = np.zeros(len(self.V), bool)
        for (a, b), fl in ef.items():
            val[a] += 1
            val[b] += 1
            if len(fl) != 2:
                open_v[a] = True
                open_v[b] = True
        return val, open_v

    def add_vertex(self, p, feat=False, corner=False, bnd=False):
        self.V = np.vstack([self.V, p[None]])
        self.feat = np.append(self.feat, feat)
        self.corner = np.append(self.corner, corner)
        self.bnd = np.append(self.bnd, bnd)
        self.comp = np.append(self.comp, self.comp[0] if len(self.comp) else 0)
        return len(self.V) - 1

    def compact(self):
        self.faces = [f for f in self.faces if f is not None and len(f) >= 3]
        used = np.zeros(len(self.V), bool)
        for f in self.faces:
            used[f] = True
        remap = np.cumsum(used) - 1
        self.V = self.V[used]
        self.feat = self.feat[used]
        self.corner = self.corner[used]
        self.bnd = self.bnd[used]
        self.comp = self.comp[used]
        self.faces = [[int(remap[v]) for v in f] for f in self.faces]
        self.feature_edges = {ekey(int(remap[a]), int(remap[b])) for a, b in self.feature_edges
                              if used[a] and used[b]}

    def face_sizes(self):
        return np.array([len(f) for f in self.faces if f is not None])

    def vertex_normals(self):
        n = np.zeros_like(self.V)
        for f in self.faces:
            if f is None or len(f) < 3:
                continue
            P = self.V[f]
            fn = np.cross(P - P.mean(0), np.roll(P, -1, axis=0) - P.mean(0)).sum(0)
            n[f] += fn
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.maximum(ln, 1e-30)
