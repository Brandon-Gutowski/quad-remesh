# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

"""Global lattice quantization on the coarse extracted mesh (QuadriFlow-style).

The extracted mesh has one vertex per lattice point. Triangulated, every edge
carries an integer lattice offset t (in its first vertex's frame) and a frame
rotation R. Around a regular triangle the offsets must sum to zero; a non-zero
sum is a dislocation "charge" that becomes a valence 3/5 pair in the output.
Charges are cancelled pairwise (or absorbed by true field singularities and
open boundaries) along cheapest dual paths, editing the offsets of the crossed
edges under capacity limits (|t| <= 1 per axis, feature edges untouched).
Quads are then rebuilt combinatorially by walking +x,+y,-x,-y around cells,
which uses the triangulation diagonals wherever a slip changed adjacency.
"""

import heapq
from collections import Counter, defaultdict, deque

import numpy as np

from .geom import dot, normalize
from .orientation import match4
from .polymesh import PolyMesh, ekey

MERGE_COST = 2.0   # extra path cost for merging two lattice points
SPLIT_COST = 2.0   # extra path cost for splitting one
_UNITS = ((1, 0), (0, 1), (-1, 0), (0, -1))
_UIDX = {u: k for k, u in enumerate(_UNITS)}


def rot_int(v, k):
    """Rotate integer 2-vector by k*90 degrees counter-clockwise."""
    k %= 4
    x, y = v
    if k == 0:
        return (x, y)
    if k == 1:
        return (-y, x)
    if k == 2:
        return (-x, -y)
    return (y, -x)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def triangulate(V, faces):
    tris = []
    for f in faces:
        if len(f) == 3:
            tris.append(list(f))
        elif len(f) == 4:
            a, b, c, d = f
            if np.linalg.norm(V[a] - V[c]) <= np.linalg.norm(V[b] - V[d]):
                tris += [[a, b, c], [a, c, d]]
            else:
                tris += [[a, b, d], [b, c, d]]
        else:
            for k in range(1, len(f) - 1):
                tris.append([f[0], f[k], f[k + 1]])
    return tris


def comb(Q, N, edges, n):
    """Rotate cross representatives to agree with a BFS parent."""
    Q = Q.copy()
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen = np.zeros(n, bool)
    for s in range(n):
        if seen[s]:
            continue
        seen[s] = True
        dq = deque([s])
        while dq:
            v = dq.popleft()
            kids = [w for w in adj[v] if not seen[w]]
            if kids:
                kids = np.array(kids)
                Q[kids] = match4(np.repeat(Q[v][None], len(kids), 0), Q[kids], N[kids])
                seen[kids] = True
                dq.extend(kids.tolist())
    return Q


def _rot_index(qa, qb, nb):
    """k such that rot^k(q_b) ~= q_a (rot = +90deg about n_b)."""
    tb = np.cross(nb, qb)
    dq, dt = float(np.dot(qa, qb)), float(np.dot(qa, tb))
    if abs(dq) >= abs(dt):
        return 0 if dq > 0 else 2
    return 1 if dt > 0 else 3


# ---------------------------------------------------------------------------
# lattice bookkeeping on the triangulated coarse mesh
# ---------------------------------------------------------------------------

class CoarseLattice:
    def __init__(self, V, N, Q, H, tris, locked_edges):
        self.V, self.N, self.Q, self.H = V, N, Q, H
        self.F = tris
        self.key = {}
        self.E = []
        self.faces_of = []
        for fi, f in enumerate(tris):
            for k in range(3):
                e = ekey(f[k], f[(k + 1) % 3])
                if e not in self.key:
                    self.key[e] = len(self.E)
                    self.E.append(e)
                    self.faces_of.append([])
                self.faces_of[self.key[e]].append(fi)
        Ea = np.array(self.E, dtype=np.int64).reshape(-1, 2)
        a, b = Ea[:, 0], Ea[:, 1]
        tb = np.cross(N[b], Q[b])
        dq = np.einsum("ij,ij->i", Q[a], Q[b])
        dt = np.einsum("ij,ij->i", Q[a], tb)
        k = np.where(np.abs(dq) >= np.abs(dt), np.where(dq > 0, 0, 2), np.where(dt > 0, 1, 3))
        ta = np.cross(N[a], Q[a])
        d = V[b] - V[a]
        h = 0.5 * (H[a] + H[b])
        tx = np.round(np.einsum("ij,ij->i", d, Q[a]) / h).astype(np.int64)
        ty = np.round(np.einsum("ij,ij->i", d, ta) / h).astype(np.int64)
        self.t = list(zip(tx.tolist(), ty.tolist()))
        self.R = ((-k) % 4).tolist()
        self.locked = [e in locked_edges for e in self.E] if locked_edges else [False] * len(self.E)
        self.corner_rot = []
        self.singular = []
        for f in tris:
            r01 = self.Rd(f[0], f[1])
            r12 = self.Rd(f[1], f[2])
            r20 = self.Rd(f[2], f[0])
            self.singular.append((r01 + r12 + r20) % 4 != 0)
            self.corner_rot.append((0, r01, (-r20) % 4))

    def Rd(self, a, b):
        """Rotation mapping b's integer coords to a's."""
        if a < b:
            return self.R[self.key[(a, b)]]
        return (-self.R[self.key[(b, a)]]) % 4

    def td(self, a, b):
        """Offset from a to b in a's coords."""
        if a < b:
            return self.t[self.key[(a, b)]]
        x, y = rot_int(self.t[self.key[(b, a)]], self.Rd(a, b))
        return (-x, -y)

    def add_td(self, a, b, w):
        if a < b:
            e = self.key[(a, b)]
            x, y = self.t[e]
            self.t[e] = (x + w[0], y + w[1])
        else:
            e = self.key[(b, a)]
            x, y = self.t[e]
            dx, dy = rot_int(w, self.Rd(b, a))
            self.t[e] = (x - dx, y - dy)

    def residual(self, T):
        f = self.F[T]
        cr = self.corner_rot[T]
        sx = sy = 0
        for k in range(3):
            x, y = rot_int(self.td(f[k], f[(k + 1) % 3]), cr[k])
            sx += x
            sy += y
        return (sx, sy)

    def orient(self, T):
        """Twice the signed area of T in integer lattice coordinates."""
        f = self.F[T]
        cr = self.corner_rot[T]
        p1 = self.td(f[0], f[1])
        x, y = rot_int(self.td(f[1], f[2]), cr[1])
        p2 = (p1[0] + x, p1[1] + y)
        return p1[0] * p2[1] - p1[1] * p2[0]

    def folded(self):
        return sum(1 for T in range(len(self.F)) if not self.singular[T] and self.residual(T) == (0, 0)
                   and self.orient(T) < 0)

    def cross(self, T, k, u):
        """Move unit charge u (T's coords) over corner edge k of T."""
        f = self.F[T]
        a, b = f[k], f[(k + 1) % 3]
        w = rot_int(u, -self.corner_rot[T][k])
        e = self.key[ekey(a, b)]
        fl = self.faces_of[e]
        if len(fl) != 2:
            return a, b, w, e, -1, None
        T2 = fl[1] if fl[0] == T else fl[0]
        f2 = self.F[T2]
        m = f2.index(b)
        if f2[(m + 1) % 3] != a:
            return a, b, w, e, -2, None
        u2 = rot_int(rot_int(w, self.Rd(b, a)), self.corner_rot[T2][m])
        return a, b, w, e, T2, u2


def cancel_charges(lat, max_cost=60.0):
    res = {}
    for T in range(len(lat.F)):
        if not lat.singular[T]:
            r = lat.residual(T)
            if r != (0, 0):
                res[T] = r
    initial = sum(abs(x) + abs(y) for x, y in res.values())
    for A in sorted(res.keys()):
        guard = 0
        while A in res and guard < 8:
            guard += 1
            rx, ry = res[A]
            u = ((1 if rx > 0 else -1), 0) if rx else (0, (1 if ry > 0 else -1))
            path = _search(lat, res, A, u, max_cost)
            if path is None:
                break
            _apply(lat, res, A, u, path)
    remaining = sum(abs(x) + abs(y) for x, y in res.values())
    return initial, remaining


def _edit_cost(lat, a, b, w, e):
    if lat.locked[e]:
        return None
    old = lat.td(a, b)
    new = (old[0] - w[0], old[1] - w[1])
    if abs(new[0]) > 1 or abs(new[1]) > 1:
        return None
    cost = 1.0
    if new == (0, 0):
        cost += MERGE_COST  # merges two lattice points
    elif old == (0, 0):
        cost += SPLIT_COST  # splits a lattice point
    return cost


def _orient_ok(lat, T, edits):
    """Would T keep a non-negative lattice orientation after `edits`?"""
    for a, b, w in edits:
        lat.add_td(a, b, (-w[0], -w[1]))
    ok = lat.orient(T) >= 0
    for a, b, w in edits:
        lat.add_td(a, b, w)
    return ok


def _entry_edit(lat, prev, st):
    p = prev[st]
    if p is None:
        return None
    Tp, cup, kp = p
    a, b, w, _e, _T2, _u2 = lat.cross(Tp, kp, cup)
    return (a, b, w)


def _search(lat, res, A, u, max_cost):
    start = (A, u)
    prev = {start: None}
    dist = {start: 0.0}
    heap = [(0.0, 0, A, u)]
    cnt = 1
    done = set()
    rA = res[A]
    start_closes = (rA[0] - u[0], rA[1] - u[1]) == (0, 0)
    while heap:
        d, _c, T, cu = heapq.heappop(heap)
        if (T, cu) in done:
            continue
        done.add((T, cu))
        if d > max_cost:
            break
        ent = _entry_edit(lat, prev, (T, cu))
        for k in range(3):
            a, b, w, e, T2, u2 = lat.cross(T, k, cu)
            if T2 == -2:
                continue
            c = _edit_cost(lat, a, b, w, e)
            if c is None:
                continue
            # T is closed once both its entry and exit edits are applied.
            if ent is not None or start_closes:
                edits = [(a, b, w)] if ent is None else [ent, (a, b, w)]
                if ent is not None and ent[:2] in ((a, b), (b, a)):
                    continue
                if not _orient_ok(lat, T, edits):
                    continue
            if T2 < 0:
                return _trace(prev, (T, cu)) + [(T, k)]
            st = (T2, u2)
            nd = d + c
            if nd >= dist.get(st, float("inf")):
                continue
            if lat.singular[T2]:
                prev[st] = (T, cu, k)
                return _trace(prev, st)
            r = res.get(T2)
            if r is not None and r[0] * u2[0] + r[1] * u2[1] < 0:
                closes = (r[0] + u2[0], r[1] + u2[1]) == (0, 0)
                if not closes or _orient_ok(lat, T2, [(a, b, w)]):
                    prev[st] = (T, cu, k)
                    return _trace(prev, st)
            dist[st] = nd
            prev[st] = (T, cu, k)
            heapq.heappush(heap, (nd, cnt, T2, u2))
            cnt += 1
    return None


def _trace(prev, st):
    steps = []
    while prev[st] is not None:
        T, cu, k = prev[st]
        steps.append((T, k))
        st = (T, cu)
    steps.reverse()
    return steps


def _apply(lat, res, A, u, path):
    r = res[A]
    r = (r[0] - u[0], r[1] - u[1])
    if r == (0, 0):
        del res[A]
    else:
        res[A] = r
    T, cu = A, u
    for (_T, k) in path:
        a, b, w, e, T2, u2 = lat.cross(T, k, cu)
        lat.add_td(a, b, (-w[0], -w[1]))
        if T2 < 0:
            return
        T, cu = T2, u2
    if lat.singular[T]:
        return
    r = res.get(T, (0, 0))
    r = (r[0] + cu[0], r[1] + cu[1])
    if r == (0, 0):
        res.pop(T, None)
    else:
        res[T] = r


# ---------------------------------------------------------------------------
# rebuild quads from the corrected lattice
# ---------------------------------------------------------------------------

def _cluster(n, adj0):
    cl = np.full(n, -1, dtype=np.int64)
    rho = np.zeros(n, dtype=np.int64)
    nc = 0
    for s in range(n):
        if cl[s] >= 0:
            continue
        cl[s] = nc
        dq = deque([s])
        while dq:
            v = dq.popleft()
            for w, r in adj0.get(v, ()):
                if cl[w] < 0:
                    cl[w] = nc
                    rho[w] = (rho[v] + r) % 4
                    dq.append(w)
        nc += 1
    return cl, rho, nc


def _relations(lat, cl, rho):
    votes = defaultdict(Counter)
    for e, (a, b) in enumerate(lat.E):
        ca, cb = cl[a], cl[b]
        if ca == cb:
            continue
        off = rot_int(lat.t[e], rho[a])
        rc = (rho[a] + lat.R[e] - rho[b]) % 4
        votes[(ca, cb)][(off, rc)] += 1
        ob = rot_int(off, -rc)
        votes[(cb, ca)][((-ob[0], -ob[1]), (-rc) % 4)] += 1
    rel = defaultdict(dict)
    for (ca, cb), cntr in votes.items():
        (off, rc), _ = cntr.most_common(1)[0]
        rel[ca][cb] = (off, rc)
    return rel


def rebuild(lat, n):
    E, t, R = lat.E, lat.t, lat.R
    adj0 = defaultdict(list)
    for e, (a, b) in enumerate(E):
        if t[e] == (0, 0):
            adj0[a].append((b, R[e]))
            adj0[b].append((a, (-R[e]) % 4))
    for _round in range(4):
        cl, rho, nc = _cluster(n, adj0)
        rel = _relations(lat, cl, rho)
        # Clusters that sit on the same lattice point (two steps apart with a
        # zero total offset) but share no edge are merged too.
        rep = np.zeros(nc, dtype=np.int64)
        rep[cl] = np.arange(n)
        cpos = np.zeros((nc, 3))
        cnt = np.bincount(cl, minlength=nc).astype(np.float64)
        for k in range(3):
            cpos[:, k] = np.bincount(cl, weights=lat.V[:, k], minlength=nc) / cnt
        chs = np.bincount(cl, weights=lat.H, minlength=nc) / cnt
        extra = 0
        for c in range(nc):
            for d, (o1, r1) in list(rel[c].items()):
                for e2, (o2, r2) in rel[d].items():
                    if e2 <= c:
                        continue
                    o = rot_int(o2, r1)
                    if (o1[0] + o[0], o1[1] + o[1]) != (0, 0):
                        continue
                    if np.linalg.norm(cpos[c] - cpos[e2]) > 0.9 * min(chs[c], chs[e2]):
                        continue
                    a_, b_ = rep[c], rep[e2]
                    r = (-rho[a_] + r1 + r2 + rho[b_]) % 4   # maps b_ coords -> a_ coords
                    adj0[a_].append((b_, r))
                    adj0[b_].append((a_, (-r) % 4))
                    extra += 1
        if not extra:
            break
    nbr = [dict() for _ in range(nc)]

    def add(c, off, d, rc):
        k = _UIDX.get(off)
        if k is not None and k not in nbr[c] and d != c:
            nbr[c][k] = (d, rc)

    for c in range(nc):
        for d, (off, rc) in rel[c].items():
            add(c, off, d, rc)
    for c in range(nc):
        for d, (o1, r1) in list(rel[c].items()):
            for e2, (o2, r2) in rel[d].items():
                if e2 != c:
                    o = rot_int(o2, r1)
                    add(c, (o1[0] + o[0], o1[1] + o[1]), e2, (r1 + r2) % 4)
    faces = []
    seen = set()
    for c in range(nc):
        for k in range(4):
            cyc = [c]
            cur, acc = c, 0
            ok = True
            for step in range(4):
                d = rot_int(_UNITS[(k + step) % 4], -acc)
                kk = _UIDX[d]
                if kk not in nbr[cur]:
                    ok = False
                    break
                nxt, rc = nbr[cur][kk]
                acc = (acc + rc) % 4
                cur = nxt
                cyc.append(cur)
            if not ok or cyc[-1] != c:
                continue
            quad = cyc[:4]
            if len(set(quad)) != 4:
                continue
            key = frozenset(quad)
            if key in seen:
                continue
            seen.add(key)
            faces.append(quad)
    used = set()
    out = []
    for f in faces:
        hes = [(f[i], f[(i + 1) % 4]) for i in range(4)]
        if any(he in used for he in hes):
            continue
        used.update(hes)
        out.append(f)
    unit_edges = {ekey(c, d) for c in range(nc) for (d, _r) in nbr[c].values()}
    return cl, nc, out, unit_edges


def solve_positions(lat, V, cl, nc, anchor=0.05, iters=400):
    """Least-squares cluster positions agreeing with the corrected offsets."""
    P0 = np.zeros((nc, 3))
    cnt = np.bincount(cl, minlength=nc).astype(np.float64)
    for k in range(3):
        P0[:, k] = np.bincount(cl, weights=V[:, k], minlength=nc) / cnt
    ii, jj, dd = [], [], []
    for e, (a, b) in enumerate(lat.E):
        ca, cb = cl[a], cl[b]
        if ca == cb:
            continue
        ta = np.cross(lat.N[a], lat.Q[a])
        h = 0.5 * (lat.H[a] + lat.H[b])
        x, y = lat.t[e]
        da = h * (x * lat.Q[a] + y * ta)
        ii.append(ca)
        jj.append(cb)
        dd.append(da)
    if not ii:
        return P0
    ii = np.array(ii)
    jj = np.array(jj)
    dd = np.array(dd)
    deg = np.bincount(np.concatenate([ii, jj]), minlength=nc).astype(np.float64)
    diag = deg + anchor

    def scatter(idx, vals):
        out = np.empty((nc, 3))
        for k in range(3):
            out[:, k] = np.bincount(idx, weights=vals[:, k], minlength=nc)
        return out

    def matvec(x):
        return diag[:, None] * x - scatter(ii, x[jj]) - scatter(jj, x[ii])

    b = anchor * P0 - scatter(ii, dd) + scatter(jj, dd)
    x = P0.copy()
    r = b - matvec(x)
    z = r / diag[:, None]
    p = z.copy()
    rz = np.einsum("ij,ij->j", r, z)
    for _ in range(iters):
        Ap = matvec(p)
        alpha = rz / np.maximum(np.einsum("ij,ij->j", p, Ap), 1e-300)
        x += p * alpha
        r -= Ap * alpha
        z = r / diag[:, None]
        rz_new = np.einsum("ij,ij->j", r, z)
        if rz_new.max() < 1e-20:
            break
        p = z + p * (rz_new / np.maximum(rz, 1e-300))
        rz = rz_new
    return x


def _fill_with_traced(P, VN, faces, unit_edges, bnd, max_len=10):
    """Cover what the combinatorial cell walk missed (around singularities)
    with faces traced geometrically on the lattice-neighbour graph."""
    from .extract import trace_faces
    used = set()
    for f in faces:
        for i in range(len(f)):
            used.add((f[i], f[(i + 1) % len(f)]))
    if not unit_edges:
        return faces
    E = np.array(sorted(unit_edges), dtype=np.int64)
    traced = trace_faces(P, VN, E[:, 0], E[:, 1], max_len=max_len)
    out = list(faces)
    for f in traced:
        hes = [(f[i], f[(i + 1) % len(f)]) for i in range(len(f))]
        if any(he in used for he in hes):
            continue
        if bnd[f].sum() > len(f) // 2:
            continue  # a real opening of the input surface
        # skip "outer" loops traced the wrong way round
        from .geom import newell_normal
        if np.dot(newell_normal(P[f]), VN[f].sum(0)) <= 0:
            continue
        used.update(hes)
        out.append(f)
    return out


def quantize_fine(mesh, N, Q, O, H, feature_edge_mask, is_feature, is_corner, is_boundary, vcomp,
                  max_cost=None):
    """Dislocation-free extraction straight from the fine mesh's position field.
    Returns the same tuple as extract.extract_mesh (minus VQ/H)."""
    n = len(mesh.V)
    tris = mesh.F.tolist()
    locked = {tuple(e) for e in mesh.edges[feature_edge_mask].tolist()}
    Qc = comb(Q, N, mesh.edges.tolist(), n)
    lat = CoarseLattice(O, N, Qc, H, tris, locked)
    if max_cost is None:
        max_cost = 100.0 * float(np.mean(H)) / max(float(mesh.edge_length.mean()), 1e-12)
    initial, remaining = cancel_charges(lat, max_cost=max_cost)
    cl, nc, faces, unit_edges = rebuild(lat, n)
    P = solve_positions(lat, O, cl, nc)
    VN0 = normalize(np.stack([np.bincount(cl, weights=N[:, k], minlength=nc) for k in range(3)], 1))
    bnd0 = np.zeros(nc, bool)
    bnd0[cl[is_boundary]] = True
    faces = _fill_with_traced(P, VN0, faces, unit_edges, bnd0)
    cnt = np.bincount(cl, minlength=nc).astype(np.float64)
    VN = normalize(np.stack([np.bincount(cl, weights=N[:, k], minlength=nc) for k in range(3)], 1))

    def agg(flags):
        f = np.zeros(nc, bool)
        f[cl[flags]] = True
        return f

    c_corner = agg(is_corner)
    cpos = np.zeros((nc, 3))
    cpos[cl[is_corner]] = mesh.V[is_corner]
    P[c_corner] = cpos[c_corner]
    c_comp = np.zeros(nc, dtype=np.int64)
    if vcomp is not None:
        c_comp[cl] = vcomp
    VQ = np.zeros((nc, 3))
    VQ[cl] = Qc
    hc = np.bincount(cl, weights=H, minlength=nc) / cnt
    return P, VN, faces, agg(is_feature), c_corner, agg(is_boundary), c_comp, VQ, hc, (initial, remaining)
