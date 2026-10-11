"""Turn the extracted quad-dominant mesh into (pure) quads."""

from collections import deque

import numpy as np

from .polymesh import PolyMesh, ekey


def _rot_to(f, v):
    k = f.index(v)
    return f[k:] + f[:k]


def merge_polys(A, B):
    """Merge two faces sharing exactly one edge; None if not mergeable."""
    sa = set(A)
    shared = [v for v in B if v in sa]
    if len(shared) != 2:
        return None
    m = len(A)
    for k in range(m):
        u, v = A[k], A[(k + 1) % m]
        if u in shared and v in shared:
            break
    else:
        return None
    # A contains u->v, so B must contain v->u.
    Ar = _rot_to(A, v)          # [v, ..., u]
    Br = _rot_to(B, u)          # [u, ..., v]
    if Br[-1] != v:
        return None
    return Ar + Br[1:-1]


def mark_feature_edges(pm, seg_proj):
    """Edges whose endpoints are feature vertices and that lie on a feature line."""
    cand = set()
    for f in pm.faces:
        m = len(f)
        for k in range(m):
            a, b = f[k], f[(k + 1) % m]
            if pm.feat[a] and pm.feat[b]:
                cand.add(ekey(a, b))
    if not cand or seg_proj is None or seg_proj.empty:
        pm.feature_edges = set()
        return
    cand = list(cand)
    E = np.array(cand)
    mid = 0.5 * (pm.V[E[:, 0]] + pm.V[E[:, 1]])
    L = np.linalg.norm(pm.V[E[:, 0]] - pm.V[E[:, 1]], axis=1)
    _, dist, dirs = seg_proj.project(mid)
    d = pm.V[E[:, 1]] - pm.V[E[:, 0]]
    cosang = np.abs(np.einsum("ij,ij->i", d, dirs)) / np.maximum(L * np.linalg.norm(dirs, axis=1), 1e-30)
    ok = (dist < 0.25 * L) & (cosang > 0.8)
    pm.feature_edges = {cand[i] for i in np.nonzero(ok)[0]}


def dissolve_valence2(pm):
    """Remove interior vertices with only two edges (they sit mid-edge)."""
    changed = True
    total = 0
    while changed:
        changed = False
        ef = pm.edge_faces()
        val, open_v = pm.valence(ef)
        vf = pm.vertex_faces()
        for v in np.nonzero((val == 2) & ~open_v & ~pm.corner & ~pm.bnd)[0]:
            fl = vf[v]
            if len(fl) != 2 or any(pm.faces[fi] is None or len(pm.faces[fi]) <= 3 for fi in fl):
                continue
            for fi in fl:
                pm.faces[fi] = [x for x in pm.faces[fi] if x != v]
            total += 1
            changed = True
        pm.faces = [f for f in pm.faces if f is not None]
    return total


def _best_quad_cut(poly, V, existing):
    n = len(poly)
    best, best_cost = None, np.inf
    per = np.mean([np.linalg.norm(V[poly[k]] - V[poly[(k + 1) % n]]) for k in range(n)])
    for k in range(n):
        a, b = poly[k], poly[(k + 3) % n]
        if ekey(a, b) in existing:
            continue
        cost = np.linalg.norm(V[a] - V[b]) / max(per, 1e-30)
        if cost < best_cost:
            best, best_cost = k, cost
    return best


def split_even(poly, V, existing):
    """Split an even polygon into quads by repeatedly cutting off a quad."""
    out = []
    poly = list(poly)
    while len(poly) > 4:
        k = _best_quad_cut(poly, V, existing)
        if k is None:
            out.append(poly)
            return out
        n = len(poly)
        quad = [poly[(k + i) % n] for i in range(4)]
        rest = [poly[(k + 3 + i) % n] for i in range(n - 2)]
        existing.add(ekey(quad[0], quad[3]))
        out.append(quad)
        poly = rest
    out.append(poly)
    return out


def split_odd_with_triangle(poly, V, existing):
    """Odd polygon (>=5) -> one triangle + quads (used when pure quads is off)."""
    n = len(poly)
    best, best_cost = 0, np.inf
    for k in range(n):
        a, b = poly[k], poly[(k + 2) % n]
        if ekey(a, b) in existing:
            continue
        c = np.linalg.norm(V[a] - V[b])
        if c < best_cost:
            best, best_cost = k, c
    tri = [poly[best], poly[(best + 1) % n], poly[(best + 2) % n]]
    rest = [poly[(best + 2 + i) % n] for i in range(n - 1)]
    existing.add(ekey(tri[0], tri[2]))
    return [tri] + split_even(rest, V, existing)


def pair_odd_faces(pm, max_depth=3):
    """Merge odd faces pairwise (directly or via a short strip of quads) into
    even polygons, then split those into quads."""
    faces = pm.faces
    ef = pm.edge_faces()
    existing = set(ef.keys())
    adj = [[] for _ in faces]
    for e, fl in ef.items():
        if len(fl) == 2 and e not in pm.feature_edges:
            a, b = fl
            adj[a].append(b)
            adj[b].append(a)
    alive = [True] * len(faces)
    odd = [i for i, f in enumerate(faces) if len(f) % 2 == 1]
    # Triangles first: they are the most damaging.
    odd.sort(key=lambda i: len(faces[i]))
    oddset = set(odd)
    new_faces = []
    for depth in range(1, max_depth + 1):
        for s in odd:
            if s not in oddset or not alive[s]:
                continue
            # BFS through quads up to `depth` faces away.
            prev = {s: None}
            dq = deque([(s, 0)])
            target = None
            while dq and target is None:
                f, d = dq.popleft()
                for g in adj[f]:
                    if g in prev or not alive[g]:
                        continue
                    prev[g] = f
                    if g in oddset and g != s:
                        target = g
                        break
                    if d + 1 < depth and len(faces[g]) == 4:
                        dq.append((g, d + 1))
            if target is None:
                continue
            path = [target]
            while prev[path[-1]] is not None:
                path.append(prev[path[-1]])
            poly = faces[path[0]]
            ok = True
            for fi in path[1:]:
                poly = merge_polys(poly, faces[fi])
                if poly is None:
                    ok = False
                    break
            if not ok or len(poly) % 2 or len(set(poly)) != len(poly):
                continue
            for fi in path:
                alive[fi] = False
                oddset.discard(fi)
            new_faces.extend(split_even(poly, pm.V, existing))
    pm.faces = [f for i, f in enumerate(faces) if alive[i]] + new_faces


def split_even_ngons(pm):
    existing = set(pm.edge_faces().keys())
    out = []
    for f in pm.faces:
        if len(f) > 4 and len(f) % 2 == 0:
            out.extend(split_even(f, pm.V, existing))
        else:
            out.append(f)
    pm.faces = out


def split_odd_ngons(pm):
    existing = set(pm.edge_faces().keys())
    out = []
    for f in pm.faces:
        if len(f) > 4 and len(f) % 2 == 1:
            out.extend(split_odd_with_triangle(f, pm.V, existing))
        else:
            out.append(f)
    pm.faces = out


def subdivide(pm):
    """One midpoint (Catmull-Clark topology) subdivision: every face -> quads."""
    ef = pm.edge_faces()
    keys = list(ef.keys())
    n = len(pm.V)
    eidx = {k: n + i for i, k in enumerate(keys)}
    E = np.array(keys, dtype=np.int64).reshape(-1, 2)
    emid = 0.5 * (pm.V[E[:, 0]] + pm.V[E[:, 1]])
    efeat = np.array([k in pm.feature_edges for k in keys], bool)
    ebnd = np.array([len(ef[k]) == 1 for k in keys], bool)
    nf = len(pm.faces)
    cbase = n + len(keys)
    cpos = np.array([pm.V[f].mean(0) for f in pm.faces]).reshape(-1, 3)
    faces = []
    for fi, f in enumerate(pm.faces):
        m = len(f)
        c = cbase + fi
        for k in range(m):
            v = f[k]
            a = eidx[ekey(v, f[(k + 1) % m])]
            b = eidx[ekey(f[k - 1], v)]
            faces.append([v, a, c, b])
    V = np.vstack([pm.V, emid, cpos])
    feat = np.concatenate([pm.feat, efeat, np.zeros(nf, bool)])
    corner = np.concatenate([pm.corner, np.zeros(len(keys) + nf, bool)])
    bnd = np.concatenate([pm.bnd, ebnd, np.zeros(nf, bool)])
    fe = set()
    for k, i in zip(keys, range(len(keys))):
        if efeat[i]:
            fe.add(ekey(k[0], n + i))
            fe.add(ekey(k[1], n + i))
    return PolyMesh(V, faces, feat, corner, bnd, fe)


def fill_holes(pm, max_len=12):
    """Close small boundary loops that are not part of the input's real boundary."""
    out_he = {}
    he = set()
    for f in pm.faces:
        m = len(f)
        for k in range(m):
            he.add((f[k], f[(k + 1) % m]))
    for a, b in he:
        if (b, a) not in he:
            # The hole runs b -> a.
            out_he.setdefault(b, []).append(a)
    used = set()
    added = 0
    for start, nxts in list(out_he.items()):
        for first in nxts:
            if (start, first) in used:
                continue
            loop = [start]
            cur, nx = start, first
            ok = True
            while True:
                used.add((cur, nx))
                if nx == start:
                    break
                loop.append(nx)
                cands = out_he.get(nx, [])
                if len(cands) != 1 or len(loop) > max_len:
                    ok = False
                    break
                cur, nx = nx, cands[0]
            if not ok or len(loop) < 3 or len(set(loop)) != len(loop):
                continue
            if pm.bnd[loop].sum() > len(loop) // 2:
                continue
            pm.faces.append(loop)
            added += 1
    return added
