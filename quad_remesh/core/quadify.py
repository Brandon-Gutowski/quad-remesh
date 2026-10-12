# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

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
    _, dist, dirs = seg_proj.project(mid, comps=pm.comp[E[:, 0]])
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
        nbrs = {}
        for (a, b) in ef:
            nbrs.setdefault(a, set()).add(b)
            nbrs.setdefault(b, set()).add(a)
        edges = set(ef.keys())
        touched = set()
        for v in np.nonzero((val == 2) & ~open_v & ~pm.corner & ~pm.bnd)[0]:
            fl = vf[v]
            if len(fl) != 2 or any(pm.faces[fi] is None or len(pm.faces[fi]) <= 3 for fi in fl):
                continue
            if any(fi in touched for fi in fl):
                continue
            nb = nbrs.get(v, set())
            if len(nb) != 2:
                continue
            u, w = nb
            if ekey(u, w) in edges:
                continue  # neighbours already joined: removing v would go non-manifold
            for fi in fl:
                pm.faces[fi] = [x for x in pm.faces[fi] if x != v]
                touched.add(fi)
            edges.add(ekey(u, w))
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


def split_odd_ngons(pm, min_size=5):
    existing = set(pm.edge_faces().keys())
    out = []
    for f in pm.faces:
        if len(f) >= min_size and len(f) % 2 == 1:
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
    comp = np.concatenate([pm.comp, pm.comp[E[:, 0]] if len(E) else np.zeros(0, np.int64),
                           np.array([pm.comp[f[0]] for f in pm.faces], dtype=np.int64)])
    fe = set()
    for k, i in zip(keys, range(len(keys))):
        if efeat[i]:
            fe.add(ekey(k[0], n + i))
            fe.add(ekey(k[1], n + i))
    return PolyMesh(V, faces, feat, corner, bnd, fe, comp)


def fill_holes(pm, max_len=12):
    """Close small boundary loops that are not part of the input's real
    boundary. Where several holes touch at a vertex, the shortest loop back
    to the start is used."""
    he = set()
    for f in pm.faces:
        m = len(f)
        for k in range(m):
            he.add((f[k], f[(k + 1) % m]))
    out_he = {}
    for a, b in he:
        if (b, a) not in he:
            out_he.setdefault(b, []).append(a)   # the hole runs b -> a
    used = set()
    added = 0
    for start in list(out_he.keys()):
        for first in out_he[start]:
            if (start, first) in used:
                continue
            # BFS over unused hole half-edges for the shortest loop to start
            prev = {first: start}
            dq = deque([first])
            found = False
            while dq and not found:
                v = dq.popleft()
                depth = 0
                w = v
                while w != start and depth <= max_len:
                    w = prev[w]
                    depth += 1
                if depth > max_len:
                    continue
                for nx in out_he.get(v, []):
                    if (v, nx) in used:
                        continue
                    if nx == start:
                        prev_start = v
                        found = True
                        break
                    if nx not in prev:
                        prev[nx] = v
                        dq.append(nx)
            if not found:
                continue
            loop = [prev_start]
            while loop[-1] != first:
                loop.append(prev[loop[-1]])
            loop.append(start)
            loop.reverse()          # start, first, ..., prev_start
            if len(loop) < 3 or len(loop) > max_len or len(set(loop)) != len(loop):
                continue
            if pm.bnd[loop].sum() > len(loop) // 2:
                continue
            for i in range(len(loop)):
                used.add((loop[i], loop[(i + 1) % len(loop)]))
            pm.faces.append(loop)
            added += 1
    return added


def close_notches(pm, max_len=10, reach=2.5):
    """Close gaps that open onto a real boundary. Such a gap merges into the
    long boundary loop, so fill_holes never sees it as a small loop; here each
    run of off-boundary vertices along a boundary loop is closed back to its
    two boundary endpoints when those are about one cell apart."""
    he = set()
    for f in pm.faces:
        m = len(f)
        for k in range(m):
            he.add((f[k], f[(k + 1) % m]))
    nxt = {}
    multi = set()
    for a, b in he:
        if (b, a) not in he:
            if b in nxt:
                multi.add(b)
            nxt[b] = a                       # boundary walk runs b -> a
    if not nxt:
        return 0
    E = np.array([[a, b] for a, b in he], dtype=np.int64)
    mean_len = float(np.linalg.norm(pm.V[E[:, 0]] - pm.V[E[:, 1]], axis=1).mean())
    existing = {ekey(a, b) for a, b in he}
    seen = set()
    added = 0
    for start in list(nxt.keys()):
        if start in seen or start in multi:
            continue
        loop = [start]
        seen.add(start)
        v = nxt[start]
        ok = True
        while v != start:
            if v in seen or v in multi or v not in nxt:
                ok = False
                break
            seen.add(v)
            loop.append(v)
            v = nxt[v]
        if not ok or len(loop) < 4 or not pm.bnd[loop].any():
            continue
        L = len(loop)
        # rotate so the loop starts on a boundary vertex
        k0 = next(i for i in range(L) if pm.bnd[loop[i]])
        loop = loop[k0:] + loop[:k0]
        i = 0
        while i < L:
            if pm.bnd[loop[i]] and not pm.bnd[loop[(i + 1) % L]]:
                j = i + 1
                while not pm.bnd[loop[j % L]] and j - i < L:
                    j += 1
                run = [loop[k % L] for k in range(i, j + 1)]
                u, w = run[0], run[-1]
                if (3 <= len(run) <= max_len and u != w and ekey(u, w) not in existing
                        and np.linalg.norm(pm.V[u] - pm.V[w]) <= reach * mean_len):
                    pm.faces.append(run)
                    existing.add(ekey(u, w))
                    added += 1
                i = j
            else:
                i += 1
    return added


def _boundary_cycles(out):
    """Split the open half-edge graph (vertex -> successors) into simple cycles.
    Where several holes touch at a vertex, each pass through it is cut out as
    its own cycle."""
    used = set()
    cycles = []
    for s in list(out.keys()):
        for t in out[s]:
            if (s, t) in used:
                continue
            path, pos, cur = [s], {s: 0}, s
            while True:
                nxt = [w for w in out.get(cur, ()) if (cur, w) not in used]
                if not nxt:
                    break
                w = nxt[0]
                used.add((cur, w))
                if w in pos:
                    k = pos[w]
                    if len(path) - k >= 3:
                        cycles.append(path[k:])
                    for v in path[k + 1:]:
                        del pos[v]
                    path = path[:k + 1]
                else:
                    pos[w] = len(path)
                    path.append(w)
                cur = w
    return cycles


def _fill_loop(pm, loop, existing, max_poly=8):
    """Close a loop (in boundary walk order). Short loops become one polygon
    (split into quads later); long ones are first cut along their shortest
    cross chord, recursively, so no high-valence fan centre is created."""
    n = len(loop)
    if n <= max_poly:
        pm.faces.append(list(loop))
        return
    P = pm.V[loop]
    best, best_d = None, np.inf
    for i in range(n):
        for k in range(n // 2 - 1, n // 2 + 2):    # roughly halve the loop
            j = (i + k) % n
            if min(k, n - k) < 3 or ekey(loop[i], loop[j]) in existing:
                continue
            d = np.linalg.norm(P[i] - P[j])
            if d < best_d:
                best, best_d = (i, j), d
    if best is None:
        pm.faces.append(list(loop))
        return
    i, j = best
    existing.add(ekey(loop[i], loop[j]))
    a = [loop[(i + t) % n] for t in range((j - i) % n + 1)]
    b = [loop[(j + t) % n] for t in range((i - j) % n + 1)]
    _fill_loop(pm, a, existing, max_poly)
    _fill_loop(pm, b, existing, max_poly)


def seal_holes(pm, bnd_proj, size_fn, tol=0.5, reach=5.0, rounds=5):
    """Close every opening of the result that is not an opening of the input.

    An open edge is *real* when its midpoint lies within tol*size of the input
    boundary (the same test the hole metric uses); cluster flags are not
    trusted. Loops made only of non-real edges are filled whatever their
    length; non-real runs along a real boundary loop are closed back to their
    endpoints when the closing edge is shorter than reach*size."""
    added = 0
    for _ in range(rounds):
        he_face = {}
        for fi, f in enumerate(pm.faces):
            m = len(f)
            for k in range(m):
                he_face[(f[k], f[(k + 1) % m])] = fi
        he = he_face.keys()
        walk = [(b, a) for a, b in he if (b, a) not in he_face]     # boundary runs b -> a
        if not walk:
            break
        W = np.array(walk, dtype=np.int64)
        mid = 0.5 * (pm.V[W[:, 0]] + pm.V[W[:, 1]])
        size = np.asarray(size_fn(mid), dtype=np.float64)
        if bnd_proj is None or bnd_proj.empty:
            real = np.zeros(len(W), bool)
        else:
            real = bnd_proj.project(mid)[1] < tol * size
        # Vertices on the real opening follow it (relax projects them onto
        # the feature / boundary curves instead of letting them drift inward).
        pm.feat[W[real].ravel()] = True
        pm.bnd[W[real].ravel()] = True
        if real.all():
            break
        is_real = dict(zip(walk, real.tolist()))
        out = {}
        for a, b in walk:
            out.setdefault(a, []).append(b)
        existing = {ekey(a, b) for a, b in he}
        before = added
        for cyc in _boundary_cycles(out):
            L = len(cyc)
            flags = [is_real[(cyc[i], cyc[(i + 1) % L])] for i in range(L)]
            if all(flags):
                continue
            if not any(flags) or L <= 4:
                _fill_loop(pm, cyc, existing)
                added += 1
                continue
            k0 = next(i for i in range(L) if flags[i])
            cyc = cyc[k0:] + cyc[:k0]
            flags = flags[k0:] + flags[:k0]
            i = 0
            while i < L:
                if flags[i]:
                    i += 1
                    continue
                j = i
                while j < L and not flags[j]:
                    j += 1
                run = [cyc[k % L] for k in range(i, j + 1)]   # vertices of the non-real run
                u, w = run[0], run[-1]
                if len(run) == 2 and bnd_proj is not None and not bnd_proj.empty:
                    # One edge cutting across a curved opening: bend it onto
                    # the real boundary through a new vertex.
                    x = bnd_proj.project(0.5 * (pm.V[u] + pm.V[w])[None])[0][0]
                    luw = np.linalg.norm(pm.V[w] - pm.V[u])
                    f = pm.faces[he_face[(w, u)]]
                    k = next((q for q in range(len(f)) if f[q] == w and f[(q + 1) % len(f)] == u), None)
                    if k is not None and min(np.linalg.norm(x - pm.V[u]), np.linalg.norm(x - pm.V[w])) > 0.2 * luw:
                        xi = pm.add_vertex(x, feat=True, bnd=True, like=u)
                        f.insert(k + 1, xi)
                        added += 1
                elif u != w and ekey(u, w) not in existing:
                    lim = reach * float(np.asarray(size_fn(pm.V[[u]]))[0])
                    if np.linalg.norm(pm.V[u] - pm.V[w]) <= lim:
                        existing.add(ekey(u, w))
                        _fill_loop(pm, run, existing)
                        added += 1
                i = j
        if added == before:
            break
    return added
