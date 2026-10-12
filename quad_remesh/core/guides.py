# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

"""Guide strokes as edge loops.

A stroke is turned into a chain of field-mesh edges running along it; the
chain's vertices are moved onto the stroke and become line constraints of
both fields (like a sharp edge, but without breaking normal smoothing or
being marked sharp on the result). The lattice then has a line through the
stroke, so the extracted mesh has an edge loop on it. A closed stroke (first
point repeated at the end) gives a closed loop, and can be surrounded by
concentric rings (offset copies one lattice cell apart) that become loops too.
"""

import heapq

import numpy as np

from .spatial import PointIndex, SegmentProjector


def resample(poly, spacing):
    """Evenly spaced points along a polyline, with unit tangents."""
    poly = np.asarray(poly, dtype=np.float64)
    if len(poly) < 2:
        return poly, np.zeros_like(poly)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] <= 0:
        return poly[:1], np.zeros((1, 3))
    t = np.linspace(0, s[-1], max(2, int(np.ceil(s[-1] / spacing)) + 1))
    pts = np.stack([np.interp(t, s, poly[:, k]) for k in range(3)], 1)
    tan = np.gradient(pts, axis=0)
    if np.linalg.norm(poly[0] - poly[-1]) < 1e-9 * max(s[-1], 1e-30):
        tan[0] = tan[-1] = pts[1] - pts[-2]          # closed: wrap the tangent
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-30)
    return pts, tan


def gauss_line(P, sigma, closed):
    """Gaussian filter along a polyline (points evenly spaced)."""
    r = int(np.ceil(3.0 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    if closed:
        ext = np.concatenate([P[-r:], P, P[:r]]) if r < len(P) else np.concatenate([P] * 3)[len(P) - r:2 * len(P) + r]
    else:
        # odd reflection about the end points keeps them (and the end direction) in place
        head = 2.0 * P[0] - P[1:r + 1][::-1]
        tail = 2.0 * P[-1] - P[-r - 1:-1][::-1]
        ext = np.concatenate([head, P, tail])
    return np.stack([np.convolve(ext[:, c], k, mode="valid") for c in range(3)], 1)


def is_closed(poly):
    poly = np.asarray(poly)
    return len(poly) > 3 and np.allclose(poly[0], poly[-1])


def _resample_closed(P, spacing):
    """Even points along a closed polyline given without its repeated end."""
    pts, _ = resample(np.vstack([P, P[:1]]), spacing)
    return pts[:-1]


def _side(P, surf, comps):
    """Unit in-surface direction across the closed polyline at each point."""
    _, N = surf.project(P, comps)
    T = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    S = np.cross(N, T)
    return S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-30)


def _self_overlap(P, tol, skip=4):
    """Points closer than tol to a point that is not a near neighbour along the ring."""
    index = PointIndex(P)
    n = len(P)
    for i, p in enumerate(P):
        for _co, j, _d in index.kd.find_range(p, tol):
            gap = abs(i - j)
            if min(gap, n - gap) > skip:
                return True
    return False


def concentric_rings(poly, surf, comp, step, outside=1, inside=0, others=None, rim_q=1.0):
    """Offset copies of a closed stroke on the surface, `step` apart (one
    lattice cell), `outside` of them away from the area it encloses and
    `inside` towards it. A side stops early (keeping the rings made so far)
    when a ring would run off the surface (an opening or a rim: more than a
    1-rim_q fraction of its points do), come within 0.75 step of another
    stroke (`others`: PointIndex over their points, or None), fold onto
    itself, or (inside) shrink below 6 cells around.
    Returns closed polylines (first point repeated)."""
    sub = 0.25 * step
    P = _resample_closed(np.asarray(poly, dtype=np.float64)[:-1], sub)
    if len(P) < 8:
        return []
    comps = np.full(len(P), comp)
    P, _ = surf.project(P, comps)
    side = _side(P, surf, comps)
    sgn = 1.0 if float(((P - P.mean(0)) * side).sum()) > 0 else -1.0
    rings = []
    for direction, count in ((sgn, outside), (-sgn, inside)):
        Q = P.copy()
        for _r in range(count):
            ok = True
            for _k in range(4):
                c = np.full(len(Q), comp)
                target = Q + direction * _side(Q, surf, c) * sub
                proj, _ = surf.project(target, c)
                if np.quantile(np.linalg.norm(proj - target, axis=1), rim_q) > 0.5 * sub:
                    ok = False                      # ran onto an opening, rim or crease
                    break
                Q = _resample_closed(proj, sub)
                if len(Q) < 8:
                    ok = False
                    break
                G = gauss_line(Q, 1.0, True)
                Q = G + (G - gauss_line(G, 1.0, True))
                Q, _ = surf.project(Q, np.full(len(Q), comp))
            if not ok:
                break
            length = float(np.linalg.norm(Q - np.roll(Q, 1, axis=0), axis=1).sum())
            if direction != sgn and length < 6.0 * step:
                break
            if _self_overlap(Q, 0.3 * step):
                break
            if others is not None and min(others.kd.find(q)[2] for q in Q) < 0.75 * step:
                break
            rings.append(np.vstack([Q, Q[:1]]))
    return rings


def on_surface(guides, surf, spacing, max_dist):
    """Resample each stroke and drop it onto the surface. Points farther than
    max_dist from the surface (strokes drawn on another part, or folded over a
    symmetry plane onto nothing) split the stroke. Returns polylines and the
    surface component each lies on."""
    out, comps = [], []
    for poly in guides:
        pts, _ = resample(poly, spacing)
        if len(pts) < 2:
            continue
        comp = surf.component_of(pts)
        proj, _ = surf.project(pts, comp)
        ok = np.linalg.norm(proj - pts, axis=1) <= max_dist
        run, rc = [], []
        for p, c, keep in zip(proj, comp, ok):
            if keep:
                run.append(p)
                rc.append(c)
            else:
                if len(run) >= 2:
                    out.append(np.array(run))
                    comps.append(int(np.bincount(rc).argmax()))
                run, rc = [], []
        if len(run) >= 2:
            out.append(np.array(run))
            comps.append(int(np.bincount(rc).argmax()))
    return out, comps


def projector(polys, comps, spacing):
    """SegmentProjector over the strokes (with component labels)."""
    segs = np.concatenate([np.stack([p[:-1], p[1:]], 1) for p in polys], 0)
    comp = np.concatenate([np.full(len(p) - 1, c, dtype=np.int64) for p, c in zip(polys, comps)])
    keep = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1) > 0
    segs, comp = segs[keep], comp[keep]
    if not len(segs):
        return None, segs
    return SegmentProjector(segs, spacing, comp), segs


def _nearest_in(index, P, comp_of_v, comp, k=16):
    """Nearest point-index entry to P that lies on component `comp`."""
    for co, i, d in index.kd.find_n(P, k):
        if comp_of_v[i] == comp:
            return i
    return index.nearest(P)


def _adjacency(n, edges, length):
    adj = [[] for _ in range(n)]
    for (a, b), L in zip(edges.tolist(), length.tolist()):
        adj[a].append((b, L))
        adj[b].append((a, L))
    return adj


def _path(adj, src, dst, cost, limit):
    """Cheapest vertex path src -> dst; cost(v, w, length) prices each edge."""
    if src == dst:
        return [src]
    dist = {src: 0.0}
    prev = {}
    heap = [(0.0, src)]
    while heap:
        d, v = heapq.heappop(heap)
        if v == dst:
            break
        if d > dist.get(v, np.inf) or d > limit:
            continue
        for w, L in adj[v]:
            nd = d + cost(v, w, L)
            if nd < dist.get(w, np.inf):
                dist[w] = nd
                prev[w] = v
                heapq.heappush(heap, (nd, w))
    if dst not in prev:
        return None
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return path[::-1]


def _stroke_cost(V, comp, gproj, off_w=4.0, turn_w=3.0):
    """Edge price for paths along a stroke: length, plus distance from the
    stroke, plus a penalty for running across it instead of along it."""
    cache = {}

    def near(v):
        if v not in cache:
            _, d, dd = gproj.project(V[v][None], comps=comp[v:v + 1])
            n = np.linalg.norm(dd[0])
            cache[v] = (float(d[0]), dd[0] / n if n > 0 else dd[0])
        return cache[v]

    def cost(v, w, L):
        # current length: V may have moved since the adjacency was built, and
        # a stale one could make the price negative (Dijkstra never ends)
        dw, tw = near(w)
        e = V[w] - V[v]
        Lc = float(np.linalg.norm(e))
        across = min(max(1.0 - abs(float(e @ tw)) / max(Lc, 1e-30), 0.0), 1.0)
        return Lc * (1.0 + turn_w * across) + off_w * dw
    return cost


def edge_chains(mesh, polys, comps, gproj, vcomp, skip):
    """Field-mesh edges along each stroke.

    Returns (vertex mask, edge mask, per-vertex unit tangent). Vertices in
    `skip` (sharp features, boundaries) keep their own constraints."""
    V = mesh.V
    n = len(V)
    vmask = np.zeros(n, bool)
    emask = np.zeros(len(mesh.edges), bool)
    tangent = np.zeros((n, 3))
    if gproj is None:
        return vmask, emask, tangent
    adj = _adjacency(n, mesh.edges, mesh.edge_length)
    index = PointIndex(V)
    med = float(np.median(mesh.edge_length))
    cost = _stroke_cost(V, vcomp, gproj)

    pairs = []
    for poly, c in zip(polys, comps):
        verts = [_nearest_in(index, p, vcomp, c) for p in poly]
        chain = [verts[0]]
        for v in verts[1:]:
            if v == chain[-1]:
                continue
            step = np.linalg.norm(V[v] - V[chain[-1]])
            p = _path(adj, chain[-1], v, cost, 20.0 * step + 8.0 * med)
            if p is None:
                chain = [v]
                continue
            for a, b in zip(p[:-1], p[1:]):
                pairs.append((a, b))
            chain.extend(p[1:])
    if not pairs:
        return vmask, emask, tangent
    pairs = np.array(pairs, dtype=np.int64)
    idx = mesh.edge_index(pairs)
    emask[idx[idx >= 0]] = True
    vmask[mesh.edges[emask].ravel()] = True
    vmask &= ~skip
    gv = np.nonzero(vmask)[0]
    _, _, dirs = gproj.project(V[gv], comps=vcomp[gv])
    tangent[gv] = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-30)
    return vmask, emask, tangent


def snap_chain(mesh, vmask, gproj, vcomp, max_move):
    """Move chain vertices onto the stroke (they become exact lattice-line
    anchors). Moves longer than max_move are skipped."""
    V = mesh.V.copy()
    gv = np.nonzero(vmask)[0]
    if not len(gv):
        return V
    P, d, _ = gproj.project(V[gv], comps=vcomp[gv])
    ok = d <= max_move
    V[gv[ok]] = P[ok]
    return V


def soft_band(mesh, gproj, vcomp, radius, strength):
    """Soft orientation hints in a band around the strokes, so the flow next to
    a loop bends to follow it. Returns (mask, direction, weight)."""
    n = len(mesh.V)
    mask = np.zeros(n, bool)
    dirs = np.zeros((n, 3))
    w = np.zeros(n)
    if gproj is None or strength <= 0:
        return mask, dirs, w
    seg_len = float(np.linalg.norm(gproj.B - gproj.A, axis=1).max())
    near = np.array([gproj.kd.find(p)[2] for p in mesh.V]) < radius + seg_len
    idx = np.nonzero(near)[0]
    if not len(idx):
        return mask, dirs, w
    _, d, dd = gproj.project(mesh.V[idx], comps=vcomp[idx])
    inside = d < radius
    idx, d, dd = idx[inside], d[inside], dd[inside]
    lam = min(50.0, strength / max(1e-3, 1.0 - strength))
    mask[idx] = True
    dirs[idx] = dd / np.maximum(np.linalg.norm(dd, axis=1, keepdims=True), 1e-30)
    w[idx] = lam * (1.0 - d / radius) ** 2
    return mask, dirs, w



def trace_loops(pm, polys, comps, gproj, size_fn):
    """Make each stroke a continuous run of edges in the result: the cheapest
    edge path along it (edges of the extracted lattice wherever it already
    follows the stroke) is moved onto the stroke and protected like a sharp
    edge. Returns the number of path vertices."""
    from .polymesh import ekey
    n = len(pm.V)
    E = set()
    for f in pm.faces:
        for k in range(len(f)):
            E.add(ekey(f[k], f[(k + 1) % len(f)]))
    E = np.array(sorted(E), dtype=np.int64).reshape(-1, 2)
    if not len(E):
        return 0
    L = np.linalg.norm(pm.V[E[:, 0]] - pm.V[E[:, 1]], axis=1)
    adj = _adjacency(n, E, L)
    index = PointIndex(pm.V)
    cost = _stroke_cost(pm.V, pm.comp, gproj, off_w=6.0)

    snapped, protected = 0, set()
    corners = np.nonzero(pm.corner)[0]
    keep_all = set(corners.tolist())
    for poly, c in zip(polys, comps):
        step = 0.5 * float(np.median(size_fn(poly)))
        pts, _ = resample(poly, step)
        verts = [_nearest_in(index, p, pm.comp, c) for p in pts]
        # snap onto this stroke only (strokes may pass close to each other)
        own = SegmentProjector(np.stack([poly[:-1], poly[1:]], 1), step / 4.0)
        # Corners of sharp features stay where they are, so a path through
        # one beside the stroke (e.g. where a lip crease ends at a mouth
        # corner) would take the loop off the stroke: go around it if possible.
        avoid = set()
        if len(corners):
            _, dc, _ = own.project(pm.V[corners])
            avoid = set(corners[dc > 0.5 * step].tolist())

        def around(v, w, L):
            return cost(v, w, L) + (40.0 * step if w in avoid else 0.0)

        path_v, path_e = set(), set()
        cur = verts[0]
        for v in verts[1:]:
            if v == cur:
                continue
            seg = np.linalg.norm(pm.V[v] - pm.V[cur])
            limit = 20.0 * seg + 8.0 * step
            p = _path(adj, cur, v, around, limit) if avoid else None
            if p is None or avoid.intersection(p):
                p = _path(adj, cur, v, cost, limit)
            if p is None:
                cur = v
                continue
            for a, b in zip(p[:-1], p[1:]):
                path_e.add(ekey(a, b))
            path_v.update(p)
            cur = v
        if not path_v:
            continue
        pv = np.array(sorted(path_v), dtype=np.int64)
        P, d, _ = own.project(pm.V[pv])
        # a corner this close is where its sharp feature crosses the stroke:
        # it moves onto the stroke (and stays a corner); farther ones stay put
        ok = np.where(pm.corner[pv], d <= 0.5 * step, d < 2.0 * np.asarray(size_fn(pm.V[pv])))
        pm.V[pv[ok]] = P[ok]
        pm.feat[pv[ok]] = True
        keep = set(pv[ok].tolist()) | keep_all
        protected |= {e for e in path_e if e[0] in keep and e[1] in keep}
        snapped += int(ok.sum())
    pm.feature_edges |= protected
    return snapped
