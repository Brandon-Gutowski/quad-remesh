# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

"""Concentric rings inside closed guide loops.

The lattice cannot give a closed guide clean rings inside it: every ring
further in is shorter, so with square cells neighbouring rings never have the
same number of edges and the spokes between them shear or end in poles. Once
the guide is an edge loop of the result, the faces it encloses are replaced
with a template: rings that all have the loop's vertex count (radial spokes by
construction), about one quad apart, around a small grid in the middle whose
four corners are the only poles. The rings are offset copies of the stroke
marched inward across the surface (as for Rings In), so they follow sockets
and lids instead of a plane; the grid is a Coons patch on the innermost ring.
"""

import numpy as np

from .polymesh import ekey


def _arc_param(P, poly):
    """Arc length along the polyline of the closest point to each P, and the distance."""
    A, Bp = poly[:-1], poly[1:]
    D = Bp - A
    L = np.linalg.norm(D, axis=1)
    s0 = np.concatenate([[0.0], np.cumsum(L)])[:-1]
    t = np.clip(((P[:, None] - A[None]) * D[None]).sum(2) / np.maximum(L * L, 1e-30)[None], 0.0, 1.0)
    C = A[None] + t[..., None] * D[None]
    d = np.linalg.norm(P[:, None] - C, axis=2)
    k = d.argmin(1)
    i = np.arange(len(P))
    return s0[k] + t[i, k] * L[k], d[i, k], float(L.sum())


def _near_graph(pm, E, poly, h, tol):
    """Result edges along the stroke: both ends within tol*h of it, running
    along it rather than across. Returns (edges, arc position per vertex,
    stroke length)."""
    cand = np.unique(E)
    s, d, total = _arc_param(pm.V[cand], poly)
    on = d < tol * h
    pos = dict(zip(cand[on].tolist(), s[on].tolist()))
    E = E[[a in pos and b in pos for a, b in E.tolist()]]
    if len(E):
        ds = np.array([pos[b] - pos[a] for a, b in E.tolist()])
        ds = np.abs((ds + 0.5 * total) % total - 0.5 * total)
        L = np.linalg.norm(pm.V[E[:, 1]] - pm.V[E[:, 0]], axis=1)
        E = E[ds > 0.5 * L]
    return E, pos, total


def _walk(pm, E, pos, total, h):
    """One turn around the stroke through the graph E: the ordered vertex
    cycle, or None."""
    adj = {}
    for a, b in E.tolist():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    # dead ends (spurs left by the path search) are not part of the loop
    leaves = [v for v, nb in adj.items() if len(nb) < 2]
    while leaves:
        v = leaves.pop()
        for u in adj.pop(v, ()):
            adj[u].discard(v)
            if len(adj[u]) < 2:
                leaves.append(u)
    if len(adj) < 8:
        return None

    def delta(a, b):
        return (pos[b] - pos[a] + 0.5 * total) % total - 0.5 * total

    # Depth-first walk forward along the stroke (nearest vertex ahead first;
    # short steps back allowed for staircase loops) until it is back at the
    # start after one turn.
    start = min(adj, key=lambda v: pos[v])
    back = 0.3 * h
    stack = [(start, [start], 0.0)]
    cyc = None
    budget = 20 * len(adj)
    while stack and budget > 0:
        budget -= 1
        v, path, run = stack.pop()
        # preferred: the traced (protected) path, then the nearest vertex
        # ahead, then the others ahead, then back steps (pushed
        # last-preferred first, so popped in that order)
        steps = sorted(((delta(v, u), u) for u in adj[v]),
                       key=lambda du: (du[0] <= 0, ekey(v, du[1]) not in pm.feature_edges, abs(du[0])),
                       reverse=True)
        for d, u in steps:
            if d < -back or (d <= 0 and u == start):
                continue
            r = run + d
            if u == start:
                if abs(r - total) < 0.05 * total and len(path) >= 8:
                    cyc = path
                    break
                continue
            if u in path or r > 1.05 * total:
                continue
            stack.append((u, path + [u], r))
        if cyc is not None:
            break
    return cyc


def _loop_cycle(pm, poly, h, tol=0.25):
    """Ordered vertex cycle of result edges running along a closed stroke (one
    turn around it, following the stroke), and every vertex on the stroke;
    None if there is no such cycle. Any edge on the stroke counts, not only
    the protected ones: the traced path can leave gaps where the lattice
    already ran along the stroke."""
    E = np.array(sorted(pm.edge_faces()), dtype=np.int64).reshape(-1, 2)
    if not len(E):
        return None
    G, pos, total = _near_graph(pm, E, poly, h, tol)
    cyc = _walk(pm, G, pos, total, h) if len(G) else None
    if cyc is None:
        return None
    return cyc, set(pos)


def _sides(pm, cyc):
    """Faces on the left of the loop (as ordered, by the faces' winding) and
    on its right, each flood-filled without crossing the loop, and the
    edge-to-faces map; None if the loop does not separate the two."""
    ef = pm.edge_faces()
    loop_e = {ekey(a, b) for a, b in zip(cyc, np.roll(cyc, -1).tolist())}
    left, right = set(), set()
    for a, b in zip(cyc, np.roll(cyc, -1).tolist()):
        fl = ef.get(ekey(a, b), [])
        if len(fl) != 2:
            return None
        for fi in fl:
            f = pm.faces[fi]
            k = f.index(a)
            (left if f[(k + 1) % len(f)] == b else right).add(fi)
    out = []
    for seed in (left, right):
        region, stack = set(seed), list(seed)
        while stack:
            f = pm.faces[stack.pop()]
            for k in range(len(f)):
                e = ekey(f[k], f[(k + 1) % len(f)])
                if e in loop_e:
                    continue
                for g in ef[e]:
                    if g not in region:
                        region.add(g)
                        stack.append(g)
        out.append(region)
    if out[0] & out[1]:
        return None
    return out, ef


def _region_ok(pm, region, cyc, ef, on_stroke, h):
    """A disk of faces with no opening and no feature curve inside it (specks
    of feature, and the stroke's own detours, are replaced along with it)."""
    verts = {v for fi in region for v in pm.faces[fi]}
    inner = verts - set(cyc)
    if not inner:
        return False
    edges = {ekey(f[k], f[(k + 1) % len(f)]) for fi in region for f in [pm.faces[fi]] for k in range(len(f))}
    if any(len(ef[e]) != 2 for e in edges if e[0] in inner or e[1] in inner):
        return False
    keep = inner - on_stroke
    feat = [e for e in pm.feature_edges if e in edges and (e[0] in keep or e[1] in keep)]
    if feat and np.linalg.norm(pm.V[[a for a, _ in feat]] - pm.V[[b for _, b in feat]], axis=1).sum() > 3.0 * h:
        return False
    return len(verts) - len(edges) + len(region) == 1


def _area(pm, region):
    a = 0.0
    for fi in region:
        P = pm.V[pm.faces[fi]]
        c = P.mean(0)
        a += 0.5 * np.linalg.norm(np.cross(P - c, np.roll(P, -1, axis=0) - c).sum(0))
    return a


def _frame(P, normals):
    """Plane through the loop: centre, in-plane axes (major first) and normal,
    right-handed so a loop wound like the faces runs counter-clockwise."""
    c = P.mean(0)
    n = normals.sum(0)
    n /= max(np.linalg.norm(n), 1e-30)
    R = P - c
    R = R - np.outer(R @ n, n)
    evec = np.linalg.eigh(R.T @ R)[1]
    e1 = evec[:, 2] - n * (evec[:, 2] @ n)
    e1 /= max(np.linalg.norm(e1), 1e-30)
    e2 = np.cross(n, e1)
    return c, e1, e2, n


def _core_xy(grid, a, b, cell, centre):
    """Plane positions of core grid points: an a x b grid of `cell` squares
    with its corners pulled in (elliptical grid mapping), so the core is
    rounded like the rings around it."""
    g = np.array(grid, dtype=np.float64).reshape(-1, 2)
    u = 2.0 * g / [a, b] - 1.0
    d = np.stack([u[:, 0] * np.sqrt(1.0 - 0.5 * u[:, 1] ** 2), u[:, 1] * np.sqrt(1.0 - 0.5 * u[:, 0] ** 2)], 1)
    return centre + d * (0.5 * cell * np.array([a, b], dtype=np.float64))


def _core_border(a, b):
    """Grid coordinates of the core border, counter-clockwise from (0,0)."""
    out = [(x, 0) for x in range(a)] + [(a, y) for y in range(b)]
    out += [(x, b) for x in range(a, 0, -1)] + [(0, y) for y in range(b, 0, -1)]
    return out


def _closest_arc(R, P):
    """Arc length along the closed polyline R (first point repeated) of the
    closest point to each P."""
    A, Bp = R[:-1], R[1:]
    D = Bp - A
    L = np.linalg.norm(D, axis=1)
    s0 = np.concatenate([[0.0], np.cumsum(L)])
    t = np.clip(((P[:, None] - A[None]) * D[None]).sum(2) / np.maximum(L * L, 1e-30)[None], 0.0, 1.0)
    d = np.linalg.norm(P[:, None] - (A[None] + t[..., None] * D[None]), axis=2)
    k = d.argmin(1)
    return s0[k] + t[np.arange(len(P)), k] * L[k], s0


def _at_arc(R, s0, t):
    """Points at arc lengths t (wrapped) along the closed polyline R."""
    t = t % s0[-1]
    return np.stack([np.interp(t, s0, R[:, k]) for k in range(3)], 1)


def _surface_rings(L0, stroke, surf, comp, h, M, core_cell=0.45, even_rate=0.35):
    """Rings inside the loop L0 (M points, counter-clockwise around the
    inside): offsets of the (smooth) stroke marched across the surface one
    quad apart, each sampled where the spokes from the ring outside it land.
    The last ring is the core's border: the one whose length suits M core
    cells of about core_cell*h. Returns a list of (M,3) arrays, or None."""
    from .guides import concentric_rings
    # a few points over a crease (mouth corners, a lid's edge) do not stop the
    # rings, as they would for Rings In
    offs = concentric_rings(stroke, surf, comp, h, outside=0, inside=64, rim_q=0.9)
    if not offs:
        return None
    lens = np.array([np.linalg.norm(np.diff(r, axis=0), axis=1).sum() for r in offs])
    ok = lens / M >= 0.25 * h
    if not ok.any():
        return None
    kc = int(np.argmin(np.where(ok, np.abs(np.log(lens / (M * core_cell * h))), np.inf)))
    rings, prev = [], L0
    for r in offs[:kc + 1]:
        # same direction as the loop (offsets keep the stroke's direction,
        # but be safe)
        t, s0 = _closest_arc(r, prev)
        total = s0[-1]
        rel = (t - t[0]) % total
        if np.any(np.diff(rel) < 0):
            t2, _ = _closest_arc(r[::-1].copy(), prev)
            rel2 = (t2 - t2[0]) % total
            if np.all(np.diff(rel2) >= 0):
                r, t, rel = r[::-1].copy(), t2, rel2
                s0 = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(r, axis=0), axis=1))])
            else:
                rel = np.maximum.accumulate(rel)
        # spokes land where they point; each ring a little more evenly spaced
        even = np.arange(M) * total / M
        rel = (1.0 - even_rate) * rel + even_rate * even
        prev = _at_arc(r, s0, t[0] + rel)
        rings.append(prev)
    return rings


def _coons(border, a, b):
    """3D grid points (x, y) for 0<x<a, 0<y<b of a Coons patch on the border
    {(x,y): point}."""
    out = {}
    P00, P10, P01, P11 = border[(0, 0)], border[(a, 0)], border[(0, b)], border[(a, b)]
    for y in range(1, b):
        for x in range(1, a):
            u, v = x / a, y / b
            out[(x, y)] = ((1 - v) * border[(x, 0)] + v * border[(x, b)] + (1 - u) * border[(0, y)]
                           + u * border[(a, y)] - ((1 - u) * (1 - v) * P00 + u * (1 - v) * P10
                                                   + (1 - u) * v * P01 + u * v * P11))
    return out


def _fill(pm, cyc, region, stroke, h, surf):
    M = len(cyc)
    if M % 2 or M < 8:
        return False
    faces = [pm.faces[fi] for fi in region]
    # plane of the loop, oriented by the region's faces
    nrm = []
    for f in faces:
        P = pm.V[f]
        c = P.mean(0)
        nrm.append(np.cross(P - c, np.roll(P, -1, axis=0) - c).sum(0))
    c3, e1, e2, n = _frame(pm.V[cyc], np.array(nrm))

    def plane(P):
        return np.stack([(P - c3) @ e1, (P - c3) @ e2], 1)

    B = plane(pm.V[cyc])
    # the loop must wind once, counter-clockwise, around its inside (small
    # steps back are staircase steps)
    ang = np.unwrap(np.arctan2(B[:, 1] - B[:, 1].mean(), B[:, 0] - B[:, 0].mean()))
    ang = np.append(ang, ang[-1] + (ang[0] - ang[-1]) % (2.0 * np.pi))
    if not (5.5 < ang[-1] - ang[0] < 7.1) or np.diff(ang).min() < -0.3:
        return False
    comp = int(pm.comp[cyc[0]])
    rings = _surface_rings(pm.V[cyc], stroke, surf, comp, h, M)
    if rings is None:
        return False

    # core: an a x b grid (aspect of the last ring) whose border is the last
    # ring, its corners where a rounded a x b grid laid on the ring falls
    C = plane(rings[-1])
    ex = 0.5 * (C[:, 0].max() - C[:, 0].min())
    ey = 0.5 * (C[:, 1].max() - C[:, 1].min())
    a = int(np.clip(round(0.5 * M * ex / max(ex + ey, 1e-30)), 1, M // 2 - 1))
    b = M // 2 - a
    cell = 0.5 * (2.0 * ex / a + 2.0 * ey / b)
    border = _core_border(a, b)
    G = _core_xy(border, a, b, cell, 0.5 * (C.max(0) + C.min(0)))
    shift = min(range(M), key=lambda s: float(((np.roll(C, -s, axis=0) - G) ** 2).sum()))

    order = np.roll(np.arange(M), -shift)
    ring_pts = [r[order] for r in rings]
    bpts = {xy: p for xy, p in zip(border, ring_pts[-1])}
    grid_inner = [(x, y) for y in range(1, b) for x in range(1, a)]
    inner_pts = _coons(bpts, a, b)
    Gi = np.array([inner_pts[xy] for xy in grid_inner]).reshape(-1, 3)
    if len(Gi):
        Gi, _ = surf.project(Gi, np.full(len(Gi), comp))
    new3 = np.vstack(ring_pts + [Gi])

    # build: rings 1..K+1 (the last is the core border), then the core interior
    base = len(pm.V)
    ids = iter(range(base, base + len(new3)))
    ring_ids = [[cyc[i] for i in order]]
    for _r in rings:
        ring_ids.append([next(ids) for _ in range(M)])
    core = {xy: v for xy, v in zip(border, ring_ids[-1])}
    for xy in grid_inner:
        core[xy] = next(ids)
    new_faces = []
    for outer, inner_r in zip(ring_ids[:-1], ring_ids[1:]):
        for j in range(M):
            j1 = (j + 1) % M
            new_faces.append([outer[j], outer[j1], inner_r[j1], inner_r[j]])
    for y in range(b):
        for x in range(a):
            new_faces.append([core[(x, y)], core[(x + 1, y)], core[(x + 1, y + 1)], core[(x, y + 1)]])
    # every new quad must face the way the surface does
    Vall = np.vstack([pm.V, new3])
    Fq = np.array(new_faces, dtype=np.int64)
    P = Vall[Fq]
    fn = np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1])
    _, sn = surf.project(P.mean(1), np.full(len(Fq), comp))
    if ((fn * sn).sum(1) <= 0).mean() > 0.02:
        return False

    for p in new3:
        pm.add_vertex(p, like=cyc[0])
    for fi in region:
        pm.faces[fi] = None
    pm.faces.extend(new_faces)
    return True


def fill_closed_guides(pm, polys, h, surf):
    """Replace the inside of every closed guide loop with concentric rings
    around a small grid. Returns how many loops were filled."""
    from .guides import is_closed
    filled = 0
    for poly in polys:
        if not is_closed(poly):
            continue
        got = _loop_cycle(pm, poly, h)
        if got is None:
            continue
        cyc, on_stroke = got
        got = _sides(pm, cyc)
        if got is None:
            continue
        (left, right), ef = got
        # the inside: the smaller disk with nothing else to keep in it
        cands = [(r, s) for r, s in ((left, cyc), (right, cyc[::-1])) if _region_ok(pm, r, s, ef, on_stroke, h)]
        if not cands:
            continue
        region, order = min(cands, key=lambda rs: _area(pm, rs[0]))
        if len(cands) == 2 or _area(pm, region) < 0.5 * _area(pm, left | right):
            if _fill(pm, list(order), region, np.asarray(poly, dtype=np.float64), h, surf):
                filled += 1
    if filled:
        pm.compact()
    return filled
