# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

"""Local topology operations on quads that reduce irregular vertices."""

import numpy as np

from .polymesh import ekey


def _energy(val, target):
    return (val - target) ** 2


def _locked(pm, open_v):
    return pm.corner | pm.bnd | open_v


def diagonal_collapse(pm, max_rounds=8):
    """Collapse quad diagonals when it lowers sum((valence-4)^2)."""
    total = 0
    for _ in range(max_rounds):
        ef = pm.edge_faces()
        val, open_v = pm.valence(ef)
        locked = _locked(pm, open_v) | pm.feat
        nbr = [set() for _ in range(len(pm.V))]
        for (x, y) in ef:
            nbr[x].add(y)
            nbr[y].add(x)
        touched = np.zeros(len(pm.V), bool)
        merges = {}
        dead = set()
        for fi, f in enumerate(pm.faces):
            if len(f) != 4:
                continue
            for k in (0, 1):
                a, b, c, d = f[k], f[k + 1], f[(k + 2) % 4], f[(k + 3) % 4]
                if locked[a] or locked[c] or touched[[a, b, c, d]].any():
                    continue
                if open_v[b] or open_v[d]:
                    continue
                if (nbr[a] & nbr[c]) != {b, d} or c in nbr[a]:
                    continue
                if touched[list(nbr[a] | nbr[c])].any():
                    continue
                before = _energy(val[a], 4) + _energy(val[c], 4) + _energy(val[b], 4) + _energy(val[d], 4)
                after = _energy(val[a] + val[c] - 2, 4) + _energy(val[b] - 1, 4) + _energy(val[d] - 1, 4)
                if val[b] - 1 < 3 or val[d] - 1 < 3 or after >= before:
                    continue
                merges[c] = a
                touched[[a, b, c, d]] = True
                touched[list(nbr[a] | nbr[c])] = True
                dead.add(fi)
                pm.V[a] = 0.5 * (pm.V[a] + pm.V[c])
                break
        if not merges:
            break
        faces = []
        for fi, f in enumerate(pm.faces):
            if fi in dead:
                continue
            g = [merges.get(v, v) for v in f]
            # remove consecutive duplicates
            h = [g[i] for i in range(len(g)) if g[i] != g[i - 1]]
            if len(h) >= 3:
                faces.append(h)
        pm.faces = faces
        total += len(merges)
    return total


def edge_rotation(pm, max_rounds=8, max_len_ratio=1.6):
    """Rotate the shared edge of two quads when it lowers the valence energy."""
    total = 0
    for _ in range(max_rounds):
        ef = pm.edge_faces()
        val, open_v = pm.valence(ef)
        locked = _locked(pm, open_v)
        touched = np.zeros(len(pm.faces), bool)
        changes = 0
        created = set()
        mean_len = np.mean([np.linalg.norm(pm.V[a] - pm.V[b]) for a, b in ef.keys()]) if ef else 1.0
        for (u, v), fl in ef.items():
            if len(fl) != 2 or touched[fl[0]] or touched[fl[1]] or ekey(u, v) in pm.feature_edges:
                continue
            F1, F2 = pm.faces[fl[0]], pm.faces[fl[1]]
            if len(F1) != 4 or len(F2) != 4:
                continue
            # Orient: F1 contains u->v.
            k = F1.index(u)
            if F1[(k + 1) % 4] != v:
                F1, F2 = F2, F1
                k = F1.index(u)
                if F1[(k + 1) % 4] != v:
                    continue
            x1, x2 = F1[(k + 2) % 4], F1[(k + 3) % 4]
            j = F2.index(v)
            if F2[(j + 1) % 4] != u:
                continue
            y1, y2 = F2[(j + 2) % 4], F2[(j + 3) % 4]
            hexv = [v, x1, x2, u, y1, y2]
            if len(set(hexv)) != 6 or locked[u] or locked[v]:
                continue
            base = sum(_energy(val[w], 4) for w in hexv)
            best = None
            for (p, q, fa, fb) in (
                (x1, y1, [x1, x2, u, y1], [y1, y2, v, x1]),
                (x2, y2, [x2, u, y1, y2], [y2, v, x1, x2]),
            ):
                if ekey(p, q) in ef or ekey(p, q) in created or locked[p] or locked[q]:
                    continue
                if np.linalg.norm(pm.V[p] - pm.V[q]) > max_len_ratio * mean_len:
                    continue
                nv = {w: val[w] for w in hexv}
                nv[u] -= 1
                nv[v] -= 1
                nv[p] += 1
                nv[q] += 1
                if nv[u] < 3 or nv[v] < 3:
                    continue
                e = sum(_energy(nv[w], 4) for w in hexv)
                if e < base and (best is None or e < best[0]):
                    best = (e, fa, fb, p, q)
            if best is None:
                continue
            pm.faces[fl[0]], pm.faces[fl[1]] = best[1], best[2]
            touched[fl[0]] = touched[fl[1]] = True
            p, q = best[3], best[4]
            val[u] -= 1
            val[v] -= 1
            val[p] += 1
            val[q] += 1
            created.add(ekey(p, q))
            changes += 1
        total += changes
        if not changes:
            break
    return total
