# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

"""Conformal scale field that makes the cross field (nearly) integrable.

A uniform lattice cannot follow a curved cross field without inserting
dislocations (valence 3/5 pairs). Letting the quad size vary as h*exp(u)
removes them when u is the harmonic-conjugate-like potential of the field's
rotation: grad(u) ~= -J * grad(theta), where grad(theta) is the field's
rotation relative to parallel transport. We solve the least-squares Poisson
problem  L u = div(-J g)  with the cotan Laplacian and conjugate gradients.
"""

import numpy as np

from .geom import dot, normalize, rotate_between
from .orientation import match4


def field_rotation(P, N, Q, i, j):
    """Signed angle (radians) the cross field turns along each edge i->j,
    measured against parallel transport, in (-pi/4, pi/4]."""
    qj = rotate_between(Q[j], N[j], N[i])
    qj = qj - N[i] * dot(qj, N[i])[:, None]
    qj = normalize(qj)
    qj = match4(Q[i], qj, N[i])
    s = dot(np.cross(Q[i], qj), N[i])
    c = dot(Q[i], qj)
    return np.arctan2(s, c)


def _cotan_weights(V, F):
    """Cotan weight per (face, corner): weight of the edge opposite the corner."""
    out = np.zeros(F.shape)
    for k in range(3):
        a = V[F[:, k]]
        b = V[F[:, (k + 1) % 3]]
        c = V[F[:, (k + 2) % 3]]
        u = b - a
        v = c - a
        cr = np.linalg.norm(np.cross(u, v), axis=1)
        out[:, k] = 0.5 * dot(u, v) / np.maximum(cr, 1e-30)
    return np.clip(out, -10.0, 1e4)


def conformal_scale(mesh, N, Q, iters=800, tol=1e-7):
    """Per-vertex log-scale u (area-weighted mean zero)."""
    V, F = mesh.V, mesh.F
    n = len(V)
    # Field rotation on each directed face edge (corner k -> k+1).
    d = np.zeros(F.shape)
    for k in range(3):
        d[:, k] = field_rotation(V, N, Q, F[:, k], F[:, (k + 1) % 3])
    # Per-face gradient g with g . e_k ~= d_k (least squares in the face plane).
    E = np.stack([V[F[:, (k + 1) % 3]] - V[F[:, k]] for k in range(3)], 1)   # (m,3,3)
    fn = mesh.face_normal
    A = np.einsum("mki,mkj->mij", E, E) + fn[:, :, None] * fn[:, None, :] * np.maximum(
        np.einsum("mki,mki->m", E, E)[:, None, None], 1e-30)
    rhs = np.einsum("mki,mk->mi", E, d)
    g = np.linalg.solve(A, rhs[..., None])[..., 0]
    target = -np.cross(fn, g)                      # -J g  (J = rotate +90deg about the normal)
    # Triangles holding a field singularity have no meaningful rotation.
    singular = np.abs(d.sum(1)) > np.pi / 4
    target[singular] = 0.0
    # Divergence of the target field at vertices: b_i = sum_f A_f * target_f . grad(phi_i)
    area = mesh.face_area
    b = np.zeros(n)
    for k in range(3):
        opp = V[F[:, (k + 2) % 3]] - V[F[:, (k + 1) % 3]]           # edge opposite corner k
        grad_phi = np.cross(fn, opp) / np.maximum(2.0 * area, 1e-30)[:, None]
        np.add.at(b, F[:, k], area * dot(target, grad_phi))
    # Cotan Laplacian (positive semi-definite) with a tiny mass term for the gauge.
    w = _cotan_weights(V, F)
    ii, jj, ww = [], [], []
    for k in range(3):
        ii.append(F[:, (k + 1) % 3])
        jj.append(F[:, (k + 2) % 3])
        ww.append(w[:, k])
    ii = np.concatenate(ii)
    jj = np.concatenate(jj)
    ww = np.concatenate(ww)
    diag = np.bincount(ii, weights=ww, minlength=n) + np.bincount(jj, weights=ww, minlength=n)
    mass = mesh.vertex_area
    eps = 1e-6 * diag.mean() / max(mass.mean(), 1e-30)
    diag_full = diag + eps * mass

    def matvec(x):
        return diag_full * x - np.bincount(ii, weights=ww * x[jj], minlength=n) \
            - np.bincount(jj, weights=ww * x[ii], minlength=n)

    b = b - mass * (b.sum() / mass.sum())          # compatibility (Neumann)
    x = np.zeros(n)
    r = b - matvec(x)
    z = r / diag_full
    p = z.copy()
    rz = r @ z
    bn = np.linalg.norm(b) + 1e-30
    for _ in range(iters):
        Ap = matvec(p)
        alpha = rz / max(p @ Ap, 1e-300)
        x += alpha * p
        r -= alpha * Ap
        if np.linalg.norm(r) / bn < tol:
            break
        z = r / diag_full
        rz_new = r @ z
        p = z + (rz_new / max(rz, 1e-300)) * p
        rz = rz_new
    x -= (x * mass).sum() / mass.sum()
    return x


def scale_field(mesh, N, Q, h, strength, sign=1.0, max_ratio=3.0):
    """Per-vertex target edge length h*exp(strength*u), normalized so the
    expected face count is unchanged, and clamped to [h/max_ratio, h*max_ratio]."""
    if strength <= 0:
        return np.full(len(mesh.V), h)
    u = sign * strength * conformal_scale(mesh, N, Q)
    lim = np.log(max_ratio)
    u = np.clip(u, -lim, lim)
    mass = mesh.vertex_area
    # keep sum(area / h_i^2) == sum(area / h^2)
    s = np.sqrt((mass * np.exp(-2 * u)).sum() / mass.sum())
    return h * np.exp(u) * s


def feature_spacing(V, edges, feat_mask, spacing, max_samples=200_000, min_sep=0.0):
    """For each feature edge, the distance to the nearest *separate* feature
    curve beside it (chains that meet at a corner do not count). Ridges,
    knurling and embossing show up as closely spaced curves.

    Returns (per-edge spacing for edges[feat_mask] (inf if none nearby),
    sample points, sample spacing). Curves closer than `min_sep` count as the
    same curve (zero-area CAD slivers lying along a crease duplicate it)."""
    from mathutils.kdtree import KDTree

    from .features import _edge_chains
    idx, lab = _edge_chains(edges, feat_mask)
    if not len(idx):
        return np.zeros(0), np.zeros((0, 3)), np.zeros(0)
    fe = edges[idx]
    A, B = V[fe[:, 0]], V[fe[:, 1]]
    L = np.linalg.norm(B - A, axis=1)
    spacing = max(spacing, float(L.sum()) / max_samples)
    ns = np.maximum(1, np.ceil(L / spacing).astype(int))
    sid = np.repeat(np.arange(len(L)), ns)
    t = (np.concatenate([np.arange(k) for k in ns]) + 0.5) / ns[sid]
    P = A[sid] + (B - A)[sid] * t[:, None]
    D = (B - A)[sid] / np.maximum(L[sid], 1e-30)[:, None]
    slab = lab[sid]
    adj = {}
    for k, (u, w) in enumerate(fe.tolist()):
        for v in (u, w):
            adj.setdefault(v, set()).add(int(lab[k]))
    near = {}
    for labs in adj.values():
        for x in labs:
            near.setdefault(x, set()).update(labs)
    kd = KDTree(len(P))
    for i, p in enumerate(P):
        kd.insert(p, i)
    kd.balance()
    sp = np.full(len(P), np.inf)
    for i in range(len(P)):
        mine = near.get(int(slab[i]), {int(slab[i])})
        for co, j, d in kd.find_n(P[i], 16):
            if d <= min_sep or int(slab[j]) in mine:
                continue
            off = np.asarray(co) - P[i]
            if abs(float(off @ D[i])) < 0.7 * d:    # beside, not further along
                sp[i] = d
                break
    per_edge = np.full(len(L), np.inf)
    np.minimum.at(per_edge, sid, sp)
    return per_edge, P, sp


def detail_size_points(dmesh, feat_mask, hv_min, hv_max, cell_per_quad, curv_factor=0.7):
    """Where the surface has detail smaller than a quad, return (points,
    lattice cell sizes) asking for smaller quads there. `dmesh` should be an
    evenly tessellated copy of the surface at about hv_min/3 resolution.

    Two kinds of detail are measured: curvature radius on smoothed normals
    (never across sharp creases, which edge loops already follow), and the
    spacing between nearby separate sharp-edge curves, which need at least
    one lattice cell between them.
    """
    from .mesh_data import smooth_normals
    V = dmesh.V
    pts, sizes = [], []
    q_min = hv_min / cell_per_quad
    med = float(np.median(dmesh.edge_length)) if len(dmesh.edge_length) else 0.0

    # Curvature radius, from normals smoothed over ~q_min/4 (scan noise and
    # tessellation facets stay out) and only over edges of reasonable length.
    Ns = smooth_normals(dmesh, feat_mask, max(0.25 * q_min, 2.0 * med))
    on_feat = np.zeros(len(V), bool)
    on_feat[dmesh.edges[feat_mask].ravel()] = True
    # Edges touching a crease see its averaged normal (half the crease angle),
    # which is not curvature of the surface.
    ok = (~feat_mask & (dmesh.edge_face_count == 2) & (dmesh.edge_length >= 0.3 * med)
          & ~on_feat[dmesh.edges[:, 0]] & ~on_feat[dmesh.edges[:, 1]])
    a, b = dmesh.edges[ok, 0], dmesh.edges[ok, 1]
    if len(a):
        ang = np.arccos(np.clip(np.einsum("ij,ij->i", Ns[a], Ns[b]), -1.0, 1.0))
        kappa = ang / dmesh.edge_length[ok]
        kv = np.zeros(len(V))
        cnt = np.zeros(len(V))
        np.add.at(kv, a, kappa)
        np.add.at(kv, b, kappa)
        np.add.at(cnt, a, 1.0)
        np.add.at(cnt, b, 1.0)
        kv /= np.maximum(cnt, 1.0)                      # mean, not max: robust to outliers
        cell = cell_per_quad * curv_factor / np.maximum(kv, 1e-30)   # chord-error bound
        sel = cell < 0.95 * hv_max
        pts.append(V[sel])
        sizes.append(np.maximum(cell[sel], hv_min))

    # Spacing between separate feature curves: one lattice cell between them.
    _, P, sp = feature_spacing(V, dmesh.edges, feat_mask, 0.5 * q_min, min_sep=0.1 * hv_min)
    if len(P):
        sel = sp < 0.95 * hv_max
        pts.append(P[sel])
        sizes.append(np.maximum(sp[sel], hv_min))
    if not pts:
        return np.zeros((0, 3)), np.zeros(0)
    return np.vstack(pts), np.concatenate(sizes)


def drop_unresolvable_features(mesh, feat_mask, hv, factor=0.9):
    """Feature curves packed closer than the local lattice cell can resolve
    (knurling, threads, embossing below the quad size) would demand
    contradictory edge loops and wreck the field; such curves are dropped
    (smoothed over). Open boundaries are always kept."""
    per_edge, _, _ = feature_spacing(mesh.V, mesh.edges, feat_mask, 0.25 * float(np.min(hv)))
    if not len(per_edge):
        return feat_mask
    from .features import _edge_chains
    idx, _lab = _edge_chains(mesh.edges, feat_mask)
    fe = mesh.edges[idx]
    local = np.minimum(hv[fe[:, 0]], hv[fe[:, 1]])
    drop = (per_edge < factor * local) & (mesh.edge_face_count[idx] == 2)
    out = feat_mask.copy()
    out[idx[drop]] = False
    return out


def apply_size_points(V, hv, pts, sizes):
    """Lower hv at the field vertices nearest to each size point."""
    hv = np.array(hv, dtype=np.float64)
    if not len(pts):
        return hv
    from mathutils.kdtree import KDTree
    kd = KDTree(len(V))
    for i, p in enumerate(V):
        kd.insert(p, i)
    kd.balance()
    idx = np.array([kd.find(p)[1] for p in pts], dtype=np.int64)
    np.minimum.at(hv, idx, sizes)
    return hv


def limit_gradient(V, edges, hv, g=0.1, max_iter=2000):
    """Let the size grow by at most g per unit distance (hv_j <= hv_i + g*|e|),
    so the lattice never has to change spacing abruptly. g = 0.1 was measured
    on CAD parts: at 0.25 graded areas got flips and holes, at 0.1 none."""
    hv = np.array(hv, dtype=np.float64)
    a, b = edges[:, 0], edges[:, 1]
    step = g * np.linalg.norm(V[a] - V[b], axis=1)
    for _ in range(max_iter):
        new = hv.copy()
        np.minimum.at(new, b, hv[a] + step)
        np.minimum.at(new, a, hv[b] + step)
        if np.array_equal(new, hv):
            break
        hv = new
    return hv
