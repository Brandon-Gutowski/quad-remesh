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


def thin_part_limit(mesh, N, hv, surf_proj, vcomp, ratio=1.2, floor=0.35, smooth_iters=6):
    """Shrink the target size where the part is thinner than a quad, so thin
    shells (ears, fins, cloth) keep valid front and back surfaces."""
    from mathutils import Vector
    V = mesh.V
    n = len(V)
    thick = np.full(n, np.inf)
    eps = 1e-4 * float(np.mean(hv))
    for i in range(n):
        tree = surf_proj.trees.get(int(vcomp[i]))
        if tree is None:
            continue
        nv = Vector(N[i])
        hit = tree.ray_cast(Vector(V[i]) - nv * eps, -nv)
        if hit[0] is not None and hit[1].dot(nv) < 0.0:   # the far side faces away
            thick[i] = hit[3]
    limit = np.maximum(ratio * thick, floor * hv)
    out = np.minimum(hv, limit)
    if not (out < hv).any():
        return hv
    a, b = mesh.edges[:, 0], mesh.edges[:, 1]
    deg = np.maximum(np.bincount(np.concatenate([a, b]), minlength=n), 1)
    for _ in range(smooth_iters):
        avg = (np.bincount(a, weights=out[b], minlength=n) + np.bincount(b, weights=out[a], minlength=n)) / deg
        out = np.minimum(out, 0.5 * (out + avg))   # only ever spreads the refinement
    return np.maximum(out, floor * hv)
