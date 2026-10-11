"""4-RoSy orientation (cross) field.

Extrinsic smoothing as in Instant Meshes: neighbouring crosses are compared
directly in 3D (best of the four rotations), averaged, and projected back into
the tangent plane. Unlike intrinsic smoothing this naturally follows principal
curvature directions and avoids curvature-induced singularities (e.g. a torus
gets a singularity-free field). Solved coarse-to-fine on the hierarchy.
"""

import numpy as np

from .geom import dot, normalize, tangent_frames
from .hierarchy import restrict_pick


def match4(q_ref, q, n):
    """Rotate each q (by k*90deg about n) to best align with q_ref."""
    t = np.cross(n, q)
    dq = dot(q_ref, q)
    dt = dot(q_ref, t)
    use_q = np.abs(dq) >= np.abs(dt)
    out = np.where(use_q[:, None], q * np.sign(dq)[:, None], t * np.sign(dt)[:, None])
    return out


def _tangent_unit(v, N, fallback):
    v = v - N * dot(v, N)[:, None]
    nv = np.linalg.norm(v, axis=1)
    bad = nv < 1e-9
    v[bad] = fallback[bad]
    return v / np.maximum(np.linalg.norm(v, axis=1), 1e-30)[:, None]


def solve_orientation(levels, hard_mask, hard_dir, soft_mask=None, soft_dir=None, soft_w=None,
                      iters_coarse=40, iters_fine=60):
    """Generator; yields progress in [0,1] and returns the fine-level field (n,3)."""
    nlev = len(levels)
    hm, hd = restrict_pick(levels, hard_mask, hard_dir)
    if soft_mask is None:
        soft_mask = np.zeros(levels[0].n, bool)
        soft_dir = np.zeros((levels[0].n, 3))
        soft_w = np.zeros(levels[0].n)
    sm, sd = restrict_pick(levels, soft_mask, np.concatenate([soft_dir, soft_w[:, None]], 1))

    q = None
    for li in range(nlev - 1, -1, -1):
        L = levels[li]
        N = L.N
        e1, _ = tangent_frames(N)
        if q is None:
            q = e1.copy()
        else:
            q = _tangent_unit(q[L.parent], N, e1)
        hmask = hm[li]
        hq = _tangent_unit(hd[li][hmask], N[hmask], e1[hmask])
        smask = sm[li]
        sq = _tangent_unit(sd[li][smask, :3], N[smask], e1[smask])
        sw = (sd[li][smask, 3] * L.deg[smask])[:, None]
        q[hmask] = hq

        iters = iters_fine if li == 0 else iters_coarse
        self_w = 0.5 * np.maximum(L.deg, 1.0)[:, None]
        for it in range(iters):
            m = match4(q[L.ei], q[L.ej], N[L.ej])
            acc = np.stack([np.bincount(L.ei, weights=m[:, k], minlength=L.n) for k in range(3)], 1)
            acc += self_w * q
            if len(sq):
                acc[smask] += sw * match4(q[smask], sq, N[smask])
            q = _tangent_unit(acc, N, q)
            q[hmask] = hq
            if it % 10 == 0:
                yield (nlev - 1 - li + it / iters) / nlev
    return q
