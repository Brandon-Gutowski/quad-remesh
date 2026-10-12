# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

"""Headless test suite.

    blender -b --factory-startup --python tests/run_tests.py
"""
import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bmesh  # noqa: E402
import bpy  # noqa: E402
import numpy as np  # noqa: E402

import meshes  # noqa: E402
import quad_remesh  # noqa: E402
from quad_remesh.core import pipeline  # noqa: E402
from quad_remesh.core.spatial import SegmentProjector  # noqa: E402


def area_of(V, F):
    t = V[F]
    return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum()


def run_core(name, density=24.0, **kw):
    V, F = getattr(meshes, name)()
    a = area_of(V, F)
    h = math.sqrt(a) / density
    res = pipeline.run(V, F, pipeline.Params(target_edge=h, **kw))
    return res, h, a, V, F


def guide_loop(res, guide, h, tol=0.15):
    """(fraction of the guide covered by edges lying along it, whether the
    edges on it close into a loop)."""
    from quad_remesh.core.spatial import SegmentProjector
    sp = SegmentProjector(np.stack([guide[:-1], guide[1:]], 1), h / 8)
    E = set()
    for f in res.faces:
        for k in range(len(f)):
            E.add(tuple(sorted((f[k], f[(k + 1) % len(f)]))))
    E = np.array(sorted(E))
    _, dv, _ = sp.project(res.V)
    on = E[(dv[E[:, 0]] < tol * h) & (dv[E[:, 1]] < tol * h)]
    _, _, dirs = sp.project(res.V[on].mean(1))
    d = res.V[on[:, 1]] - res.V[on[:, 0]]
    L = np.linalg.norm(d, axis=1)
    along = np.abs((d * dirs).sum(1)) > 0.85 * L * np.linalg.norm(dirs, axis=1)
    cover = L[along].sum() / np.linalg.norm(np.diff(guide, axis=0), axis=1).sum()
    # one connected piece that contains a cycle
    par = {}

    def find(x):
        while par.setdefault(x, x) != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b in on:
        par[find(a)] = find(b)
    roots = {find(v) for v in np.unique(on)}
    cycle = len(roots) == 1 and len(on) >= len(np.unique(on))
    return cover, cycle


class CoreQuality(unittest.TestCase):
    def check_common(self, res, h, area, closed=True, max_poles=None):
        m = res.metrics
        self.assertEqual(m["quad_ratio"], 1.0, "pure quads expected")
        self.assertEqual(m["nonmanifold_edges"], 0)
        if closed:
            self.assertEqual(m["boundary_edges"], 0, "unexpected holes")
        est = area / (h * h)
        self.assertGreater(m["faces"], 0.6 * est)
        self.assertLess(m["faces"], 1.4 * est)
        self.assertLess(m["max_dev"], 0.25 * h)
        if max_poles is not None:
            self.assertLessEqual(m["v5"] + m["v6plus"], max_poles)
        self.assertEqual(m["v6plus"], 0, "no 6+ valence stars")

    def test_cube_is_perfect_grid(self):
        res, h, a, _, _ = run_core("cube")
        self.check_common(res, h, a)
        m = res.metrics
        self.assertEqual(m["v3"], 8)
        self.assertEqual(m["v5"], 0)
        self.assertLess(m["edge_cv"], 0.05)

    def test_plane_open_boundary(self):
        res, h, a, _, _ = run_core("plane")
        self.check_common(res, h, a, closed=False)
        self.assertEqual(res.metrics["irregular"], 0)

    def test_sphere(self):
        res, h, a, _, _ = run_core("sphere")
        self.check_common(res, h, a, max_poles=8)
        self.assertLessEqual(res.metrics["irregular"], 24)
        self.assertLess(res.metrics["edge_cv"], 0.25)

    def test_torus_is_regular(self):
        res, h, a, _, _ = run_core("torus")
        self.check_common(res, h, a, max_poles=2)
        self.assertLessEqual(res.metrics["irregular"], 4)

    def test_noisy_scan(self):
        res, h, a, _, _ = run_core("noisy_sphere")
        self.check_common(res, h, a, max_poles=6)
        self.assertLessEqual(res.metrics["irregular"], 20)
        self.assertEqual(res.metrics.get("flipped", 0), 0)

    def test_beveled_box(self):
        res, h, a, _, _ = run_core("beveled_box", density=30)
        self.check_common(res, h, a, max_poles=6)
        self.assertLessEqual(res.metrics["irregular"], 16)

    def test_cylinder_follows_rims(self):
        res, h, a, V, F = run_core("cylinder")
        self.check_common(res, h, a, max_poles=4)
        # Every output vertex near a rim should lie on the rim.
        r = np.linalg.norm(res.V[:, :2], axis=1)
        z = np.abs(res.V[:, 2])
        near_rim = (np.abs(z - 1.5) < 1e-3) | (np.abs(r - 1.0) < 1e-3)
        on_both = (np.abs(z - 1.5) < 0.02 * h) & (np.abs(r - 1.0) < 0.02 * h)
        self.assertGreater(on_both.sum(), 40, "edge loops should run along both rims")
        del near_rim

    def test_marked_sharp_edges_followed(self):
        # A sphere with a marked "equator" crease: the output must contain an
        # edge loop on the equator.
        V, F = meshes.sphere(seg=48, rings=24)
        eq = np.nonzero(np.abs(V[:, 2]) < 1e-6)[0]
        ang = np.arctan2(V[eq, 1], V[eq, 0])
        eq = eq[np.argsort(ang)]
        pairs = np.stack([eq, np.roll(eq, -1)], 1)
        a = area_of(V, F)
        h = math.sqrt(a) / 24
        res = pipeline.run(V, F, pipeline.Params(target_edge=h, detect_sharp=False, marked_pairs=pairs))
        self.assertEqual(res.metrics["quad_ratio"], 1.0)
        on_eq = np.abs(res.V[:, 2]) < 1e-4
        self.assertGreater(on_eq.sum(), int(2 * math.pi / h * 0.8))

    def test_guides_steer_flow(self):
        # A tilted great circle drawn on a sphere: edges near the stroke must
        # run along it (or across it), which the unguided field does not do.
        V, F = meshes.sphere(seg=64, rings=32)
        a = area_of(V, F)
        h = math.sqrt(a) / 24
        tilt = math.radians(35)
        s = np.linspace(0, 2 * math.pi, 200)
        circle = np.stack([np.cos(s), np.sin(s) * math.cos(tilt), np.sin(s) * math.sin(tilt)], 1)

        def alignment(res):
            E = set()
            for f in res.faces:
                for k in range(len(f)):
                    E.add(tuple(sorted((f[k], f[(k + 1) % len(f)]))))
            E = np.array(list(E))
            mid = res.V[E].mean(1)
            nrm = np.array([0.0, -math.sin(tilt), math.cos(tilt)])   # circle plane normal
            near = np.abs(mid @ nrm) < 0.6 * h
            d = res.V[E[near, 1]] - res.V[E[near, 0]]
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            along = np.abs(d @ nrm)          # 0 = along the circle, 1 = across it
            return ((along < 0.35) | (along > 0.9)).mean()

        guided = pipeline.run(V, F, pipeline.Params(target_edge=h, guides=[circle], guide_strength=1.0))
        self.assertEqual(guided.metrics["quad_ratio"], 1.0)
        self.assertGreater(alignment(guided), 0.85)

    def test_ring_guide_gets_full_outer_ring(self):
        # The feedback case: a closed guide around an eye gets one more full
        # loop around it (Rings Outside = 1), and both are closed edge loops.
        from quad_remesh.core.spatial import SurfaceProjector
        V, F = meshes.blob_head()
        h = math.sqrt(area_of(V, F)) / 40
        sp = SurfaceProjector(V, F)
        s = np.linspace(0, 2 * math.pi, 240)
        cp, cn = sp.project(np.array([(0.38, -1, 0.3)], float))
        cp, cn = cp[0], cn[0] / np.linalg.norm(cn[0])
        u = np.cross(cn, [0.0, 0.0, 1.0]); u /= np.linalg.norm(u)
        w = np.cross(cn, u)
        eye = sp.project(cp + 0.27 * (np.cos(s)[:, None] * u + np.sin(s)[:, None] * w))[0]
        eye[-1] = eye[0]
        params = pipeline.Params(target_edge=h, guides=[eye], guide_rings_out=1)
        rings = pipeline._with_rings(V, F, params).guides[1:]
        self.assertEqual(len(rings), 1)
        res = pipeline.run(V, F, params)
        self.assertEqual(res.metrics["quad_ratio"], 1.0)
        self.assertEqual(res.metrics.get("holes", 0), 0)
        for g in [eye] + rings:
            cover, cycle = guide_loop(res, g, h)
            self.assertTrue(cycle, "guide and its ring must both be closed edge loops")
            self.assertGreater(cover, 0.85)

    def test_closed_guide_is_filled_with_rings(self):
        # Inside a closed guide: concentric rings with radial spokes around a
        # small grid, so its four corners are the only poles inside the loop.
        from quad_remesh.core.polymesh import ekey
        from quad_remesh.core.spatial import SurfaceProjector
        V, F = meshes.blob_head()
        h = math.sqrt(area_of(V, F)) / 40
        sp = SurfaceProjector(V, F)
        s = np.linspace(0, 2 * math.pi, 240)
        cp, cn = sp.project(np.array([(0.38, -1, 0.3)], float))
        cp, cn = cp[0], cn[0] / np.linalg.norm(cn[0])
        u = np.cross(cn, [0.0, 0.0, 1.0]); u /= np.linalg.norm(u)
        w = np.cross(cn, u)
        eye = sp.project(cp + 0.4 * (np.cos(s)[:, None] * u + 0.8 * np.sin(s)[:, None] * w))[0]
        eye[-1] = eye[0]

        def inside_valences(res):
            # vertices inside the ring (seen along its normal), off the loop itself
            def plane(P):
                return np.stack([(P - cp) @ u, (P - cp) @ w], 1)
            L, P = plane(eye[:-1]), plane(res.V)
            Lq = np.roll(L, 1, axis=0)
            hit = ((L[None, :, 1] > P[:, None, 1]) != (Lq[None, :, 1] > P[:, None, 1])) & (
                P[:, None, 0] < (Lq[:, 0] - L[:, 0])[None] * (P[:, None, 1] - L[None, :, 1])
                / np.where(Lq[:, 1] == L[:, 1], 1e-30, Lq[:, 1] - L[:, 1])[None] + L[None, :, 0])
            gp = SegmentProjector(np.stack([eye[:-1], eye[1:]], 1), h / 8)
            _, d, _ = gp.project(res.V)
            ins = (hit.sum(1) % 2 == 1) & (np.linalg.norm(res.V - cp, axis=1) < 1.0) & (d > 0.3 * h)
            E = {ekey(f[k], f[(k + 1) % len(f)]) for f in res.faces for k in range(len(f))}
            val = np.bincount(np.array(sorted(E)).ravel(), minlength=len(res.V))
            return val[ins]

        res = pipeline.run(V, F, pipeline.Params(target_edge=h, guides=[eye]))
        self.assertEqual(res.metrics["quad_ratio"], 1.0)
        self.assertEqual(res.metrics.get("holes", 0), 0)
        cover, cycle = guide_loop(res, eye, h)
        self.assertTrue(cycle, "the guide must stay a closed edge loop")
        val = inside_valences(res)
        self.assertGreater(len(val), 20)
        self.assertEqual(int((val == 3).sum()), 4, "the core grid's corners are the only poles inside")
        self.assertEqual(int((val != 4).sum()), 4)
        # the old behaviour is still there
        off = pipeline.run(V, F, pipeline.Params(target_edge=h, guides=[eye], guide_fill_inside=False))
        self.assertTrue(guide_loop(off, eye, h)[1])

    def test_guide_rings_become_closed_loops(self):
        # Closed strokes (first point repeated) must come out as one continuous
        # edge loop lying on the stroke: a small ring on a sphere, and rings
        # around the "eye" and "mouth" of a sculpt-like head.
        from quad_remesh.core.spatial import SurfaceProjector
        s = np.linspace(0, 2 * math.pi, 240)

        def ring_on(V, F, c, r, sx=1.0):
            sp = SurfaceProjector(V, F)
            cp, cn = sp.project(np.array([c], float))
            cp, cn = cp[0], cn[0] / np.linalg.norm(cn[0])
            u = np.cross(cn, [0.0, 0.0, 1.0])
            u /= np.linalg.norm(u)
            w = np.cross(cn, u)
            R = sp.project(cp + r * (sx * np.cos(s)[:, None] * u + np.sin(s)[:, None] * w))[0]
            R[-1] = R[0]
            return R

        V, F = meshes.sphere(seg=64, rings=32)
        h = math.sqrt(area_of(V, F)) / 24
        ring = ring_on(V, F, (0.0, -0.34, 0.94), 0.5)
        res = pipeline.run(V, F, pipeline.Params(target_edge=h, guides=[ring]))
        cover, cycle = guide_loop(res, ring, h)
        self.assertGreater(cover, 0.95)
        self.assertTrue(cycle, "ring guide must give a closed edge loop")

        V, F = meshes.blob_head()
        h = math.sqrt(area_of(V, F)) / 40
        eyes = [ring_on(V, F, (x, -1, 0.3), 0.3) for x in (0.35, -0.35)]
        mouth = ring_on(V, F, (0, -1, -0.45), 0.2, 1.8)
        res = pipeline.run(V, F, pipeline.Params(target_edge=h, guides=eyes + [mouth]))
        self.assertEqual(res.metrics["quad_ratio"], 1.0)
        self.assertEqual(res.metrics.get("holes", 0), 0)
        for g in eyes + [mouth]:
            cover, cycle = guide_loop(res, g, h)
            self.assertTrue(cycle, "ring guide must give a closed edge loop")
            self.assertGreater(cover, 0.85)   # a few short jogs across the stroke are allowed

    def test_concentric_rings(self):
        from quad_remesh.core import guides as G
        from quad_remesh.core.spatial import PointIndex, SurfaceProjector
        V, F = meshes.sphere(seg=96, rings=48)
        surf = SurfaceProjector(V, F)
        t = np.linspace(0, 2 * math.pi, 200)
        r0 = 0.5                                     # small circle around the north pole
        ring = np.stack([r0 * np.cos(t), r0 * np.sin(t), np.full_like(t, math.sqrt(1 - r0 ** 2))], 1)
        ring[-1] = ring[0]
        step = 0.15
        geo = lambda P: np.arccos(np.clip(P[:, 2], -1, 1))     # geodesic distance from the pole
        base = math.asin(r0)
        out = G.concentric_rings(ring, surf, 0, step, outside=2, inside=1)
        self.assertEqual(len(out), 3)
        for R in out:
            self.assertTrue(np.allclose(R[0], R[-1]), "rings are closed")
        d = sorted(float(np.mean(geo(R))) - base for R in out)
        self.assertAlmostEqual(d[0], -step, delta=0.25 * step)    # inside
        self.assertAlmostEqual(d[1], step, delta=0.25 * step)
        self.assertAlmostEqual(d[2], 2 * step, delta=0.25 * step)
        for R in out:
            self.assertLess(np.ptp(geo(R)), 0.2 * step, "rings stay concentric")
        # another stroke 1.5 steps outside: the second ring would cross it
        r1 = math.sin(base + 1.5 * step)
        other = np.stack([r1 * np.cos(t), r1 * np.sin(t), np.full_like(t, math.sqrt(1 - r1 ** 2))], 1)
        out = G.concentric_rings(ring, surf, 0, step, outside=2, inside=0, others=PointIndex(other))
        self.assertEqual(len(out), 0, "a ring within 3/4 cell of another stroke is left out")

    def test_rings_of_neighbouring_guides_drop_together(self):
        # Two rings side by side with room for neither guide's outer ring:
        # both are left out (not just the second one drawn).
        V, F = meshes.sphere(seg=96, rings=48)
        h = 0.06
        t = np.linspace(0, 2 * math.pi, 160)

        def cap(axis_x, r=0.3):
            a = np.array([axis_x, 0.0, math.sqrt(1 - axis_x ** 2)])
            u = np.cross(a, [0.0, 1.0, 0.0]); u /= np.linalg.norm(u)
            w = np.cross(a, u)
            P = math.cos(math.asin(r)) * a + r * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * w)
            P[-1] = P[0]
            return P
        left, right = cap(-0.34), cap(0.34)
        params = pipeline.Params(target_edge=h, guides=[left, right], guide_rings_out=1)
        self.assertEqual(len(pipeline._with_rings(V, F, params).guides), 2)
        far = pipeline.Params(target_edge=h, guides=[cap(-0.7), cap(0.7)], guide_rings_out=1)
        self.assertEqual(len(pipeline._with_rings(V, F, far).guides), 4, "with room, both get their ring")

    def test_symmetry_folds_mirrored_guides_once(self):
        from quad_remesh.core.symmetry import fold_guides
        p0, n = np.zeros(3), np.array([1.0, 0.0, 0.0])
        t = np.linspace(0, 2 * math.pi, 101)
        ring = np.stack([0.5 + 0.2 * np.cos(t), np.full_like(t, -1.0), 0.2 * np.sin(t)], 1)
        ring[-1] = ring[0]
        mirrored = ring * np.array([-1.0, 1.0, 1.0])
        out = fold_guides([ring, mirrored], p0, n)
        self.assertEqual(len(out), 1, "a stroke and its mirror image fold onto one guide")
        self.assertTrue(np.allclose(out[0][0], out[0][-1]), "a ring stays closed")
        out = fold_guides([mirrored], p0, n)
        self.assertEqual(len(out), 1)
        self.assertTrue((out[0][:, 0] > 0).all(), "a - side ring is reflected over")
        mouth = np.stack([0.3 * np.cos(t), np.full_like(t, -1.0), -0.5 + 0.1 * np.sin(t)], 1)
        mouth[-1] = mouth[0]
        out = fold_guides([mouth], p0, n)
        self.assertEqual(len(out), 1, "a ring across the plane gives one arc, not a doubled one")
        self.assertTrue((out[0][:, 0] >= -1e-9).all())
        self.assertAlmostEqual(abs(out[0][0, 0]), 0.0, places=6)
        self.assertAlmostEqual(abs(out[0][-1, 0]), 0.0, places=6)
        L = np.linalg.norm(np.diff(out[0], axis=0), axis=1).sum()
        full = np.linalg.norm(np.diff(mouth, axis=0), axis=1).sum()
        self.assertAlmostEqual(L, full / 2, delta=0.02 * full)

    def test_symmetry_with_mirrored_guide_rings(self):
        # Eye rings drawn mirrored (as Draw Guides does with Symmetry on) and a
        # mouth ring across the mirror plane: all closed loops in the result.
        from quad_remesh.core.spatial import SurfaceProjector
        V, F = meshes.blob_head()
        h = math.sqrt(area_of(V, F)) / 40
        sp = SurfaceProjector(V, F)
        s = np.linspace(0, 2 * math.pi, 240)

        def ring_on(c, r, sx=1.0):
            cp, cn = sp.project(np.array([c], float))
            cp, cn = cp[0], cn[0] / np.linalg.norm(cn[0])
            u = np.cross(cn, [0.0, 0.0, 1.0])
            u /= np.linalg.norm(u)
            w = np.cross(cn, u)
            R = sp.project(cp + r * (sx * np.cos(s)[:, None] * u + np.sin(s)[:, None] * w))[0]
            R[-1] = R[0]
            return R

        eye = ring_on((0.38, -1, 0.3), 0.27)
        eye_m = eye * np.array([-1.0, 1.0, 1.0])
        mouth = ring_on((0, -1, -0.45), 0.2, 1.8)
        res = pipeline.run(V, F, pipeline.Params(target_edge=h, guides=[eye, eye_m, mouth],
                                                 symmetry_plane=((0, 0, 0), (1, 0, 0))))
        self.assertEqual(res.metrics["quad_ratio"], 1.0)
        self.assertEqual(res.metrics["boundary_edges"], 0)
        for g in (eye, eye_m, mouth):
            cover, cycle = guide_loop(res, g, h)
            self.assertTrue(cycle, "ring guide must give a closed edge loop")
            self.assertGreater(cover, 0.85)
        self._check_symmetric(res, 0)

    def _check_symmetric(self, res, axis):
        from mathutils.kdtree import KDTree
        kd = KDTree(len(res.V))
        for i, p in enumerate(res.V):
            kd.insert(p, i)
        kd.balance()
        flip = np.ones(3)
        flip[axis] = -1.0
        worst = max(kd.find(p * flip)[2] for p in res.V)
        self.assertLess(worst, 1e-6, "result must be exactly mirror symmetric")

    def test_symmetry_x(self):
        V, F = meshes.smooth_monkey()
        a = area_of(V, F)
        h = math.sqrt(a) / 30
        res = pipeline.run(V, F, pipeline.Params(target_edge=h, symmetry_plane=((0, 0, 0), (1, 0, 0))))
        self.assertEqual(res.metrics["quad_ratio"], 1.0)
        self.assertEqual(res.metrics["nonmanifold_edges"], 0)
        self.assertEqual(res.metrics["v6plus"], 0)
        self._check_symmetric(res, 0)

    def test_symmetry_closed_mesh_is_watertight(self):
        for name, axis in (("sphere", 2), ("beveled_box", 1), ("torus", 0)):
            res, h, a, _, _ = run_core(name, density=30, symmetry_plane=((0, 0, 0), tuple(np.eye(3)[axis])))
            self.assertEqual(res.metrics["quad_ratio"], 1.0, name)
            self.assertEqual(res.metrics["boundary_edges"], 0, name + " seam must be welded")
            self.assertEqual(res.metrics["nonmanifold_edges"], 0, name)
            self._check_symmetric(res, axis)

    def test_quad_dominant_mode(self):
        res, h, a, _, _ = run_core("sphere", pure_quads=False)
        self.assertGreater(res.metrics["quad_ratio"], 0.95)
        self.assertEqual(res.metrics["ngons"], 0)


def run_ui_defaults(name, density=20.0, **kw):
    """Run like the add-on does by default: input prep, thin-part refinement
    and Detail 0.5."""
    from quad_remesh import prep
    V, F = getattr(meshes, name)()
    a = area_of(V, F)
    h = math.sqrt(a) / density
    params = dict(prepare_fn=prep.prepare, adapt_thin=True, detail=0.5)
    params.update(kw)
    return pipeline.run(V, F, pipeline.Params(target_edge=h, **params)), h


class CadRobustness(unittest.TestCase):
    """Tessellated-CAD situations: thin shells, dense ridges, sliver triangles."""

    def check_clean(self, m, max_flips, max_stars=0):
        self.assertEqual(m["quad_ratio"], 1.0)
        self.assertEqual(m.get("holes", 0), 0, "no holes")
        self.assertEqual(m["nonmanifold_edges"], 0)
        self.assertLessEqual(m["v6plus"], max_stars, "no stars")
        self.assertLessEqual(m.get("flipped", 0), max_flips)

    def test_thin_shell_is_refined_as_a_whole(self):
        res, h = run_ui_defaults("thin_cap")
        # one star may remain where four creases meet (rim corners)
        self.check_clean(res.metrics, max_flips=10, max_stars=1)
        self.assertLess(res.metrics["edge_cv"], 0.2, "one even size across the thin part")

    def test_knurling_finer_than_quads(self):
        res, h = run_ui_defaults("knurled_cap")
        self.check_clean(res.metrics, max_flips=3)

    def test_sliver_triangulation(self):
        res, h = run_ui_defaults("sliver_box")
        self.check_clean(res.metrics, max_flips=6)

    def test_organic_mesh_has_no_holes(self):
        # Dense subdivided Suzanne with every option on: the extraction used to
        # leave dozens of open edges (some far from the eye openings).
        res, h = run_ui_defaults("big_monkey", density=50)
        self.assertEqual(res.metrics["quad_ratio"], 1.0)
        self.assertEqual(res.metrics.get("interior_holes", 0), 0, "no holes away from real openings")
        self.assertLessEqual(res.metrics.get("holes", 0), 2, "at most a sliver along an opening")
        self.assertEqual(res.metrics["nonmanifold_edges"], 0)

    @unittest.skipUnless(os.path.exists(os.path.join(ROOT, "test_geo", "Diageo_750ml_Gilbeys_Bottle_v001.fbx")),
                         "CAD sample not present")
    def test_bottle_fbx_parts(self):
        import subprocess
        out = subprocess.run(
            [bpy.app.binary_path, "-b", "--factory-startup", "--python", os.path.join(ROOT, "tests", "cad_compare.py"),
             "--", os.path.join(ROOT, "test_geo", "Diageo_750ml_Gilbeys_Bottle_v001.fbx"), "20",
             "detail=0.5", "adapt_thin=True"], capture_output=True, text=True, timeout=1800).stdout
        rows = [ln for ln in out.splitlines() if " tris " in ln]
        self.assertEqual(len(rows), 6, out)
        for ln in rows:
            holes = int(ln.split("holes")[1].split()[0])
            nm = int(ln.split(" nm ")[1].split()[0])
            self.assertEqual(nm, 0, ln)
            self.assertLessEqual(holes, 10, ln)


class AddonIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        quad_remesh.register()

    @classmethod
    def tearDownClass(cls):
        quad_remesh.unregister()

    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def _make_cube(self):
        me = bpy.data.meshes.new("Cube")
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=2.0)
        bm.to_mesh(me)
        bm.free()
        ob = bpy.data.objects.new("Cube", me)
        bpy.context.scene.collection.objects.link(ob)
        bpy.context.view_layer.objects.active = ob
        ob.select_set(True)
        return ob

    def test_operator_creates_quad_object(self):
        ob = self._make_cube()
        ob.scale = (2.0, 1.0, 1.0)
        bpy.context.view_layer.update()
        ob.quad_remesh.density = 16
        r = bpy.ops.quad_remesh.remesh('EXEC_DEFAULT')
        self.assertEqual(r, {'FINISHED'})
        res = bpy.data.objects.get("Cube_Quad")
        self.assertIsNotNone(res)
        sizes = {len(p.vertices) for p in res.data.polygons}
        self.assertEqual(sizes, {4})
        self.assertTrue(ob.hide_get())
        # World-space size preserved.
        bb = np.array([res.matrix_world @ v.co for v in res.data.vertices])
        self.assertAlmostEqual(bb[:, 0].max() - bb[:, 0].min(), 4.0, places=2)
        # Running again replaces the mesh instead of creating a new object.
        bpy.context.view_layer.objects.active = ob
        ob.hide_set(False)
        r = bpy.ops.quad_remesh.remesh('EXEC_DEFAULT')
        self.assertEqual(len([o for o in bpy.data.objects if o.get("qr_source")]), 1)

    def test_marked_sharp_attribute_is_read(self):
        ob = self._make_cube()
        st = ob.quad_remesh
        st.detect_sharp = False
        st.follow_marked_sharp = True
        attr = ob.data.attributes.new("sharp_edge", 'BOOLEAN', 'EDGE')
        attr.data.foreach_set("value", [True] * len(ob.data.edges))
        r = bpy.ops.quad_remesh.remesh('EXEC_DEFAULT')
        self.assertEqual(r, {'FINISHED'})
        res = bpy.data.objects.get("Cube_Quad")
        sharp = res.data.attributes.get("sharp_edge")
        self.assertIsNotNone(sharp, "feature edges should be marked sharp on the result")
        vals = [d.value for d in sharp.data]
        self.assertGreater(sum(vals), 12)

    def test_symmetry_operator_on_transformed_object(self):
        me = bpy.data.meshes.new("Monkey")
        bm = bmesh.new()
        bmesh.ops.create_monkey(bm)
        bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=1, use_grid_fill=True)
        bm.to_mesh(me)
        bm.free()
        ob = bpy.data.objects.new("Monkey", me)
        bpy.context.scene.collection.objects.link(ob)
        bpy.context.view_layer.objects.active = ob
        ob.location = (1.0, 2.0, 0.5)
        ob.rotation_euler = (0.3, 0.2, 0.7)
        ob.scale = (1.5, 1.0, 0.8)
        bpy.context.view_layer.update()
        st = ob.quad_remesh
        st.use_symmetry = True
        st.symmetry_axis = 'X'
        st.density = 25
        r = bpy.ops.quad_remesh.remesh('EXEC_DEFAULT')
        self.assertEqual(r, {'FINISHED'})
        res = bpy.data.objects.get("Monkey_Quad")
        self.assertEqual({len(p.vertices) for p in res.data.polygons}, {4})
        co = np.array([v.co[:] for v in res.data.vertices])     # local space
        from mathutils.kdtree import KDTree
        kd = KDTree(len(co))
        for i, p in enumerate(co):
            kd.insert(p, i)
        kd.balance()
        worst = max(kd.find((-p[0], p[1], p[2]))[2] for p in co)
        self.assertLess(worst, 1e-4)

    def test_mirror_drawing_helpers(self):
        from mathutils import Vector
        from quad_remesh.guides import crosses_plane, mirror_point
        p0, n = Vector((0, 0, 0)), Vector((1, 0, 0))
        self.assertTrue(np.allclose(tuple(mirror_point(Vector((0.4, 1, 2)), p0, n)), (-0.4, 1.0, 2.0)))
        one_side = [Vector((x, 0, 0)) for x in (0.2, 0.3, 0.4)]
        across = [Vector((x, 0, 0)) for x in (-0.2, 0.0, 0.3)]
        self.assertFalse(crosses_plane(one_side, p0, n), "one-sided strokes get a mirrored copy")
        self.assertTrue(crosses_plane(across, p0, n), "strokes across the plane are kept as drawn")

    def test_strokes_join_at_mirror_line(self):
        from mathutils import Vector
        from quad_remesh.guides import join_at_plane
        p0, n, snap = Vector((0, 0, 0)), Vector((1, 0, 0)), 0.1
        t = np.linspace(-math.pi / 2, math.pi / 2, 40)

        def is_mirror_symmetric(P):
            M = P * np.array([-1.0, 1.0, 1.0])
            d = np.linalg.norm(M[:, None] - P[None], axis=2).min(1)
            return d.max() < 1e-6

        # half a mouth ring from just short of the line, around, and past it
        half = [Vector((0.3 * math.cos(a) + 0.03, 0.0, 0.2 * math.sin(a) * 1.1)) for a in t]
        half[0] = Vector((0.05, 0.0, -0.2))
        half[-1] = Vector((-0.04, 0.0, 0.2))
        pts, ring = join_at_plane(half, False, p0, n, snap)
        P = np.array([p[:] for p in pts])
        self.assertTrue(ring, "both ends at the line: one closed ring")
        self.assertTrue(is_mirror_symmetric(P))
        self.assertGreater(P[:, 0].min(), -0.4)
        # a stroke from the side to the line: one open stroke through it
        brow = [Vector((x, 0.0, 0.5 + 0.1 * x)) for x in np.linspace(0.6, 0.05, 20)]
        pts, ring = join_at_plane(brow, False, p0, n, snap)
        P = np.array([p[:] for p in pts])
        self.assertFalse(ring)
        self.assertTrue(is_mirror_symmetric(P))
        self.assertAlmostEqual(P[:, 0].max(), 0.6, places=6)
        self.assertAlmostEqual(P[:, 0].min(), -0.6, places=6)
        # a ring drawn all the way across (closed): symmetric ring of its larger half
        a = np.linspace(0, 2 * math.pi, 60, endpoint=False)
        full = [Vector((0.05 + 0.3 * math.cos(x), 0.0, 0.2 * math.sin(x))) for x in a]
        pts, ring = join_at_plane(full, True, p0, n, snap)
        self.assertTrue(ring)
        self.assertTrue(is_mirror_symmetric(np.array([p[:] for p in pts])))
        # far from the line, or along it: not joined
        side = [Vector((x, 0.0, 0.0)) for x in np.linspace(0.5, 0.9, 10)]
        self.assertIsNone(join_at_plane(side, False, p0, n, snap))
        centre = [Vector((0.02, 0.0, z)) for z in np.linspace(-0.5, 0.5, 10)]
        self.assertIsNone(join_at_plane(centre, False, p0, n, snap))

    def test_strokes_snap_to_open_strokes(self):
        from mathutils import Vector
        from quad_remesh.guides import smooth_junctions, snap_to_strokes

        def line(a, b, k=11):
            return [Vector(a).lerp(Vector(b), t) for t in np.linspace(0, 1, k)]

        A = line((0, 0, 0), (1, 0, 0))
        B = line((2, 1, 0), (2, 2, 0))
        # starts near A's end: A continues into the new stroke
        new = line((1.05, 0.02, 0), (1.9, 0.9, 0))
        pts, closed, used, junctions = snap_to_strokes(new, [("A", A)], 0.1)
        self.assertEqual(used, ["A"])
        self.assertFalse(closed)
        self.assertEqual(len(pts), len(A) + len(new) - 1)
        self.assertEqual(tuple(pts[0]), tuple(A[0]))
        self.assertEqual(tuple(pts[len(A) - 1]), tuple(A[-1]), "the new stroke starts exactly at A's end")
        self.assertEqual(junctions, [len(A) - 1])
        # ...and ends near B's start: A + new + B in one stroke
        pts, closed, used, junctions = snap_to_strokes(new + [Vector((2.02, 0.97, 0))],
                                                       [("A", A), ("B", B)], 0.1)
        self.assertEqual(sorted(used), ["A", "B"])
        self.assertEqual(tuple(pts[-1]), tuple(B[-1]))
        self.assertEqual(len(junctions), 2)
        # both ends on the two ends of one stroke: a ring
        U = line((0, 0, 0), (0, 1, 0))
        arc = [Vector((0.03, 1.02, 0)), Vector((0.5, 1.2, 0)), Vector((0.8, 0.5, 0)),
               Vector((0.5, -0.2, 0)), Vector((0.02, -0.03, 0))]
        pts, closed, used, junctions = snap_to_strokes(arc, [("U", U)], 0.1)
        self.assertTrue(closed)
        self.assertEqual(used, ["U"])
        self.assertEqual(len(pts), len(arc) + len(U) - 2)
        # nothing near: unchanged
        far = line((5, 5, 0), (6, 5, 0))
        pts, closed, used, junctions = snap_to_strokes(far, [("A", A)], 0.1)
        self.assertEqual((used, closed, len(pts)), ([], False, len(far)))
        # junction smoothing leaves points far from the join alone
        P = [Vector((x, 0.0, 0.0)) for x in np.linspace(0, 1, 21)] + \
            [Vector((1.0, y, 0.0)) for y in np.linspace(0.05, 1, 20)]
        out = smooth_junctions(P, False, [20], 1.5)
        self.assertGreater((out[20] - P[20]).length, 0.01, "the corner is rounded")
        self.assertLess((out[2] - P[2]).length, 1e-6)
        self.assertLess((out[-3] - P[-3]).length, 1e-6)

    def test_draw_tool_joins_strokes_and_mirrors(self):
        # The draw tool's release plan, on real guide splines: continuing an
        # open stroke merges into it (and, with symmetry, the mirrored copy
        # into the mirrored stroke); Ctrl keeps the stroke as drawn.
        import types
        from mathutils import Vector
        from quad_remesh import guides as GD
        ob = self._make_cube()
        g = GD.guide_object_for(bpy.context, ob, create=True)
        GD._add_spline(g, [Vector((x, 0.5, 1.0)) for x in np.linspace(0.4, 1.0, 13)])
        GD._add_spline(g, [Vector((-x, 0.5, 1.0)) for x in np.linspace(0.4, 1.0, 13)])
        Op = GD.QR_OT_draw_guides

        def tool(stroke, free=False, plane=True):
            f = types.SimpleNamespace(_stroke=stroke, _close_dist=0.1, _spacing=0.05, _smooth=0.5,
                                      _obj=ob, _free=free, _on_surface=lambda p: p,
                                      _plane=(Vector((0, 0, 0)), Vector((1, 0, 0))) if plane else None)
            for m in ("_closes", "_final", "_mirrored", "_open_strokes"):
                setattr(f, m, (lambda name: (lambda *a: getattr(Op, name)(f, *a)))(m))
            return f

        new = [Vector((1.03, 0.52, 1.0))] + [Vector((1.0 + 0.04 * k, 0.5 - 0.03 * k, 1.0)) for k in range(1, 12)]
        strokes, replaced = Op._plan(tool(new), bpy.context)
        self.assertEqual(len(strokes), 2)
        self.assertEqual(len(replaced), 2, "both the stroke and its mirror partner are merged")
        xs = sorted((min(q.x for q in p), max(q.x for q in p)) for p, _c, _a in strokes)
        self.assertAlmostEqual(xs[0][1], -0.4, places=5)
        self.assertAlmostEqual(xs[1][0], 0.4, places=5)
        strokes, replaced = Op._plan(tool(new, free=True), bpy.context)
        self.assertEqual((len(strokes), len(replaced)), (2, 0), "Ctrl: no snapping")
        strokes, replaced = Op._plan(tool(new, plane=False), bpy.context)
        self.assertEqual((len(strokes), len(replaced)), (1, 1))

    def test_stroke_smoothing(self):
        from mathutils import Vector
        from quad_remesh.guides import smooth_stroke
        rng = np.random.default_rng(1)
        t = np.linspace(0, 2 * math.pi, 160, endpoint=False)
        # hand wobble: wiggles ~2 quads long plus mouse jitter (points ~1/4 quad apart)
        r = 1.0 + 0.03 * np.sin(20 * t) + 0.01 * rng.standard_normal(len(t))
        ring = [Vector((ri * math.cos(a), ri * math.sin(a), 0.0)) for ri, a in zip(r, t)]
        out = np.array([p[:] for p in smooth_stroke(ring, True, 0.5, 0.04)])
        rad = np.linalg.norm(out[:, :2], axis=1)
        self.assertLess(rad.std(), 0.25 * r.std(), "wobble is smoothed out")
        self.assertAlmostEqual(rad.mean(), 1.0, delta=0.02)   # rings keep their size
        line = [Vector((x, 0.03 * rng.standard_normal(), 0.0)) for x in np.linspace(0, 1, 50)]
        out = smooth_stroke(line, False, 0.5, 0.02)
        self.assertTrue(np.allclose(out[0][:], line[0][:]) and np.allclose(out[-1][:], line[-1][:]),
                        "open strokes keep their end points")
        same = smooth_stroke(line, False, 0.0, 0.02)
        self.assertEqual(len(same), len(line), "0 keeps the stroke as drawn")

    def test_guide_curve_is_used(self):
        ob = self._make_cube()
        cu = bpy.data.curves.new("g", 'CURVE')
        cu.dimensions = '3D'
        spl = cu.splines.new('POLY')
        spl.points.add(1)
        spl.points[0].co = (-1.0, -1.0, 1.0, 1.0)
        spl.points[1].co = (1.0, 1.0, 1.0, 1.0)
        g = bpy.data.objects.new("g", cu)
        bpy.context.scene.collection.objects.link(g)
        ob.quad_remesh.guide_object = g
        from quad_remesh import bridge
        polys = bridge.guide_polylines(g)
        self.assertEqual(len(polys), 1)
        self.assertEqual(len(polys[0]), 2)
        spl.use_cyclic_u = True
        ring = bridge.guide_polylines(g)[0]
        self.assertEqual(len(ring), 3, "a cyclic spline repeats its first point")
        self.assertTrue(np.allclose(ring[0], ring[-1]))
        r = bpy.ops.quad_remesh.remesh('EXEC_DEFAULT')
        self.assertEqual(r, {'FINISHED'})


if __name__ == "__main__":
    argv = [sys.argv[0]] + (sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    result = unittest.main(argv=argv, exit=False, verbosity=2).result
    sys.exit(0 if result.wasSuccessful() else 1)
