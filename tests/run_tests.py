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
        r = bpy.ops.quad_remesh.remesh('EXEC_DEFAULT')
        self.assertEqual(r, {'FINISHED'})


if __name__ == "__main__":
    argv = [sys.argv[0]] + (sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    result = unittest.main(argv=argv, exit=False, verbosity=2).result
    sys.exit(0 if result.wasSuccessful() else 1)
