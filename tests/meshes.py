"""Test mesh factory (runs inside Blender)."""
import bmesh
import bpy
import numpy as np


def _from_bm(bm):
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    V = np.array([v.co[:] for v in bm.verts])
    F = np.array([[v.index for v in f.verts] for f in bm.faces])
    bm.free()
    return V, F


def cube(size=2.0):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=size)
    bm.verts.index_update()
    return _from_bm(bm)


def sphere(r=1.0, seg=48, rings=24):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=rings, radius=r)
    bm.verts.index_update()
    return _from_bm(bm)


def cylinder(r=1.0, depth=3.0, seg=48):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=seg, radius1=r, radius2=r, depth=depth)
    bm.verts.index_update()
    return _from_bm(bm)


def monkey():
    bm = bmesh.new()
    bmesh.ops.create_monkey(bm)
    bm.verts.index_update()
    return _from_bm(bm)


def torus():
    me = bpy.data.meshes.new("t")
    bm = bmesh.new()
    import math
    R, r, nu, nv = 1.0, 0.35, 64, 24
    verts = []
    for i in range(nu):
        for j in range(nv):
            u, v = 2 * math.pi * i / nu, 2 * math.pi * j / nv
            verts.append(bm.verts.new(((R + r * math.cos(v)) * math.cos(u), (R + r * math.cos(v)) * math.sin(u), r * math.sin(v))))
    for i in range(nu):
        for j in range(nv):
            a = verts[i * nv + j]; b = verts[((i + 1) % nu) * nv + j]
            c = verts[((i + 1) % nu) * nv + (j + 1) % nv]; d = verts[i * nv + (j + 1) % nv]
            bm.faces.new((a, b, c, d))
    bm.verts.index_update()
    return _from_bm(bm)


def plane(size=2.0, sub=8):
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=sub, y_segments=sub, size=size / 2)
    bm.verts.index_update()
    return _from_bm(bm)


def noisy_sphere(n_sub=6, amp=0.01, seed=0):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=n_sub, radius=1.0)
    bm.verts.index_update()
    V, F = _from_bm(bm)
    rng = np.random.default_rng(seed)
    V = V * (1 + amp * rng.standard_normal(len(V)))[:, None]
    return V, F


def beveled_box():
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    bmesh.ops.bevel(bm, geom=bm.edges[:] + bm.verts[:], offset=0.2, segments=3, affect='EDGES', profile=0.5)
    bm.verts.index_update()
    return _from_bm(bm)
