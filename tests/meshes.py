# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

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


def smooth_monkey(levels=2):
    me = bpy.data.meshes.new("sm")
    bm = bmesh.new()
    bmesh.ops.create_monkey(bm)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new("sm", me)
    bpy.context.scene.collection.objects.link(ob)
    mod = ob.modifiers.new("s", 'SUBSURF')
    mod.levels = levels
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    m2 = ev.to_mesh()
    bm = bmesh.new()
    bm.from_mesh(m2)
    ev.to_mesh_clear()
    bpy.data.objects.remove(ob)
    bm.verts.index_update()
    return _from_bm(bm)


def big_monkey():
    return smooth_monkey(levels=3)


def _evaluated(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    m2 = ev.to_mesh()
    bm = bmesh.new()
    bm.from_mesh(m2)
    ev.to_mesh_clear()
    me = ob.data
    bpy.data.objects.remove(ob)
    bpy.data.meshes.remove(me)
    bm.verts.index_update()
    return _from_bm(bm)


def thin_cap(radius=1.0, height=0.6, wall=0.04, seg=96):
    """Bottle-cap shell: open at the bottom, wall about 0.2 of a quad at density 20."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=seg, radius1=radius,
                          radius2=radius, depth=height)
    bottom = min(bm.faces, key=lambda f: f.calc_center_median().z)
    bmesh.ops.delete(bm, geom=[bottom], context='FACES_ONLY')
    me = bpy.data.meshes.new("cap")
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new("cap", me)
    bpy.context.scene.collection.objects.link(ob)
    mod = ob.modifiers.new("s", 'SOLIDIFY')
    mod.thickness = wall
    return _evaluated(ob)


def knurled_cap(radius=1.0, height=0.6, ridges=120, depth=0.03, rings=12):
    """Closed cylinder with sharp knurling ridges far finer than a quad."""
    import math
    bm = bmesh.new()
    n = ridges * 2
    rows = []
    for j in range(rings + 1):
        z = -height / 2 + height * j / rings
        row = []
        for i in range(n):
            a = 2 * math.pi * i / n
            r = radius + (depth if i % 2 else 0.0)
            row.append(bm.verts.new((r * math.cos(a), r * math.sin(a), z)))
        rows.append(row)
    for j in range(rings):
        for i in range(n):
            bm.faces.new((rows[j][i], rows[j][(i + 1) % n], rows[j + 1][(i + 1) % n], rows[j + 1][i]))
    bm.faces.new(list(reversed(rows[0])))
    bm.faces.new(rows[-1])
    bm.normal_update()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.verts.index_update()
    return _from_bm(bm)


def sliver_box(size=2.0, cuts=24):
    """Cube whose faces are fan-triangulated into long needle triangles, like
    a CAD export."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=size)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=cuts, use_grid_fill=False)
    bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method='FIXED', ngon_method='EAR_CLIP')
    bm.verts.index_update()
    return _from_bm(bm)


def blob_head(subdivisions=6):
    """Dense, closed, sculpt-like head: an egg with eye bulges in sockets, a
    nose and a mouth dip (for guide rings around facial features)."""
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdivisions, radius=1.0)
    V, F = _from_bm(bm)
    n = V / np.linalg.norm(V, axis=1, keepdims=True)
    V = V * np.array([0.85, 1.0, 1.15])

    def bump(c, r, a):
        c = np.array(c, dtype=np.float64) / np.linalg.norm(c)
        return a * np.exp(-(np.linalg.norm(n - c, axis=1) / r) ** 2)

    disp = (bump((0.35, -1, 0.25), 0.18, 0.08) + bump((-0.35, -1, 0.25), 0.18, 0.08)
            - bump((0.35, -1, 0.3), 0.32, 0.06) - bump((-0.35, -1, 0.3), 0.32, 0.06)
            + bump((0, -1, -0.05), 0.12, 0.12) - bump((0, -1, -0.45), 0.2, 0.05))
    return V + n * disp[:, None], F
