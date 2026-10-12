# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

"""Input preparation with Blender's C tools (needs bpy / bmesh).

Tessellated CAD models are often far denser than the remesh needs (a million
triangles for a few hundred quads). `prepare` builds the *field mesh* the
solvers run on:

1. Collapse-decimate to about the triangle count the target edge length needs.
2. Make it even (isotropic): Blender's decimator leaves long needle triangles
   on smooth surfaces (edges spanning a whole bottle), so long edges are split,
   short ones collapsed and edges in nearly flat areas flipped, all in C.
   Vertices are never smoothed or re-projected: on thin double-walled parts
   that snapped them onto the opposite wall.
3. Repair: drop loose and non-manifold leftovers and fill tiny holes (real
   openings, which are long loops, stay open).

Inputs that are not much denser than needed are used as they are (re-meshing
a CAD tessellation that is already at the right density only made thin shells
worse). The original mesh stays the reference for projection and features.
"""

import math

import bmesh
import bpy
import numpy as np

# Edges bending less than this may be flipped to improve triangle shape.
# Well below the sharp-edge angle, so creases are never touched.
FLIP_ANGLE = math.radians(15.0)

def _mesh_from_arrays(V, F, name="_qr_prep"):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", np.asarray(V, dtype=np.float32).ravel())
    nf = len(F)
    me.loops.add(nf * 3)
    me.loops.foreach_set("vertex_index", np.asarray(F, dtype=np.int32).ravel())
    me.polygons.add(nf)
    me.polygons.foreach_set("loop_start", np.arange(0, nf * 3, 3, dtype=np.int32))
    me.update(calc_edges=True)
    return me


def _decimate(me, ratio):
    """Collapse-decimate a mesh datablock via a temporary evaluated object."""
    scene = bpy.context.scene
    ob = bpy.data.objects.new("_qr_prep", me)
    scene.collection.objects.link(ob)
    try:
        mod = ob.modifiers.new("decimate", 'DECIMATE')
        mod.decimate_type = 'COLLAPSE'
        mod.ratio = float(ratio)
        mod.use_collapse_triangulate = True
        depsgraph = bpy.context.evaluated_depsgraph_get()
        ob_eval = ob.evaluated_get(depsgraph)
        tmp = ob_eval.to_mesh()
        bm = bmesh.new()
        bm.from_mesh(tmp)
        ob_eval.to_mesh_clear()
    finally:
        bpy.data.objects.remove(ob)
    return bm


def _sharp(e):
    return len(e.link_faces) != 2 or e.calc_face_angle(math.pi) >= FLIP_ANGLE


def _collapse_short(bm, edge, short=0.8, long=1.33):
    """Safe edge collapses (isotropic-remeshing rules): never on boundaries
    or creases, only when the link condition holds (no pinching / non-manifold
    result), no surrounding face flips and no new edge gets too long."""
    lo, hi = short * edge, long * edge
    cand = sorted((e for e in bm.edges if e.calc_length() < lo), key=lambda e: e.calc_length())
    # Collapses whose neighbourhoods do not touch are applied together in one
    # weld (one bmesh operator call per batch: per-edge calls cost O(mesh)).
    locked = set()
    targetmap = {}
    for e in cand:
        v1, v2 = e.verts
        if v1 in locked or v2 in locked:
            continue
        if any(_sharp(x) for x in v1.link_edges) or any(_sharp(x) for x in v2.link_edges):
            continue
        n1 = {x.other_vert(v1) for x in v1.link_edges}
        n2 = {x.other_vert(v2) for x in v2.link_edges}
        if len(n1 & n2) != 2:
            continue                                   # link condition
        mid = (v1.co + v2.co) * 0.5
        ring = (n1 | n2) - {v1, v2}
        if any(w in locked for w in ring):
            continue
        if any((w.co - mid).length > hi for w in ring):
            continue
        ok = True
        for v in (v1, v2):
            for f in v.link_faces:
                if v1 in f.verts and v2 in f.verts:
                    continue
                pts = [mid if (x is v1 or x is v2) else x.co for x in f.verts]
                n_new = (pts[1] - pts[0]).cross(pts[2] - pts[0])
                if n_new.length < 1e-20 or n_new.normalized().dot(f.normal) < 0.5:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        v2.co = mid
        targetmap[v1] = v2
        locked.update(ring)
        locked.update((v1, v2))
    if targetmap:
        bmesh.ops.weld_verts(bm, targetmap=targetmap)
    return len(targetmap)


def _make_even(bm, edge, max_tris, iterations=10):
    """Split long / collapse short / flip, until edges are about `edge` long.
    Returns False when the triangle count runs past `max_tris` (the decimated
    mesh was too uneven to even out in reasonable time)."""
    for _ in range(iterations):
        if len(bm.faces) > max_tris:
            return False
        long = [e for e in bm.edges if e.calc_length() > 1.33 * edge]
        if long:
            bmesh.ops.subdivide_edges(bm, edges=long, cuts=1, use_grid_fill=False)
            bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
        collapsed = _collapse_short(bm, edge)
        bm.normal_update()
        flat = [e for e in bm.edges if not _sharp(e)]
        if flat:
            bmesh.ops.beautify_fill(bm, faces=bm.faces[:], edges=flat)
        bm.normal_update()
        if len(long) < 0.005 * max(len(bm.edges), 1) and collapsed < 0.005 * max(len(bm.edges), 1):
            break
    return len(bm.faces) <= max_tris


def _boundary_loops(bm):
    nxt = {}
    for e in bm.edges:
        if len(e.link_faces) == 1:
            a, b = e.verts
            nxt.setdefault(a, []).append(e)
            nxt.setdefault(b, []).append(e)
    seen = set()
    loops = []
    for e0 in [e for e in bm.edges if len(e.link_faces) == 1]:
        if e0 in seen:
            continue
        loop, stack = [], [e0]
        while stack:
            e = stack.pop()
            if e in seen:
                continue
            seen.add(e)
            loop.append(e)
            for v in e.verts:
                stack.extend(x for x in nxt[v] if x not in seen)
        loops.append(loop)
    return loops


def _repair(bm, real_boundary, tol):
    """Remove what the edge collapses can leave behind, and close every hole
    that is not a real opening of the original mesh."""
    loose_e = [e for e in bm.edges if not e.link_faces]
    if loose_e:
        bmesh.ops.delete(bm, geom=loose_e, context='EDGES')
    bad = {f for e in bm.edges if len(e.link_faces) > 2 for f in e.link_faces[2:]}
    if bad:
        bmesh.ops.delete(bm, geom=list(bad), context='FACES_ONLY')
    loose_v = [v for v in bm.verts if not v.link_faces]
    if loose_v:
        bmesh.ops.delete(bm, geom=loose_v, context='VERTS')
    for loop in _boundary_loops(bm):
        verts = {v for e in loop for v in e.verts}
        if real_boundary is not None:
            near = sum(1 for v in verts if real_boundary.find(v.co)[2] <= tol)
            if near >= 0.5 * len(verts):
                continue          # a real opening: keep it open
        bmesh.ops.holes_fill(bm, edges=loop, sides=len(loop) + 1)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])


def _real_boundary(me):
    """KD-tree of the original mesh's open-edge vertices (None if closed)."""
    from mathutils.kdtree import KDTree
    E = np.empty(len(me.edges) * 2, dtype=np.int64)
    me.edges.foreach_get("vertices", E)
    E = E.reshape(-1, 2)
    loops_e = np.empty(len(me.loops), dtype=np.int64)
    me.loops.foreach_get("edge_index", loops_e)
    use = np.bincount(loops_e, minlength=len(E))
    bv = np.unique(E[use == 1].ravel())
    if not len(bv):
        return None
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    kd = KDTree(len(bv))
    for i, v in enumerate(bv):
        kd.insert(co[v], i)
    kd.balance()
    return kd


def _voxel(V, F, edge):
    """Even, watertight field mesh of a closed surface via Blender's voxel
    remesher (OpenVDB). Rounds sharp edges slightly, which is fine for the field
    mesh: features and projection still come from the original."""
    me = _mesh_from_arrays(V, F)
    ob = bpy.data.objects.new("_qr_voxel", me)
    bpy.context.scene.collection.objects.link(ob)
    try:
        mod = ob.modifiers.new("remesh", 'REMESH')
        mod.mode = 'VOXEL'
        mod.voxel_size = float(edge)
        mod.adaptivity = 0.0
        ob.modifiers.new("tri", 'TRIANGULATE')
        ob_eval = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
        tmp = ob_eval.to_mesh()
        bm = bmesh.new()
        bm.from_mesh(tmp)
        ob_eval.to_mesh_clear()
    finally:
        bpy.data.objects.remove(ob)
        bpy.data.meshes.remove(me)
    try:
        return _arrays(bm)
    finally:
        bm.free()


def _arrays(bm):
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    V = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
    F = np.array([[v.index for v in f.verts] for f in bm.faces if len(f.verts) == 3],
                 dtype=np.int64).reshape(-1, 3)
    return V, F


def prepare(V, F, edge_target):
    """Return (V, F) of a field mesh with edges about `edge_target` long (the
    input itself when it is not much denser than that)."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    if not len(F):
        return V, F
    t = V[F]
    area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum()
    # Equilateral triangles with edge e cover sqrt(3)/4 e^2 each.
    target_tris = max(200, int(area / (0.433 * edge_target * edge_target)))
    if len(F) <= 2.0 * target_tris:
        return V, F
    me = _mesh_from_arrays(V, F)
    try:
        real = _real_boundary(me)
        bm = _decimate(me, target_tris / len(F))
    finally:
        bpy.data.meshes.remove(me)
    try:
        if not _make_even(bm, edge_target, 3 * target_tris) and real is None:
            bm.free()
            bm = None
            return _voxel(V, F, edge_target)
        _repair(bm, real, 0.5 * edge_target)
        return _arrays(bm)
    finally:
        if bm is not None:
            bm.free()
