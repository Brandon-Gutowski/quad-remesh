# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

"""Conversion between Blender data and the NumPy arrays used by the core."""

import math

import bpy
import numpy as np


# ---------------------------------------------------------------------------
# Evaluated-mesh cache (shared by the checker preview and the face estimate)
# ---------------------------------------------------------------------------

_cache = {}


class MeshArrays:
    """Local-space triangle soup of an object's evaluated mesh."""

    def __init__(self, co, tris, normals, edges, sharp):
        self.co = co            # (n,3) float32, object space
        self.tris = tris        # (m,3) int32
        self.normals = normals  # (n,3) float32, object space
        self.edges = edges      # (k,2) int32
        self.sharp = sharp      # (k,) bool or None
        self.batch = None       # GPU batch, filled lazily by preview
        self._area = None

    def world_area(self, matrix):
        key = tuple(round(v, 9) for row in matrix for v in row)
        if self._area is None or self._area[0] != key:
            M = np.array(matrix, dtype=np.float64)
            P = self.co.astype(np.float64) @ M[:3, :3].T + M[:3, 3]
            t = P[self.tris]
            a = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum()
            self._area = (key, float(a))
        return self._area[1]


def _read_mesh(me):
    me.calc_loop_triangles()
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    tris = np.empty(len(me.loop_triangles) * 3, dtype=np.int32)
    me.loop_triangles.foreach_get("vertices", tris)
    nrm = np.empty(n * 3, dtype=np.float32)
    if hasattr(me, "vertex_normals"):
        me.vertex_normals.foreach_get("vector", nrm)
    else:  # pragma: no cover - very old versions
        me.vertices.foreach_get("normal", nrm)
    edges = np.empty(len(me.edges) * 2, dtype=np.int32)
    me.edges.foreach_get("vertices", edges)
    sharp = None
    attr = me.attributes.get("sharp_edge")
    if attr is not None and attr.domain == 'EDGE' and len(me.edges):
        sharp = np.zeros(len(me.edges), dtype=bool)
        attr.data.foreach_get("value", sharp)
    return MeshArrays(co.reshape(-1, 3), tris.reshape(-1, 3), nrm.reshape(-1, 3), edges.reshape(-1, 2), sharp)


def mesh_arrays(context, obj):
    """Cached arrays of obj's evaluated mesh (invalidated on geometry change)."""
    key = obj.name_full
    ma = _cache.get(key)
    if ma is None:
        depsgraph = context.evaluated_depsgraph_get()
        ob_eval = obj.evaluated_get(depsgraph)
        me = ob_eval.to_mesh()
        try:
            ma = _read_mesh(me)
        finally:
            ob_eval.to_mesh_clear()
        _cache[key] = ma
    return ma


def invalidate(name=None):
    if name is None:
        _cache.clear()
    else:
        _cache.pop(name, None)


@bpy.app.handlers.persistent
def _on_depsgraph_update(scene, depsgraph):
    for upd in depsgraph.updates:
        idd = upd.id
        if isinstance(idd, bpy.types.Object) and upd.is_updated_geometry:
            _cache.pop(idd.original.name_full, None)
        elif isinstance(idd, bpy.types.Mesh):
            # Mesh data changed: drop every object using it.
            for name in [k for k in _cache]:
                ob = bpy.data.objects.get(name)
                if ob is None or ob.data == idd.original:
                    _cache.pop(name, None)


@bpy.app.handlers.persistent
def _on_load(*_args):
    _cache.clear()


# ---------------------------------------------------------------------------
# Remesh input / output
# ---------------------------------------------------------------------------

def target_edge_length(context, obj):
    ma = mesh_arrays(context, obj)
    area = ma.world_area(obj.matrix_world)
    density = obj.quad_remesh.density
    if area <= 0:
        return 0.0, 0.0
    return math.sqrt(area) / density, area


def remesh_inputs(context, obj):
    """World-space vertices, triangles and marked-sharp vertex pairs."""
    ma = mesh_arrays(context, obj)
    M = np.array(obj.matrix_world, dtype=np.float64)
    V = ma.co.astype(np.float64) @ M[:3, :3].T + M[:3, 3]
    F = ma.tris.astype(np.int64)
    marked = None
    if ma.sharp is not None and ma.sharp.any():
        marked = ma.edges[ma.sharp].astype(np.int64)
    return V, F, marked


def symmetry_plane(obj, axis):
    """World-space (point, unit normal) of the object's local mirror plane."""
    k = "XYZ".index(axis)
    M = np.array(obj.matrix_world, dtype=np.float64)
    local = np.zeros(3)
    local[k] = 1.0
    n = np.linalg.inv(M[:3, :3]).T @ local       # normals transform by the inverse transpose
    return M[:3, 3].copy(), n / np.linalg.norm(n)


def guide_polylines(guide_obj):
    """World-space point lists of every spline in a curve object. Cyclic
    splines repeat their first point at the end (a closed ring)."""
    if guide_obj is None or guide_obj.type != 'CURVE':
        return []
    M = guide_obj.matrix_world
    out = []
    for spl in guide_obj.data.splines:
        pts = spl.bezier_points if spl.type == 'BEZIER' else spl.points
        if len(pts) < 2:
            continue
        P = [(M @ p.co.to_3d())[:] for p in pts]
        if spl.use_cyclic_u:
            P.append(P[0])
        out.append(np.array(P, dtype=np.float64))
    return out


def _find_result(src):
    for ob in bpy.data.objects:
        if ob.get("qr_source") == src.name_full and ob.type == 'MESH':
            return ob
    return None


def write_result(context, src, V_world, faces, feature_edges=()):
    """Create (or update) the '<name>_Quad' object for src."""
    M = src.matrix_world
    Minv = np.array(M.inverted(), dtype=np.float64)
    V = V_world @ Minv[:3, :3].T + Minv[:3, 3]

    me = bpy.data.meshes.new(src.name + "_Quad")
    me.from_pydata(V.tolist(), [], [list(map(int, f)) for f in faces])
    me.validate(clean_customdata=False)
    me.update()
    if feature_edges:
        fe = set(feature_edges)
        E = np.empty(len(me.edges) * 2, dtype=np.int64)
        me.edges.foreach_get("vertices", E)
        E = E.reshape(-1, 2)
        mask = np.array([(min(a, b), max(a, b)) in fe for a, b in E], dtype=bool)
        if mask.any():
            attr = me.attributes.get("sharp_edge") or me.attributes.new("sharp_edge", 'BOOLEAN', 'EDGE')
            attr.data.foreach_set("value", mask)
    if src.type == 'MESH':
        for mat in src.data.materials:
            me.materials.append(mat)
        polys = src.data.polygons
        if len(polys):
            smooth = np.empty(len(polys), dtype=bool)
            polys.foreach_get("use_smooth", smooth)
            if smooth.mean() > 0.5:
                me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))

    ob = _find_result(src)
    if ob is not None:
        old = ob.data
        ob.data = me
        if old.users == 0:
            bpy.data.meshes.remove(old)
        ob.name = src.name + "_Quad"
    else:
        ob = bpy.data.objects.new(src.name + "_Quad", me)
        cols = src.users_collection or (context.scene.collection,)
        for col in cols:
            col.objects.link(ob)
        ob["qr_source"] = src.name_full
    ob.matrix_world = M.copy()
    me.name = ob.name
    return ob


def register():
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)
    bpy.app.handlers.load_post.append(_on_load)


def unregister():
    for lst, fn in ((bpy.app.handlers.depsgraph_update_post, _on_depsgraph_update),
                    (bpy.app.handlers.load_post, _on_load)):
        if fn in lst:
            lst.remove(fn)
    _cache.clear()
