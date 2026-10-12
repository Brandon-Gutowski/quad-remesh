# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

"""Render one mesh from a file, before and/or after remeshing.

blender -b --factory-startup --python tests/render_cad.py -- file.fbx ObjectName density out.png [source] [view=x,y,z] [key=value ...]
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402

from quad_remesh import prep  # noqa: E402
from quad_remesh.core import metrics, pipeline  # noqa: E402
from quad_remesh.core.spatial import SurfaceProjector  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
path, name, density, out = argv[0], argv[1], float(argv[2]), argv[3]
view = Vector((1.0, -1.4, 0.8))
kw = {}
for a in argv[4:]:
    if a.startswith("view="):
        view = Vector(tuple(float(x) for x in a[5:].split(",")))
    elif "=" in a:
        k, v = a.split("=", 1)
        kw[k] = eval(v)
show_source = "source" in argv
use_prep = kw.pop("prep", True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=path)
src = bpy.data.objects[name]
dg = bpy.context.evaluated_depsgraph_get()
me = src.evaluated_get(dg).to_mesh()
me.calc_loop_triangles()
M = np.array(src.matrix_world)
co = np.empty(len(me.vertices) * 3)
me.vertices.foreach_get("co", co)
V = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
F = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
me.loop_triangles.foreach_get("vertices", F)
F = F.reshape(-1, 3)
src.evaluated_get(dg).to_mesh_clear()
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
t = V[F]
area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum()
h = math.sqrt(area) / density
sc = bpy.context.scene


def add(nm, verts, faces, color, wire=None):
    mesh = bpy.data.meshes.new(nm)
    mesh.from_pydata(verts.tolist(), [], faces)
    ob = bpy.data.objects.new(nm, mesh)
    sc.collection.objects.link(ob)
    ob.color = color
    if wire:
        mod = ob.modifiers.new("w", 'WIREFRAME')
        mod.thickness = wire
        mod.use_relative_offset = False
    return ob


if show_source:
    RV, RF = V, F.tolist()
    add("src", RV, RF, (0.8, 0.8, 0.8, 1))
    add("srcw", RV, RF, (0.05, 0.05, 0.05, 1), wire=h * 0.01)
else:
    res = pipeline.run(V, F, pipeline.Params(target_edge=h, prepare_fn=prep.prepare if use_prep else None, **kw))
    print(metrics.summary(res.metrics))
    RV = res.V
    bad = set(metrics.flipped_faces(RV, res.faces, SurfaceProjector(V, F)))
    good = [list(map(int, f)) for i, f in enumerate(res.faces) if i not in bad]
    badf = [list(map(int, f)) for i, f in enumerate(res.faces) if i in bad]
    add("solid", RV, good, (0.82, 0.82, 0.82, 1))
    if badf:
        add("bad", RV, badf, (1, 0.1, 0.1, 1))
    add("wire", RV, [list(map(int, f)) for f in res.faces], (0.05, 0.05, 0.05, 1),
        wire=float(np.median(np.linalg.norm(RV[[f[0] for f in res.faces]] - RV[[f[1] for f in res.faces]], axis=1))) * 0.04)
sc.render.engine = 'BLENDER_WORKBENCH'
sc.display.shading.color_type = 'OBJECT'
sc.render.resolution_x, sc.render.resolution_y = 1100, 900
c = Vector(RV.mean(0))
ext = float(np.linalg.norm(RV.max(0) - RV.min(0)))
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
cam.data.clip_start = ext * 1e-3
cam.data.clip_end = ext * 100
cam.location = c + view.normalized() * ext * 1.4
cam.rotation_euler = (c - cam.location).to_track_quat('-Z', 'Z').to_euler()
sc.camera = cam
sc.render.filepath = out
bpy.ops.render.render(write_still=True)
