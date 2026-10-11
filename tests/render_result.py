"""Render remesh results to PNG for visual inspection.

blender -b --factory-startup --python tests/render_result.py -- <mesh> <density> <out.png> [nosharp] [view=x,y,z]
"""
import math, os, sys
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.dirname(os.path.abspath(__file__))]
import bpy
import numpy as np
from mathutils import Vector
import meshes
from quad_remesh.core import pipeline, metrics

argv = sys.argv[sys.argv.index("--") + 1:]
name, density, out = argv[0], float(argv[1]), argv[2]
nosharp = "nosharp" in argv
view = Vector((1.0, -1.6, 0.9))
for a in argv:
    if a.startswith("view="):
        view = Vector(tuple(float(x) for x in a[5:].split(",")))
V, F = getattr(meshes, name)()
t = V[F]
area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum()
h = math.sqrt(area) / density
if "source" in argv:
    class _R: pass
    res = _R(); res.V = V; res.faces = F.tolist()
    res.metrics = metrics.compute(V, res.faces)
else:
    res = pipeline.run(V, F, pipeline.Params(target_edge=h, detect_sharp=not nosharp, untangle="untangle" in argv))
print(metrics.summary(res.metrics), "bnd", res.metrics["boundary_edges"])

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene

def add(nm, verts, faces, color):
    me = bpy.data.meshes.new(nm)
    me.from_pydata(verts.tolist(), [], faces)
    ob = bpy.data.objects.new(nm, me)
    sc.collection.objects.link(ob)
    ob.color = color
    return ob

def color_faces(ob):
    # colour non-quads red, and faces touching irregular vertices
    pass

from quad_remesh.core.spatial import SurfaceProjector
bad = set(metrics.flipped_faces(res.V, res.faces, SurfaceProjector(V, F))) if "source" not in argv else set()
good_f = [list(map(int, f)) for i, f in enumerate(res.faces) if i not in bad]
bad_f = [list(map(int, f)) for i, f in enumerate(res.faces) if i in bad]
solid = add("solid", res.V, good_f, (0.85, 0.85, 0.85, 1))
if bad_f:
    add("bad", res.V, bad_f, (1.0, 0.1, 0.1, 1))
wire = add("wire", res.V, [list(map(int, f)) for f in res.faces], (0.05, 0.05, 0.05, 1))
mod = wire.modifiers.new("w", 'WIREFRAME')
mod.thickness = h * 0.04
mod.use_relative_offset = False
mod.offset = 0.0
# Irregular vertex markers
from quad_remesh.core.polymesh import ekey
val = np.zeros(len(res.V), int); cnt = {}
for f in res.faces:
    for k in range(len(f)):
        e = ekey(f[k], f[(k + 1) % len(f)]); cnt[e] = cnt.get(e, 0) + 1
openv = np.zeros(len(res.V), bool)
for (a, b), c in cnt.items():
    val[a] += 1; val[b] += 1
    if c != 2: openv[a] = openv[b] = True
for vv, col in ((3, (0.1, 0.4, 1.0, 1)), (5, (1.0, 0.2, 0.1, 1)), (6, (1, 0, 1, 1))):
    sel = np.nonzero((val == vv if vv < 6 else val >= 6) & ~openv)[0]
    if len(sel):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=h * 0.18, segments=8, ring_count=4)
        proto = bpy.context.active_object
        proto.color = col
        for i, vi in enumerate(sel):
            o = proto if i == 0 else proto.copy()
            if i:
                sc.collection.objects.link(o)
            o.location = Vector(res.V[vi])
# source ghost for comparison
src = add("src", V, F.tolist(), (1.0, 0.6, 0.2, 1))
src.display_type = 'WIRE'
src.hide_render = True

sc.render.engine = 'BLENDER_WORKBENCH'
sc.display.shading.color_type = 'OBJECT'
sc.display.shading.light = 'STUDIO'
sc.render.resolution_x = 1100
sc.render.resolution_y = 900
sc.render.film_transparent = False
c = res.V.mean(0)
ext = np.linalg.norm(res.V.max(0) - res.V.min(0))
cam_data = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cam_data)
sc.collection.objects.link(cam)
cam.location = Vector(c) + view.normalized() * ext * 1.5
cam.rotation_euler = (Vector(c) - cam.location).to_track_quat('-Z', 'Y').to_euler()
sc.camera = cam
sc.render.filepath = out
bpy.ops.render.render(write_still=True)
