# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

"""Remesh every mesh in an FBX/OBJ/blend file and print a quality table.

blender -b --factory-startup --python tests/cad_compare.py -- path.fbx [density] [names...] [key=value ...]
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from quad_remesh import prep  # noqa: E402
from quad_remesh.core import pipeline  # noqa: E402
from quad_remesh.core.spatial import SurfaceProjector  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
path = argv[0]
density = float(argv[1]) if len(argv) > 1 and argv[1].replace(".", "").isdigit() else 20.0
kw = {}
names = []
for a in argv[1:]:
    if "=" in a:
        k, v = a.split("=", 1)
        kw[k] = eval(v)
    elif not a.replace(".", "").isdigit():
        names.append(a)
use_prep = kw.pop("prep", True)

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
if path.lower().endswith(".fbx"):
    bpy.ops.import_scene.fbx(filepath=path)
elif path.lower().endswith(".obj"):
    bpy.ops.wm.obj_import(filepath=path)
else:
    bpy.ops.wm.open_mainfile(filepath=path)

dg = bpy.context.evaluated_depsgraph_get()
meshes = [o for o in bpy.data.objects if o.type == 'MESH' and (not names or o.name in names)]
for ob in meshes:
    me = ob.evaluated_get(dg).to_mesh()
    me.calc_loop_triangles()
    M = np.array(ob.matrix_world)
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    V = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    F = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", F)
    F = F.reshape(-1, 3)
    marked = None
    attr = me.attributes.get("sharp_edge")
    if attr is not None and attr.domain == 'EDGE':
        s = np.zeros(len(me.edges), bool)
        attr.data.foreach_get("value", s)
        E = np.empty(len(me.edges) * 2, dtype=np.int64)
        me.edges.foreach_get("vertices", E)
        marked = E.reshape(-1, 2)[s] if s.any() else None
    ob.evaluated_get(dg).to_mesh_clear()
    t = V[F]
    area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum()
    h = math.sqrt(area) / density
    params = pipeline.Params(target_edge=h, marked_pairs=marked,
                             prepare_fn=prep.prepare if use_prep else None, **kw)
    t0 = time.time()
    try:
        r = pipeline.run(V, F, params)
    except Exception as exc:
        print(f"{ob.name:16s} FAILED {type(exc).__name__}: {exc}")
        continue
    dt = time.time() - t0
    m = r.metrics
    quads = [f for f in r.faces if len(f) == 4]
    tris = [[f[0], f[1], f[2]] for f in quads] + [[f[0], f[2], f[3]] for f in quads]
    out = SurfaceProjector(r.V, np.array(tris))
    idx = np.linspace(0, len(V) - 1, min(4000, len(V))).astype(int)
    P, _ = out.project(V[idx])
    d = np.linalg.norm(P - V[idx], axis=1) / h
    print(f"{ob.name:16s} {len(F):8d} tris {dt:6.1f}s | faces {m['faces']:6d} v3 {m['v3']:4d} v5 {m['v5']:4d} "
          f"v6+ {m['v6plus']:3d} | cv {m['edge_cv']:.2f} flip {m.get('flipped', 0):4d} holes {m.get('holes', 0):4d} "
          f"nm {m['nonmanifold_edges']} | src->out/h p95 {np.percentile(d, 95):.2f} max {d.max():.2f} | {m.get('method')}",
          flush=True)
