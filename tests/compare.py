"""Compare parameter settings across meshes.
blender -b --python tests/compare.py -- "mesh:density,..." key=v1|v2 ...
"""
import math, os, sys, time
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.dirname(os.path.abspath(__file__))]
import numpy as np
import meshes
from quad_remesh.core import pipeline, metrics
argv = sys.argv[sys.argv.index("--") + 1:]
cases = [c.split(":") for c in argv[0].split(",")]
grid = [{}]
for a in argv[1:]:
    k, vs = a.split("=")
    grid = [dict(g, **{k: eval(v)}) for g in grid for v in vs.split("|")]
for name, d in cases:
    V, F = getattr(meshes, name)()
    t = V[F]; area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum()
    h = math.sqrt(area) / float(d)
    for g in grid:
        t0 = time.time()
        r = pipeline.run(V, F, pipeline.Params(target_edge=h, **g))
        m = r.metrics
        print(f"{name:14s} {str(g):40s} {time.time()-t0:5.1f}s  faces {m['faces']:5d}/{int(area/h/h):5d}  v3 {m['v3']:3d} v5 {m['v5']:3d} v6+ {m['v6plus']:2d}  cv {m['edge_cv']:.2f} ang {m['angle_dev']:4.1f} flip {m.get('flipped',0):2d} bnd {m['boundary_edges']}")
