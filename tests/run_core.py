"""Run the core pipeline on test meshes and print metrics.

blender -b --factory-startup --python tests/run_core.py -- [names...] [--density N]
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import meshes
from quad_remesh.core import pipeline, metrics

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
density = 12.0
if "--density" in argv:
    i = argv.index("--density"); density = float(argv[i + 1]); del argv[i:i + 2]
names = argv or ["torus", "sphere", "cube", "cylinder", "monkey", "plane"]

for name in names:
    V, F = getattr(meshes, name)()
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
    h = math.sqrt(area) / density
    t0 = time.time()
    res = pipeline.run(V, F, pipeline.Params(target_edge=h))
    dt = time.time() - t0
    est = area / h / h
    print(f"[{name}] {dt:.2f}s est {est:.0f} | {metrics.summary(res.metrics)} | maxdev/h {res.metrics.get('max_dev', 0)/h:.3f} nm {res.metrics['nonmanifold_edges']} bnd {res.metrics['boundary_edges']}")
    out = os.environ.get("QR_DUMP")
    if out:
        np.savez(os.path.join(out, name + ".npz"), V=res.V, F=np.array([f + [-1] * (8 - len(f)) for f in res.faces]))
