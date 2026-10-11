import math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import meshes
from quad_remesh.core import pipeline
from quad_remesh.core.mesh_data import TriMesh, clean, split_long_edges
from quad_remesh.core.features import Features
from quad_remesh.core.hierarchy import build_hierarchy
from quad_remesh.core.orientation import solve_orientation
from quad_remesh.core.position import solve_position
from quad_remesh.core.extract import extract_mesh
from quad_remesh.core.geom import normalize, dot

def drain(gen):
    while True:
        try: next(gen)
        except StopIteration as s: return s.value

def match(qa, na, qb):
    # rotate qb (any of 4) to best align with qa in plane na
    qb = qb - na * dot(qb, na)[:, None]
    qb = normalize(qb)
    tb = np.cross(na, qb)
    c = np.stack([qb, tb, -qb, -tb], 1)
    k = np.argmax(np.einsum('kij,kj->ki', c, qa), 1)
    return c[np.arange(len(qa)), k]

def singularities(mesh, Q):
    F = mesh.F
    N = mesh.vertex_normal
    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    qa = Q[a]
    na = N[a]
    qb = match(qa, na, Q[b])
    qc = match(qb, na, Q[c])
    qa2 = match(qc, na, Q[a])
    return np.sum(dot(qa2, qa) < 0.5)

argv = sys.argv[sys.argv.index("--") + 1:]
name, density = argv[0], float(argv[1])
V, F = getattr(meshes, name)()
tri = V[F]
area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
h = math.sqrt(area) / density
scale = 2 * h
V, F, _ = clean(V, F)
V, F, _ = split_long_edges(V, F, scale / 3)
mesh = TriMesh(V, F)
print("verts", len(V), "scale", scale, "mean edge", mesh.edge_length.mean())
feats = Features(mesh, None if os.environ.get("NOSHARP") else math.radians(40))
levels = build_hierarchy(mesh.V, mesh.vertex_normal, mesh.vertex_area, mesh.ei, mesh.ej)
print("levels", [L.n for L in levels])
line = feats.is_feature & ~feats.is_corner
Q = drain(solve_orientation(levels, line, feats.tangent))
print("orientation singular tris:", singularities(mesh, Q))
O, Q2 = drain(solve_position(levels, Q, scale, line, feats.tangent, feats.is_corner))
VP, VN, faces, cf, cc, cb, _cc = extract_mesh(mesh.V, mesh.vertex_normal, Q2, O, mesh.edges, scale, feats.is_feature, feats.is_corner, feats.is_boundary)
sizes = np.bincount([len(f) for f in faces])
print("extract verts", len(VP), "faces", len(faces), "sizes", dict(enumerate(sizes)), "expected", area / scale**2)
from quad_remesh.core.position import round4
T = np.cross(mesh.vertex_normal, Q2)
i, j = mesh.edges[:, 0], mesh.edges[:, 1]
A = round4(O[i], Q2[i], T[i], O[j], scale)
err = np.linalg.norm(A - O[j], axis=1) / scale
print("position mismatch edges >0.1:", (err > 0.1).sum(), "of", len(err))
# position singularities per triangle: integer translation sum around triangle
def tvec(a, b):
    Ab = round4(O[b], Q2[b], T[b], O[a], scale)  # b's lattice point nearest O_a
    d = O[b] - Ab
    return np.stack([np.round(dot(d, Q2[a]) / scale), np.round(dot(d, T[a]) / scale)], 1)
Fm = mesh.F
s = tvec(Fm[:,0], Fm[:,1]) + tvec(Fm[:,1], Fm[:,2]) + tvec(Fm[:,2], Fm[:,0])
print("position-singular tris:", int((np.abs(s).sum(1) > 0).sum()))
from quad_remesh.core.polymesh import PolyMesh
from quad_remesh.core import quadify, cleanup as cl, metrics as mt
def stats(pm, tag):
    m = mt.compute(pm.V, pm.faces)
    print(f"  {tag}: faces {m['faces']} tris {m['tris']} quads {m['quads']} ngons {m['ngons']} v3 {m['v3']} v5 {m['v5']} v6+ {m['v6plus']} bnd {m['boundary_edges']}")
pm = PolyMesh(VP, faces, cf, cc, cb); pm.compact(); stats(pm, "extracted")
quadify.dissolve_valence2(pm); stats(pm, "dissolve2")
quadify.pair_odd_faces(pm); stats(pm, "pair_odd")
quadify.split_even_ngons(pm); stats(pm, "split_even")
cl.edge_rotation(pm); stats(pm, "edge_rot")
cl.diagonal_collapse(pm); stats(pm, "diag_collapse")
quadify.dissolve_valence2(pm); quadify.pair_odd_faces(pm); pm.compact(); stats(pm, "pair2")
pm2 = quadify.subdivide(pm); stats(pm2, "subdiv")
tris = [f for f in faces if len(f) == 3]
VPa = np.asarray(VP)
# singular triangle centers
Fm = mesh.F; Nn = mesh.vertex_normal
qa = Q[Fm[:,0]]; qb = match(qa, Nn[Fm[:,0]], Q[Fm[:,1]]); qc = match(qb, Nn[Fm[:,0]], Q[Fm[:,2]]); qa2 = match(qc, Nn[Fm[:,0]], Q[Fm[:,0]])
sing_c = mesh.V[Fm[dot(qa2, qa) < 0.5]].mean(1)
for f in tris[:12]:
    p = VPa[f]
    L = [np.linalg.norm(p[k] - p[(k+1)%3]) / scale for k in range(3)]
    c = p.mean(0)
    ds = np.min(np.linalg.norm(sing_c - c, axis=1)) / scale if len(sing_c) else -1
    print("tri edges/scale", np.round(L, 2), "dist to singularity/scale", round(ds, 2))
