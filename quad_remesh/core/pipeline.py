"""End-to-end remeshing pipeline (bpy-free). `remesh` is a generator that
yields (stage, progress) tuples and returns a Result."""

import math
from dataclasses import dataclass, field

import numpy as np

from . import cleanup, metrics, quadify
from .extract import _components, extract_mesh
from .features import Features
from .hierarchy import build_hierarchy
from .mesh_data import TriMesh, clean, smooth_normals, split_long_edges
from .orientation import solve_orientation
from .polymesh import PolyMesh
from .position import solve_position
from .relax import relax, untangle
from .spatial import PointIndex, SegmentProjector, SurfaceProjector


@dataclass
class Params:
    target_edge: float
    detect_sharp: bool = True
    sharp_angle: float = math.radians(40.0)
    marked_pairs: np.ndarray = None
    guides: list = field(default_factory=list)
    guide_strength: float = 0.7
    pure_quads: bool = True
    relax_iterations: int = 12
    cleanup: bool = True
    untangle: bool = True


@dataclass
class Result:
    V: np.ndarray
    faces: list
    metrics: dict
    feature_edges: set = field(default_factory=set)


class Cancelled(Exception):
    pass


def _resample(poly, spacing):
    poly = np.asarray(poly, dtype=np.float64)
    if len(poly) < 2:
        return poly, np.zeros_like(poly)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] <= 0:
        return poly[:1], np.zeros((1, 3))
    t = np.linspace(0, s[-1], max(2, int(np.ceil(s[-1] / spacing)) + 1))
    pts = np.stack([np.interp(t, s, poly[:, k]) for k in range(3)], 1)
    tan = np.gradient(pts, axis=0)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-30)
    return pts, tan


def guide_constraints(mesh, guides, radius, strength):
    n = len(mesh.V)
    soft_mask = np.zeros(n, bool)
    soft_dir = np.zeros((n, 3))
    soft_w = np.zeros(n)
    hard_mask = np.zeros(n, bool)
    if not guides:
        return soft_mask, soft_dir, soft_w, hard_mask
    index = PointIndex(mesh.V)
    best = np.full(n, np.inf)
    lam = min(50.0, strength / max(1e-3, 1.0 - strength))
    for poly in guides:
        pts, tan = _resample(poly, radius / 4.0)
        for p, t in zip(pts, tan):
            if not t.any():
                continue
            for v in index.within(p, radius):
                d = np.linalg.norm(mesh.V[v] - p)
                if d < best[v]:
                    best[v] = d
                    soft_dir[v] = t
                    soft_w[v] = lam * (1.0 - d / radius) ** 2
                    soft_mask[v] = True
    if strength >= 0.999:
        hard_mask = soft_mask & (best < 0.35 * radius)
    return soft_mask, soft_dir, soft_w, hard_mask


def remesh(V, F, params):
    h = float(params.target_edge)
    scale = 2.0 * h if params.pure_quads else h

    yield ("Preparing mesh", 0.0)
    V, F, marked = clean(V, F, params.marked_pairs)
    if not len(F):
        raise ValueError("Mesh has no faces")
    V, F, marked = split_long_edges(V, F, scale / 3.0, marked)
    mesh = TriMesh(V, F)
    lab = _components(len(mesh.V), mesh.edges[:, 0], mesh.edges[:, 1])
    _, vcomp = np.unique(lab, return_inverse=True)
    vcomp = vcomp.reshape(-1)
    surf = SurfaceProjector(mesh.V, mesh.F, vcomp[mesh.F[:, 0]])

    yield ("Detecting sharp edges", 0.04)
    feats = Features(mesh, params.sharp_angle if params.detect_sharp else None, marked, scale=scale)
    seg = SegmentProjector(feats.segments, scale / 8.0, vcomp[feats.feature_edges[:, 0]]) if feats.any else None

    yield ("Building hierarchy", 0.08)
    N = smooth_normals(mesh, feats.edge_mask, 0.25 * scale)
    levels = build_hierarchy(mesh.V, N, mesh.vertex_area, mesh.ei, mesh.ej)

    line_mask = feats.is_feature & ~feats.is_corner
    hard_mask = line_mask.copy()
    hard_dir = feats.tangent.copy()
    soft = guide_constraints(mesh, params.guides, 0.75 * scale, params.guide_strength)
    soft_mask, soft_dir, soft_w, ghard = soft
    gh = ghard & ~hard_mask
    hard_mask |= gh
    hard_dir[gh] = soft_dir[gh]
    soft_mask &= ~hard_mask

    gen = solve_orientation(levels, hard_mask, hard_dir, soft_mask, soft_dir, soft_w)
    while True:
        try:
            p = next(gen)
            yield ("Solving orientation field", 0.1 + 0.25 * p)
        except StopIteration as stop:
            Q = stop.value
            break

    gen = solve_position(levels, Q, scale, line_mask, feats.tangent, feats.is_corner)
    while True:
        try:
            p = next(gen)
            yield ("Solving position field", 0.35 + 0.3 * p)
        except StopIteration as stop:
            O, Q = stop.value
            break

    yield ("Extracting quads", 0.66)
    VP, VN, faces, c_feat, c_corner, c_bnd, c_comp = extract_mesh(
        mesh.V, N, Q, O, mesh.edges, scale,
        feats.is_feature, feats.is_corner, feats.is_boundary, vcomp)
    if not faces:
        raise ValueError("Extraction produced no faces; try a lower density")
    pm = PolyMesh(VP, faces, c_feat, c_corner, c_bnd, comp=c_comp)
    pm.compact()
    snap = pm.feat & ~pm.corner
    if seg is not None and snap.any():
        pm.V[snap], _, _ = seg.project(pm.V[snap], comps=pm.comp[snap])
    on_surf = ~pm.feat
    pm.V[on_surf], _ = surf.project(pm.V[on_surf], pm.comp[on_surf], pm.vertex_normals()[on_surf])

    yield ("Fixing irregular faces", 0.72)
    quadify.fill_holes(pm)
    quadify.mark_feature_edges(pm, seg)
    quadify.dissolve_valence2(pm)
    quadify.pair_odd_faces(pm)
    quadify.split_even_ngons(pm)
    if params.cleanup:
        cleanup.edge_rotation(pm)
        cleanup.diagonal_collapse(pm)
        quadify.dissolve_valence2(pm)
        quadify.pair_odd_faces(pm)
    pm.compact()
    pm.V = relax(pm.V, pm.faces, scale, surf, seg, pm.feat, pm.corner, iterations=4, comp=pm.comp)

    if params.pure_quads:
        yield ("Subdividing to pure quads", 0.8)
        pm = quadify.subdivide(pm)
    else:
        quadify.split_odd_ngons(pm)

    yield ("Relaxing", 0.85)
    pm.V = relax(pm.V, pm.faces, h, surf, seg, pm.feat, pm.corner,
                 iterations=max(1, params.relax_iterations // 2), comp=pm.comp)
    if params.cleanup:
        cleanup.edge_rotation(pm)
        cleanup.diagonal_collapse(pm)
        quadify.dissolve_valence2(pm)
        quadify.pair_odd_faces(pm, max_depth=1)
        pm.compact()
    pm.V = relax(pm.V, pm.faces, h, surf, seg, pm.feat, pm.corner,
                 iterations=max(1, params.relax_iterations - params.relax_iterations // 2), comp=pm.comp)

    if params.untangle:
        pm.V = untangle(pm.V, pm.faces, surf, seg, pm.feat, pm.corner, pm.comp)

    yield ("Measuring", 0.98)
    m = metrics.compute(pm.V, pm.faces, surf, comp=pm.comp)
    return Result(pm.V, pm.faces, m, set(pm.feature_edges))


def run(V, F, params, progress=None):
    """Run the pipeline to completion (no cancellation)."""
    gen = remesh(V, F, params)
    while True:
        try:
            stage, p = next(gen)
            if progress:
                progress(stage, p)
        except StopIteration as stop:
            return stop.value
