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
from .scale import scale_field
from .quantize import quantize_fine
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
    size_flexibility: float = 0.0
    quantize: bool = True


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


def _dbg(pm, tag):
    if not __import__("os").environ.get("QR_DEBUG_STAGES"):
        return
    m = metrics.compute(pm.V, pm.faces)
    print(f"  [{tag}] faces {m['faces']} tris {m['tris']} ngons {m['ngons']} v3 {m['v3']} v5 {m['v5']} "
          f"v6+ {m['v6plus']} bnd {m['boundary_edges']} nm {m['nonmanifold_edges']}")


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

    yield ("Fitting quad sizes", 0.35)
    hv = scale_field(mesh, N, Q, scale, params.size_flexibility)
    index = PointIndex(mesh.V)

    def size_at(pts, factor=1.0):
        return np.array([hv[index.nearest(p)] for p in pts]) * factor

    gen = solve_position(levels, Q, hv, line_mask, feats.tangent, feats.is_corner)
    while True:
        try:
            p = next(gen)
            yield ("Solving position field", 0.35 + 0.3 * p)
        except StopIteration as stop:
            O, Q = stop.value
            break

    ctx = dict(mesh=mesh, N=N, Q=Q, O=O, hv=hv, feats=feats, vcomp=vcomp, surf=surf, seg=seg,
               size_at=size_at, h=h, scale=scale, params=params)
    expected = mesh.area / (h * h)
    if params.quantize:
        result = yield from _finish(ctx, quantized=True, p0=0.67, p1=0.82)
        if _suspicious(result.metrics, expected):
            alt = yield from _finish(ctx, quantized=False, p0=0.82, p1=0.97)
            if _score(alt.metrics, expected) < _score(result.metrics, expected):
                result = alt
    else:
        result = yield from _finish(ctx, quantized=False, p0=0.67, p1=0.97)
    return result


def _suspicious(m, expected):
    return (m["v6plus"] > 0 or m.get("flipped", 0) > 0 or m["edge_cv"] > 0.3
            or abs(m["faces"] / max(expected, 1.0) - 1.0) > 0.15)


def _score(m, expected):
    """Lower is better: poles first, then defects that look broken."""
    s = m["v3"] + m["v5"] + 4.0 * m["v6plus"]
    s += 5.0 * m.get("flipped", 0)
    s += 0.5 * m["boundary_edges"]
    s += 60.0 * max(0.0, m["edge_cv"] - 0.25)
    s += 80.0 * max(0.0, abs(m["faces"] / max(expected, 1.0) - 1.0) - 0.15)
    return s


def _finish(ctx, quantized, p0, p1):
    """Extraction and all topology / geometry clean-up after the fields."""
    mesh, N, Q, O, hv = ctx["mesh"], ctx["N"], ctx["Q"], ctx["O"], ctx["hv"]
    feats, vcomp, surf, seg = ctx["feats"], ctx["vcomp"], ctx["surf"], ctx["seg"]
    size_at, h, scale, params = ctx["size_at"], ctx["h"], ctx["scale"], ctx["params"]

    def prog(f):
        return p0 + (p1 - p0) * f

    yield ("Extracting quads", prog(0.0))
    if quantized:
        VP, VN, faces, c_feat, c_corner, c_bnd, c_comp, VQ, hc, _q = quantize_fine(
            mesh, N, Q, O, hv, feats.edge_mask, feats.is_feature, feats.is_corner, feats.is_boundary, vcomp)
    else:
        VP, VN, faces, c_feat, c_corner, c_bnd, c_comp, VQ, hc = extract_mesh(
            mesh.V, N, Q, O, mesh.edges, hv,
            feats.is_feature, feats.is_corner, feats.is_boundary, vcomp)
    if not faces:
        raise ValueError("Extraction produced no faces; try a lower density")
    pm = PolyMesh(VP, faces, c_feat, c_corner, c_bnd, comp=c_comp)
    pm.attrs = {"N": VN, "Q": VQ, "H": hc}
    pm.compact()
    snap = pm.feat & ~pm.corner
    if seg is not None and snap.any():
        pm.V[snap], _, _ = seg.project(pm.V[snap], comps=pm.comp[snap])
    on_surf = ~pm.feat
    pm.V[on_surf], _ = surf.project(pm.V[on_surf], pm.comp[on_surf], pm.vertex_normals()[on_surf])

    yield ("Fixing irregular faces", prog(0.25))
    _dbg(pm, "extracted")
    quadify.fill_holes(pm)
    quadify.mark_feature_edges(pm, seg)
    _dbg(pm, "filled")
    quadify.dissolve_valence2(pm)
    quadify.pair_odd_faces(pm, max_depth=6)
    quadify.split_even_ngons(pm)
    _dbg(pm, "paired")
    if params.cleanup:
        cleanup.edge_rotation(pm)
        cleanup.diagonal_collapse(pm)
        quadify.dissolve_valence2(pm)
        quadify.pair_odd_faces(pm)
    pm.compact()
    _dbg(pm, "cleaned")
    pm.V = relax(pm.V, pm.faces, scale, surf, seg, pm.feat, pm.corner, iterations=4, comp=pm.comp,
                 size_fn=size_at)

    if params.pure_quads:
        yield ("Subdividing to pure quads", prog(0.45))
        quadify.split_odd_ngons(pm, min_size=7)   # avoid valence 7+ face centres
        pm = quadify.subdivide(pm)
    else:
        quadify.split_odd_ngons(pm)

    _dbg(pm, "subdivided")
    yield ("Relaxing", prog(0.55))

    def final_size(pts):
        return size_at(pts, h / scale)

    pm.V = relax(pm.V, pm.faces, h, surf, seg, pm.feat, pm.corner,
                 iterations=max(1, params.relax_iterations // 2), comp=pm.comp, size_fn=final_size)
    if params.cleanup:
        cleanup.edge_rotation(pm)
        cleanup.diagonal_collapse(pm)
        quadify.dissolve_valence2(pm)
        quadify.pair_odd_faces(pm, max_depth=1)
        pm.compact()
    pm.V = relax(pm.V, pm.faces, h, surf, seg, pm.feat, pm.corner,
                 iterations=max(1, params.relax_iterations - params.relax_iterations // 2), comp=pm.comp,
                 size_fn=final_size)

    _dbg(pm, "final-clean")
    if params.untangle:
        yield ("Untangling", prog(0.9))
        pm.V = untangle(pm.V, pm.faces, surf, seg, pm.feat, pm.corner, pm.comp)

    m = metrics.compute(pm.V, pm.faces, surf, comp=pm.comp)
    m["method"] = "quantized" if quantized else "classic"
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
