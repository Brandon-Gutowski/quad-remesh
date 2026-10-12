# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

"""End-to-end remeshing pipeline (bpy-free). `remesh` is a generator that
yields (stage, progress) tuples and returns a Result."""

import math
from dataclasses import dataclass, field, replace

import numpy as np

from . import cleanup, guides as guide_mod, metrics, quadify, symmetry
from .extract import _components, extract_mesh
from .features import Features, transfer_features
from .hierarchy import build_hierarchy
from .mesh_data import TriMesh, clean, smooth_normals, split_long_edges
from .orientation import solve_orientation
from .polymesh import PolyMesh
from .position import solve_position
from .relax import relax, untangle
from .scale import (apply_size_points, detail_size_points, drop_unresolvable_features,
                    limit_gradient, scale_field)
from .quantize import quantize_fine
from .spatial import PointIndex, SegmentProjector, SurfaceProjector


@dataclass
class Params:
    target_edge: float
    detect_sharp: bool = True
    sharp_angle: float = math.radians(40.0)
    marked_pairs: np.ndarray = None
    guides: list = field(default_factory=list)   # polylines; closed ones repeat the first point
    guide_strength: float = 0.7   # how far the flow beside a guide loop bends to follow it
    guide_rings_out: int = 1      # extra concentric loops around each closed guide, one cell apart
    guide_rings_in: int = 0       # ... and inside it
    pure_quads: bool = True
    relax_iterations: int = 12
    cleanup: bool = True
    untangle: bool = True
    size_flexibility: float = 0.0
    adapt_thin: bool = False
    quantize: bool = True
    symmetry_plane: tuple = None   # (point, unit normal); keeps the + side and mirrors it
    detail: float = 0.0            # 0 = uniform quads; 1 = down to 1/4 size where detail needs it
    prepare_fn: object = None      # (V, F, edge_target, even=True) -> (V, F): simplify/clean the field mesh


@dataclass
class Result:
    V: np.ndarray
    faces: list
    metrics: dict
    feature_edges: set = field(default_factory=set)


class Cancelled(Exception):
    pass


def _with_rings(V, F, params):
    """Add the concentric rings of every closed guide as guides of their own
    (on the whole input, before any symmetry cut)."""
    closed = [g for g in params.guides if guide_mod.is_closed(g)]
    if not closed or (params.guide_rings_out <= 0 and params.guide_rings_in <= 0):
        return params
    Vc, Fc, _ = clean(V, F)
    if not len(Fc):
        return params
    mesh = TriMesh(Vc, Fc)
    comp = _components_of(len(Vc), mesh.edges)
    surf = SurfaceProjector(Vc, Fc, comp[Fc[:, 0]])
    step = (2.0 if params.pure_quads else 1.0) * float(params.target_edge)
    guides = [np.asarray(g, dtype=np.float64) for g in params.guides]
    pts = [guide_mod.resample(g, 0.25 * step)[0] for g in guides]
    # Rings of each guide, stopped by the drawn strokes...
    cands = {}
    for gi, g in enumerate(guides):
        if not guide_mod.is_closed(g):
            continue
        rest = [q for k, q in enumerate(pts) if k != gi]
        others = PointIndex(np.vstack(rest)) if rest else None
        c = int(np.bincount(surf.component_of(g)).argmax())
        cands[gi] = guide_mod.concentric_rings(g, surf, c, step, params.guide_rings_out,
                                               params.guide_rings_in, others)
    # ...then where rings of two guides run into each other, both go (with the
    # rings beyond them), so neighbouring guides are treated alike.
    flat = [(gi, k, r) for gi, rs in cands.items() for k, r in enumerate(rs)]
    keep = {(gi, k) for gi, k, _r in flat}
    trees = {(gi, k): PointIndex(r) for gi, k, r in flat}
    for i, (ga, ka, ra) in enumerate(flat):
        for gb, kb, _rb in flat[i + 1:]:
            if ga != gb and min(trees[(gb, kb)].kd.find(q)[2] for q in ra) < 0.75 * step:
                keep.discard((ga, ka))
                keep.discard((gb, kb))
    added = []
    for gi, rs in cands.items():
        for k, r in enumerate(rs):
            # an outer ring only stays if every ring inside it (same side) stayed
            if all((gi, j) in keep for j in range(k + 1)):
                added.append(r)
    return replace(params, guides=guides + added) if added else params


def remesh(V, F, params):
    params = _with_rings(V, F, params)
    if params.symmetry_plane is None:
        return (yield from _remesh_parts(V, F, params))
    p0 = np.asarray(params.symmetry_plane[0], dtype=np.float64)
    nrm = np.asarray(params.symmetry_plane[1], dtype=np.float64)
    nrm = nrm / np.linalg.norm(nrm)
    V0, F0 = np.asarray(V, dtype=np.float64), np.asarray(F, dtype=np.int64)
    Vh, Fh, marked = symmetry.cut_half(V0, F0, p0, nrm, params.marked_pairs)
    if not len(Fh):
        raise ValueError("Nothing on the positive side of the symmetry plane")
    half = replace(params, symmetry_plane=None, marked_pairs=marked,
                   guides=symmetry.fold_guides(params.guides, p0, nrm))
    res = yield from _remesh_parts(Vh, Fh, half)
    yield ("Mirroring", 0.99)
    Vm, faces, fe = symmetry.mirror_weld(res.V, res.faces, res.feature_edges, p0, nrm,
                                         seam_tol=0.6 * params.target_edge)
    # Seam vertices left with no interior edge become valence-2 after welding:
    # dissolve them and merge the two resulting triangles back into one quad.
    pm = PolyMesh(Vm, faces, np.zeros(len(Vm), bool), np.zeros(len(Vm), bool),
                  np.zeros(len(Vm), bool), fe)
    if quadify.dissolve_valence2(pm):
        quadify.pair_odd_faces(pm, max_depth=1)
        pm.compact()
    Vm, faces, fe = pm.V, pm.faces, pm.feature_edges
    surf = SurfaceProjector(*clean(V0, F0)[:2])
    m = metrics.compute(Vm, faces, surf)
    m["method"] = res.metrics.get("method", "")
    return Result(Vm, faces, m, fe)


def _components_of(n, edges):
    lab = _components(n, edges[:, 0], edges[:, 1])
    return np.unique(lab, return_inverse=True)[1].reshape(-1)


def _map_components(mesh, surf, n_samples=24):
    """Label each vertex of `mesh` with the reference component it lies on
    (majority vote per connected piece of `mesh`)."""
    own = _components_of(len(mesh.V), mesh.edges)
    out = np.zeros(len(mesh.V), dtype=np.int64)
    for c in np.unique(own):
        idx = np.nonzero(own == c)[0]
        pick = idx[np.linspace(0, len(idx) - 1, min(n_samples, len(idx))).astype(int)]
        out[idx] = np.bincount(surf.component_of(mesh.V[pick])).argmax()
    return out


def _scaled(gen, p0, p1):
    """Re-map a sub-generator's progress into [p0, p1] and return its value."""
    while True:
        try:
            stage, p = next(gen)
            yield (stage, p0 + (p1 - p0) * p)
        except StopIteration as stop:
            return stop.value


def _remesh_parts(V, F, params, cell_per_thickness=1.3, thin_frac=0.6, floor=0.15):
    """Whole-part refinement: pieces mostly thinner than a lattice cell are
    remeshed on their own at a smaller *uniform* size (no size transitions
    inside a piece, so no holes or extra poles); everything else together at
    the requested size."""
    if not params.adapt_thin:
        return (yield from _remesh(V, F, params))
    yield ("Measuring thickness", 0.0)
    V, F, marked = clean(V, F, params.marked_pairs)
    if not len(F):
        raise ValueError("Mesh has no faces")
    mesh = TriMesh(V, F)
    comp = _components_of(len(V), mesh.edges)
    surf = SurfaceProjector(V, F, comp[F[:, 0]])
    h = float(params.target_edge)
    cell = 2.0 * h if params.pure_quads else h
    # Sample points spread over the faces (by area) with their face normals:
    # vertex samples sit on rims and corners of coarse CAD meshes, where the
    # averaged normal points diagonally out of the wall.
    rng = np.random.default_rng(0)
    ns = min(30000, max(2000, 4 * len(F)))
    fi = rng.choice(len(F), size=ns, p=mesh.face_area / mesh.face_area.sum())
    bary = rng.dirichlet((1.0, 1.0, 1.0), size=ns)
    P = np.einsum("ij,ijk->ik", bary, V[F[fi]])
    thick = surf.ray_thickness(P, mesh.face_normal[fi], comp[F[fi, 0]])
    need = cell_per_thickness * thick
    scomp = comp[F[fi, 0]]
    groups = {}                       # target edge -> list of components
    for c in np.unique(comp):
        m = scomp == c
        thin = need[m] < cell
        if m.any() and thin.mean() >= thin_frac:
            size = max(float(np.median(need[m][thin])), floor * cell)
            groups.setdefault(h * size / cell, []).append(c)
        else:
            groups.setdefault(h, []).append(c)
    if len(groups) == 1 and h in groups:
        return (yield from _remesh(V, F, replace(params, marked_pairs=marked)))

    results = []
    keys = sorted(groups)
    for gi, he in enumerate(keys):
        keep_v = np.isin(comp, groups[he])
        keep_f = keep_v[F[:, 0]]
        remap = np.cumsum(keep_v) - 1
        Vg, Fg = V[keep_v], remap[F[keep_f]]
        mg = None
        if marked is not None and len(marked):
            mm = marked[keep_v[marked[:, 0]] & keep_v[marked[:, 1]]]
            mg = remap[mm] if len(mm) else None
        # Thin pieces are already meshed far finer than any detail floor;
        # graded detail on top of that only destabilises them.
        sub = replace(params, target_edge=he, marked_pairs=mg,
                      detail=params.detail if he == h else 0.0)
        res = yield from _scaled(_remesh(Vg, Fg, sub), gi / len(keys), (gi + 1) / len(keys))
        results.append(res)
    if len(results) == 1:
        return results[0]
    # Combine the pieces.
    Vs, faces, fe, off = [], [], set(), 0
    for r in results:
        Vs.append(r.V)
        faces += [[v + off for v in f] for f in r.faces]
        fe |= {(a + off, b + off) for a, b in r.feature_edges}
        off += len(r.V)
    Vall = np.vstack(Vs)
    m = metrics.compute(Vall, faces, surf)
    m["holes"] = sum(r.metrics.get("holes", 0) for r in results)
    m["method"] = "+".join(sorted({r.metrics.get("method", "") for r in results}))
    return Result(Vall, faces, m, fe)


def _remesh(V, F, params):
    h = float(params.target_edge)
    scale = 2.0 * h if params.pure_quads else h
    cell_per_quad = scale / h

    yield ("Preparing mesh", 0.0)
    V, F, marked = clean(V, F, params.marked_pairs)
    if not len(F):
        raise ValueError("Mesh has no faces")

    # Reference surface: the original, used for projection, feature curves,
    # thickness and detail. The solvers run on a separate field mesh.
    use_ref = params.prepare_fn is not None
    # (Thin pieces were already given their own uniform size by _remesh_parts.)
    sizing = params.detail > 0
    detail_min = scale * (1.0 - 0.75 * min(max(params.detail, 0.0), 1.0))
    size_pts, size_vals = [], []

    def measure_sizes(dmesh, dmask):
        """Size requests (per point) from surface detail."""
        p, v = detail_size_points(dmesh, dmask, detail_min, scale, cell_per_quad)
        size_pts.append(p)
        size_vals.append(v)

    if use_ref:
        ref = TriMesh(V, F)
        ref_comp = _components_of(len(ref.V), ref.edges)
        surf = SurfaceProjector(ref.V, ref.F, ref_comp[ref.F[:, 0]])
        yield ("Detecting sharp edges", 0.02)
        ref_feats = Features(ref, params.sharp_angle if params.detect_sharp else None, marked, scale=scale)
        seg = (SegmentProjector(ref_feats.segments, scale / 8.0, ref_comp[ref_feats.feature_edges[:, 0]])
               if ref_feats.any else None)
        if sizing:
            yield ("Measuring detail", 0.03)
            # copy at the finest resolution detail may need (curvature does not
            # need it evened out, which is the expensive part)
            dV, dF = params.prepare_fn(ref.V, ref.F, 0.5 * detail_min / cell_per_quad, even=False)
            dmesh = TriMesh(*clean(dV, dF)[:2])
            measure_sizes(dmesh, transfer_features(dmesh, seg))
        finest = min([scale] + [float(v.min()) for v in size_vals if len(v)])
        yield ("Simplifying input", 0.05)
        Vf, Ff = params.prepare_fn(ref.V, ref.F, finest / 3.0)
        Vf, Ff, _ = clean(Vf, Ff)
        Vf, Ff, _ = split_long_edges(Vf, Ff, scale / 3.0)
        mesh = TriMesh(Vf, Ff)
        vcomp = _map_components(mesh, surf)
    else:
        V, F, marked = split_long_edges(V, F, scale / 3.0, marked)
        mesh = TriMesh(V, F)
        vcomp = _components_of(len(mesh.V), mesh.edges)
        surf = SurfaceProjector(mesh.V, mesh.F, vcomp[mesh.F[:, 0]])
        ref, ref_feats = mesh, None

    def field_features(mesh, vcomp, marked):
        if use_ref:
            mask = transfer_features(mesh, seg, vcomp)
            return Features(mesh, mask=mask, scale=scale)
        return Features(mesh, params.sharp_angle if params.detect_sharp else None, marked, scale=scale)

    yield ("Detecting sharp edges", 0.06)
    feats = field_features(mesh, vcomp, marked)
    if not use_ref:
        seg = (SegmentProjector(feats.segments, scale / 8.0, vcomp[feats.feature_edges[:, 0]])
               if feats.any else None)
        ref_feats = feats
        if sizing:
            yield ("Measuring detail", 0.07)
            measure_sizes(mesh, feats.edge_mask)

    # Size field (lattice cell per vertex): thin parts and fine detail ask for
    # smaller cells; the gradient limit keeps every change gradual.
    hv = np.full(len(mesh.V), scale)
    if sizing:
        for p, v in zip(size_pts, size_vals):
            hv = apply_size_points(mesh.V, hv, p, v)
        hv = limit_gradient(mesh.V, mesh.edges, hv)
        if hv.min() < 0.9 * scale:
            # Refine the field mesh where cells got smaller, and put the new
            # vertices on the real surface so fine detail is seen by the fields.
            n0 = len(mesh.V)
            out = split_long_edges(mesh.V, mesh.F, None, None if use_ref else marked, vlimit=hv / 3.0)
            Vr, Fr, marked_r, lim = out
            hv = 3.0 * lim
            if not use_ref:
                marked = marked_r
            mesh0 = TriMesh(Vr, Fr)
            comps_new = _map_components(mesh0, surf) if use_ref else _components_of(len(Vr), mesh0.edges)
            if len(Vr) > n0:
                Vr[n0:], _ = surf.project(Vr[n0:], comps_new[n0:])
            mesh = TriMesh(Vr, Fr)
            vcomp = comps_new
            feats = field_features(mesh, vcomp, marked)
            if not use_ref:
                seg = (SegmentProjector(feats.segments, scale / 8.0, vcomp[feats.feature_edges[:, 0]])
                       if feats.any else None)
            hv = limit_gradient(mesh.V, mesh.edges, hv)

    # Feature curves packed closer than the lattice can resolve are smoothed over.
    kept = drop_unresolvable_features(mesh, feats.edge_mask, hv)
    if (kept != feats.edge_mask).any():
        feats = Features(mesh, mask=kept, scale=scale)
    if use_ref or (kept != feats.edge_mask).any():
        seg = (SegmentProjector(feats.segments, scale / 8.0, vcomp[feats.feature_edges[:, 0]])
               if feats.any else None)

    # Guide strokes: chains of field-mesh edges moved onto the strokes, which
    # become line constraints so an edge loop runs along each stroke.
    gproj, gsegs, polys, pcomps = None, np.zeros((0, 2, 3)), [], []
    gverts = np.zeros(len(mesh.V), bool)
    gedges = np.zeros(len(mesh.edges), bool)
    gtan = np.zeros((len(mesh.V), 3))
    if params.guides:
        yield ("Placing guide loops", 0.08)
        med = float(np.median(mesh.edge_length))
        polys, pcomps = guide_mod.on_surface(params.guides, surf, 0.5 * med, 0.5 * scale)
        if polys:
            gproj, gsegs = guide_mod.projector(polys, pcomps, scale / 8.0)
        if gproj is not None:
            skip = feats.is_feature | feats.is_boundary
            gverts, gedges, _ = guide_mod.edge_chains(mesh, polys, pcomps, gproj, vcomp, skip)
            mesh = TriMesh(guide_mod.snap_chain(mesh, gverts, gproj, vcomp, med), mesh.F)
            gverts, gedges, gtan = guide_mod.edge_chains(mesh, polys, pcomps, gproj, vcomp, skip)

    yield ("Building hierarchy", 0.08)
    N = smooth_normals(mesh, feats.edge_mask, 0.25 * float(np.median(hv)))
    levels = build_hierarchy(mesh.V, N, mesh.vertex_area, mesh.ei, mesh.ej)

    line_mask = (feats.is_feature & ~feats.is_corner) | gverts
    tangent = feats.tangent.copy()
    tangent[gverts] = gtan[gverts]
    hard_mask = line_mask.copy()
    hard_dir = tangent
    band = (1.0 + 3.0 * min(max(params.guide_strength, 0.0), 1.0)) * scale
    soft_mask, soft_dir, soft_w = guide_mod.soft_band(mesh, gproj, vcomp, band, params.guide_strength)
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
    if params.size_flexibility > 0:
        hv = hv * (scale_field(mesh, N, Q, scale, params.size_flexibility) / scale)
    index = PointIndex(mesh.V)

    def size_at(pts, factor=1.0):
        return np.array([hv[index.nearest(p)] for p in pts]) * factor

    gen = solve_position(levels, Q, hv, line_mask, tangent, feats.is_corner)
    while True:
        try:
            p = next(gen)
            yield ("Solving position field", 0.35 + 0.3 * p)
        except StopIteration as stop:
            O, Q = stop.value
            break

    bnd_e = ref.edges[ref.edge_face_count == 1]
    bnd_proj = (SegmentProjector(np.stack([ref.V[bnd_e[:, 0]], ref.V[bnd_e[:, 1]]], 1), scale / 8.0)
                if len(bnd_e) else None)
    # Sharp features and guide loops alike are snapped to and protected during
    # clean-up; only sharp ones are marked on the result.
    seg_all = seg
    if gproj is not None:
        sharp_segs = feats.segments if seg is not None else np.zeros((0, 2, 3))
        sharp_comp = seg.seg_comp if seg is not None else np.zeros(0, np.int64)
        seg_all = SegmentProjector(np.concatenate([sharp_segs, gsegs], 0), scale / 8.0,
                                   np.concatenate([sharp_comp, gproj.seg_comp]))
    ctx = dict(mesh=mesh, N=N, Q=Q, O=O, hv=hv, feats=feats, vcomp=vcomp, surf=surf, seg=seg,
               seg_all=seg_all, gproj=gproj, gpolys=(polys, pcomps), is_feature=feats.is_feature | gverts, edge_mask=feats.edge_mask | gedges,
               bnd_proj=bnd_proj,
               size_at=size_at, h=h, scale=scale, params=params)
    expected = float((mesh.vertex_area / (hv / cell_per_quad) ** 2).sum())
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
    return (m.get("guide_coverage", 1.0) < 0.97 or m["v6plus"] > 0 or m.get("flipped", 0) > 0 or m.get("holes", 0) > 0 or m["edge_cv"] > 0.3
            or abs(m["faces"] / max(expected, 1.0) - 1.0) > 0.15)


def _score(m, expected):
    """Lower is better: interior holes above all, then poles and defects that look broken."""
    s = 1e6 * m.get("interior_holes", 0) + 10.0 * m.get("holes", 0)
    # every quad length of a guide stroke without an edge loop on it
    s += 20.0 * (1.0 - m.get("guide_coverage", 1.0)) * m.get("guide_cells", 0.0)
    s += m["v3"] + m["v5"] + 4.0 * m["v6plus"]
    s += 5.0 * m.get("flipped", 0)
    s += 60.0 * max(0.0, m["edge_cv"] - 0.25)
    s += 80.0 * max(0.0, abs(m["faces"] / max(expected, 1.0) - 1.0) - 0.15)
    return s


def _finish(ctx, quantized, p0, p1):
    """Extraction and all topology / geometry clean-up after the fields."""
    mesh, N, Q, O, hv = ctx["mesh"], ctx["N"], ctx["Q"], ctx["O"], ctx["hv"]
    feats, vcomp, surf, seg = ctx["feats"], ctx["vcomp"], ctx["surf"], ctx["seg_all"]
    size_at, h, scale, params = ctx["size_at"], ctx["h"], ctx["scale"], ctx["params"]
    is_feature, edge_mask = ctx["is_feature"], ctx["edge_mask"]

    def prog(f):
        return p0 + (p1 - p0) * f

    def final_size(pts):
        return size_at(pts, h / scale)

    yield ("Extracting quads", prog(0.0))
    if quantized:
        VP, VN, faces, c_feat, c_corner, c_bnd, c_comp, VQ, hc, _q = quantize_fine(
            mesh, N, Q, O, hv, edge_mask, is_feature, feats.is_corner, feats.is_boundary, vcomp)
    else:
        VP, VN, faces, c_feat, c_corner, c_bnd, c_comp, VQ, hc = extract_mesh(
            mesh.V, N, Q, O, mesh.edges, hv,
            is_feature, feats.is_corner, feats.is_boundary, vcomp)
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
    quadify.fill_holes(pm)
    quadify.close_notches(pm)
    quadify.seal_holes(pm, ctx["bnd_proj"], final_size)
    quadify.mark_feature_edges(pm, seg)
    quadify.dissolve_valence2(pm)
    quadify.pair_odd_faces(pm, max_depth=6)
    quadify.split_even_ngons(pm)
    if params.cleanup:
        cleanup.edge_rotation(pm)
        cleanup.diagonal_collapse(pm)
        quadify.dissolve_valence2(pm)
        quadify.pair_odd_faces(pm)
    pm.compact()
    pm.V = relax(pm.V, pm.faces, scale, surf, seg, pm.feat, pm.corner, iterations=4, comp=pm.comp,
                 size_fn=size_at)

    if params.pure_quads:
        yield ("Subdividing to pure quads", prog(0.45))
        quadify.split_odd_ngons(pm, min_size=7)   # avoid valence 7+ face centres
        pm = quadify.subdivide(pm)
    else:
        quadify.split_odd_ngons(pm)
    if ctx["gproj"] is not None:
        # close whatever the lattice left open in each guide loop
        guide_mod.trace_loops(pm, *ctx["gpolys"], ctx["gproj"], final_size)
    yield ("Relaxing", prog(0.55))

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
    # Stars left over: rotate one of their edges away (judged on relaxed
    # geometry, where the new diagonal's length is meaningful), then settle.
    if cleanup.break_stars(pm):
        pm.V = relax(pm.V, pm.faces, h, surf, seg, pm.feat, pm.corner, iterations=3,
                     comp=pm.comp, size_fn=final_size)
    if params.untangle:
        yield ("Untangling", prog(0.9))
        pm.V = untangle(pm.V, pm.faces, surf, seg, pm.feat, pm.corner, pm.comp)

    m = metrics.compute(pm.V, pm.faces, surf, comp=pm.comp)
    m["holes"] = metrics.unexpected_boundary(pm.V, pm.faces, ctx["bnd_proj"], final_size)
    # open edges away from every opening of the input: real holes, not slivers along a rim
    m["interior_holes"] = metrics.unexpected_boundary(pm.V, pm.faces, ctx["bnd_proj"], final_size, far=2.0)
    if ctx["gproj"] is not None:
        m["guide_coverage"] = metrics.guide_coverage(pm.V, pm.faces, ctx["gproj"], final_size)
        m["guide_cells"] = float(np.linalg.norm(ctx["gproj"].B - ctx["gproj"].A, axis=1).sum()) / h
    m["method"] = "quantized" if quantized else "classic"
    fe = set(pm.feature_edges)
    if seg is not ctx["seg"]:
        # guide loops were protected as features; only sharp edges are marked
        quadify.mark_feature_edges(pm, ctx["seg"])
        fe = set(pm.feature_edges)
    return Result(pm.V, pm.faces, m, fe)


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
