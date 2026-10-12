# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

"""Draw edge-flow guide strokes directly on the mesh surface."""

import bpy
import gpu
import numpy as np
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from . import bridge
from .core.guides import gauss_line as _gauss

GUIDE_PREFIX = "QR_Guides_"


def guide_object_for(context, obj, create=False):
    st = obj.quad_remesh
    g = st.guide_object
    if g is not None and g.name in bpy.data.objects:
        return g
    if not create:
        return None
    cu = bpy.data.curves.new(GUIDE_PREFIX + obj.name, 'CURVE')
    cu.dimensions = '3D'
    g = bpy.data.objects.new(GUIDE_PREFIX + obj.name, cu)
    for col in obj.users_collection or (context.scene.collection,):
        col.objects.link(g)
    g.parent = obj
    g.matrix_parent_inverse = Matrix.Identity(4)
    g.matrix_world = obj.matrix_world.copy()
    g.show_in_front = True
    g.hide_render = True
    g.color = (1.0, 0.55, 0.05, 1.0)
    st.guide_object = g
    return g


def _add_spline(g, pts_world, cyclic=False):
    Minv = g.matrix_world.inverted()
    spl = g.data.splines.new('POLY')
    spl.use_cyclic_u = cyclic
    spl.points.add(len(pts_world) - 1)
    co = []
    for p in pts_world:
        q = Minv @ p
        co.extend((q.x, q.y, q.z, 1.0))
    spl.points.foreach_set("co", co)


def mirror_point(p, p0, n):
    """Reflection of p over the plane through p0 with unit normal n."""
    return p - 2.0 * (p - p0).dot(n) * n


def smooth_stroke(pts, closed, amount, spacing, project=None):
    """Take the hand wobble out of a stroke: resample it evenly, low-pass it
    along its length and put it back on the surface with `project`.
    `amount` 0..1 sets how long a wiggle still counts as wobble (up to ~1.5
    quads at 1; points are ~1/4 quad apart). The filter is shrink-corrected,
    so rings keep their size; open strokes keep their end points."""
    P = np.array([p[:] for p in pts], dtype=np.float64)
    if len(P) < 3 or amount <= 0:
        return [Vector(p) for p in P]
    if closed:
        P = np.vstack([P, P[:1]])
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(4, int(np.ceil(s[-1] / spacing)) + 1)
    t = np.linspace(0, s[-1], n)
    P = np.stack([np.interp(t, s, P[:, k]) for k in range(3)], 1)
    if closed:
        P = P[:-1]
    sigma = min(1.0 + 5.0 * min(amount, 1.0), 0.15 * len(P))
    if sigma >= 0.5 and (closed or len(P) > 2 * int(np.ceil(3.0 * sigma)) + 2):
        G = _gauss(P, sigma, closed)
        P = G + (G - _gauss(G, sigma, closed))     # undo the low-frequency shrink
        if not closed:
            P[0], P[-1] = pts[0][:], pts[-1][:]
    if project is not None:
        P = np.array([project(Vector(p))[:] for p in P])
    return [Vector(p) for p in P]


def join_at_plane(pts, closed, p0, n, snap):
    """Join a stroke that reaches the mirror plane with its mirror image, the
    way a stroke ending at its start closes into a ring.

    The part on the stroke's main side (the side it reaches farthest into) is
    kept; where it crosses the plane, or ends within `snap` of it, it is cut /
    extended exactly onto the plane and continued by its reflection. Both ends
    on the plane give one closed symmetric ring (half a mouth drawn), one end an
    open stroke through the plane. Returns (points, closed), or None when the
    stroke does not reach the plane or runs along it (a centre line)."""
    P = np.array([p[:] for p in pts], dtype=np.float64)
    p0 = np.array(p0[:], dtype=np.float64)
    n = np.array(n[:], dtype=np.float64)
    if len(P) < 2:
        return None
    sd = (P - p0) @ n
    if np.abs(sd).max() <= snap:
        return None
    i = int(np.argmax(np.abs(sd)))
    s0 = np.sign(sd[i])
    main = s0 * sd > 0
    N = len(P)
    if closed:
        if main.all():
            return None                      # a ring on one side
        k = int(np.nonzero(~main)[0][0])     # start the walk off the main side
        P, sd, main = np.roll(P, -k, 0), np.roll(sd, -k), np.roll(main, -k)
        i = (i - k) % N
    a = i
    while a > 0 and main[a - 1]:
        a -= 1
    b = i
    while b < N - 1 and main[b + 1]:
        b += 1

    def on_plane(j, k):                      # P[j] off the main side, P[k] on it
        t = sd[j] / (sd[j] - sd[k])
        return P[j] + t * (P[k] - P[j])

    run = list(P[a:b + 1])
    start_on = end_on = False
    if a > 0:
        run.insert(0, on_plane(a - 1, a))
        start_on = True
    elif s0 * sd[a] <= snap:
        run.insert(0, P[a] - sd[a] * n)
        start_on = True
    if b < N - 1 or closed:
        run.append(on_plane((b + 1) % N, b))
        end_on = True
    elif s0 * sd[b] <= snap:
        run.append(P[b] - sd[b] * n)
        end_on = True
    if not (start_on or end_on):
        return None
    H = np.array(run)
    M = H - 2.0 * np.outer((H - p0) @ n, n)
    if start_on and end_on:
        out, ring = np.vstack([H, M[::-1][1:-1]]), True
    elif end_on:
        out, ring = np.vstack([H, M[::-1][1:]]), False
    else:
        out, ring = np.vstack([M[::-1][:-1], H]), False
    return [Vector(p) for p in out], ring


def snap_to_strokes(new, others, snap):
    """Join an open stroke to open strokes whose end points it starts or ends
    near (within `snap`). `others` is [(key, points)]. Its ends land exactly
    on the other strokes' ends; both ends on the two ends of one stroke close
    it into a ring. Returns (points, closed, used keys, junction indices)."""
    new = [Vector(p) for p in new]
    if len(new) < 2 or not others:
        return new, False, [], []

    def nearest(q, skip=None):
        best = None
        for key, pts in others:
            for end, e in ((0, pts[0]), (1, pts[-1])):
                if skip == (key, end):
                    continue
                d = (e - q).length
                if d <= snap and (best is None or d < best[2]):
                    best = (key, end, d)
        return best

    a = nearest(new[0])
    b = nearest(new[-1])
    if a and b and a[:2] == b[:2]:                 # both ends at the same end point
        if a[2] <= b[2]:
            b = nearest(new[-1], skip=a[:2])
        else:
            a = nearest(new[0], skip=b[:2])
    lookup = dict(others)

    def oriented(hit, toward_end):
        """Other stroke's points running so that its snapped end comes last
        (toward_end) or first."""
        pts = list(lookup[hit[0]])
        at_end = hit[1] == 1
        return pts if at_end == toward_end else pts[::-1]

    out, junctions, used, closed = list(new), [], [], False
    if a and b and a[0] == b[0]:
        # both ends on the two ends of one stroke: a ring
        other = oriented(b, toward_end=False)     # starts at the end near new[-1]
        out[0], out[-1] = other[-1], other[0]
        junctions = [len(out) - 1, 0]
        out = out + other[1:-1]
        return out, True, [a[0]], junctions
    if a:
        head = oriented(a, toward_end=True)        # ends at the end near new[0]
        out = head[:-1] + [head[-1]] + out[1:]
        junctions.append(len(head) - 1)
        used.append(a[0])
    if b:
        tail = oriented(b, toward_end=False)       # starts at the end near new[-1]
        junctions.append(len(out) - 1)
        out = out[:-1] + tail
        used.append(b[0])
    return out, closed, used, junctions


def smooth_junctions(pts, closed, junctions, sigma):
    """Round off the kinks where strokes were joined, leaving the rest of the
    strokes as they were (smoothing fades out within ~3 sigma of a junction)."""
    P = np.array([p[:] for p in pts], dtype=np.float64)
    r = int(np.ceil(3.0 * sigma))
    if not junctions or len(P) < 2 * r + 3:
        return [Vector(p) for p in P]
    G = _gauss(P, sigma, closed)
    G = G + (G - _gauss(G, sigma, closed))
    idx = np.arange(len(P))
    w = np.zeros(len(P))
    for j in junctions:
        d = np.abs(idx - j)
        if closed:
            d = np.minimum(d, len(P) - d)
        w = np.maximum(w, np.exp(-0.5 * (d / sigma) ** 2))
    if not closed:
        w[0] = w[-1] = 0.0
    P = P + w[:, None] * (G - P)
    return [Vector(p) for p in P]


def crosses_plane(pts, p0, n):
    """True when a stroke reaches both sides of the plane (a ring around the
    mouth, a centre line): it already spans both halves and is not copied."""
    side = [(p - p0).dot(n) for p in pts]
    return min(side) < 0.0 < max(side)


class QR_OT_draw_guides(bpy.types.Operator):
    """Draw strokes on the surface; each one gets an edge loop along it. End a stroke where it
started to make a closed ring (eyes, mouth). With Symmetry on, strokes are mirrored across
the symmetry axis; a stroke reaching the mirror line joins its mirror image there (half a
mouth ring drawn from the centre line and back becomes the whole ring).
A stroke starting or ending at the end of another open stroke is joined to it.
Hold Ctrl while drawing to keep a stroke exactly as drawn (no snapping or joining).
LMB drag: draw  |  Ctrl: no snapping  |  Shift+LMB drag: erase  |  Enter / RMB / Esc: finish"""
    bl_idname = "quad_remesh.draw_guides"
    bl_label = "Draw Guides"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.type == 'MESH' and context.mode == 'OBJECT' and context.area and context.area.type == 'VIEW_3D'

    def invoke(self, context, event):
        obj = context.active_object
        ma = bridge.mesh_arrays(context, obj)
        M = np.array(obj.matrix_world, dtype=np.float64)
        Vw = ma.co.astype(np.float64) @ M[:3, :3].T + M[:3, 3]
        self._bvh = BVHTree.FromPolygons(Vw.tolist(), ma.tris.tolist(), all_triangles=True)
        h, _ = bridge.target_edge_length(context, obj)
        diag = float(np.linalg.norm(Vw.max(0) - Vw.min(0))) if len(Vw) else 1.0
        self._spacing = max(h * 0.25, diag * 1e-3)
        # a stroke ending this close to its start (and long enough) is closed into a ring
        self._close_dist = max(1.5 * h, 6.0 * self._spacing)
        self._lift = diag * 5e-4
        self._obj = obj
        self._stroke = []
        self._mode = None
        self._free = False          # Ctrl held: no snapping of any kind
        st = obj.quad_remesh
        self._smooth = st.guide_smoothing
        self._plane = None
        if st.use_symmetry:
            p0, nrm = bridge.symmetry_plane(obj, st.symmetry_axis)
            self._plane = (Vector(p0), Vector(nrm))
        self._handle = bpy.types.SpaceView3D.draw_handler_add(self._draw, (context,), 'WINDOW', 'POST_VIEW')
        context.window_manager.modal_handler_add(self)
        mirror = f"   Mirrored across {st.symmetry_axis}" if self._plane else ""
        context.workspace.status_text_set("Draw Guides:  LMB drag draw (end at the start for a ring)   "
                                          "Ctrl no snapping   Shift+LMB drag erase   Enter/RMB/Esc finish"
                                          + mirror)
        return {'RUNNING_MODAL'}

    # -- helpers --------------------------------------------------------------
    def _closes(self):
        st = self._stroke
        if self._free or len(st) < 4:
            return False
        length = sum((st[i + 1] - st[i]).length for i in range(len(st) - 1))
        return (st[-1] - st[0]).length <= self._close_dist and length > 3.0 * self._close_dist

    def _on_surface(self, p):
        loc, nrm, _idx, _d = self._bvh.find_nearest(p)
        return loc + nrm * self._lift if loc is not None else p

    def _final(self):
        """The current stroke as it will be stored: (points, closed, add a
        mirrored copy)."""
        pts, closed = self._stroke, self._closes()
        mirror = False
        if self._plane:
            p0, n = self._plane
            joined = None
            st = self._stroke
            if self._free:
                pass
            elif not closed:
                joined = join_at_plane(st, False, p0, n, self._close_dist)
            elif crosses_plane(st, p0, n):
                joined = join_at_plane(st, True, p0, n, self._close_dist)   # a ring drawn across
            elif max(abs((q - p0).dot(n)) for q in (st[0], st[-1])) <= self._close_dist:
                joined = join_at_plane(st, False, p0, n, self._close_dist)  # half a ring, ends at the line
            if joined is not None:
                pts, closed = joined
            else:
                # strokes along the centre line are kept as drawn
                far = max(abs((q - p0).dot(n)) for q in st) > self._close_dist
                mirror = far and not crosses_plane(st, p0, n)
        pts = smooth_stroke(pts, closed, self._smooth, self._spacing, self._on_surface)
        return pts, closed, mirror

    def _open_strokes(self, context):
        """[(spline, world points)] of the guide object's open strokes."""
        g = guide_object_for(context, self._obj)
        if g is None:
            return []
        M = g.matrix_world
        out = []
        for spl in g.data.splines:
            if spl.use_cyclic_u or spl.type != 'POLY' or len(spl.points) < 2:
                continue
            out.append((spl, [M @ p.co.to_3d() for p in spl.points]))
        return out

    def _plan(self, context):
        """What releasing the stroke now would store: [(points, closed, alpha)]
        to add, and the existing splines they replace."""
        pts, closed, mirror = self._final()
        strokes = [(pts, closed)]
        if mirror:
            strokes.append((self._mirrored(pts), closed))
        if self._free:
            return [(p, c, 1.0 if k == 0 else 0.55) for k, (p, c) in enumerate(strokes)], []
        # one read: Blender hands out new wrapper objects on every access
        existing = self._open_strokes(context)
        others = [(("spline", i), p) for i, (_spl, p) in enumerate(existing)]
        by_key = {("spline", i): spl for i, (spl, _p) in enumerate(existing)}
        out, replaced = [], []
        for k, (p, c) in enumerate(strokes):
            if not c:
                p, c, used, junctions = snap_to_strokes(p, others, self._close_dist)
                if used:
                    sigma = 1.0 + 3.0 * self._smooth
                    p = [self._on_surface(q) for q in smooth_junctions(p, c, junctions, sigma)]
                    for key in used:
                        if key in by_key:
                            replaced.append(by_key[key])
                        else:                    # joined to the stroke planned before it
                            out = [o for o in out if id(o) != key]
                    others = [(key, q) for key, q in others if key not in used]
            entry = (p, c, 1.0 if k == 0 else 0.55)
            out.append(entry)
            if not c:
                others.append((id(entry), p))
        return out, replaced

    def _mirrored(self, pts):
        """Pts reflected over the symmetry plane and put back on the surface
        (so the copy still lies on a mesh that is not exactly symmetric)."""
        p0, n = self._plane
        return [self._on_surface(mirror_point(p, p0, n)) for p in pts]

    def _hit(self, context, event):
        region, rv3d = context.region, context.region_data
        co = (event.mouse_region_x, event.mouse_region_y)
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, co)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, co)
        loc, nrm, _idx, _d = self._bvh.ray_cast(origin, direction)
        if loc is None:
            return None
        return loc + nrm * self._lift

    def _erase(self, context, event):
        g = guide_object_for(context, self._obj)
        if g is None:
            return
        region, rv3d = context.region, context.region_data
        cursors = [Vector((event.mouse_region_x, event.mouse_region_y))]
        hit = self._hit(context, event) if self._plane else None
        if hit is not None:
            # the mirrored partner of a stroke goes too
            m = view3d_utils.location_3d_to_region_2d(region, rv3d, self._mirrored([hit])[0])
            if m is not None:
                cursors.append(m)
        M = g.matrix_world
        doomed = []
        for spl in g.data.splines:
            for p in spl.points:
                s = view3d_utils.location_3d_to_region_2d(region, rv3d, M @ p.co.to_3d())
                if s is not None and any((s - c).length < 18 for c in cursors):
                    doomed.append(spl)
                    break
        for spl in doomed:
            g.data.splines.remove(spl)

    def _finish(self, context):
        bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
        context.workspace.status_text_set(None)
        context.area.tag_redraw()

    # -- modal ----------------------------------------------------------------
    def modal(self, context, event):
        context.area.tag_redraw()
        self._free = event.ctrl
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM'} \
                or event.type.startswith('NDOF'):
            return {'PASS_THROUGH'}
        if event.type in {'RIGHTMOUSE', 'ESC', 'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            self._finish(context)
            return {'FINISHED'}
        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self._mode = 'ERASE' if event.shift else 'DRAW'
                self._stroke = []
                if self._mode == 'ERASE':
                    self._erase(context, event)
                else:
                    p = self._hit(context, event)
                    if p is not None:
                        self._stroke.append(p)
            elif event.value == 'RELEASE':
                if self._mode == 'DRAW' and len(self._stroke) >= 2:
                    strokes, replaced = self._plan(context)
                    g = guide_object_for(context, self._obj, create=True)
                    for spl in replaced:
                        g.data.splines.remove(spl)
                    for pts, closed, _alpha in strokes:
                        _add_spline(g, pts, cyclic=closed)
                self._stroke = []
                self._mode = None
            return {'RUNNING_MODAL'}
        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'} and self._mode:
            if self._mode == 'ERASE':
                self._erase(context, event)
            else:
                p = self._hit(context, event)
                if p is not None and (not self._stroke or (p - self._stroke[-1]).length >= self._spacing):
                    self._stroke.append(p)
            return {'RUNNING_MODAL'}
        return {'RUNNING_MODAL'}

    def _draw(self, context):
        if len(self._stroke) < 2:
            return
        shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        strokes, _replaced = self._plan(context)    # mirrored copy fainter
        gpu.state.blend_set('ALPHA')
        shader.bind()
        shader.uniform_float("viewportSize", (context.region.width, context.region.height))
        shader.uniform_float("lineWidth", 3.0)
        for stroke, closes, alpha in strokes:
            pts = [p[:] for p in stroke]
            if closes:
                pts.append(pts[0])      # show that releasing here makes a ring / join
            batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": pts})
            shader.uniform_float("color", (1.0, 0.55, 0.05, alpha))
            batch.draw(shader)
        gpu.state.blend_set('NONE')


class QR_OT_clear_guides(bpy.types.Operator):
    """Delete all guide strokes of the active object"""
    bl_idname = "quad_remesh.clear_guides"
    bl_label = "Clear Guides"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.type == 'MESH' and ob.quad_remesh.guide_object is not None

    def execute(self, context):
        g = context.active_object.quad_remesh.guide_object
        g.data.splines.clear()
        return {'FINISHED'}


classes = (QR_OT_draw_guides, QR_OT_clear_guides)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
