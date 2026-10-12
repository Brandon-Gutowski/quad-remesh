# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

import time

import blf
import bpy

from . import bridge, prep
from .core import metrics, pipeline


def _params(context, obj):
    st = obj.quad_remesh
    h, _area = bridge.target_edge_length(context, obj)
    V, F, marked = bridge.remesh_inputs(context, obj)
    guides = bridge.guide_polylines(st.guide_object) if st.use_guides else []
    params = pipeline.Params(
        target_edge=h,
        detect_sharp=st.detect_sharp,
        sharp_angle=st.sharp_angle,
        marked_pairs=marked if st.follow_marked_sharp else None,
        guides=guides,
        guide_strength=st.guide_strength,
        guide_rings_out=st.guide_rings_out,
        guide_rings_in=st.guide_rings_in,
        pure_quads=st.pure_quads,
        relax_iterations=st.relax_iterations,
        cleanup=st.cleanup,
        quantize=st.cleanup,
        adapt_thin=st.adapt_thin,
        detail=st.detail,
        prepare_fn=prep.prepare,
        symmetry_plane=bridge.symmetry_plane(obj, st.symmetry_axis) if st.use_symmetry else None,
    )
    return V, F, params


# Shared with the panel, which draws a progress bar in place of the button.
PROGRESS = {"active": False, "source": "", "frac": 0.0, "stage": "", "mouse": None}

CURSOR_TEXT = "Doing hard stuff"


def accent_color():
    """The theme's accent (selection) colour, which the progress bar also uses."""
    ui = bpy.context.preferences.themes[0].user_interface
    return tuple(ui.wcol_progress.item)


def _redraw_views(context):
    screen = getattr(context, "screen", None)
    for area in (screen.areas if screen else ()):
        if area.type == 'VIEW_3D':
            area.tag_redraw()


def _draw_cursor_label():
    """Text next to the mouse while a remesh runs (instead of a % on the cursor)."""
    mouse = PROGRESS["mouse"]
    region = bpy.context.region
    if not PROGRESS["active"] or mouse is None or region is None:
        return
    x, y = mouse[0] - region.x, mouse[1] - region.y
    if not (0 <= x < region.width and 0 <= y < region.height):
        return
    scale = bpy.context.preferences.system.ui_scale
    font = 0
    blf.size(font, 13 * scale)
    full_w, text_h = blf.dimensions(font, CURSOR_TEXT + "...")
    tx, ty = x + 18 * scale, y - 26 * scale
    pad = 6 * scale
    _draw_round_rect(tx - pad, ty - pad, full_w + 2 * pad, text_h + 2 * pad, 4 * scale,
                     (0.06, 0.06, 0.06, 0.82))
    dots = "." * (1 + int(time.time() * 2.5) % 3)
    blf.color(font, *accent_color())
    blf.position(font, tx, ty, 0)
    blf.draw(font, CURSOR_TEXT + dots)


def _draw_round_rect(x, y, w, h, r, color, seg=4):
    import math
    import gpu
    from gpu_extras.batch import batch_for_shader
    pts = []
    for cx, cy, a0 in ((x + w - r, y + h - r, 0.0), (x + r, y + h - r, 90.0),
                       (x + r, y + r, 180.0), (x + w - r, y + r, 270.0)):
        for k in range(seg + 1):
            a = math.radians(a0 + 90.0 * k / seg)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    center = (x + w / 2, y + h / 2)
    verts = [center] + pts
    idx = [(0, i, i % len(pts) + 1) for i in range(1, len(pts) + 1)]
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'TRIS', {"pos": verts}, indices=idx)
    gpu.state.blend_set('ALPHA')
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.blend_set('NONE')


class QR_OT_remesh(bpy.types.Operator):
    """Remesh the active object into evenly sized quads that follow sharp edges and guides"""
    bl_idname = "quad_remesh.remesh"
    bl_label = "Quad Remesh"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _gen = None
    _label = None

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.type == 'MESH' and context.mode == 'OBJECT'

    def _start(self, context):
        self._src = context.active_object
        V, F, params = _params(context, self._src)
        if params.target_edge <= 0:
            raise ValueError("Mesh has no surface area")
        est = len(F)
        self._t0 = time.time()
        self._gen = pipeline.remesh(V, F, params)
        return est

    def _finish(self, context, result):
        src = self._src
        ob = bridge.write_result(context, src, result.V, result.faces, result.feature_edges)
        st = src.quad_remesh
        st.show_checker = False
        if st.hide_original:
            src.hide_set(True)
        for o in context.selected_objects:
            o.select_set(False)
        ob.hide_set(False)
        ob.select_set(True)
        context.view_layer.objects.active = ob
        msg = f"{metrics.summary(result.metrics)} ({time.time() - self._t0:.1f}s)"
        st.last_report = msg
        self.report({'INFO'}, msg)

    def execute(self, context):
        try:
            self._start(context)
            while True:
                try:
                    next(self._gen)
                except StopIteration as stop:
                    result = stop.value
                    break
            self._finish(context, result)
        except Exception as exc:  # report instead of a traceback popup
            self.report({'ERROR'}, f"Quad Remesh failed: {exc}")
            return {'CANCELLED'}
        return {'FINISHED'}

    def invoke(self, context, event):
        try:
            self._start(context)
        except Exception as exc:
            self.report({'ERROR'}, f"Quad Remesh failed: {exc}")
            return {'CANCELLED'}
        wm = context.window_manager
        PROGRESS.update(active=True, source=self._src.name_full, frac=0.0, stage="Starting",
                        mouse=(event.mouse_x, event.mouse_y))
        self._label = bpy.types.SpaceView3D.draw_handler_add(_draw_cursor_label, (), 'WINDOW', 'POST_PIXEL')
        self._timer = wm.event_timer_add(0.02, window=context.window)
        wm.modal_handler_add(self)
        _redraw_views(context)
        return {'RUNNING_MODAL'}

    def _cleanup_modal(self, context):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        if self._label is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._label, 'WINDOW')
            self._label = None
        PROGRESS.update(active=False, frac=0.0, stage="", mouse=None)
        context.workspace.status_text_set(None)
        _redraw_views(context)

    def modal(self, context, event):
        if event.type == 'ESC':
            self._gen.close()
            self._cleanup_modal(context)
            self.report({'WARNING'}, "Quad Remesh cancelled")
            return {'CANCELLED'}
        if event.type != 'TIMER':
            if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
                PROGRESS["mouse"] = (event.mouse_x, event.mouse_y)
                _redraw_views(context)
            if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM'}:
                return {'PASS_THROUGH'}
            return {'RUNNING_MODAL'}
        deadline = time.time() + 0.15
        try:
            while time.time() < deadline:
                stage, frac = next(self._gen)
                PROGRESS["frac"], PROGRESS["stage"] = frac, stage
                context.workspace.status_text_set(f"Quad Remesh: {stage}   Esc to cancel")
            _redraw_views(context)
        except StopIteration as stop:
            self._cleanup_modal(context)
            try:
                self._finish(context, stop.value)
            except Exception as exc:
                self.report({'ERROR'}, f"Quad Remesh failed: {exc}")
                return {'CANCELLED'}
            return {'FINISHED'}
        except Exception as exc:
            self._cleanup_modal(context)
            self.report({'ERROR'}, f"Quad Remesh failed: {exc}")
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}


class QR_OT_toggle_original(bpy.types.Operator):
    """Toggle between the remeshed result and its source object"""
    bl_idname = "quad_remesh.toggle_original"
    bl_label = "Toggle Original"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and "qr_source" in ob

    def execute(self, context):
        res = context.active_object
        src = bpy.data.objects.get(res["qr_source"])
        if src is None:
            self.report({'WARNING'}, "Source object not found")
            return {'CANCELLED'}
        show_src = src.hide_get()
        src.hide_set(not show_src)
        res.hide_set(show_src)
        return {'FINISHED'}


classes = (QR_OT_remesh, QR_OT_toggle_original)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
