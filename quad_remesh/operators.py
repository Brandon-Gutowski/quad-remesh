import time

import bpy

from . import bridge
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
        pure_quads=st.pure_quads,
        relax_iterations=st.relax_iterations,
        cleanup=st.cleanup,
        quantize=st.cleanup,
    )
    return V, F, params


class QR_OT_remesh(bpy.types.Operator):
    """Remesh the active object into evenly sized quads that follow sharp edges and guides"""
    bl_idname = "quad_remesh.remesh"
    bl_label = "Quad Remesh"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _gen = None

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
        wm.progress_begin(0, 100)
        self._timer = wm.event_timer_add(0.02, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _cleanup_modal(self, context):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        wm.progress_end()
        context.workspace.status_text_set(None)

    def modal(self, context, event):
        if event.type == 'ESC':
            self._gen.close()
            self._cleanup_modal(context)
            self.report({'WARNING'}, "Quad Remesh cancelled")
            return {'CANCELLED'}
        if event.type != 'TIMER':
            if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM'}:
                return {'PASS_THROUGH'}
            return {'RUNNING_MODAL'}
        deadline = time.time() + 0.15
        try:
            while time.time() < deadline:
                stage, frac = next(self._gen)
                context.window_manager.progress_update(int(frac * 100))
                context.workspace.status_text_set(f"Quad Remesh: {stage} ({int(frac * 100)}%)   Esc to cancel")
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
