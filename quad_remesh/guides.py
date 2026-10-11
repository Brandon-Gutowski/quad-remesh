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


def _add_spline(g, pts_world):
    Minv = g.matrix_world.inverted()
    spl = g.data.splines.new('POLY')
    spl.points.add(len(pts_world) - 1)
    co = []
    for p in pts_world:
        q = Minv @ p
        co.extend((q.x, q.y, q.z, 1.0))
    spl.points.foreach_set("co", co)


class QR_OT_draw_guides(bpy.types.Operator):
    """Draw strokes on the surface to steer the quad edge flow.
LMB drag: draw  |  Shift+LMB drag: erase  |  Enter / RMB / Esc: finish"""
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
        self._lift = diag * 5e-4
        self._obj = obj
        self._stroke = []
        self._mode = None
        self._handle = bpy.types.SpaceView3D.draw_handler_add(self._draw, (context,), 'WINDOW', 'POST_VIEW')
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set("Draw Guides:  LMB drag draw   Shift+LMB drag erase   Enter/RMB/Esc finish")
        return {'RUNNING_MODAL'}

    # -- helpers --------------------------------------------------------------
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
        mouse = Vector((event.mouse_region_x, event.mouse_region_y))
        M = g.matrix_world
        doomed = []
        for spl in g.data.splines:
            for p in spl.points:
                s = view3d_utils.location_3d_to_region_2d(region, rv3d, M @ p.co.to_3d())
                if s is not None and (s - mouse).length < 18:
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
                    g = guide_object_for(context, self._obj, create=True)
                    _add_spline(g, self._stroke)
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
        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": [p[:] for p in self._stroke]})
        shader.bind()
        shader.uniform_float("viewportSize", (context.region.width, context.region.height))
        shader.uniform_float("lineWidth", 3.0)
        shader.uniform_float("color", (1.0, 0.55, 0.05, 1.0))
        gpu.state.blend_set('ALPHA')
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
