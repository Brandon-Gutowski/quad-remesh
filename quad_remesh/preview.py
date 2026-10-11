# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

"""Viewport checker overlay: squares the size of the target output quads."""

import bpy
import gpu
from gpu_extras.batch import batch_for_shader

from . import bridge

_handle = None
_shader = None

_VERT = """
void main()
{
    vec4 wp = modelMatrix * vec4(pos, 1.0);
    wpos = wp.xyz;
    wnor = normalize(mat3(normalMatrix) * nor);
    gl_Position = viewProjectionMatrix * wp;
    gl_Position.z -= 0.00015 * gl_Position.w;
}
"""

_FRAG = """
void main()
{
    vec3 n = normalize(wnor);
    vec3 an = abs(n);
    vec2 uv;
    if (an.x >= an.y && an.x >= an.z) {
        uv = wpos.yz;
    }
    else if (an.y >= an.z) {
        uv = wpos.xz;
    }
    else {
        uv = wpos.xy;
    }
    uv /= cell;
    float c = mod(floor(uv.x) + floor(uv.y), 2.0);
    vec3 col = mix(colorA.rgb, colorB.rgb, c);
    float shade = 0.6 + 0.4 * abs(dot(n, normalize(vec3(0.35, 0.45, 0.82))));
    FragColor = vec4(col * shade, alpha);
}
"""


def _get_shader():
    global _shader
    if _shader is not None:
        return _shader
    iface = gpu.types.GPUStageInterfaceInfo("qr_checker_iface")
    iface.smooth('VEC3', "wpos")
    iface.smooth('VEC3', "wnor")
    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', "viewProjectionMatrix")
    info.push_constant('MAT4', "modelMatrix")
    info.push_constant('MAT4', "normalMatrix")
    info.push_constant('VEC4', "colorA")
    info.push_constant('VEC4', "colorB")
    info.push_constant('FLOAT', "cell")
    info.push_constant('FLOAT', "alpha")
    info.vertex_in(0, 'VEC3', "pos")
    info.vertex_in(1, 'VEC3', "nor")
    info.vertex_out(iface)
    info.fragment_out(0, 'VEC4', "FragColor")
    info.vertex_source(_VERT)
    info.fragment_source(_FRAG)
    _shader = gpu.shader.create_from_info(info)
    return _shader


def _draw():
    context = bpy.context
    obj = context.active_object
    if obj is None or obj.type != 'MESH' or not obj.visible_get():
        return
    st = getattr(obj, "quad_remesh", None)
    if st is None or not st.show_checker:
        return
    try:
        ma = bridge.mesh_arrays(context, obj)
        h, _area = bridge.target_edge_length(context, obj)
    except Exception:
        return
    if h <= 0 or not len(ma.tris):
        return
    shader = _get_shader()
    if ma.batch is None:
        ma.batch = batch_for_shader(shader, 'TRIS', {"pos": ma.co, "nor": ma.normals}, indices=ma.tris)
    M = obj.matrix_world
    shader.bind()
    shader.uniform_float("viewProjectionMatrix", context.region_data.perspective_matrix)
    shader.uniform_float("modelMatrix", M)
    shader.uniform_float("normalMatrix", M.inverted_safe().transposed())
    shader.uniform_float("colorA", (0.93, 0.93, 0.93, 1.0))
    shader.uniform_float("colorB", (0.16, 0.42, 0.85, 1.0))
    shader.uniform_float("cell", h)
    shader.uniform_float("alpha", 0.92)
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)
    gpu.state.blend_set('ALPHA')
    ma.batch.draw(shader)
    gpu.state.blend_set('NONE')
    gpu.state.depth_mask_set(True)
    gpu.state.depth_test_set('NONE')


def register():
    global _handle
    if bpy.app.background:
        return
    _handle = bpy.types.SpaceView3D.draw_handler_add(_draw, (), 'WINDOW', 'POST_VIEW')


def unregister():
    global _handle, _shader
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, 'WINDOW')
        _handle = None
    _shader = None
