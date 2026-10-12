# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

import bpy

from . import bridge, operators


class QR_PT_main(bpy.types.Panel):
    bl_label = "Quad Remesh"
    bl_idname = "QR_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Quad Remesh"

    def draw(self, context):
        layout = self.layout
        ob = context.active_object
        if ob is not None and "qr_source" in ob and ob.type == 'MESH':
            layout.operator("quad_remesh.toggle_original", icon='HIDE_OFF')
            src = bpy.data.objects.get(ob["qr_source"])
            if src is None:
                return
            st = src.quad_remesh
            if st.last_report:
                box = layout.box()
                for part in st.last_report.split(" | "):
                    box.label(text=part)
            layout.label(text=f"Settings live on '{src.name}'", icon='INFO')
            return
        if ob is None or ob.type != 'MESH':
            layout.label(text="Select a mesh object", icon='INFO')
            return
        st = ob.quad_remesh

        col = layout.column(align=True)
        col.prop(st, "density", slider=False)
        col.prop(st, "show_checker", toggle=True, icon='TEXTURE')
        try:
            h, area = bridge.target_edge_length(context, ob)
            if h > 0:
                more = "+" if (st.detail > 0 or st.adapt_thin) else ""
                layout.label(text=f"~{int(round(area / (h * h))):,}{more} quads, edge {h:.3g} m")
        except Exception:
            pass

        box = layout.box()
        box.label(text="Sharp Edges", icon='MOD_EDGESPLIT')
        row = box.row(align=True)
        row.prop(st, "detect_sharp", text="By Angle")
        sub = row.row(align=True)
        sub.active = st.detect_sharp
        sub.prop(st, "sharp_angle", text="")
        box.prop(st, "follow_marked_sharp")

        box = layout.box()
        split = box.split(factor=0.42)
        split.prop(st, "use_symmetry")
        sub = split.row(align=True)
        sub.active = st.use_symmetry
        sub.prop(st, "symmetry_axis", expand=True)

        box = layout.box()
        box.label(text="Edge Flow Guides", icon='GP_SELECT_STROKES')
        box.prop(st, "use_guides")
        sub = box.column(align=True)
        sub.active = st.use_guides
        row = sub.row(align=True)
        row.operator("quad_remesh.draw_guides", icon='GREASEPENCIL')
        row.operator("quad_remesh.clear_guides", text="", icon='TRASH')
        sub.prop(st, "guide_smoothing", slider=True)
        sub.prop(st, "guide_object", text="")
        sub.prop(st, "guide_strength", slider=True)
        row = sub.row(align=True)
        row.prop(st, "guide_rings_out", text="Rings Out")
        col = row.row(align=True)
        col.active = not st.guide_fill_inside
        col.prop(st, "guide_rings_in", text="In")
        sub.prop(st, "guide_fill_inside")

        box = layout.box()
        box.label(text="Quality", icon='MESH_GRID')
        box.prop(st, "pure_quads")
        box.prop(st, "cleanup")
        box.prop(st, "adapt_thin")
        box.prop(st, "detail", slider=True)
        box.prop(st, "relax_iterations")
        box.prop(st, "hide_original")

        layout.separator()
        row = layout.row()
        row.scale_y = 1.6
        prog = operators.PROGRESS
        if prog["active"] and prog["source"] == ob.name_full:
            pct = int(round(prog["frac"] * 100))
            row.progress(factor=prog["frac"], type='BAR', text=f"{prog['stage']}  {pct}%")
        else:
            row.operator("quad_remesh.remesh", icon='MOD_REMESH')
        if st.last_report:
            box = layout.box()
            for part in st.last_report.split(" | "):
                box.label(text=part)


classes = (QR_PT_main,)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
