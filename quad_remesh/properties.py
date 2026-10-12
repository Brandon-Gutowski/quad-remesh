# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

import math

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty


def _redraw(self, context):
    screen = getattr(context, "screen", None)
    if screen is None:
        return
    for area in screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()


def _poll_curve(self, obj):
    return obj.type == 'CURVE'


class QR_Settings(bpy.types.PropertyGroup):
    density: FloatProperty(
        name="Density",
        description="How many quads fit across the mesh. The checker preview shows "
                    "the size of each output quad",
        default=20.0, min=1.0, soft_min=2.0, soft_max=150.0, max=2000.0,
        step=50, precision=1, update=_redraw,
    )
    show_checker: BoolProperty(
        name="Show Checker",
        description="Overlay a checker grid whose squares match the target quad size",
        default=False, update=_redraw,
    )
    detect_sharp: BoolProperty(
        name="Detect Sharp Edges",
        description="Make edge flow follow edges whose angle exceeds the threshold",
        default=True,
    )
    sharp_angle: FloatProperty(
        name="Angle",
        description="Edges sharper than this become hard features the edge flow follows",
        subtype='ANGLE', default=math.radians(40.0), min=math.radians(1.0), max=math.radians(180.0),
    )
    follow_marked_sharp: BoolProperty(
        name="Follow Marked Sharp Edges",
        description="Make edge flow follow edges marked Sharp (Edge > Mark Sharp)",
        default=True,
    )
    use_guides: BoolProperty(
        name="Use Guides",
        description="Put an edge loop along every drawn guide stroke. A stroke that ends where it "
                    "started (or a cyclic spline) becomes a closed ring, e.g. around an eye or mouth",
        default=True,
    )
    guide_strength: FloatProperty(
        name="Influence",
        description="How far around each guide the edge flow bends to follow it. The edge loop on "
                    "the guide itself is always kept. 0 = only the loop; 1 = flow runs parallel to "
                    "the guide up to about 4 quads away (concentric rings)",
        default=0.7, min=0.0, max=1.0, subtype='FACTOR',
    )
    guide_rings_out: IntProperty(
        name="Rings Outside",
        description="Extra full edge loops around every closed guide (ring), one quad row pair "
                    "apart, like the loops around an eye or mouth. A ring is left out where it "
                    "would run into an opening, another guide or another guide's rings",
        default=1, min=0, max=5,
    )
    guide_rings_in: IntProperty(
        name="Rings Inside",
        description="Extra full edge loops inside every closed guide (when Fill Inside is off). "
                    "Left out when the ring would get too small or run into an opening",
        default=0, min=0, max=5,
    )
    guide_fill_inside: BoolProperty(
        name="Fill Inside",
        description="Rebuild the inside of every closed guide as concentric loops, one quad "
                    "apart, around a small grid: spokes run straight in and the grid's four "
                    "corners are the only poles inside. Skipped for a ring around an opening "
                    "or a sharp edge",
        default=True,
    )
    guide_smoothing: FloatProperty(
        name="Stroke Smoothing",
        description="How much hand wobble is taken out of newly drawn guide strokes "
                    "(rings keep their size). 0 = keep strokes exactly as drawn",
        default=0.5, min=0.0, max=1.0, subtype='FACTOR',
    )
    guide_object: PointerProperty(
        name="Guides",
        description="Curve object whose splines guide the edge flow",
        type=bpy.types.Object, poll=_poll_curve,
    )
    use_symmetry: BoolProperty(
        name="Symmetry",
        description="Remesh one half and mirror it, so the result is exactly symmetric "
                    "with an edge loop on the mirror plane",
        default=False,
    )
    symmetry_axis: EnumProperty(
        name="Axis",
        description="Mirror across the plane through the object origin perpendicular to "
                    "this local axis. The + side is remeshed and mirrored",
        items=(('X', "X", "Mirror across the local YZ plane"),
               ('Y', "Y", "Mirror across the local XZ plane"),
               ('Z', "Z", "Mirror across the local XY plane")),
        default='X',
    )
    pure_quads: BoolProperty(
        name="Pure Quads",
        description="Guarantee an all-quad result (no triangles or n-gons)",
        default=True,
    )
    relax_iterations: IntProperty(
        name="Smoothing",
        description="Passes that even out quad size and squareness after the quads are built. "
                    "Vertices stay on the original surface, sharp edges and guides. 0 = raw "
                    "output; more = more even quads, but small surface detail softens a little",
        default=12, min=0, soft_max=50, max=200,
    )
    cleanup: BoolProperty(
        name="Reduce Poles",
        description="Globally straighten the quad grid and clean up topology to remove "
                    "unnecessary irregular vertices (poles/stars)",
        default=True,
    )
    adapt_thin: BoolProperty(
        name="Refine Thin Parts",
        description="Separate pieces that are thinner than a quad (caps, shells, labels, fins) are "
                    "remeshed at their own smaller, even quad size so both sides and the rim stay "
                    "clean. Adds faces on those pieces only",
        default=True,
    )
    detail: FloatProperty(
        name="Detail",
        description="Keep detail that is smaller than a quad (small fillets, ridges, embossing) by "
                    "using gradually smaller quads there, down to 1/4 size. 0 keeps every quad the "
                    "same size (best for organic / character meshes); detail too fine to keep is "
                    "smoothed over",
        default=0.0, min=0.0, max=1.0, subtype='FACTOR',
    )
    hide_original: BoolProperty(
        name="Hide Original",
        description="Hide the source object after remeshing",
        default=True,
    )
    last_report: StringProperty(name="Last Result", default="")


classes = (QR_Settings,)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Object.quad_remesh = PointerProperty(type=QR_Settings)


def unregister():
    del bpy.types.Object.quad_remesh
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
