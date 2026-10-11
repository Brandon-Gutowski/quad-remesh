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
        description="Steer the edge flow along drawn guide strokes",
        default=True,
    )
    guide_strength: FloatProperty(
        name="Strength",
        description="How strongly guide strokes steer the flow (1 = hard constraint)",
        default=0.7, min=0.0, max=1.0, subtype='FACTOR',
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
        name="Relax",
        description="Smoothing iterations that even out quad shapes and sizes",
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
        description="Use smaller quads where the mesh is thinner than a quad (ears, fins, "
                    "cloth) so both sides stay clean. Adds a few poles at the size transitions",
        default=True,
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
