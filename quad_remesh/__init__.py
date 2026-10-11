"""Quad Remesh: field-aligned quad remeshing for Blender."""

from . import bridge, guides, operators, preview, properties, ui

_modules = (properties, bridge, operators, guides, ui, preview)


def register():
    for m in _modules:
        m.register()


def unregister():
    for m in reversed(_modules):
        m.unregister()
