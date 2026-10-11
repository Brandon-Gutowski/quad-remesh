# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

"""Quad Remesh: field-aligned quad remeshing for Blender."""

from . import bridge, guides, operators, preview, properties, ui

_modules = (properties, bridge, operators, guides, ui, preview)


def register():
    for m in _modules:
        m.register()


def unregister():
    for m in reversed(_modules):
        m.unregister()
