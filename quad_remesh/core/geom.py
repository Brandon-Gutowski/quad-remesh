# SPDX-FileCopyrightText: 2026 Brandon Gutowski
# SPDX-License-Identifier: GPL-3.0-or-later

"""Small vectorized geometry helpers shared by the core modules."""

import numpy as np


def dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def norm(v):
    return np.sqrt(np.einsum("ij,ij->i", v, v))


def normalize(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def tangent_frames(N):
    """Arbitrary orthonormal tangent frame (e1, e2) for each unit normal."""
    a = np.zeros_like(N)
    a[np.arange(len(N)), np.argmin(np.abs(N), axis=1)] = 1.0
    e1 = normalize(np.cross(N, a))
    e2 = np.cross(N, e1)
    return e1, e2


def rotate_between(v, n_from, n_to):
    """Apply the minimal rotation taking n_from onto n_to to the vectors v."""
    k = np.cross(n_from, n_to)
    c = np.maximum(dot(n_from, n_to), -0.999999)
    return v * c[:, None] + np.cross(k, v) + k * (dot(k, v) / (1.0 + c))[:, None]


def project_tangent(v, N):
    return v - N * dot(v, N)[:, None]


def newell_normal(P):
    """Newell normal (unnormalized) of a polygon given as (k,3) points."""
    nxt = np.roll(P, -1, axis=0)
    return np.array([
        np.sum((P[:, 1] - nxt[:, 1]) * (P[:, 2] + nxt[:, 2])),
        np.sum((P[:, 2] - nxt[:, 2]) * (P[:, 0] + nxt[:, 0])),
        np.sum((P[:, 0] - nxt[:, 0]) * (P[:, 1] + nxt[:, 1])),
    ])
