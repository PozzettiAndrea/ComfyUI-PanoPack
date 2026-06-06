"""Stitch 6 cube-map faces into an equirectangular panorama via grid_sample.

Faces are rendered with a FOV slightly wider than 90° so neighbouring faces
overlap, and the overlap is blended with a smooth feather weight. This removes
the visible seams you get from hard-cutting each face at exactly its 90° edge
(where antialiased face-vs-background pixels otherwise leave stitch lines).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


# Render each cube face at this FOV (degrees). >90° gives an overlap margin of
# (FACE_FOV_DEG - 90)/2 degrees on every side for the feather blend to work in.
# Renderers import this so the render FOV and the stitch geometry stay in sync.
FACE_FOV_DEG = 100.0


# Cube face order: +X, -X, +Y, -Y, +Z, -Z
# Each entry: (right, up, forward) as unit vectors — defines the camera
# basis for each cube face, looking INWARD from origin.
_CUBE_FACES = [
    # +X: look right
    {"right": np.array([0, 0, -1]), "up": np.array([0, 1, 0]), "fwd": np.array([1, 0, 0])},
    # -X: look left
    {"right": np.array([0, 0, 1]),  "up": np.array([0, 1, 0]), "fwd": np.array([-1, 0, 0])},
    # +Y: look up
    {"right": np.array([1, 0, 0]),  "up": np.array([0, 0, -1]), "fwd": np.array([0, 1, 0])},
    # -Y: look down
    {"right": np.array([1, 0, 0]),  "up": np.array([0, 0, 1]),  "fwd": np.array([0, -1, 0])},
    # +Z: look forward
    {"right": np.array([1, 0, 0]),  "up": np.array([0, 1, 0]),  "fwd": np.array([0, 0, 1])},
    # -Z: look backward
    {"right": np.array([-1, 0, 0]), "up": np.array([0, 1, 0]),  "fwd": np.array([0, 0, -1])},
]


def cube_faces_to_equirect(
    faces: list[np.ndarray],
    erp_width: int,
    erp_height: int,
    face_fov_deg: float = FACE_FOV_DEG,
    feather: float = 0.25,
) -> np.ndarray:
    """Stitch 6 cube-map faces [+X, -X, +Y, -Y, +Z, -Z] into an ERP image.

    Each face is (H_face, W_face, C) float32 rendered at ``face_fov_deg`` FOV
    (square). Output is (erp_height, erp_width, C).

    ``feather`` is the fraction of each face (from its rendered edge inward) over
    which its contribution ramps from 0 to 1. Overlapping faces are combined as a
    weighted average, so the ramps form a smooth partition of unity across seams.
    """
    assert len(faces) == 6
    C = faces[0].shape[2]

    # Half-extent of the rendered face in tan space. A direction maps to the face
    # edge (|coord| == 1) when d_perp / d_fwd == tan(fov/2).
    half_tan = math.tan(math.radians(face_fov_deg) / 2.0)
    feather = max(float(feather), 1e-4)

    # Build ERP pixel directions
    u = np.linspace(0.5, erp_width - 0.5, erp_width, dtype=np.float32)
    v = np.linspace(0.5, erp_height - 0.5, erp_height, dtype=np.float32)
    vv, uu = np.meshgrid(v, u, indexing="ij")

    # ERP coords -> spherical -> 3D direction (Y-up)
    lon = (uu / erp_width) * 2.0 * math.pi - math.pi      # [-pi, pi]
    lat = (vv / erp_height) * math.pi - (math.pi / 2.0)    # [-pi/2, pi/2]

    # 3D direction (Y-up, right-handed)
    dx = np.cos(lat) * np.sin(lon)
    dy = -np.sin(lat)  # Y-up: top of image = +Y
    dz = np.cos(lat) * np.cos(lon)
    dirs = np.stack([dx, dy, dz], axis=-1)  # (H, W, 3)

    # For each pixel, accumulate every face it falls inside, weighted by feather.
    erp = np.zeros((erp_height, erp_width, C), dtype=np.float32)
    weight = np.zeros((erp_height, erp_width, 1), dtype=np.float32)

    for i, face_info in enumerate(_CUBE_FACES):
        fwd = face_info["fwd"].astype(np.float32)
        right = face_info["right"].astype(np.float32)
        up = face_info["up"].astype(np.float32)

        # Project direction onto this face's coordinate system
        d_fwd = (dirs * fwd).sum(axis=-1)    # depth along face forward
        d_right = (dirs * right).sum(axis=-1)
        d_up = (dirs * up).sum(axis=-1)

        mask = d_fwd > 1e-6
        d_fwd_safe = np.where(mask, d_fwd, 1.0)
        # Normalised face coords: [-1, 1] spans the rendered FOV.
        face_x = (d_right / d_fwd_safe) / half_tan
        face_y = (d_up / d_fwd_safe) / half_tan

        # Smooth feather: 1 well inside the face, ramping to 0 at the rendered
        # edge over `feather` of the half-extent. Where faces overlap, both ramps
        # are non-zero and the weighted average blends them with no hard seam.
        m = np.maximum(np.abs(face_x), np.abs(face_y))
        edge = 1.0 - m                         # 0 at rendered edge, 1 at centre
        w = np.clip(edge / feather, 0.0, 1.0)
        w = w * w * (3.0 - 2.0 * w)            # smoothstep
        w = np.where(mask & (m <= 1.0), w, 0.0).astype(np.float32)

        if not np.any(w > 0.0):
            continue

        # Sample from this face using grid_sample (grid in [-1, 1]).
        face_t = torch.from_numpy(np.ascontiguousarray(faces[i])).permute(2, 0, 1).unsqueeze(0)
        gx = torch.from_numpy(face_x)
        gy = torch.from_numpy(-face_y)  # flip Y (image Y is top-down)
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # (1, H, W, 2)

        sampled = F.grid_sample(
            face_t, grid, mode="bilinear", padding_mode="border", align_corners=False,
        )  # (1, C, H, W)
        sampled_np = sampled[0].permute(1, 2, 0).numpy()  # (H, W, C)

        w3 = w[..., None]
        erp += sampled_np * w3
        weight += w3

    # Normalize the feathered weighted average.
    weight = np.maximum(weight, 1e-8)
    return erp / weight
