"""Stitch 6 cube-map faces into an equirectangular panorama via grid_sample."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


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
) -> np.ndarray:
    """Stitch 6 cube-map faces [+X, -X, +Y, -Y, +Z, -Z] into an ERP image.

    Each face is (H_face, W_face, C) float32. Output is (erp_height, erp_width, C).
    Uses PyTorch grid_sample for bilinear interpolation.
    """
    assert len(faces) == 6
    face_size = faces[0].shape[0]
    C = faces[0].shape[2]

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

    # For each pixel, find which cube face it belongs to and sample
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

        # Only pixels facing this face (d_fwd > 0) and within the 90° FOV
        mask = d_fwd > 1e-6
        # Cube face coords: [-1, 1] within the face
        face_x = np.where(mask, d_right / d_fwd, 0.0)
        face_y = np.where(mask, d_up / d_fwd, 0.0)
        in_face = mask & (np.abs(face_x) <= 1.0) & (np.abs(face_y) <= 1.0)

        if not in_face.any():
            continue

        # Sample from this face using grid_sample
        face_t = torch.from_numpy(faces[i]).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        # grid_sample expects grid in [-1, 1]
        gx = torch.from_numpy(face_x).unsqueeze(0).unsqueeze(0)   # (1, 1, H, W) — but we need (1, H, W, 2)
        gy = torch.from_numpy(-face_y).unsqueeze(0).unsqueeze(0)  # flip Y (image Y is top-down)

        grid = torch.stack([gx.squeeze(0).squeeze(0),
                            gy.squeeze(0).squeeze(0)], dim=-1).unsqueeze(0)  # (1, H, W, 2)

        sampled = F.grid_sample(
            face_t, grid, mode="bilinear", padding_mode="border", align_corners=False,
        )  # (1, C, H, W)
        sampled_np = sampled[0].permute(1, 2, 0).numpy()  # (H, W, C)

        in_face_f = in_face.astype(np.float32)[..., None]
        erp += sampled_np * in_face_f
        weight += in_face_f

    # Normalize overlapping regions (cube edges/corners)
    weight = np.maximum(weight, 1e-8)
    return erp / weight
