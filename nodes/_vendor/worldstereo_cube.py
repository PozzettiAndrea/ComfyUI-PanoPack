"""Tight-scope vendor of three WorldStereo helpers used by PanoramaCubeSplit.

The upstream lives at
  worldstereo/ComfyUI/custom_nodes/ComfyUI-WorldStereo/nodes/worldstereo/src/panorama_utils.py

We only need:
  - `split_panorama_image(image, ext, K, h, w, interp)` — rectangular crop
    (h ≠ w supported), which the MoGe-vendored `split_panorama_image`
    in this pack's `_vendor/moge_panorama.py` doesn't do (it only takes
    a single `resolution` arg for square crops).
  - `directions_to_spherical_uv` — call-graph dependency of the above.
  - `rotate_around_z_axis` — clockwise rotation around +Z used to derive
    look-at points for the fixed pitch/yaw grid.

Copied verbatim from upstream so behavior matches exactly; if upstream
ever diverges, this file is what to update.
"""

from __future__ import annotations

import cv2
import numpy as np
import utils3d


def rotate_around_z_axis(points, angle_deg):
    """Rotate 3D points clockwise around the Z axis."""
    angle_rad = np.radians(angle_deg)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    # Right-handed system → clockwise == negative counter-clockwise.
    rotation_matrix = np.array([
        [cos_theta, sin_theta, 0],
        [-sin_theta, cos_theta, 0],
        [0, 0, 1],
    ])
    return np.dot(points, rotation_matrix.T)


def directions_to_spherical_uv(directions: np.ndarray):
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    u = 1 - np.arctan2(directions[..., 1], directions[..., 0]) / (2 * np.pi) % 1.0
    v = np.arccos(directions[..., 2]) / np.pi
    return np.stack([u, v], axis=-1)


def split_panorama_image(image: np.ndarray, extrinsics: np.ndarray,
                         intrinsics: np.ndarray, h: int, w: int, interp):
    """Sample N rectangular crops from an equirect panorama.

    Differs from `_vendor/moge_panorama.split_panorama_image` in accepting
    separate (h, w) — the cube-split grid wants 832×480 (16:9), not square.
    """
    height, width = image.shape[:2]
    safe_height = height // 2
    safe_width = int(round(safe_height / h * w))
    if interp == cv2.INTER_AREA:
        # remap doesn't support area downsampling — remap to a safe
        # resolution first to avoid frequency artifacts, then resize.
        uv = utils3d.np.uv_map((safe_height, safe_width))
    else:
        uv = utils3d.np.uv_map((h, w))
    splitted_images = []
    for i in range(len(extrinsics)):
        spherical_uv = directions_to_spherical_uv(
            utils3d.np.unproject_cv(
                uv, np.ones_like(uv[..., 0]),
                extrinsics=extrinsics[i], intrinsics=intrinsics[i],
            )
        )
        pixels = utils3d.np.uv_to_pixel(spherical_uv, (height, width)).astype(np.float32)
        if interp == cv2.INTER_AREA:
            splitted = cv2.remap(image, pixels[..., 0], pixels[..., 1],
                                 interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            splitted = cv2.resize(splitted, (w, h), interpolation=interp)
        else:
            splitted = cv2.remap(image, pixels[..., 0], pixels[..., 1],
                                 interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        splitted_images.append(splitted)
    return splitted_images
