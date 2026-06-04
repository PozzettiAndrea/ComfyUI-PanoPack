"""PanoCrop — extract a perspective (rectilinear) crop from an equirect."""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import utils3d
from comfy_api.latest import io

from .utils import PANORAMA_TYPE, normalize_pano_tensor, unwrap_panorama_to_image
from ._vendor.worldstereo_cube import split_panorama_image


class PanoCrop(io.ComfyNode):
    """Crop a perspective view from an equirectangular panorama."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PanoCrop",
            display_name="Panorama Crop",
            category="PanoPack",
            description=(
                "Extract a rectilinear (perspective) crop from an "
                "equirectangular panorama.\n\n"
                "Specify the center of the crop in equirectangular UV "
                "coordinates (u=0..1 horizontal, v=0..1 vertical), the "
                "horizontal and vertical field of view, and the output "
                "image resolution."
            ),
            inputs=[
                io.Custom(PANORAMA_TYPE).Input(
                    "panorama",
                    tooltip="The equirectangular panorama to crop from."),
                io.Float.Input(
                    "center_u", default=0.5, min=0.0, max=1.0, step=0.01,
                    tooltip="Horizontal center of the crop in UV space. "
                            "0.0 = left edge, 0.5 = center (front), 1.0 = right edge."),
                io.Float.Input(
                    "center_v", default=0.5, min=0.0, max=1.0, step=0.01,
                    tooltip="Vertical center of the crop in UV space. "
                            "0.0 = top (north pole), 0.5 = horizon, 1.0 = bottom (south pole)."),
                io.Float.Input(
                    "fov_h", default=90.0, min=10.0, max=170.0, step=1.0,
                    tooltip="Horizontal field of view in degrees."),
                io.Float.Input(
                    "fov_v", default=90.0, min=10.0, max=170.0, step=1.0,
                    tooltip="Vertical field of view in degrees."),
                io.Int.Input(
                    "width", default=512, min=64, max=4096, step=64,
                    tooltip="Output image width in pixels."),
                io.Int.Input(
                    "height", default=512, min=64, max=4096, step=64,
                    tooltip="Output image height in pixels."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        panorama,
        center_u: float = 0.5,
        center_v: float = 0.5,
        fov_h: float = 90.0,
        fov_v: float = 90.0,
        width: int = 512,
        height: int = 512,
    ):
        # Panorama to numpy
        img_t = unwrap_panorama_to_image(panorama)
        img_t = normalize_pano_tensor(img_t)
        np_img = img_t[0].cpu().numpy()  # (H, W, C) float32 [0, 1]

        # UV -> spherical angles -> 3D look direction
        # u: 0..1 maps to longitude (WorldNav convention: u=0 -> theta=2pi, u=1 -> theta=0)
        # v: 0..1 maps to latitude  (v=0 -> north pole phi=0, v=1 -> south pole phi=pi)
        theta = (1.0 - float(center_u)) * 2.0 * math.pi  # azimuth
        phi = float(center_v) * math.pi                    # polar angle from +Z

        look_dir = np.array([
            math.sin(phi) * math.cos(theta),
            math.sin(phi) * math.sin(theta),
            math.cos(phi),
        ], dtype=np.float32)

        # Build extrinsics (world-to-camera) with pole-safe up vector
        eye = np.zeros(3, dtype=np.float32)
        target = look_dir.reshape(1, 3)
        up_default = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        up_fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        # If looking straight up/down, swap up vector
        if abs(np.dot(look_dir, up_default)) > 0.999:
            up = up_fallback
        else:
            up = up_default
        extrinsics = utils3d.numpy.extrinsics_look_at(eye, target, up).astype(np.float32)

        # Build intrinsics from FOV
        intrinsics = utils3d.numpy.intrinsics_from_fov(
            fov_x=math.radians(float(fov_h)),
            fov_y=math.radians(float(fov_v)),
        ).astype(np.float32)
        intrinsics = intrinsics.reshape(1, 3, 3)

        # Sample the crop
        crops = split_panorama_image(
            np_img, extrinsics, intrinsics,
            h=int(height), w=int(width),
            interp=cv2.INTER_LINEAR,
        )
        crop_np = crops[0]  # (H, W, C) float32

        # Ensure [0, 1] range and return as IMAGE
        crop_t = torch.from_numpy(crop_np).unsqueeze(0).clamp(0, 1)  # (1, H, W, C)
        return io.NodeOutput(crop_t)


NODE_CLASS_MAPPINGS = {"PanoCrop": PanoCrop}
NODE_DISPLAY_NAME_MAPPINGS = {"PanoCrop": "Panorama Crop"}
