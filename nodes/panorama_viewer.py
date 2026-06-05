"""PanoramaViewer — save the equirect + interactive 360° viewer in ComfyUI."""

from __future__ import annotations

import math
import os
import time

import cv2
import numpy as np
import torch
import utils3d
from comfy_api.latest import io

from .utils import (
    PANORAMA_TYPE,
    normalize_pano_tensor,
    unwrap_panorama_to_image,
)
from ._vendor.worldstereo_cube import split_panorama_image

try:
    import folder_paths
    _OUTPUT_DIR = folder_paths.get_output_directory()
except ImportError:
    _OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def _crop_perspective(np_img, center_u, center_v, fov_h, fov_v, out_w, out_h):
    """Extract a single perspective crop from a float32 equirect image."""
    theta = (1.0 - float(center_u)) * 2.0 * math.pi
    phi = float(center_v) * math.pi
    look_dir = np.array([
        math.sin(phi) * math.cos(theta),
        math.sin(phi) * math.sin(theta),
        math.cos(phi),
    ], dtype=np.float32)

    eye = np.zeros(3, dtype=np.float32)
    up_default = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    up_fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    up = up_fallback if abs(np.dot(look_dir, up_default)) > 0.999 else up_default

    extrinsics = utils3d.numpy.extrinsics_look_at(
        eye, look_dir.reshape(1, 3), up,
    ).astype(np.float32)
    intrinsics = utils3d.numpy.intrinsics_from_fov(
        fov_x=math.radians(fov_h), fov_y=math.radians(fov_v),
    ).astype(np.float32).reshape(1, 3, 3)

    crops = split_panorama_image(
        np_img, extrinsics, intrinsics,
        h=out_h, w=out_w, interp=cv2.INTER_LINEAR,
    )
    return crops[0]


class PanoramaViewer(io.ComfyNode):
    """Save a panorama as PNG, display in 360° viewer, and output a perspective crop."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PanoramaViewer",
            display_name="Preview Panorama",
            category="PanoPack",
            is_output_node=True,
            description=(
                "Save the input panorama to ComfyUI/output/ and display it "
                "in an interactive 360° viewer.\n\n"
                "Also outputs a perspective crop of the current view "
                "(based on initial_yaw / initial_pitch and crop_fov). "
                "Use this to preview what a camera at that angle sees.\n\n"
                "Controls: mouse drag to rotate, scroll to zoom, "
                "arrow keys / WASD to navigate, preset view buttons "
                "(Front, Back, Left, Right, Top, Bottom)."
            ),
            inputs=[
                io.Custom(PANORAMA_TYPE).Input(
                    "panorama",
                    tooltip="The panorama to display."),
                io.Int.Input(
                    "viewer_width", default=512, min=128, max=4096, step=64,
                    tooltip="Width of the 360° viewer widget in pixels."),
                io.Int.Input(
                    "viewer_height", default=512, min=128, max=4096, step=64,
                    tooltip="Height of the 360° viewer widget in pixels."),
                io.Float.Input(
                    "initial_yaw", default=0.0, min=-180.0, max=180.0, step=1.0,
                    tooltip="Initial yaw (longitude) in degrees for the "
                            "viewer and crop output. 0 = front."),
                io.Float.Input(
                    "initial_pitch", default=0.0, min=-90.0, max=90.0, step=1.0,
                    tooltip="Initial pitch (latitude) in degrees for the "
                            "viewer and crop output. 0 = horizon."),
                io.Float.Input(
                    "crop_fov", default=90.0, min=10.0, max=170.0, step=1.0,
                    tooltip="Field of view for the image crop output (degrees)."),
                io.String.Input(
                    "filename_prefix", default="panorama", optional=True,
                    tooltip="Prefix for the saved PNG filename. A "
                            "timestamp suffix is appended automatically."),
            ],
            outputs=[
                io.Image.Output(display_name="crop_image"),
                io.String.Output(display_name="saved_path"),
            ],
        )

    @classmethod
    def execute(
        cls,
        panorama,
        viewer_width: int = 512,
        viewer_height: int = 512,
        initial_yaw: float = 0.0,
        initial_pitch: float = 0.0,
        crop_fov: float = 90.0,
        filename_prefix: str = "panorama",
    ):
        from PIL import Image as PILImage

        img = unwrap_panorama_to_image(panorama)
        img = normalize_pano_tensor(img)
        np_img = img[0].clamp(0, 1).cpu().numpy()
        if np_img.shape[-1] == 1:
            np_img = np.repeat(np_img, 3, axis=-1)
        elif np_img.shape[-1] == 4:
            np_img = np_img[..., :3]
        uint8_img = (np_img * 255.0).round().clip(0, 255).astype(np.uint8)

        # Save panorama
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        timestamp = int(time.time() * 1000)
        fname = f"{filename_prefix}_{timestamp}.png"
        out_path = os.path.join(_OUTPUT_DIR, fname)
        PILImage.fromarray(uint8_img).save(out_path, format="PNG", optimize=False)

        # Perspective crop at initial_yaw / initial_pitch.
        # Convert yaw/pitch (degrees) to equirect UV. The +0.5 (half turn) on
        # U matches the three.js viewer convention, where yaw=0 looks at the
        # centre of the equirect image (image U=0.5), not its left edge.
        center_u = ((float(initial_yaw) / 360.0) + 0.5) % 1.0
        center_v = 0.5 - (float(initial_pitch) / 180.0)
        crop_np = _crop_perspective(
            np_img, center_u, center_v,
            fov_h=float(crop_fov), fov_v=float(crop_fov),
            out_w=int(viewer_width), out_h=int(viewer_height),
        )
        crop_t = torch.from_numpy(crop_np).unsqueeze(0).clamp(0, 1)

        # Note: intentionally NOT including an "images" key here — that would
        # make ComfyUI render a full-size preview below the node. The 360°
        # viewer is driven entirely by the "panorama" payload below.
        ui_payload = {
            "panorama": [{
                "filename": fname,
                "subfolder": "",
                "type": "output",
                "width": int(np_img.shape[1]),
                "height": int(np_img.shape[0]),
                "viewer_width": int(viewer_width),
                "viewer_height": int(viewer_height),
                "initial_yaw": float(initial_yaw),
                "initial_pitch": float(initial_pitch),
                "crop_fov": float(crop_fov),
            }],
        }

        return io.NodeOutput(crop_t, out_path, ui=ui_payload)


NODE_CLASS_MAPPINGS = {"PanoramaViewer": PanoramaViewer}
NODE_DISPLAY_NAME_MAPPINGS = {"PanoramaViewer": "Preview Panorama"}
