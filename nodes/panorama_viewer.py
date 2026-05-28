"""PanoramaViewer — save the equirect + show it in ComfyUI's preview lane.

v0.1 emits a regular IMAGE preview (so you can at least see the panorama).
A drag-rotate 360° viewer is planned for a follow-up (web/ extension that
detects this node's output and replaces the static preview with a
pannellum.js embed).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
from comfy_api.latest import io

from ..utils import (
    PANORAMA_TYPE,
    normalize_pano_tensor,
    unwrap_panorama_to_image,
)

try:
    import folder_paths
    _OUTPUT_DIR = folder_paths.get_output_directory()
except ImportError:
    _OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


class PanoramaViewer(io.ComfyNode):
    """Save a panorama as PNG and surface it in ComfyUI's UI."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PanoramaViewer",
            display_name="Panorama Viewer",
            category="PanoPack",
            is_output_node=True,
            description=(
                "Save the input panorama to ComfyUI/output/ and show it "
                "in the node's preview lane.\n\n"
                "v0.1: standard image preview. A drag-rotate 360° viewer "
                "(pannellum.js-based) is planned for a future release."
            ),
            inputs=[
                io.Custom(PANORAMA_TYPE).Input(
                    "panorama",
                    tooltip="The panorama to display."),
                io.String.Input(
                    "filename_prefix", default="panorama", optional=True,
                    tooltip="Prefix for the saved PNG filename. A "
                            "timestamp suffix is appended automatically."),
            ],
            outputs=[
                io.String.Output(display_name="saved_path"),
            ],
        )

    @classmethod
    def execute(cls, panorama, filename_prefix: str = "panorama"):
        from PIL import Image as PILImage

        img = unwrap_panorama_to_image(panorama)
        img = normalize_pano_tensor(img)
        # Take batch[0] for the preview if a multi-pano batch was passed.
        np_img = img[0].clamp(0, 1).cpu().numpy()
        if np_img.shape[-1] == 1:
            np_img = np.repeat(np_img, 3, axis=-1)
        elif np_img.shape[-1] == 4:
            np_img = np_img[..., :3]
        uint8_img = (np_img * 255.0).round().clip(0, 255).astype(np.uint8)

        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        timestamp = int(time.time() * 1000)
        fname = f"{filename_prefix}_{timestamp}.png"
        out_path = os.path.join(_OUTPUT_DIR, fname)
        PILImage.fromarray(uint8_img).save(out_path, format="PNG", optimize=False)

        # ComfyUI UI hook: emit the file so it shows up in the node's
        # preview lane. `type=output` + filename only is the canonical
        # form for ComfyUI to serve the image via /view.
        ui_payload = {
            "images": [{
                "filename": fname,
                "subfolder": "",
                "type": "output",
            }],
            # Extra payload the future panorama_viewer.js extension
            # will pick up to swap in a 360 viewer:
            "panorama": [{
                "filename": fname,
                "subfolder": "",
                "type": "output",
                "width": int(np_img.shape[1]),
                "height": int(np_img.shape[0]),
            }],
        }

        return io.NodeOutput(out_path, ui=ui_payload)


NODE_CLASS_MAPPINGS = {"PanoramaViewer": PanoramaViewer}
NODE_DISPLAY_NAME_MAPPINGS = {"PanoramaViewer": "Panorama Viewer"}
