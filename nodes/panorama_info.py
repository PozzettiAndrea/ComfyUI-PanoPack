"""PanoramaInfo — diagnostics + quality scoring for an equirect."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from .utils import (
    PANORAMA_TYPE,
    border_continuity_score,
    normalize_pano_tensor,
    unwrap_panorama_to_image,
)


class PanoramaInfo(io.ComfyNode):
    """Report panorama shape, dynamic range, and border continuity."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PanoramaInfo",
            display_name="Panorama Info",
            category="PanoPack",
            is_output_node=True,
            description=(
                "Inspect an equirect panorama: resolution, aspect, value "
                "range, and a border-continuity quality score.\n\n"
                "  seam ∈ [0, 1]: 1.0 = pixel-perfect ±π wraparound, "
                "0 = visible seam. Computed as 1 − mean(|left_col − "
                "right_col|) across all channels.\n"
                "  north_pole_var / south_pole_var ∈ [0, 1]: 1.0 = "
                "perfectly uniform polar row (a true pole is a "
                "singularity — all pixels image the same world point). "
                "Lower scores indicate the panorama wasn't generated / "
                "captured with pole-awareness."
            ),
            inputs=[
                io.Custom(PANORAMA_TYPE).Input(
                    "panorama",
                    tooltip="The panorama to inspect."),
            ],
            outputs=[
                io.String.Output(display_name="info"),
                io.Float.Output(display_name="seam_score"),
                io.Float.Output(display_name="north_pole_score"),
                io.Float.Output(display_name="south_pole_score"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(cls, panorama):
        img = unwrap_panorama_to_image(panorama)
        img = normalize_pano_tensor(img)
        b, h, w, c = img.shape

        scores = border_continuity_score(img)
        vmin = float(img.min())
        vmax = float(img.max())
        vmed = float(img.median())

        info = (
            f"Panorama Info\n"
            f"=============\n"
            f"Shape:          {b} × {h} × {w} × {c}\n"
            f"Aspect ratio:   {scores['aspect_ratio']:.4f}  (2.0 = perfect equirect)\n"
            f"Value range:    min={vmin:.4f}  median={vmed:.4f}  max={vmax:.4f}\n"
            f"\n"
            f"Border continuity\n"
            f"-----------------\n"
            f"  ±π seam        : {scores['seam']:.4f}   "
            f"({'seamless' if scores['seam'] > 0.95 else 'visible seam' if scores['seam'] < 0.7 else 'mild seam'})\n"
            f"  north pole var : {scores['north_pole_var']:.4f}   "
            f"({'uniform' if scores['north_pole_var'] > 0.95 else 'not pole-aware' if scores['north_pole_var'] < 0.7 else 'mild'})\n"
            f"  south pole var : {scores['south_pole_var']:.4f}   "
            f"({'uniform' if scores['south_pole_var'] > 0.95 else 'not pole-aware' if scores['south_pole_var'] < 0.7 else 'mild'})\n"
        )

        return io.NodeOutput(
            info,
            scores["seam"],
            scores["north_pole_var"],
            scores["south_pole_var"],
            w, h,
            ui={"text": [info]},
        )


NODE_CLASS_MAPPINGS = {"PanoramaInfo": PanoramaInfo}
NODE_DISPLAY_NAME_MAPPINGS = {"PanoramaInfo": "Panorama Info"}
