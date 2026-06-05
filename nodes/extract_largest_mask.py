"""ExtractLargestMask — keep only the largest connected blob of a mask."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from comfy_api.latest import io


def _largest_component(binary: np.ndarray) -> np.ndarray:
    """Return a binary array keeping only the largest 8-connected blob of 1s."""
    # connectedComponentsWithStats wants a uint8 single-channel image.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8,
    )
    if num_labels <= 1:
        # Only background — nothing to keep.
        return np.zeros_like(binary, dtype=np.float32)
    # Label 0 is the background; pick the largest among the rest by pixel area.
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(np.argmax(areas)) + 1
    return (labels == best_label).astype(np.float32)


class ExtractLargestMask(io.ComfyNode):
    """Threshold a mask at 0.5, then keep only its largest connected region."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ExtractLargestMask",
            display_name="Extract Largest Mask",
            category="PanoPack",
            description=(
                "Binarize a mask (values ≥ 0.5 → 1, < 0.5 → 0), find the "
                "largest connected patch of 1s, and return just that patch "
                "as 1 over a 0 background. Each mask in the batch is "
                "processed independently. Uses 8-connectivity."
            ),
            inputs=[
                io.Mask.Input(
                    "mask",
                    tooltip="Mask to filter. Thresholded at 0.5."),
            ],
            outputs=[
                io.Mask.Output(display_name="mask"),
            ],
        )

    @classmethod
    def execute(cls, mask):
        # ComfyUI masks are [B, H, W]; tolerate a bare [H, W] too.
        squeeze_back = False
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
            squeeze_back = True

        np_mask = mask.detach().cpu().numpy()
        binary = (np_mask >= 0.5)

        out = np.stack([_largest_component(binary[i]) for i in range(binary.shape[0])], axis=0)
        out_t = torch.from_numpy(out).to(mask.device)

        if squeeze_back:
            out_t = out_t[0]
        return io.NodeOutput(out_t)


NODE_CLASS_MAPPINGS = {"ExtractLargestMask": ExtractLargestMask}
NODE_DISPLAY_NAME_MAPPINGS = {"ExtractLargestMask": "Extract Largest Mask"}
