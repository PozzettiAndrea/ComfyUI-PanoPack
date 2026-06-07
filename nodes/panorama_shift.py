"""PanoramaShift - rotate an equirect by yaw and pitch angles in degrees."""

from __future__ import annotations

from comfy_api.latest import io

from .utils import (
    PANORAMA_TYPE,
    panorama_shift_horizontal,
    panorama_shift_pitch,
    unwrap_panorama_to_image,
    wrap_image_as_panorama,
)


class PanoramaShift(io.ComfyNode):
    """Rotate an equirect panorama by yaw and pitch angles."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PanoramaShift",
            display_name="Panorama Shift",
            category="PanoPack",
            description=(
                "Rotate an equirect panorama by yaw (horizontal) and pitch "
                "(vertical) angles in degrees.\n\n"
                "Yaw: rotation around the vertical axis. "
                "Implementation uses integer-pixel torch.roll along the W "
                "axis (lossless for whole-pixel shifts) plus a sub-pixel "
                "bilinear residual via grid_sample. Wraps cleanly across "
                "the +/-pi seam.\n\n"
                "Pitch: rotation around the horizontal axis. "
                "Implementation uses a spherical-coordinate remap via "
                "grid_sample.\n\n"
                "Sign convention: POSITIVE yaw rotates the SCENE right "
                "(camera turns LEFT). POSITIVE pitch tilts the camera UP "
                "(scene moves down)."
            ),
            inputs=[
                io.Custom(PANORAMA_TYPE).Input(
                    "panorama",
                    tooltip="The panorama to rotate."),
                io.Float.Input(
                    "yaw_degrees", default=90.0, min=-360.0, max=360.0, step=1.0,
                    tooltip="Rotation in degrees around the vertical "
                            "axis. Positive = scene rotates right / "
                            "camera turns left."),
                io.Float.Input(
                    "pitch_degrees", default=0.0, min=-90.0, max=90.0, step=1.0,
                    tooltip="Rotation in degrees around the horizontal "
                            "axis. Positive = camera tilts up / "
                            "scene moves down."),
            ],
            outputs=[
                io.Custom(PANORAMA_TYPE).Output(display_name="panorama"),
            ],
        )

    @classmethod
    def execute(cls, panorama, yaw_degrees: float = 90.0, pitch_degrees: float = 0.0):
        img = unwrap_panorama_to_image(panorama)
        rotated = panorama_shift_horizontal(img, yaw_degrees)
        rotated = panorama_shift_pitch(rotated, pitch_degrees)
        meta = panorama.get("meta", {}).copy() if isinstance(panorama, dict) else {}
        meta["yaw_shift_deg"] = float(meta.get("yaw_shift_deg", 0.0)) + float(yaw_degrees)
        meta["pitch_shift_deg"] = float(meta.get("pitch_shift_deg", 0.0)) + float(pitch_degrees)
        out = wrap_image_as_panorama(rotated, meta=meta)
        return io.NodeOutput(out)


NODE_CLASS_MAPPINGS = {"PanoramaShift": PanoramaShift}
NODE_DISPLAY_NAME_MAPPINGS = {"PanoramaShift": "Panorama Shift"}
