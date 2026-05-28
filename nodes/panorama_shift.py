"""PanoramaShift — rotate an equirect by a yaw angle in degrees."""

from __future__ import annotations

from comfy_api.latest import io

from .utils import (
    PANORAMA_TYPE,
    panorama_shift_horizontal,
    unwrap_panorama_to_image,
    wrap_image_as_panorama,
)


class PanoramaShift(io.ComfyNode):
    """Rotate an equirect panorama horizontally by a yaw angle."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PanoramaShift",
            display_name="Panorama Shift (yaw rotation)",
            category="PanoPack",
            description=(
                "Rotate an equirect panorama around the vertical axis "
                "by a yaw angle in degrees.\n\n"
                "Implementation: integer-pixel torch.roll along the W "
                "axis (lossless for whole-pixel shifts) plus a sub-pixel "
                "bilinear residual via grid_sample. Wraps cleanly across "
                "the ±π seam.\n\n"
                "Sign convention: POSITIVE yaw rotates the SCENE right "
                "(content at u_pixel=0 moves toward u_pixel=+shift). "
                "Equivalently, the camera turns LEFT.\n\n"
                "180° rotates the back of the scene to face forward; "
                "any multiple of 360° is a no-op."
            ),
            inputs=[
                io.Custom(PANORAMA_TYPE).Input(
                    "panorama",
                    tooltip="The panorama to rotate."),
                io.Float.Input(
                    "yaw_degrees", default=0.0, min=-360.0, max=360.0, step=1.0,
                    tooltip="Rotation in degrees around the vertical "
                            "axis. Positive = scene rotates right / "
                            "camera turns left."),
            ],
            outputs=[
                io.Custom(PANORAMA_TYPE).Output(display_name="panorama"),
            ],
        )

    @classmethod
    def execute(cls, panorama, yaw_degrees: float = 0.0):
        img = unwrap_panorama_to_image(panorama)
        rotated = panorama_shift_horizontal(img, yaw_degrees)
        meta = panorama.get("meta", {}).copy() if isinstance(panorama, dict) else {}
        meta["yaw_shift_deg"] = float(meta.get("yaw_shift_deg", 0.0)) + float(yaw_degrees)
        out = wrap_image_as_panorama(rotated, meta=meta)
        return io.NodeOutput(out)


NODE_CLASS_MAPPINGS = {"PanoramaShift": PanoramaShift}
NODE_DISPLAY_NAME_MAPPINGS = {"PanoramaShift": "Panorama Shift (yaw rotation)"}
