"""PanoramaWrap / PanoramaUnwrap - IMAGE <-> PANORAMA conversion nodes."""

from __future__ import annotations

from comfy_api.latest import io

from .utils import (
    PANORAMA_TYPE,
    is_two_to_one,
    normalize_pano_tensor,
    wrap_image_as_panorama,
    unwrap_panorama_to_image,
)


class PanoramaWrap(io.ComfyNode):
    """Wrap a generic IMAGE as a typed PANORAMA. Validates 2:1 aspect."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PanoramaWrap",
            display_name="Panorama Wrap (IMAGE -> PANORAMA)",
            category="PanoPack",
            description=(
                "Tag an IMAGE as a PANORAMA - a typed socket for "
                "equirectangular 2:1 panoramas. Downstream PanoPack "
                "nodes accept this type. Raises if the input isn't 2:1 "
                "(W must be exactly twice H, within 1% tolerance)."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Equirectangular image, shape [B, H, W, C] "
                            "or [H, W, C]. Must satisfy W = 2.H."),
            ],
            outputs=[
                io.Custom(PANORAMA_TYPE).Output(display_name="panorama"),
            ],
        )

    @classmethod
    def execute(cls, image: torch.Tensor):
        import torch
        pano = wrap_image_as_panorama(image)
        return io.NodeOutput(pano)


class PanoramaUnwrap(io.ComfyNode):
    """Unwrap a PANORAMA back to a generic IMAGE socket."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PanoramaUnwrap",
            display_name="Panorama Unwrap (PANORAMA -> IMAGE)",
            category="PanoPack",
            description=(
                "Convert a PANORAMA back to an IMAGE socket so it can "
                "flow into generic ComfyUI image nodes (Preview, Save, "
                "etc.). Idempotent if the source is already an IMAGE."
            ),
            inputs=[
                io.Custom(PANORAMA_TYPE).Input(
                    "panorama",
                    tooltip="The PANORAMA to unwrap."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, panorama):
        img = unwrap_panorama_to_image(panorama)
        img = normalize_pano_tensor(img)
        return io.NodeOutput(img)


NODE_CLASS_MAPPINGS = {
    "PanoramaWrap": PanoramaWrap,
    "PanoramaUnwrap": PanoramaUnwrap,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PanoramaWrap": "Panorama Wrap (IMAGE -> PANORAMA)",
    "PanoramaUnwrap": "Panorama Unwrap (PANORAMA -> IMAGE)",
}
