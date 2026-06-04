"""PanoRenderGaussians — render a gaussian splat PLY as an ERP panorama."""

from __future__ import annotations

import math
import sys

import numpy as np
import torch
from comfy_api.latest import io

from .utils import PANORAMA_TYPE, wrap_image_as_panorama
from .utils.cube_to_equirect import cube_faces_to_equirect

_C0 = 0.28209479177387814


def _p(msg: str) -> None:
    print(f"[PanoRenderGaussians] {msg}", file=sys.stderr, flush=True)


def _load_gaussian_ply(path: str):
    """Load a 3DGS .ply file. Returns (xyz, rgb, scale, opacity, rot)."""
    # Reuse ComfyUI's robust PLY parser
    sys_path = sys.path.copy()
    try:
        from comfy_extras.nodes_gaussian_splat import _parse_ply_gaussian
    finally:
        sys.path = sys_path

    with open(path, "rb") as f:
        data = f.read()
    xyz, scale, rot, opacity, sh = _parse_ply_gaussian(data)
    # SH DC -> base RGB
    rgb = np.clip(sh[:, 0, :] * _C0 + 0.5, 0, 1)
    return xyz, rgb, scale, opacity, rot


# 6 cube face cameras: position at origin, looking along each axis
# Returns (yaw, pitch) pairs for render_splat, and camera_info dicts for _render_gaussian
_CUBE_VIEWS = [
    {"name": "+X", "yaw": -90, "pitch": 0},
    {"name": "-X", "yaw": 90,  "pitch": 0},
    {"name": "+Y", "yaw": 0,   "pitch": -90},
    {"name": "-Y", "yaw": 0,   "pitch": 90},
    {"name": "+Z", "yaw": 0,   "pitch": 0},
    {"name": "-Z", "yaw": 180, "pitch": 0},
]


def _render_6_faces_cpu(xyz, rgb, scale, opacity, face_size):
    """Render 6 cube faces using the numpy render_splat (CPU, no CUDA needed)."""
    from comfy.ldm.triposplat.preview import render_splat

    faces = []
    for view in _CUBE_VIEWS:
        img = render_splat(
            xyz, rgb, scale, opacity=opacity,
            yaw=float(view["yaw"]),
            pitch=float(view["pitch"]),
            size=face_size, fov=90.0, dist=0.0,
            min_px=1, max_px=12, gain=1.5,
            min_opacity=0.01,
        )
        face_np = np.array(img, dtype=np.float32) / 255.0
        faces.append(face_np)
    return faces


def _render_6_faces_gpu(xyz, rgb, scale, opacity, rot, face_size):
    """Render 6 cube faces using the GPU gaussian rasterizer."""
    import comfy.model_management
    from comfy_extras.nodes_gaussian_splat import (
        _render_gaussian,
        _lookat_camera_info,
    )

    dev = comfy.model_management.get_torch_device()
    origin = torch.zeros(3, device=dev)
    bg = torch.zeros(3)

    faces = []
    for view in _CUBE_VIEWS:
        yaw_rad = math.radians(view["yaw"])
        pitch_rad = math.radians(view["pitch"])
        # Compute look-at target 1 unit away
        target = torch.tensor([
            -math.cos(pitch_rad) * math.sin(yaw_rad),
            math.sin(pitch_rad),
            math.cos(pitch_rad) * math.cos(yaw_rad),
        ], device=dev)
        cam_info = _lookat_camera_info(origin, target, fov=90.0, dev=dev)
        img, _mask = _render_gaussian(
            xyz, rgb, opacity, scale, rot,
            width=face_size, height=face_size,
            splat_scale=1.0, bg=bg,
            camera_info=cam_info,
            render_style="color",
        )
        faces.append(img.numpy().astype(np.float32))
    return faces


class PanoRenderGaussians(io.ComfyNode):
    """Render a gaussian splat PLY file as an equirectangular panorama."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PanoRenderGaussians",
            display_name="Pano Render Gaussians",
            category="PanoPack",
            description=(
                "Load a 3D Gaussian Splat .ply file and render it as an "
                "equirectangular panorama from the world origin.\n\n"
                "Two backends:\n"
                "  gpu — full anisotropic gaussian rasterizer (CUDA). "
                "Fast and high quality.\n"
                "  cpu — simple disk-splat z-buffer (numpy). Works on "
                "any hardware (AMD, CPU, Metal) but lower quality."
            ),
            inputs=[
                io.String.Input(
                    "gaussian_ply_path",
                    tooltip="Path to a 3DGS .ply file."),
                io.Int.Input(
                    "width", default=2048, min=256, max=8192, step=64,
                    tooltip="Output panorama width. Height = width / 2."),
                io.Int.Input(
                    "face_resolution", default=1024, min=256, max=4096, step=64,
                    tooltip="Resolution of each cube face render. Higher = "
                            "better quality but slower."),
                io.Combo.Input(
                    "backend",
                    options=["gpu", "cpu"],
                    default="gpu",
                    tooltip="gpu: CUDA anisotropic rasterizer (fast, high "
                            "quality). cpu: numpy disk-splat (works everywhere)."),
            ],
            outputs=[
                io.Custom(PANORAMA_TYPE).Output(display_name="panorama"),
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, gaussian_ply_path, width=2048, face_resolution=1024,
                backend="gpu"):
        import time
        t0 = time.perf_counter()

        erp_w = int(width)
        erp_h = erp_w // 2
        face_size = int(face_resolution)

        _p(f"loading {gaussian_ply_path}")
        xyz, rgb, scale, opacity, rot = _load_gaussian_ply(gaussian_ply_path)
        _p(f"  {len(xyz)} gaussians loaded")

        if backend == "gpu" and torch.cuda.is_available():
            _p(f"rendering 6 cube faces @ {face_size}px (GPU)")
            faces = _render_6_faces_gpu(xyz, rgb, scale, opacity, rot, face_size)
        else:
            if backend == "gpu":
                _p("CUDA not available, falling back to CPU backend")
            _p(f"rendering 6 cube faces @ {face_size}px (CPU)")
            faces = _render_6_faces_cpu(xyz, rgb, scale, opacity, face_size)

        _p(f"stitching to {erp_w}x{erp_h} equirect")
        erp = cube_faces_to_equirect(faces, erp_w, erp_h)
        erp = np.clip(erp, 0, 1).astype(np.float32)

        _p(f"done in {time.perf_counter() - t0:.2f}s")

        erp_t = torch.from_numpy(erp).unsqueeze(0)  # (1, H, W, 3)
        pano = wrap_image_as_panorama(erp_t)
        return io.NodeOutput(pano, erp_t)


NODE_CLASS_MAPPINGS = {"PanoRenderGaussians": PanoRenderGaussians}
NODE_DISPLAY_NAME_MAPPINGS = {"PanoRenderGaussians": "Pano Render Gaussians"}
