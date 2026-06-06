"""PanoRenderGaussians — render a gaussian splat PLY as an ERP panorama."""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
from comfy_api.latest import io

from .utils import PANORAMA_TYPE, wrap_image_as_panorama
from .utils.cube_to_equirect import FACE_FOV_DEG, cube_faces_to_equirect

_C0 = 0.28209479177387814


def _p(msg: str) -> None:
    print(f"[PanoRenderGaussians] {msg}", file=sys.stderr, flush=True)


def _resolve_ply_path(p: str) -> str:
    """Resolve a PLY path: as-given, then relative to the ComfyUI base dir, then input dir.

    Lets workflows pass base-relative paths like ``/input/3d/x.ply`` (which the worker's CWD
    can't see) while still honoring genuine absolute paths.
    """
    if os.path.isfile(p):
        return p
    import folder_paths

    rel = p.lstrip("/\\")
    candidates = [
        os.path.join(folder_paths.base_path, rel),   # /input/3d/x.ply -> <base>/input/3d/x.ply
        os.path.join(folder_paths.get_input_directory(), rel),  # 3d/x.ply -> <input>/3d/x.ply
    ]
    try:
        candidates.insert(1, folder_paths.get_annotated_filepath(rel))  # honors [input]/[output] tags
    except Exception:
        pass
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"PanoRenderGaussians: PLY not found: {p!r} (tried {[p, *candidates]})")


def _load_gaussian_ply(path: str):
    """Load a 3DGS .ply file. Returns (xyz, rgb, scale, opacity, rot)."""
    path = _resolve_ply_path(path)
    # Reuse ComfyUI's robust PLY parser
    sys_path = sys.path.copy()
    try:
        from comfy_extras.nodes_gaussian_splat import _parse_ply_gaussian
    finally:
        sys.path = sys_path

    with open(path, "rb") as f:
        data = f.read()
    _p(f"  parsing {len(data)} bytes from {path}")
    xyz, scale, rot, opacity, sh = _parse_ply_gaussian(data)
    _p(f"  parsed: xyz{xyz.shape} scale{scale.shape} rot{rot.shape} "
       f"opacity{opacity.shape} sh{sh.shape}")
    _p(f"  raw sh[:,0] (DC) min={float(sh[:, 0].min()):.3f} mean={float(sh[:, 0].mean()):.3f} "
       f"max={float(sh[:, 0].max()):.3f}")
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


def _render_6_faces_cpu(xyz, rgb, scale, opacity, face_size, gain=2.0, max_px=40):
    """Render 6 cube faces from the origin using the numpy disk-splat renderer.

    This is the only renderer that works for inside-out panoramic capture: the
    GPU anisotropic rasterizer is an orbit-preview tool and produces nothing when
    the camera sits inside the cloud at the world origin.
    """
    from comfy.ldm.triposplat.preview import render_splat

    # render_splat expects a scalar per-splat radius; 3DGS PLYs store a 3-axis
    # (anisotropic) scale, so collapse to an isotropic mean. Passing (N,3) raises
    # a broadcast error inside render_splat.
    scale = np.asarray(scale, dtype=np.float32)
    if scale.ndim == 2:
        scale = scale.mean(axis=1)

    _p(f"  CPU disk-splat: fov={FACE_FOV_DEG} dist=0.0 min_opacity=0.01 "
       f"gain={gain} min_px=1 max_px={max_px} over {len(xyz)} splats "
       f"(scale scalar med={float(np.median(scale)):.4f})")
    faces = []
    for view in _CUBE_VIEWS:
        img = render_splat(
            xyz, rgb, scale, opacity=opacity,
            yaw=float(view["yaw"]),
            pitch=float(view["pitch"]),
            size=face_size, fov=FACE_FOV_DEG, dist=0.0,
            min_px=1, max_px=max_px, gain=gain,
            min_opacity=0.01,
        )
        face_np = np.array(img, dtype=np.float32) / 255.0
        nz = int((face_np.reshape(-1, face_np.shape[-1]).max(-1) > 1e-4).sum())
        _p(f"  CPU face {view['name']} (yaw={view['yaw']},pitch={view['pitch']}): "
           f"nonblack={nz} max={float(face_np.max()):.3f}")
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
    _p(f"  GPU backend: device={dev} fov={FACE_FOV_DEG} splat_scale=1.0 bg=black "
       f"over {len(xyz)} splats")

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
        cam_info = _lookat_camera_info(origin, target, fov=FACE_FOV_DEG, dev=dev)
        img, mask = _render_gaussian(
            xyz, rgb, opacity, scale, rot,
            width=face_size, height=face_size,
            splat_scale=1.0, bg=bg,
            camera_info=cam_info,
            render_style="color",
        )
        face_np = img.numpy().astype(np.float32)
        nz = int((face_np.reshape(-1, face_np.shape[-1]).max(-1) > 1e-4).sum())
        cov = float(mask.float().mean()) if hasattr(mask, "float") else float("nan")
        _p(f"  GPU face {view['name']} target={[round(float(x), 2) for x in target.tolist()]}: "
           f"nonblack={nz} coverage={cov:.4f} max={float(face_np.max()):.3f}")
        faces.append(face_np)
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
                "Rendered with a numpy disk-splat renderer. (The CUDA "
                "anisotropic rasterizer is an orbit-preview tool and cannot "
                "render from inside the cloud, so it is not used here.)"
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
                io.Float.Input(
                    "gain", default=2.0, min=0.5, max=15.0, step=0.5,
                    tooltip="Splat footprint scale. Higher = thicker splats / "
                            "fuller coverage (fewer black gaps), lower = finer."),
                io.Combo.Input(
                    "backend",
                    options=["cpu", "gpu"],
                    default="cpu",
                    tooltip="Renderer. 'cpu' is the numpy disk-splat renderer "
                            "(the only one that works for panoramic capture). "
                            "'gpu' falls back to it with a warning."),
            ],
            outputs=[
                io.Custom(PANORAMA_TYPE).Output(display_name="panorama"),
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, gaussian_ply_path, width=2048, face_resolution=1024,
                gain=2.0, backend="cpu"):
        import time
        t0 = time.perf_counter()

        erp_w = int(width)
        erp_h = erp_w // 2
        face_size = int(face_resolution)

        _p(f"=== PanoRenderGaussians: backend={backend} gain={gain} "
           f"width={width} face_res={face_resolution} ===")
        resolved = _resolve_ply_path(gaussian_ply_path)
        _p(f"loading {gaussian_ply_path} -> {resolved}")
        xyz, rgb, scale, opacity, rot = _load_gaussian_ply(resolved)
        _p(f"  {len(xyz)} gaussians loaded")
        if len(xyz):
            # Diagnostics: a black render usually means the splats are all far
            # outside the origin cameras, fully transparent, or sub-pixel small.
            lo, hi = xyz.min(0), xyz.max(0)
            dist = np.linalg.norm(xyz, axis=1)
            _p(f"  xyz bbox min={lo.round(3).tolist()} max={hi.round(3).tolist()}")
            _p(f"  |xyz| from origin: min={dist.min():.3f} med={np.median(dist):.3f} max={dist.max():.3f}")
            _p(f"  opacity: min={opacity.min():.3f} med={np.median(opacity):.3f} max={opacity.max():.3f}")
            _p(f"  scale:   min={scale.min():.4f} med={np.median(scale):.4f} max={scale.max():.4f}")
            _p(f"  rgb:     min={rgb.min():.3f} mean={rgb.mean():.3f} max={rgb.max():.3f}")

        if backend == "gpu":
            _p("backend='gpu' requested, but the CUDA anisotropic rasterizer is an "
               "orbit-preview tool and renders nothing from inside the cloud at the "
               "origin. Falling back to the CPU disk-splat renderer.")
        _p(f"rendering 6 cube faces @ {face_size}px (CPU disk-splat, gain={gain})")
        faces = _render_6_faces_cpu(xyz, rgb, scale, opacity, face_size, gain=float(gain))

        nonblack = [int((f.reshape(-1, f.shape[-1]).max(-1) > 1e-4).sum()) for f in faces]
        _p(f"  non-black px per face [+X,-X,+Y,-Y,+Z,-Z]: {nonblack}")

        _p(f"stitching to {erp_w}x{erp_h} equirect (FOV={FACE_FOV_DEG})")
        erp = cube_faces_to_equirect(faces, erp_w, erp_h)
        erp = np.clip(erp, 0, 1).astype(np.float32)
        nz = int((erp.reshape(-1, 3).max(-1) > 1e-4).sum())
        _p(f"  ERP: nonblack={nz}/{erp_w * erp_h} ({100.0 * nz / (erp_w * erp_h):.1f}%) "
           f"min={float(erp.min()):.3f} mean={float(erp.mean()):.3f} max={float(erp.max()):.3f}")

        _p(f"done in {time.perf_counter() - t0:.2f}s")

        erp_t = torch.from_numpy(erp).unsqueeze(0)  # (1, H, W, 3)
        pano = wrap_image_as_panorama(erp_t)
        return io.NodeOutput(pano, erp_t)


NODE_CLASS_MAPPINGS = {"PanoRenderGaussians": PanoRenderGaussians}
NODE_DISPLAY_NAME_MAPPINGS = {"PanoRenderGaussians": "Pano Render Gaussians"}
