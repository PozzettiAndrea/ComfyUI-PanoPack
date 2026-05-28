"""PanoramaSplit — equirect panorama → N rectilinear faces.

Lets the user run any rectilinear depth model (MoGe2, DepthAnythingV3, …)
on a panorama without that model needing to be panorama-aware: split the
sphere into N evenly-tiled cube/icosahedron faces, feed each face through
the depth model, then stitch the per-face depths back via WorldNavDepthMerge.

Outputs:
- face_images IMAGE (B=N, H, W, 3) — rectilinear views, fov_x=fov_y=90°
- extrinsics EXTRINSICS [N, 4, 4]
- intrinsics INTRINSICS [N, 3, 3]
- fov_x_deg FLOAT (90) — convenience for the rectilinear depth node's fov widget

User then wires `face_images → MoGe2Inference` (or similar) and pipes the
per-face depths + the extrinsics + intrinsics back into WorldNavDepthMerge.
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np
import torch
from comfy_api.latest import io


def _p(msg: str) -> None:
    print(f"[PanoramaSplit] {msg}", file=sys.stderr, flush=True)


def _hsv_color_bgr(idx: int, total: int) -> tuple[int, int, int]:
    """Distinct BGR color per face via HSV color wheel (cv2 expects BGR)."""
    import cv2
    hue = int(round(180.0 * idx / max(total, 1))) % 180
    hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _make_pano_debug_overlay(
    panorama_u8: np.ndarray,
    extrinsics: np.ndarray,
    fov_rad: float,
) -> np.ndarray:
    """Draw each face's frustum edges on the panorama (WorldNav convention).

    For each face: sample N=64 points along each of 4 frustum edges at
    z=1 in camera space (corners at (±tan(fov/2), ±tan(fov/2), 1)),
    transform to world via R_c2w = R_w2c^T, normalize, convert to
    (theta, phi) in WorldNav's spherical convention, then to equirect
    pixel coords. Draw a polyline per edge with a distinct HSV-wheel hue.
    Split any edge that crosses the seam into separate segments.

    WorldNav convention (matches `spherical_uv_to_directions` upstream):
      world up = +Z
      theta = atan2(ry, rx)            azimuth around Z, in [0, 2π)
      phi   = arccos(rz)               polar angle from +Z, in [0, π]
      u_erp = (1 - theta / (2π)) · (W - 1)
      v_erp = (phi / π)         · (H - 1)

    Args:
        panorama_u8: (H, W, 3) uint8 RGB panorama.
        extrinsics:  (N, 4, 4) world-to-camera (from utils3d look-at).
        fov_rad:     per-face horizontal+vertical FOV (square views).

    Returns:
        (H, W, 3) uint8 RGB panorama with N frustums drawn on it.
    """
    import cv2
    H, W = panorama_u8.shape[:2]
    debug = panorama_u8.copy()
    N = int(extrinsics.shape[0])
    ext_np = np.asarray(extrinsics, dtype=np.float32)

    S = 64
    half = math.tan(fov_rad / 2.0)
    edge_corners = [
        ([-half, -half], [+half, -half]),  # top
        ([+half, -half], [+half, +half]),  # right
        ([+half, +half], [-half, +half]),  # bottom
        ([-half, +half], [-half, -half]),  # left
    ]

    for i in range(N):
        # extrinsics is w2c; invert (orthonormal rotation) for c2w.
        R_c2w = ext_np[i, :3, :3].T
        color = _hsv_color_bgr(i, N)
        for (p0, p1) in edge_corners:
            t = np.linspace(0.0, 1.0, S, dtype=np.float32)
            xs = p0[0] + t * (p1[0] - p0[0])
            ys = p0[1] + t * (p1[1] - p0[1])
            cam_dirs = np.stack(
                [xs, ys, np.ones_like(xs)], axis=-1,
            )  # (S, 3)
            world_dirs = cam_dirs @ R_c2w.T   # row * R^T == R @ col
            world_dirs /= np.maximum(
                np.linalg.norm(world_dirs, axis=-1, keepdims=True), 1e-12,
            )
            rx, ry, rz = world_dirs[:, 0], world_dirs[:, 1], world_dirs[:, 2]
            theta = np.arctan2(ry, rx) % (2.0 * np.pi)             # [0, 2π)
            phi = np.arccos(np.clip(rz, -1.0, 1.0))                # [0, π]
            u = (1.0 - theta / (2.0 * np.pi)) * (W - 1)
            v = (phi / np.pi) * (H - 1)
            pts = np.stack([u, v], axis=-1).astype(np.float32)

            # Split at azimuth wraparound: any consecutive pair with
            # |Δu| > W/2 wrapped the seam.
            du = np.abs(np.diff(pts[:, 0]))
            breaks = np.where(du > W * 0.5)[0]
            segments = np.split(pts, breaks + 1) if len(breaks) else [pts]
            for seg in segments:
                if len(seg) < 2:
                    continue
                seg_int = seg.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(
                    debug, [seg_int], isClosed=False,
                    color=color, thickness=2, lineType=cv2.LINE_AA,
                )
    return debug


class PanoramaSplit(io.ComfyNode):
    """Panorama → N rectilinear face images + per-face extrinsics/intrinsics."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PanoramaSplit",
            display_name="Panorama Split",
            category="PanoPack",
            description=(
                "Split an equirectangular panorama into N rectilinear views "
                "(90° fov each). Pipe the views through any rectilinear depth "
                "model (e.g. MoGe2Inference), then merge the per-view depths "
                "back into an equirect depth map via WorldNavDepthMerge.\n\n"
                "Mirrors upstream HY-World's pred_pano_depth split step."
            ),
            inputs=[
                io.Custom("WORLDSTEREO_PANORAMA").Input(
                    "panorama",
                    tooltip="Equirectangular RGB panorama (2:1). "
                            "WORLDSTEREO_PANORAMA = IMAGE wrapped in a "
                            "custom socket type. Wire from "
                            "WorldStereoLoadPanorama OR a native IMAGE via "
                            "WorldStereoPanoramaWrap."),
                io.Int.Input(
                    "resolution", default=512, min=128, max=2048, step=64,
                    tooltip="Per-face image resolution (square). Default 512 matches upstream."),
                io.Combo.Input(
                    "subdivision",
                    options=["icosahedron_12", "icosahedron_42"],
                    default="icosahedron_42",
                    tooltip="Tiling density. 12 = base icosahedron (faster), "
                            "42 = subdivided (better polar coverage)."),
                io.Float.Input(
                    "fov_degrees", default=90.0, min=30.0, max=170.0, step=1.0,
                    optional=True,
                    tooltip="Per-face FOV in degrees (square — same FOV "
                            "horizontal and vertical). Default 90° matches "
                            "upstream HY-World and gives full sphere coverage "
                            "with icosahedron_42.\n\n"
                            "Coverage trade-off: narrower FOV (e.g. 60°) "
                            "gives sharper per-face content (MoGe2 / other "
                            "rectilinear depth models stay in-distribution) "
                            "but may leave sphere gaps with icosahedron_12. "
                            "Wider FOV (e.g. 120°) gives more redundancy but "
                            "per-face image quality drops past ~110° as "
                            "rectilinear distortion grows near corners."),
                io.Boolean.Input(
                    "use_gpu", default=True,
                    optional=True,
                    tooltip="Batched panorama→face resampling via "
                            "torch.nn.functional.grid_sample (bilinear). "
                            "All N faces sampled in one kernel launch. "
                            "Falls back to CPU cv2.remap loop if CUDA "
                            "isn't available. Math identical within fp32 "
                            "round-off."),
            ],
            outputs=[
                io.Image.Output(display_name="face_images"),
                io.Custom("EXTRINSICS").Output(display_name="extrinsics"),
                io.Custom("INTRINSICS").Output(display_name="intrinsics"),
                io.Float.Output(display_name="fov_x_deg"),
                io.Image.Output(
                    display_name="debug_image",
                    tooltip=(
                        "Original panorama with each face's frustum edges "
                        "drawn on it (HSV-colored polylines, one color per "
                        "face in icosahedron-vertex order). Useful for "
                        "visually verifying coverage and per-face "
                        "orientation."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, panorama, resolution=512, subdivision="icosahedron_42",
                fov_degrees=90.0, use_gpu=True):
        from PIL import Image
        import cv2
        from ._vendor.moge_panorama import (
            get_panorama_cameras, split_panorama_image,
            split_panorama_image_gpu,
        )
        from ._vendor.worldgen.src.panorama_utils import subdivide_icosahedron
        import utils3d

        t_total = time.perf_counter()

        # --- panorama → numpy uint8 (H, W, 3) ---
        arr = panorama.detach().cpu().numpy() if isinstance(panorama, torch.Tensor) else np.asarray(panorama)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        H, W = arr.shape[:2]

        # --- pick the icosahedron vertices ---
        t_geom = time.perf_counter()
        if subdivision == "icosahedron_12":
            vertices, _ = utils3d.np.create_icosahedron_mesh()
        elif subdivision == "icosahedron_42":
            vertices = subdivide_icosahedron(subdivisions=1)
        else:
            raise ValueError(f"PanoramaSplit: unknown subdivision {subdivision!r}")
        N = len(vertices)

        # User-selectable square FOV. Default 90° matches upstream HY-World.
        fov_rad = math.radians(float(fov_degrees))
        intrinsics_one = utils3d.np.intrinsics_from_fov(
            fov_x=fov_rad, fov_y=fov_rad,
        )
        # utils3d.np.extrinsics_look_at(eye, target, up=[0,0,1]) silently
        # produces NaN extrinsics when forward is parallel to up
        # (cross(forward, up) = 0 -> zero right-vector -> divide by zero).
        # The level-1 icosahedron has exactly 2 vertices on the +/- Z poles
        # (edge midpoints of (0,1,phi)<->(0,-1,phi) normalize to (0,0,1)),
        # so 2 frames come back black. Upstream HY-World has the same bug
        # in get_panorama_cameras_v2 -- they tolerate it by masking the
        # garbage MoGe2 outputs downstream, but we'd rather not waste the
        # compute or trigger utils3d's divide-by-zero warning.
        #
        # Fix: pick `up` per vertex. If forward is within ~2.5 deg of the
        # default up (cos > 0.999), swap to an orthogonal up. Identical to
        # default for non-pole vertices.
        _UP_DEFAULT = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        _UP_FALLBACK = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        _PARALLEL_COS = 0.999
        eye = np.zeros(3, dtype=np.float32)
        verts = np.asarray(vertices, dtype=np.float32)
        forwards = verts - eye
        forwards = forwards / np.maximum(
            np.linalg.norm(forwards, axis=-1, keepdims=True), 1e-12)
        parallel = np.abs(forwards @ _UP_DEFAULT) > _PARALLEL_COS  # (N,) bool
        extrinsics = np.empty((N, 4, 4), dtype=np.float32)
        # Batched call for the non-parallel majority.
        if (~parallel).any():
            extrinsics[~parallel] = utils3d.np.extrinsics_look_at(
                eye, verts[~parallel], _UP_DEFAULT,
            ).astype(np.float32)
        # Per-vertex call with the fallback up for poles.
        if parallel.any():
            extrinsics[parallel] = utils3d.np.extrinsics_look_at(
                eye, verts[parallel], _UP_FALLBACK,
            ).astype(np.float32)
            _p(f"  swapped up=[0,1,0] for {int(parallel.sum())} pole vertices "
               f"(forward||[0,0,1] within {np.degrees(np.arccos(_PARALLEL_COS)):.1f}°)")
        intrinsics = np.stack([intrinsics_one] * N, axis=0).astype(np.float32)
        t_geom_done = time.perf_counter()

        # --- rasterize the panorama into N face images ---
        t_split = time.perf_counter()
        if use_gpu and torch.cuda.is_available():
            splitted = split_panorama_image_gpu(arr, extrinsics, intrinsics, resolution)
            backend = "grid_sample (GPU)"
        else:
            splitted = split_panorama_image(arr, extrinsics, intrinsics, resolution)
            backend = ("cv2.remap (CPU, cuda unavailable)"
                       if use_gpu else "cv2.remap (CPU, use_gpu=False)")
        t_split_done = time.perf_counter()
        # splitted is list of (resolution, resolution, 3) uint8 arrays. Stack + normalize.
        face_stack = np.stack(splitted, axis=0).astype(np.float32) / 255.0  # (N, R, R, 3)
        face_t = torch.from_numpy(face_stack)

        # --- Debug overlay: original panorama with each face's frustum
        # edges drawn as a colored polyline. `arr` is already uint8 RGB.
        debug_np = _make_pano_debug_overlay(arr, extrinsics, fov_rad)
        debug_t = (
            torch.from_numpy(debug_np.astype(np.float32) / 255.0).unsqueeze(0)
        )  # [1, H, W, 3]

        _p(f"({W}×{H} RGB) → {N} faces @ {resolution}×{resolution}, "
           f"fov={fov_degrees:.1f}° via {backend}; "
           f"cameras {t_geom_done - t_geom:.3f}s, split {t_split_done - t_split:.3f}s, "
           f"total {time.perf_counter() - t_total:.3f}s")

        return io.NodeOutput(
            face_t,
            torch.from_numpy(extrinsics),
            torch.from_numpy(intrinsics),
            float(fov_degrees),
            debug_t,
        )


NODE_CLASS_MAPPINGS = {"PanoramaSplit": PanoramaSplit}
NODE_DISPLAY_NAME_MAPPINGS = {"PanoramaSplit": "Panorama Split"}
