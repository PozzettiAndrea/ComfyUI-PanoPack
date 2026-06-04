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

from .utils import PANORAMA_TYPE, unwrap_panorama_to_image


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
                io.Custom(PANORAMA_TYPE).Input(
                    "panorama",
                    tooltip="Equirectangular RGB panorama (2:1). PANORAMA = "
                            "IMAGE wrapped in PanoPack's typed socket. Wire "
                            "from PanoramaWrap or any node that emits a "
                            "PANORAMA."),
                io.Int.Input(
                    "resolution", default=952, min=128, max=2048, step=1,
                    tooltip="Per-face image resolution (square). Any integer."),
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
                io.Boolean.Input(
                    "create_masks", default=False,
                    optional=True,
                    tooltip="Generate per-face masks. Wire into SHARP's "
                            "mask input to delete overlapping gaussians."),
                io.Combo.Input(
                    "mask_method",
                    options=["closest_to"],
                    default="closest_to",
                    optional=True,
                    tooltip="closest_to: each pixel in a crop is 1 only if "
                            "it's closer to this crop's center than to any "
                            "other crop's center (Voronoi on the sphere). "
                            "Eliminates overlap between adjacent faces."),
            ],
            outputs=[
                io.Image.Output(display_name="face_images"),
                io.Custom("EXTRINSICS").Output(display_name="extrinsics"),
                io.Custom("INTRINSICS").Output(display_name="intrinsics"),
                io.Float.Output(display_name="fov_x_deg"),
                io.Mask.Output(
                    display_name="face_masks",
                    tooltip="Per-face masks (N, H, W). Only emitted when "
                            "create_masks is enabled; otherwise zeros."),
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
                io.Image.Output(
                    display_name="debug_masks",
                    tooltip="Equirect image with each pixel colored by its "
                            "Voronoi cell (which face owns it). Only "
                            "populated when create_masks is enabled."),
            ],
        )

    @classmethod
    def execute(cls, panorama, resolution=512, subdivision="icosahedron_42",
                fov_degrees=90.0, use_gpu=True, create_masks=False,
                mask_method="closest_to"):
        from PIL import Image
        import cv2
        from ._vendor.moge_panorama import (
            get_panorama_cameras, split_panorama_image,
            split_panorama_image_gpu,
        )
        from ._vendor.worldgen.src.panorama_utils import subdivide_icosahedron
        import utils3d

        t_total = time.perf_counter()

        # --- panorama → numpy (H, W, 3), preserving value range ---
        # Three input cases:
        #   1. uint8 [0, 255]                     — visual RGB
        #   2. float32 [0, 1]                     — visual RGB (ComfyUI IMAGE convention)
        #   3. float32 with values outside [0, 1] — DATA panorama (depth in meters,
        #      ||point|| ray-distance, scalar field, etc.)
        # The split helpers (`split_panorama_image` via cv2.remap,
        # `split_panorama_image_gpu` via grid_sample) handle uint8 and float32
        # alike; they preserve the input dtype. Previously this node UNCONDITIONALLY
        # quantized non-uint8 input to uint8 [0, 255] via `np.clip(arr, 0, 1) * 255`
        # — fine for visual RGB, FATAL for metric depth panoramas (0.5–50 m): values
        # outside [0, 1] saturate, and the post-split /255 recovers a [0, 1] float
        # with no magnitude. Symptom: gaussian splat collapses to a sphere because
        # all gaussians end up at ~uniform distance from the camera.
        pano_t = unwrap_panorama_to_image(panorama)
        arr_orig = pano_t.detach().cpu().numpy() if isinstance(pano_t, torch.Tensor) else np.asarray(pano_t)
        if arr_orig.ndim == 4:
            arr_orig = arr_orig[0]

        if arr_orig.dtype == np.uint8:
            # Case 1: visual RGB uint8. Pass straight through; recover [0, 1]
            # float at the end by dividing by 255.
            arr = arr_orig
            face_norm_divisor = 255.0
            arr_for_overlay_u8 = arr_orig
            data_panorama = False
        elif np.issubdtype(arr_orig.dtype, np.floating):
            arr = arr_orig.astype(np.float32, copy=False)
            arr_min = float(arr.min())
            arr_max = float(arr.max())
            # 1e-3 slack so a visual panorama at exactly 1.0 doesn't get
            # mistaken for a data panorama.
            in_unit_range = arr_min >= -1e-3 and arr_max <= 1.0 + 1e-3
            if in_unit_range:
                # Case 2: visual RGB float [0, 1]. Helpers return float [0, 1];
                # no rescale needed at the end.
                face_norm_divisor = 1.0
                arr_for_overlay_u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                data_panorama = False
            else:
                # Case 3: DATA panorama (depth, etc.). Preserve raw float values
                # through the split — helpers will return float32 with the same
                # metric magnitudes. For the debug overlay (which uses cv2 to
                # draw colored polylines and needs uint8 RGB), build a
                # min-max-normalized visualization.
                face_norm_divisor = 1.0
                if arr_max > arr_min:
                    vis = ((arr - arr_min) / (arr_max - arr_min) * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    vis = np.zeros(arr.shape, dtype=np.uint8)
                if vis.ndim == 2:
                    vis = np.repeat(vis[..., None], 3, axis=-1)
                elif vis.shape[-1] == 1:
                    vis = np.repeat(vis, 3, axis=-1)
                arr_for_overlay_u8 = np.ascontiguousarray(vis)
                data_panorama = True
                _p(f"  data panorama detected (float, range [{arr_min:.3g}, {arr_max:.3g}]) "
                   f"→ preserving raw values through split; overlay uses min-max normalize")
        else:
            raise TypeError(
                f"PanoramaSplit: unsupported panorama dtype {arr_orig.dtype} "
                f"(expected uint8 or float)"
            )
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
        # Stack + recover the ComfyUI IMAGE convention (float32, range
        # depends on input case):
        #   - uint8 in  → splitted is uint8 → /255.0 → float [0, 1]
        #   - float [0,1] in → splitted is float [0, 1] → /1.0 → float [0, 1]
        #   - data panorama in → splitted is float (metric units) → /1.0 →
        #     PRESERVED magnitudes (this is the whole point of the dtype branching).
        face_stack = np.stack(splitted, axis=0).astype(np.float32) / face_norm_divisor  # (N, R, R, C)
        face_t = torch.from_numpy(face_stack)

        # --- Debug overlay: original panorama with each face's frustum
        # edges drawn as a colored polyline. Uses the uint8 RGB
        # visualization we built above (raw for visual panoramas;
        # min-max normalized for data panoramas).
        debug_np = _make_pano_debug_overlay(arr_for_overlay_u8, extrinsics, fov_rad)
        debug_t = (
            torch.from_numpy(debug_np.astype(np.float32) / 255.0).unsqueeze(0)
        )  # [1, H, W, 3]

        # --- Per-face masks (Voronoi on sphere) ---
        if create_masks and mask_method == "closest_to":
            _p("generating closest_to masks...")
            t_mask = time.perf_counter()
            # For each face i, for each pixel (u, v) in that face:
            #   1. Unproject (u, v) to a 3D direction on the unit sphere
            #   2. Check which face center (icosahedron vertex) is closest
            #   3. mask[i, v, u] = 1.0 if face i is the closest, else 0.0
            R = resolution
            uv = utils3d.np.uv_map((R, R))  # (R, R, 2) normalized pixel coords
            face_dirs = verts / np.maximum(
                np.linalg.norm(verts, axis=-1, keepdims=True), 1e-12
            )  # (N, 3) unit direction per face center

            masks = np.zeros((N, R, R), dtype=np.float32)
            for i in range(N):
                # Unproject this face's pixel grid to world 3D directions
                pixel_dirs = utils3d.np.unproject_cv(
                    uv, np.ones_like(uv[..., 0]),
                    extrinsics=extrinsics[i], intrinsics=intrinsics[i],
                )  # (R, R, 3)
                pixel_dirs = pixel_dirs / np.maximum(
                    np.linalg.norm(pixel_dirs, axis=-1, keepdims=True), 1e-12
                )
                # Dot product with all face centers: (R, R, N)
                dots = np.einsum("hwc,nc->hwn", pixel_dirs, face_dirs)
                # This pixel belongs to face i if face i has the highest dot product
                closest_face = np.argmax(dots, axis=-1)  # (R, R)
                masks[i] = (closest_face == i).astype(np.float32)

            face_masks_t = torch.from_numpy(masks)  # (N, R, R)
            _p(f"masks done in {time.perf_counter() - t_mask:.3f}s; "
               f"avg coverage per face: {masks.mean():.1%}")

            # Debug masks: color each equirect pixel by its Voronoi cell
            _p("generating debug_masks equirect image...")
            from .panorama_split_adaptive import _equirect_ray_dirs
            eq_rays = _equirect_ray_dirs(H, W).reshape(-1, 3)
            eq_dots = eq_rays @ face_dirs.T
            eq_assignment = np.argmax(eq_dots, axis=1)

            hsv_colors = np.zeros((N, 3), dtype=np.float32)
            for i in range(N):
                hue = float(i) / max(N, 1)
                h6 = hue * 6.0
                c = 0.8
                x = c * (1 - abs(h6 % 2 - 1))
                if h6 < 1:   r, g, b = c, x, 0
                elif h6 < 2: r, g, b = x, c, 0
                elif h6 < 3: r, g, b = 0, c, x
                elif h6 < 4: r, g, b = 0, x, c
                elif h6 < 5: r, g, b = x, 0, c
                else:        r, g, b = c, 0, x
                hsv_colors[i] = [r + 0.2, g + 0.2, b + 0.2]

            debug_masks_np = hsv_colors[eq_assignment].reshape(H, W, 3)
            pano_f = arr_for_overlay_u8[..., :3].astype(np.float32) / 255.0
            debug_masks_np = np.clip(0.5 * debug_masks_np + 0.5 * pano_f, 0, 1)
            debug_masks_t = torch.from_numpy(debug_masks_np.astype(np.float32)).unsqueeze(0)
        else:
            face_masks_t = torch.zeros(N, resolution, resolution)
            debug_masks_t = torch.zeros((1, H, W, 3), dtype=torch.float32)

        _p(f"({W}×{H} RGB) → {N} faces @ {resolution}×{resolution}, "
           f"fov={fov_degrees:.1f}° via {backend}; "
           f"cameras {t_geom_done - t_geom:.3f}s, split {t_split_done - t_split:.3f}s, "
           f"total {time.perf_counter() - t_total:.3f}s")

        return io.NodeOutput(
            face_t,
            torch.from_numpy(extrinsics),
            torch.from_numpy(intrinsics),
            float(fov_degrees),
            face_masks_t,
            debug_t,
            debug_masks_t,
        )


NODE_CLASS_MAPPINGS = {"PanoramaSplit": PanoramaSplit}
NODE_DISPLAY_NAME_MAPPINGS = {"PanoramaSplit": "Panorama Split"}
