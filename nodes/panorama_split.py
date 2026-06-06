"""PanoramaSplit - equirect panorama -> N rectilinear crops (3 methods).

One node, three camera layouts via the `split_method` dropdown:
  - icosahedron_12 : 12 base-icosahedron vertices, square crops (for per-face
    rectilinear depth, e.g. MoGe2 -> PanoramaDepthMerge).
  - icosahedron_42 : level-1 subdivided icosahedron (42 dirs), square crops.
  - cube_split     : fixed pitch x yaw grid (3 pitches x N yaws), RECTANGULAR
    crops at separate fov_x/fov_y (seeds a WorldStereo memory bank with the
    crops the model was trained against).

All methods share one core: directions -> per-camera extrinsics/intrinsics ->
sample crops (GPU grid_sample / CPU cv2.remap) -> optional spherical-Voronoi
masks -> memory-bank entries -> debug overlay. Every output is produced for
every method (none are method-specific), so downstream pipelines (MoGe depth,
WorldStereo bank) each just read the sockets they need.

Intrinsics conventions: the `intrinsics` output is NORMALIZED (uv in [0,1],
what PanoramaDepthMerge expects); the `entries` dict carries PIXEL-scale
intrinsics (the WorldStereo memory-bank convention).
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
    fov_x_rad: float,
    fov_y_rad: float,
) -> np.ndarray:
    """Draw each crop's 4 frustum edges on the panorama (rectangular frustum).

    Camera-space corners use separate half-FOVs on x vs y so the frustum
    matches the crop aspect (square for icosahedron, 832x480-ish for cube).
    Spherical convention matches `spherical_uv_to_directions`:
      theta = atan2(ry, rx), phi = arccos(rz),
      u = (1 - theta/2pi).(W-1), v = (phi/pi).(H-1).
    """
    import cv2
    H, W = panorama_u8.shape[:2]
    debug = panorama_u8.copy()
    N = int(extrinsics.shape[0])
    ext_np = np.asarray(extrinsics, dtype=np.float32)

    S = 64
    half_x = math.tan(fov_x_rad / 2.0)
    half_y = math.tan(fov_y_rad / 2.0)
    edge_corners = [
        ([-half_x, -half_y], [+half_x, -half_y]),  # top
        ([+half_x, -half_y], [+half_x, +half_y]),  # right
        ([+half_x, +half_y], [-half_x, +half_y]),  # bottom
        ([-half_x, +half_y], [-half_x, -half_y]),  # left
    ]

    for i in range(N):
        R_c2w = ext_np[i, :3, :3].T   # w2c -> c2w (orthonormal)
        color = _hsv_color_bgr(i, N)
        for (p0, p1) in edge_corners:
            t = np.linspace(0.0, 1.0, S, dtype=np.float32)
            xs = p0[0] + t * (p1[0] - p0[0])
            ys = p0[1] + t * (p1[1] - p0[1])
            cam_dirs = np.stack([xs, ys, np.ones_like(xs)], axis=-1)
            world_dirs = cam_dirs @ R_c2w.T
            world_dirs /= np.maximum(
                np.linalg.norm(world_dirs, axis=-1, keepdims=True), 1e-12)
            rx, ry, rz = world_dirs[:, 0], world_dirs[:, 1], world_dirs[:, 2]
            theta = np.arctan2(ry, rx) % (2.0 * np.pi)
            phi = np.arccos(np.clip(rz, -1.0, 1.0))
            u = (1.0 - theta / (2.0 * np.pi)) * (W - 1)
            v = (phi / np.pi) * (H - 1)
            pts = np.stack([u, v], axis=-1).astype(np.float32)
            du = np.abs(np.diff(pts[:, 0]))
            breaks = np.where(du > W * 0.5)[0]
            segments = np.split(pts, breaks + 1) if len(breaks) else [pts]
            for seg in segments:
                if len(seg) < 2:
                    continue
                seg_int = seg.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(debug, [seg_int], isClosed=False,
                              color=color, thickness=2, lineType=cv2.LINE_AA)
    return debug


def _camera_directions(method, *, resolution, fov_degrees,
                       image_w, image_h, fov_x_deg, fov_y_deg,
                       rot_deg, pitch_up, pitch_down):
    """Return (verts (N,3), crop_h, crop_w, fov_x_rad, fov_y_rad, fov_x_out)
    for the chosen split method."""
    import utils3d
    if method in ("icosahedron_12", "icosahedron_42"):
        if method == "icosahedron_12":
            verts, _ = utils3d.np.create_icosahedron_mesh()
        else:
            from ._vendor.worldgen.src.panorama_utils import subdivide_icosahedron
            verts = subdivide_icosahedron(subdivisions=1)
        verts = np.asarray(verts, dtype=np.float32)
        r = int(resolution)
        f = math.radians(float(fov_degrees))
        return verts, r, r, f, f, float(fov_degrees)

    if method == "cube_split":
        from ._vendor.worldstereo_cube import rotate_around_z_axis
        start_points = [
            np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            np.array([-1.0, 0.0, float(pitch_up)], dtype=np.float32),
            np.array([-1.0, 0.0, float(pitch_down)], dtype=np.float32),
        ]
        n_view = int(360 / max(1, int(rot_deg)))
        direct = list(start_points)
        for sp in start_points:
            for i in range(1, n_view):
                direct.append(rotate_around_z_axis(sp.reshape(1, 3), rot_deg * i)[0])
        verts = np.stack(direct, axis=0).astype(np.float32)
        return (verts, int(image_h), int(image_w),
                math.radians(float(fov_x_deg)), math.radians(float(fov_y_deg)),
                float(fov_x_deg))

    raise ValueError(f"PanoramaSplit: unknown split_method {method!r}")


class PanoramaSplit(io.ComfyNode):
    """Panorama -> N rectilinear crops + cameras + masks + memory-bank entries."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PanoramaSplit",
            display_name="Panorama Split",
            category="PanoPack",
            description=(
                "Split an equirect panorama into N rectilinear crops. Three "
                "camera layouts via split_method:\n"
                "  icosahedron_12 / icosahedron_42 - even sphere tiling, square "
                "crops at fov_degrees (feed per-crop depth e.g. MoGe2 -> "
                "PanoramaDepthMerge).\n"
                "  cube_split - 3 pitch x N yaw grid, rectangular crops at "
                "fov_x/fov_y (seeds a WorldStereo memory bank).\n\n"
                "Outputs are produced for every method: face_images, cameras, "
                "optional Voronoi masks, a debug overlay, and a MEMORY_BANK_"
                "ENTRIES dict."
            ),
            inputs=[
                io.Custom(PANORAMA_TYPE).Input(
                    "panorama",
                    tooltip="Equirectangular RGB panorama (2:1). Wire from "
                            "PanoramaWrap or any node that emits a PANORAMA."),
                io.DynamicCombo.Input(
                    "split_method",
                    tooltip="Camera layout. icosahedron_* = square crops (depth); "
                            "cube_split = rectangular pitch/yaw grid (memory bank).",
                    options=[
                        io.DynamicCombo.Option("icosahedron_42", []),
                        io.DynamicCombo.Option("icosahedron_12", []),
                        io.DynamicCombo.Option("cube_split", [
                            io.Int.Input("image_w", default=832, min=224, max=2048, step=16,
                                         tooltip="Crop width (832 = WorldStereo anchor)."),
                            io.Int.Input("image_h", default=480, min=224, max=2048, step=16,
                                         tooltip="Crop height (480 = WorldStereo anchor)."),
                            io.Float.Input("fov_x_deg", default=120.0, min=30.0, max=170.0, step=1.0,
                                           tooltip="Horizontal FOV (120 matches upstream)."),
                            io.Float.Input("fov_y_deg", default=90.0, min=30.0, max=170.0, step=1.0,
                                           tooltip="Vertical FOV (90 matches upstream)."),
                            io.Float.Input("rot_deg", default=40.0, min=10.0, max=120.0, step=5.0,
                                           tooltip="Yaw step. 40 -> 9 yaws/pitch -> 27 crops."),
                            io.Float.Input("pitch_up", default=0.5, min=0.0, max=1.0, step=0.05,
                                           tooltip="Z of the upper-hemisphere look-at points."),
                            io.Float.Input("pitch_down", default=-0.5, min=-1.0, max=0.0, step=0.05,
                                           tooltip="Z of the lower-hemisphere look-at points."),
                        ]),
                    ]),
                io.Int.Input(
                    "resolution", default=952, min=128, max=2048, step=1,
                    tooltip="Square per-crop resolution (icosahedron methods)."),
                io.Float.Input(
                    "fov_degrees", default=90.0, min=30.0, max=170.0, step=1.0,
                    tooltip="Square per-crop FOV in degrees (icosahedron methods). "
                            "90 matches upstream HY-World; full coverage with "
                            "icosahedron_42."),
                io.Boolean.Input(
                    "use_gpu", default=True, optional=True,
                    tooltip="Batched panorama->crop resampling via grid_sample "
                            "(all crops in one kernel). Falls back to CPU "
                            "cv2.remap if CUDA is unavailable."),
                io.Boolean.Input(
                    "create_masks", default=False, optional=True,
                    tooltip="Generate per-crop spherical-Voronoi masks (each "
                            "pixel kept only if its crop center is the nearest "
                            "of all crop centers) + a colored debug_masks image."),
                io.String.Input(
                    "fname", default="pano_bank", multiline=False, optional=True,
                    tooltip="Provenance tag stored per entry in the "
                            "MEMORY_BANK_ENTRIES output."),
            ],
            outputs=[
                io.Image.Output(display_name="face_images"),
                io.Custom("EXTRINSICS").Output(display_name="extrinsics"),
                io.Custom("INTRINSICS").Output(
                    display_name="intrinsics",
                    tooltip="NORMALIZED per-crop intrinsics [N,3,3] (uv in [0,1]) "
                            "- what PanoramaDepthMerge expects."),
                io.Float.Output(display_name="fov_x_deg"),
                io.Mask.Output(
                    display_name="face_masks",
                    tooltip="Per-crop Voronoi masks (N,H,W). Only when "
                            "create_masks is on; otherwise zeros."),
                io.Image.Output(
                    display_name="debug_image",
                    tooltip="Panorama with each crop's frustum edges drawn "
                            "(one HSV color per crop)."),
                io.Image.Output(
                    display_name="debug_masks",
                    tooltip="Equirect colored by Voronoi cell (adjacency-aware "
                            "palette so touching cells contrast). Only when "
                            "create_masks is on."),
                io.Custom("MEMORY_BANK_ENTRIES").Output(
                    display_name="entries",
                    tooltip="frames + extrinsics + PIXEL intrinsics + image_size "
                            "+ fnames, depths=None. Feed into MemoryBankAdd."),
            ],
        )

    @classmethod
    def execute(cls, panorama, split_method=None,
                resolution=952, fov_degrees=90.0,
                use_gpu=True, create_masks=False, fname="pano_bank"):
        import utils3d
        from ._vendor.moge_panorama import split_panorama_image_gpu
        from ._vendor.worldstereo_cube import split_panorama_image as split_rect

        t_total = time.perf_counter()

        sm = split_method if isinstance(split_method, dict) else {}
        method = sm.get("split_method", "icosahedron_42")

        # --- panorama -> numpy (H, W, C), preserving value range ---
        # uint8 visual / float[0,1] visual / float DATA (depth etc., kept raw).
        pano_t = unwrap_panorama_to_image(panorama)
        arr_orig = pano_t.detach().cpu().numpy() if isinstance(pano_t, torch.Tensor) else np.asarray(pano_t)
        if arr_orig.ndim == 4:
            arr_orig = arr_orig[0]
        if arr_orig.dtype == np.uint8:
            arr = arr_orig
            face_norm_divisor = 255.0
            arr_for_overlay_u8 = arr_orig
        elif np.issubdtype(arr_orig.dtype, np.floating):
            arr = arr_orig.astype(np.float32, copy=False)
            amin, amax = float(arr.min()), float(arr.max())
            if amin >= -1e-3 and amax <= 1.0 + 1e-3:
                face_norm_divisor = 1.0
                arr_for_overlay_u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            else:
                # DATA panorama: keep raw float through the split; overlay uses
                # a min-max-normalized visualization.
                face_norm_divisor = 1.0
                if amax > amin:
                    vis = ((arr - amin) / (amax - amin) * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    vis = np.zeros(arr.shape, dtype=np.uint8)
                if vis.ndim == 2:
                    vis = np.repeat(vis[..., None], 3, axis=-1)
                elif vis.shape[-1] == 1:
                    vis = np.repeat(vis, 3, axis=-1)
                arr_for_overlay_u8 = np.ascontiguousarray(vis)
                _p(f"  data panorama (float [{amin:.3g}, {amax:.3g}]) -> raw values preserved")
        else:
            raise TypeError(f"PanoramaSplit: unsupported panorama dtype {arr_orig.dtype}")
        H, W = arr.shape[:2]

        # --- camera directions + crop geometry for the chosen method ---
        t_geom = time.perf_counter()
        verts, crop_h, crop_w, fov_x_rad, fov_y_rad, fov_x_out = _camera_directions(
            method, resolution=resolution, fov_degrees=fov_degrees,
            image_w=int(sm.get("image_w", 832)), image_h=int(sm.get("image_h", 480)),
            fov_x_deg=float(sm.get("fov_x_deg", 120.0)), fov_y_deg=float(sm.get("fov_y_deg", 90.0)),
            rot_deg=float(sm.get("rot_deg", 40.0)),
            pitch_up=float(sm.get("pitch_up", 0.5)), pitch_down=float(sm.get("pitch_down", -0.5)),
        )
        N = len(verts)

        intrinsics_one = utils3d.np.intrinsics_from_fov(
            fov_x=fov_x_rad, fov_y=fov_y_rad).astype(np.float32)   # normalized
        intrinsics = np.stack([intrinsics_one] * N, axis=0).astype(np.float32)

        # extrinsics_look_at(eye, fwd, up=[0,0,1]) gives NaN when forward || up
        # (the 2 icosahedron pole vertices). Swap to an orthogonal up there.
        _UP_DEFAULT = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        _UP_FALLBACK = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        eye = np.zeros(3, dtype=np.float32)
        fwd = verts / np.maximum(np.linalg.norm(verts, axis=-1, keepdims=True), 1e-12)
        parallel = np.abs(fwd @ _UP_DEFAULT) > 0.999
        extrinsics = np.empty((N, 4, 4), dtype=np.float32)
        if (~parallel).any():
            extrinsics[~parallel] = utils3d.np.extrinsics_look_at(
                eye, verts[~parallel], _UP_DEFAULT).astype(np.float32)
        if parallel.any():
            extrinsics[parallel] = utils3d.np.extrinsics_look_at(
                eye, verts[parallel], _UP_FALLBACK).astype(np.float32)
            _p(f"  swapped up=[0,1,0] for {int(parallel.sum())} pole vertices")
        t_geom_done = time.perf_counter()

        # --- sample N rectilinear crops (GPU grid_sample / CPU cv2.remap) ---
        t_split = time.perf_counter()
        if use_gpu and torch.cuda.is_available():
            splitted = split_panorama_image_gpu(arr, extrinsics, intrinsics, crop_h, crop_w)
            backend = "grid_sample (GPU)"
        else:
            import cv2
            splitted = split_rect(arr, extrinsics, intrinsics, crop_h, crop_w, cv2.INTER_AREA)
            backend = "cv2.remap (CPU, cuda unavailable)" if use_gpu else "cv2.remap (CPU)"
        t_split_done = time.perf_counter()
        face_stack = np.stack(splitted, axis=0).astype(np.float32) / face_norm_divisor
        face_t = torch.from_numpy(face_stack)

        # --- debug overlay ---
        debug_np = _make_pano_debug_overlay(arr_for_overlay_u8, extrinsics, fov_x_rad, fov_y_rad)
        debug_t = torch.from_numpy(debug_np.astype(np.float32) / 255.0).unsqueeze(0)

        # --- per-crop spherical-Voronoi masks + adjacency-aware debug image ---
        if create_masks:
            _p("generating closest_to masks...")
            face_dirs = verts / np.maximum(np.linalg.norm(verts, axis=-1, keepdims=True), 1e-12)
            uv = utils3d.np.uv_map((crop_h, crop_w))
            masks = np.zeros((N, crop_h, crop_w), dtype=np.float32)
            for i in range(N):
                pixel_dirs = utils3d.np.unproject_cv(
                    uv, np.ones_like(uv[..., 0]),
                    extrinsics=extrinsics[i], intrinsics=intrinsics_one)
                pixel_dirs /= np.maximum(np.linalg.norm(pixel_dirs, axis=-1, keepdims=True), 1e-12)
                dots = np.einsum("hwc,nc->hwn", pixel_dirs, face_dirs)
                masks[i] = (np.argmax(dots, axis=-1) == i).astype(np.float32)
            face_masks_t = torch.from_numpy(masks)

            from .panorama_split_adaptive import _equirect_ray_dirs, _adjacency_aware_cell_colors
            eq_rays = _equirect_ray_dirs(H, W).reshape(-1, 3)
            eq_assignment = np.argmax(eq_rays @ face_dirs.T, axis=1)
            cell_colors = _adjacency_aware_cell_colors(eq_assignment.reshape(H, W), N)
            debug_masks_np = cell_colors[eq_assignment].reshape(H, W, 3)
            pano_f = arr_for_overlay_u8[..., :3].astype(np.float32) / 255.0
            debug_masks_np = np.clip(0.5 * debug_masks_np + 0.5 * pano_f, 0, 1)
            debug_masks_t = torch.from_numpy(debug_masks_np.astype(np.float32)).unsqueeze(0)
        else:
            face_masks_t = torch.zeros(N, crop_h, crop_w)
            debug_masks_t = torch.zeros((1, H, W, 3), dtype=torch.float32)

        # --- memory-bank entries (PIXEL-scale intrinsics) ---
        K_pixel = intrinsics_one.copy()
        K_pixel[0] *= float(crop_w)
        K_pixel[1] *= float(crop_h)
        Ks_pixel = np.broadcast_to(K_pixel, (N, 3, 3)).astype(np.float32).copy()
        extrinsics_t = torch.from_numpy(extrinsics)
        entries = {
            "frames":     face_t,
            "extrinsics": extrinsics_t,
            "intrinsics": torch.from_numpy(Ks_pixel),
            "depths":     None,
            "fnames":     [str(fname or "pano_bank")] * N,
            "frame_idx":  list(range(N)),
            "image_size": [int(crop_w), int(crop_h)],
        }

        _p(f"{method}: ({W}x{H}) -> {N} crops @ {crop_w}x{crop_h}, "
           f"fov=({math.degrees(fov_x_rad):.0f},{math.degrees(fov_y_rad):.0f}) via {backend}; "
           f"cameras {t_geom_done - t_geom:.3f}s, split {t_split_done - t_split:.3f}s, "
           f"total {time.perf_counter() - t_total:.3f}s")

        return io.NodeOutput(
            face_t,
            extrinsics_t,
            torch.from_numpy(intrinsics),   # normalized
            float(fov_x_out),
            face_masks_t,
            debug_t,
            debug_masks_t,
            entries,
        )


NODE_CLASS_MAPPINGS = {"PanoramaSplit": PanoramaSplit}
NODE_DISPLAY_NAME_MAPPINGS = {"PanoramaSplit": "Panorama Split"}
