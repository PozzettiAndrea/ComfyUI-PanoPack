"""PanoramaSplitAdaptive - depth+normals-conditioned adaptive panorama split.

Keeps a FIXED number of cameras (the chosen seed layout: icosahedron_12 = 12,
icosahedron_42 = 42, cube_split = 27) but RELOCATES them so they cluster on
high-world-area regions. Distant/grazing surfaces (far walls, steep ceilings)
end up covered by more, smaller cells -> more crop pixels there -> more
gaussians downstream; the camera count never changes.

Algorithm:
  1. Compute per-pixel world-space area weight from depth + normals:
     weight = depth^2 / max(cos(theta), eps) where theta = angle(normal, view_ray)
  2. Initial Voronoi seeds = the chosen layout (icosahedron_12 / _42 / cube).
  3. Weighted spherical Lloyd relaxation (centroidal Voronoi tessellation):
     assign each equirect pixel to its nearest seed, move each seed to the
     area-weight-weighted centroid of its cell (re-normalized to the sphere),
     iterate to convergence. Count stays fixed; seeds migrate toward high
     weight, balancing total area-weight per cell.
  4. Generate rectilinear crops from the relaxed camera directions.
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np
from comfy_api.latest import io

from .utils import PANORAMA_TYPE, unwrap_panorama_to_image


def _p(msg: str) -> None:
    print(f"[PanoramaSplitAdaptive] {msg}", file=sys.stderr, flush=True)


def _hue_to_rgb(hue: float, c: float = 0.8, bump: float = 0.2) -> np.ndarray:
    """HSV-wheel hue (in [0,1)) -> RGB in roughly [bump, c+bump]."""
    h6 = (hue % 1.0) * 6.0
    x = c * (1 - abs(h6 % 2 - 1))
    if h6 < 1:   r, g, b = c, x, 0
    elif h6 < 2: r, g, b = x, c, 0
    elif h6 < 3: r, g, b = 0, c, x
    elif h6 < 4: r, g, b = 0, x, c
    elif h6 < 5: r, g, b = x, 0, c
    else:        r, g, b = c, 0, x
    return np.array([r + bump, g + bump, b + bump], dtype=np.float32)


def _adjacency_aware_cell_colors(eq_assignment_2d: np.ndarray, n_cells: int) -> np.ndarray:
    """Color the Voronoi cells so SPATIALLY-ADJACENT cells never look alike.

    The cell-adjacency graph is planar (dual of a sphere Delaunay triangulation),
    so by the four-color theorem a DSATUR greedy colors it with a small K (~4-6)
    classes in O(V+E). Adjacent cells land in DIFFERENT classes; mapping the K
    classes to K maximally-spread hues then guarantees every touching pair differs
    by at least one full hue class (worst-case contrast ~1/K, flat in N) -- the
    cheapest scheme that maximizes the worst adjacent pair (textbook map coloring).

    A small per-cell VALUE (brightness) jitter is layered on so same-class cells
    that are far apart stay distinguishable; it never crosses a hue band, so the
    adjacency contrast guarantee is preserved.

    `eq_assignment_2d` is the (H, W) per-pixel cell-index map (longitude wraps in
    x). Returns (n_cells, 3) float32 RGB.
    """
    a = eq_assignment_2d
    adj = [set() for _ in range(n_cells)]

    def _edges(p, q):
        d = p != q
        if not d.any():
            return
        # unique (u, v) boundary pairs -- collapses millions of pixels to the
        # handful of distinct cell adjacencies.
        pairs = np.unique(np.stack([p[d].ravel(), q[d].ravel()], axis=1), axis=0)
        for u, v in pairs:
            u, v = int(u), int(v)
            adj[u].add(v); adj[v].add(u)

    _edges(a, np.roll(a, -1, axis=1))   # right neighbor (longitude wraps)
    _edges(a[:-1, :], a[1:, :])         # down neighbor (latitude, no wrap)

    # DSATUR: repeatedly color the uncolored cell with the most distinct colors
    # among its neighbors (ties broken by degree), giving it the smallest free
    # color index. Produces K ~ 4-6 proper classes on these planar graphs.
    color = [-1] * n_cells
    sat = [set() for _ in range(n_cells)]
    deg = [len(adj[i]) for i in range(n_cells)]
    for _ in range(n_cells):
        i = max((v for v in range(n_cells) if color[v] == -1),
                key=lambda v: (len(sat[v]), deg[v]))
        used = {color[n] for n in adj[i] if color[n] != -1}
        c = 0
        while c in used:
            c += 1
        color[i] = c
        for n in adj[i]:
            sat[n].add(c)
    K = max(color) + 1 if n_cells else 1

    # K classes -> K maximally-spread hues (guaranteed >= 1/K hue gap on every
    # edge) + a small deterministic brightness jitter for same-class readability.
    colors = np.zeros((n_cells, 3), dtype=np.float32)
    for i in range(n_cells):
        rgb = _hue_to_rgb(color[i] / max(K, 1))
        value = 0.82 + 0.18 * ((i * 0.6180339887) % 1.0)  # in [0.82, 1.0]
        colors[i] = np.clip(rgb * value, 0.0, 1.0)
    return colors


def _equirect_ray_dirs(H: int, W: int) -> np.ndarray:
    """Compute unit ray direction for each equirect pixel. Returns (H, W, 3).
    Convention: +Z forward, +Y up, theta=0 at center column."""
    u = np.linspace(0.5, W - 0.5, W, dtype=np.float32) / W  # [0, 1]
    v = np.linspace(0.5, H - 0.5, H, dtype=np.float32) / H  # [0, 1]
    vv, uu = np.meshgrid(v, u, indexing="ij")

    theta = (1.0 - uu) * 2.0 * np.pi  # azimuth [0, 2pi], right-to-left
    phi = vv * np.pi                    # polar [0, pi], top-to-bottom

    x = np.sin(phi) * np.cos(theta)
    y = np.cos(phi)                     # +Y = up (north pole)
    z = np.sin(phi) * np.sin(theta)

    return np.stack([x, y, z], axis=-1)  # (H, W, 3)


def _compute_area_weights(
    depth: np.ndarray, normals: np.ndarray, rays: np.ndarray,
) -> np.ndarray:
    """Per-pixel world-space area weight. (H, W) float32.

    weight ~ depth^2 / cos(theta) where theta = angle between surface normal
    and the viewing ray. Pixels with normals facing away or zero depth
    get weight 0.
    """
    # cos(theta) = dot(normal, -ray) (ray points outward, normal points toward camera)
    cos_theta = np.sum(normals * (-rays), axis=-1)  # (H, W)
    cos_theta = np.clip(cos_theta, 0.05, 1.0)       # clamp grazing to avoid infinity

    weight = (depth ** 2) / cos_theta
    weight = np.where(depth > 1e-6, weight, 0.0)
    return weight.astype(np.float32)


def _spherical_voronoi_assign(
    rays: np.ndarray, seeds: np.ndarray,
) -> np.ndarray:
    """Assign each pixel (via its ray direction) to the nearest seed.

    rays: (H, W, 3) unit directions
    seeds: (K, 3) unit directions
    Returns: (H, W) int32 - index of nearest seed per pixel.
    """
    # dots: (H, W, K) = rays . seeds^T
    dots = np.einsum("hwc,kc->hwk", rays, seeds)
    return np.argmax(dots, axis=-1).astype(np.int32)


def _initial_seeds(method: str) -> np.ndarray:
    """Initial Voronoi seed directions for the chosen layout. (N, 3) unit.

    The layout fixes the camera COUNT (icosahedron_12 -> 12, icosahedron_42 ->
    42, cube_split -> 27); Lloyd relaxation then moves these seeds without
    changing the count.
    """
    import utils3d
    if method == "icosahedron_12":
        verts, _ = utils3d.np.create_icosahedron_mesh()
    elif method == "icosahedron_42":
        from ._vendor.worldgen.src.panorama_utils import subdivide_icosahedron
        verts = subdivide_icosahedron(subdivisions=1)
    elif method == "cube_split":
        from ._vendor.worldstereo_cube import rotate_around_z_axis
        start = [np.array([-1.0, 0.0, 0.0], np.float32),
                 np.array([-1.0, 0.0, 0.5], np.float32),
                 np.array([-1.0, 0.0, -0.5], np.float32)]
        n_view = 9  # 40deg yaw step -> 9 yaws/pitch -> 27 total
        d = list(start)
        for sp in start:
            for i in range(1, n_view):
                d.append(rotate_around_z_axis(sp.reshape(1, 3), 40.0 * i)[0])
        verts = np.stack(d, axis=0)
    else:
        raise ValueError(f"PanoramaSplitAdaptive: unknown split_method {method!r}")
    verts = np.asarray(verts, dtype=np.float32)
    return verts / np.maximum(np.linalg.norm(verts, axis=-1, keepdims=True), 1e-12)


def _cell_pixels_per_area(seeds, rays, weights, pixels_per_face):
    """Per-cell pixels-per-(world-area-weight). Each crop has a fixed
    `pixels_per_face` budget covering its Voronoi cell, whose total area weight
    is W_k, so pixels/area = pixels_per_face / W_k. Low value = under-sampled
    (a large-world-area cell starved of pixels). Returns (K,) float64."""
    a = np.argmax(rays.reshape(-1, 3) @ seeds.T, axis=1)
    cw = np.zeros(len(seeds), dtype=np.float64)
    np.add.at(cw, a, weights.flatten().astype(np.float64))
    return float(pixels_per_face) / np.maximum(cw, 1e-9)


def _lloyd_relax(
    rays: np.ndarray,
    weights: np.ndarray,
    initial_seeds: np.ndarray,
    n_iters: int = 24,
    tag: str = "",
) -> np.ndarray:
    """Weighted spherical Lloyd relaxation (centroidal Voronoi tessellation).

    Keeps the seed COUNT fixed. Each iteration assigns every pixel (by ray) to
    its nearest seed, then moves each seed to the area-weight-weighted centroid
    of its Voronoi cell (re-normalized to the sphere). Seeds migrate toward
    high-weight (large world-area) regions, so those regions end up covered by
    more, smaller cells -> more crop pixels there -> more gaussians downstream,
    while the total camera count stays equal to the seed layout.

    Returns (K, 3) unit seeds (K == len(initial_seeds)).
    """
    seeds = initial_seeds.astype(np.float32).copy()
    seeds /= np.maximum(np.linalg.norm(seeds, axis=1, keepdims=True), 1e-12)
    K = len(seeds)
    rays_flat = rays.reshape(-1, 3).astype(np.float32)
    w_flat = weights.flatten().astype(np.float64)

    for it in range(n_iters):
        assignment = np.argmax(rays_flat @ seeds.T, axis=1)
        new_seeds = seeds.copy()
        max_move = 0.0
        for k in range(K):
            m = assignment == k
            if not m.any():
                continue  # empty cell: leave the seed where it is
            wk = w_flat[m]
            if wk.sum() < 1e-9:
                continue
            c = (rays_flat[m] * wk[:, None]).sum(axis=0)
            nrm = np.linalg.norm(c)
            if nrm < 1e-12:
                continue
            c = (c / nrm).astype(np.float32)
            max_move = max(max_move, float(np.linalg.norm(c - seeds[k])))
            new_seeds[k] = c
        seeds = new_seeds
        if max_move < 1e-4:
            _p(f"  {tag}Lloyd converged at iter {it + 1} (max move {max_move:.2e})")
            break
    else:
        _p(f"  {tag}Lloyd ran {n_iters} iters (max move {max_move:.2e})")
    return seeds


class PanoramaSplitAdaptive(io.ComfyNode):
    """Depth+normals-conditioned adaptive panorama split."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PanoramaSplitAdaptive",
            display_name="Panorama Split (Depth+Normals Conditioned)",
            category="PanoPack",
            description=(
                "Adaptive panorama split: places cameras so each crop "
                "covers approximately equal world-space surface area. "
                "Distant/grazing surfaces (far walls, ceilings at steep "
                "angles) get more crops; nearby frontal surfaces get "
                "fewer.\n\n"
                "Requires a depth panorama and normals panorama from "
                "a first pass (e.g. MoGe2 -> DepthMerge)."
            ),
            inputs=[
                io.Custom(PANORAMA_TYPE).Input(
                    "panorama",
                    tooltip="Equirectangular RGB panorama (2:1)."),
                io.Image.Input(
                    "depth_panorama",
                    tooltip="Equirectangular distance map (from DepthMerge). "
                            "Single-channel or 3-channel (uses first channel)."),
                io.Image.Input(
                    "normals_panorama",
                    tooltip="Equirectangular surface normals (3-channel, "
                            "world-space, normalized). From a normals "
                            "estimation pass or from MoGe2's normals output."),
                io.Combo.Input(
                    "split_method",
                    options=["icosahedron_42", "icosahedron_12", "cube_split"],
                    default="icosahedron_42",
                    tooltip="Initial camera layout (same set as PanoramaSplit). "
                            "Fixes the camera COUNT: icosahedron_12 = 12, "
                            "icosahedron_42 = 42, cube_split = 27. The count "
                            "does NOT change - the cameras are RELOCATED via "
                            "weighted Lloyd relaxation so they cluster on "
                            "high-area (far/grazing) regions, giving those "
                            "regions more crops' pixels -> more gaussians, "
                            "while keeping the same number of cameras as the "
                            "chosen layout."),
                io.Int.Input(
                    "resolution", default=952, min=128, max=2048, step=1,
                    tooltip="Per-face image resolution (square)."),
                io.Float.Input(
                    "fov_degrees", default=90.0, min=30.0, max=170.0, step=1.0,
                    optional=True,
                    tooltip="Per-face FOV in degrees."),
                io.Boolean.Input(
                    "use_gpu", default=True, optional=True,
                    tooltip="GPU-accelerated face resampling via grid_sample."),
                io.Boolean.Input(
                    "create_masks", default=False, optional=True,
                    tooltip="Generate per-face Voronoi masks."),
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
                io.Image.Output(
                    display_name="face_images_depth",
                    tooltip="Per-face depth crops (N, R, R, 3). Same "
                            "adaptive cameras, sampled from depth_panorama. "
                            "3-channel (depth replicated) for IMAGE compat."),
                io.Image.Output(
                    display_name="face_images_normals",
                    tooltip="Per-face normals crops (N, R, R, 3). Same "
                            "adaptive cameras, sampled from normals_panorama."),
                io.Custom("EXTRINSICS").Output(display_name="extrinsics"),
                io.Custom("INTRINSICS").Output(display_name="intrinsics"),
                io.Float.Output(display_name="fov_x_deg"),
                io.Mask.Output(display_name="face_masks"),
                io.Image.Output(display_name="debug_image"),
                io.Image.Output(
                    display_name="debug_masks",
                    tooltip="Equirect image with each pixel colored by its "
                            "Voronoi cell (which face owns it). Only "
                            "populated when create_masks is enabled."),
            ],
        )

    @classmethod
    def execute(cls, panorama, depth_panorama, normals_panorama,
                split_method="icosahedron_42", resolution=952, fov_degrees=90.0,
                use_gpu=True, create_masks=False, mask_method="closest_to"):
        import torch
        import cv2
        import utils3d
        from ._vendor.moge_panorama import (
            split_panorama_image, split_panorama_image_gpu,
        )

        t_total = time.perf_counter()

        # --- Unpack panorama ---
        pano_t = unwrap_panorama_to_image(panorama)
        arr = pano_t.detach().cpu().numpy() if isinstance(pano_t, torch.Tensor) else np.asarray(pano_t)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.dtype == np.uint8:
            face_norm_divisor = 255.0
            arr_for_overlay = arr
        else:
            arr = arr.astype(np.float32)
            face_norm_divisor = 1.0 if arr.max() > 1.01 else 1.0
            arr_for_overlay = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.01 else \
                ((arr - arr.min()) / max(arr.max() - arr.min(), 1e-6) * 255).clip(0, 255).astype(np.uint8)
        H, W = arr.shape[:2]

        # --- Unpack depth ---
        d = depth_panorama.detach().cpu().numpy() if isinstance(depth_panorama, torch.Tensor) else np.asarray(depth_panorama)
        if d.ndim == 4:
            d = d[0]
        if d.ndim == 3:
            d = d[..., 0]
        depth_np = d.astype(np.float32)
        # Resize depth to panorama size if needed
        if depth_np.shape != (H, W):
            depth_np = cv2.resize(depth_np, (W, H), interpolation=cv2.INTER_LINEAR)

        # --- Unpack normals ---
        n = normals_panorama.detach().cpu().numpy() if isinstance(normals_panorama, torch.Tensor) else np.asarray(normals_panorama)
        if n.ndim == 4:
            n = n[0]
        normals_np = n.astype(np.float32)
        if normals_np.shape[:2] != (H, W):
            normals_np = cv2.resize(normals_np, (W, H), interpolation=cv2.INTER_LINEAR)
        # Normalize
        norms = np.linalg.norm(normals_np, axis=-1, keepdims=True)
        normals_np = normals_np / np.maximum(norms, 1e-8)

        # --- Compute ray directions and area weights ---
        _p(f"computing area weights from depth ({H}x{W}) + normals...")
        rays = _equirect_ray_dirs(H, W)
        weights = _compute_area_weights(depth_np, normals_np, rays)
        _p(f"  weight range: [{weights.min():.3g}, {weights.max():.3g}], "
           f"mean={weights.mean():.3g}")

        # --- Initial seeds from the chosen layout (fixes the camera count) ---
        initial_seeds = _initial_seeds(split_method)
        N = len(initial_seeds)

        # --- Adaptive RELOCATION: weighted Lloyd relaxation (count stays N) ---
        _p(f"adaptive relax: {split_method} ({N} cameras), weighted Lloyd...")
        t_split_start = time.perf_counter()
        seeds = _lloyd_relax(rays, weights, initial_seeds, tag="")
        _p(f"  Lloyd relaxation done: {N} camera directions in "
           f"{time.perf_counter() - t_split_start:.3f}s")

        # --- Build extrinsics/intrinsics from adaptive seeds ---
        fov_rad = math.radians(float(fov_degrees))
        intrinsics_one = utils3d.np.intrinsics_from_fov(fov_x=fov_rad, fov_y=fov_rad)

        _UP_DEFAULT = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        _UP_FALLBACK = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        eye = np.zeros(3, dtype=np.float32)

        extrinsics = np.empty((N, 4, 4), dtype=np.float32)
        parallel = np.abs(seeds @ _UP_DEFAULT) > 0.999
        if (~parallel).any():
            extrinsics[~parallel] = utils3d.np.extrinsics_look_at(
                eye, seeds[~parallel], _UP_DEFAULT).astype(np.float32)
        if parallel.any():
            extrinsics[parallel] = utils3d.np.extrinsics_look_at(
                eye, seeds[parallel], _UP_FALLBACK).astype(np.float32)
        intrinsics = np.stack([intrinsics_one] * N, axis=0).astype(np.float32)

        # --- Rasterize crops ---
        _p(f"rasterizing {N} faces @ {resolution}x{resolution}...")
        t_raster = time.perf_counter()
        if use_gpu and torch.cuda.is_available():
            splitted = split_panorama_image_gpu(arr, extrinsics, intrinsics, resolution)
            backend = "GPU"
        else:
            splitted = split_panorama_image(arr, extrinsics, intrinsics, resolution)
            backend = "CPU"
        face_stack = np.stack(splitted, axis=0).astype(np.float32) / face_norm_divisor
        face_t = torch.from_numpy(face_stack)
        _p(f"  rasterized RGB in {time.perf_counter() - t_raster:.3f}s ({backend})")

        # --- Rasterize depth crops (same cameras) ---
        _p(f"rasterizing depth crops...")
        # Depth is single-channel; make it 3-channel for the split helpers
        depth_3ch = np.stack([depth_np, depth_np, depth_np], axis=-1)
        if use_gpu and torch.cuda.is_available():
            depth_splitted = split_panorama_image_gpu(depth_3ch, extrinsics, intrinsics, resolution)
        else:
            depth_splitted = split_panorama_image(depth_3ch, extrinsics, intrinsics, resolution)
        face_depth_stack = np.stack(depth_splitted, axis=0).astype(np.float32)
        face_depth_t = torch.from_numpy(face_depth_stack)

        # --- Rasterize normals crops (same cameras) ---
        _p(f"rasterizing normals crops...")
        if use_gpu and torch.cuda.is_available():
            normals_splitted = split_panorama_image_gpu(normals_np, extrinsics, intrinsics, resolution)
        else:
            normals_splitted = split_panorama_image(normals_np, extrinsics, intrinsics, resolution)
        face_normals_stack = np.stack(normals_splitted, axis=0).astype(np.float32)
        face_normals_t = torch.from_numpy(face_normals_stack)

        # --- Debug overlay ---
        from .panorama_split import _make_pano_debug_overlay
        debug_np = _make_pano_debug_overlay(arr_for_overlay, extrinsics, fov_rad)
        debug_t = torch.from_numpy(
            debug_np.astype(np.float32) / 255.0).unsqueeze(0)

        # --- Masks (closest_to Voronoi) + debug_masks equirect image ---
        if create_masks and mask_method == "closest_to":
            _p("generating closest_to masks...")
            R = resolution
            uv = utils3d.np.uv_map((R, R))
            face_dirs = seeds / np.maximum(
                np.linalg.norm(seeds, axis=-1, keepdims=True), 1e-12)
            masks = np.zeros((N, R, R), dtype=np.float32)
            for i in range(N):
                pixel_dirs = utils3d.np.unproject_cv(
                    uv, np.ones_like(uv[..., 0]),
                    extrinsics=extrinsics[i], intrinsics=intrinsics[i])
                pixel_dirs /= np.maximum(
                    np.linalg.norm(pixel_dirs, axis=-1, keepdims=True), 1e-12)
                dots = np.einsum("hwc,nc->hwn", pixel_dirs, face_dirs)
                masks[i] = (np.argmax(dots, axis=-1) == i).astype(np.float32)
            face_masks_t = torch.from_numpy(masks)

            # Debug masks: color each equirect pixel by its Voronoi cell
            _p("generating debug_masks equirect image...")
            rays_flat = rays.reshape(-1, 3)
            eq_dots = rays_flat @ face_dirs.T  # (H*W, N)
            eq_assignment = np.argmax(eq_dots, axis=1)  # (H*W,)

            # Adjacency-aware coloring: spatially-touching Voronoi cells get
            # high-contrast colors (no two similar colors end up next to each
            # other), instead of hue-by-index which ignores layout.
            hsv_colors = _adjacency_aware_cell_colors(eq_assignment.reshape(H, W), N)

            debug_masks_np = hsv_colors[eq_assignment].reshape(H, W, 3)
            # Blend 50/50 with the original panorama for context
            if arr_for_overlay.ndim == 3 and arr_for_overlay.shape[-1] >= 3:
                pano_f = arr_for_overlay[..., :3].astype(np.float32) / 255.0
                debug_masks_np = 0.5 * debug_masks_np + 0.5 * pano_f
            debug_masks_np = np.clip(debug_masks_np, 0, 1).astype(np.float32)
            debug_masks_t = torch.from_numpy(debug_masks_np).unsqueeze(0)
        else:
            face_masks_t = torch.zeros(N, resolution, resolution)
            debug_masks_t = torch.zeros((1, H, W, 3), dtype=torch.float32)

        # Final summary: how much the relaxation balanced per-camera pixel
        # budget (resolution^2 each) against world area, before vs after.
        _ppf = float(resolution) ** 2
        _ppa0 = _cell_pixels_per_area(initial_seeds, rays, weights, _ppf)
        _ppa1 = _cell_pixels_per_area(seeds, rays, weights, _ppf)
        _p(f"pixels/area: lowest {_ppa0.min():.3g} -> {_ppa1.min():.3g} "
           f"(+{(_ppa1.min() / max(_ppa0.min(), 1e-9) - 1) * 100:.0f}% on the "
           f"worst-sampled cell), highest {_ppa0.max():.3g} -> {_ppa1.max():.3g}; "
           f"spread max/min {_ppa0.max() / max(_ppa0.min(), 1e-9):.1f}x -> "
           f"{_ppa1.max() / max(_ppa1.min(), 1e-9):.1f}x")
        _p(f"done: {N} adaptive faces, {time.perf_counter() - t_total:.3f}s total")

        return io.NodeOutput(
            face_t,
            face_depth_t,
            face_normals_t,
            torch.from_numpy(extrinsics),
            torch.from_numpy(intrinsics),
            float(fov_degrees),
            face_masks_t,
            debug_t,
            debug_masks_t,
        )


NODE_CLASS_MAPPINGS = {"PanoramaSplitAdaptive": PanoramaSplitAdaptive}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PanoramaSplitAdaptive": "Panorama Split (Depth+Normals Conditioned)",
}
