"""PanoramaSplitAdaptive - depth+normals-conditioned adaptive panorama split.

Instead of uniform icosahedron tiling, places cameras adaptively so each
crop captures approximately equal world-space surface area. Areas with
distant/grazing surfaces (walls at steep angles, far ceilings) get more
crops; nearby frontal surfaces get fewer.

Algorithm:
  1. Compute per-pixel world-space area weight from depth + normals:
     weight = depth^2 / max(cos(theta), eps) where theta = angle(normal, view_ray)
  2. Start with icosahedron_12 directions as initial Voronoi seeds
  3. Assign each equirect pixel to its nearest seed (spherical Voronoi)
  4. Compute total area-weight per cell
  5. Repeatedly split the highest-weight cell (bisect along its
     principal axis) until target_faces is reached
  6. Recompute Voronoi assignments after each split
  7. Generate rectilinear crops from the final camera directions
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


def _adaptive_split(
    rays: np.ndarray,
    weights: np.ndarray,
    initial_seeds: np.ndarray,
    target_faces: int,
) -> np.ndarray:
    """Iteratively split the highest-area Voronoi cell until target_faces.

    Returns (target_faces, 3) unit direction seeds.
    """
    seeds = initial_seeds.copy()
    K = len(seeds)

    if K >= target_faces:
        return seeds[:target_faces]

    H, W = rays.shape[:2]
    rays_flat = rays.reshape(-1, 3)       # (H*W, 3)
    weights_flat = weights.flatten()       # (H*W,)

    for iteration in range(target_faces - K):
        # Assign pixels to nearest seed
        dots = rays_flat @ seeds.T  # (H*W, K_current)
        assignment = np.argmax(dots, axis=1)  # (H*W,)

        # Compute total weight per cell
        K_cur = len(seeds)
        cell_weights = np.zeros(K_cur, dtype=np.float64)
        np.add.at(cell_weights, assignment, weights_flat)

        # Find the heaviest cell
        heaviest = int(np.argmax(cell_weights))

        # Find pixels in this cell
        cell_mask = (assignment == heaviest)
        cell_rays = rays_flat[cell_mask]     # (M, 3)
        cell_w = weights_flat[cell_mask]      # (M,)

        if cell_rays.shape[0] < 2:
            # Degenerate: duplicate the seed with a small offset
            perturb = np.random.randn(3).astype(np.float32) * 0.01
            new_seed = seeds[heaviest] + perturb
            new_seed /= np.linalg.norm(new_seed)
            seeds = np.vstack([seeds, new_seed[None]])
            continue

        # Weighted centroid of the cell (on the sphere)
        centroid = (cell_rays * cell_w[:, None]).sum(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-12)

        # PCA on weighted cell rays to find principal spread axis
        centered = cell_rays - centroid[None]
        cov = (centered * cell_w[:, None]).T @ centered  # (3, 3)
        _, _, Vt = np.linalg.svd(cov)
        split_axis = Vt[0]  # direction of maximum spread

        # Split: two new seeds offset along the principal axis
        # Replace the heaviest seed with two children
        offset = split_axis * 0.15  # ~8.5deg offset
        child_a = centroid + offset
        child_a /= np.linalg.norm(child_a)
        child_b = centroid - offset
        child_b /= np.linalg.norm(child_b)

        seeds[heaviest] = child_a
        seeds = np.vstack([seeds, child_b[None]])

        if (iteration + 1) % 10 == 0 or (iteration + 1) == target_faces - K:
            _p(f"  split {iteration + 1}/{target_faces - K}: "
               f"{len(seeds)} seeds, heaviest cell weight ratio: "
               f"{cell_weights[heaviest] / max(cell_weights.sum(), 1e-12):.1%}")

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
                io.Int.Input(
                    "target_faces", default=42, min=12, max=200, step=1,
                    tooltip="Target number of output crops. The algorithm "
                            "starts from icosahedron_12 and subdivides the "
                            "highest-area cells until this count is reached. "
                            "42 matches icosahedron_42; higher values give "
                            "denser coverage of complex areas."),
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
                target_faces=42, resolution=952, fov_degrees=90.0,
                use_gpu=True, create_masks=False, mask_method="closest_to"):
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

        # --- Initial seeds: icosahedron_12 ---
        initial_verts, _ = utils3d.np.create_icosahedron_mesh()
        initial_seeds = initial_verts.astype(np.float32)
        initial_seeds /= np.maximum(
            np.linalg.norm(initial_seeds, axis=-1, keepdims=True), 1e-12)

        # --- Adaptive splitting ---
        _p(f"adaptive split: 12 -> {target_faces} faces...")
        t_split_start = time.perf_counter()
        seeds = _adaptive_split(rays, weights, initial_seeds, target_faces)
        N = len(seeds)
        _p(f"  adaptive split done: {N} camera directions in "
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
