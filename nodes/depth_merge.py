"""PanoramaDepthMerge — per-face rectilinear point maps → equirect depth.

Stitches per-face 3D POINT MAPS (e.g. MoGe-2's `points_raw` output) back
into a single equirect distance map. Uses upstream MoGe's sparse LSMR
formulation (Laplacian + gradient terms in log-distance space) — vendored
under `_vendor/moge_panorama.py`.

CONVENTION: consumes 3D POINT MAPS, not raw depth, because the merger
operates in EUCLIDEAN RAY DISTANCE (`||points||`), which is invariant
across overlapping faces. Per-face PLANAR Z (depth_raw / points[..., 2])
is face-axis-dependent and produces face-shaped seam facets when fused
across overlapping views. The node computes `np.linalg.norm(points,
axis=-1)` internally before LSMR — mirroring upstream MoGe
`infer_panorama.py` and HY-World 2.0's `pred_pano_depth`.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from comfy_api.latest import io


def _p(msg: str) -> None:
    print(f"[PanoramaDepthMerge] {msg}", file=sys.stderr, flush=True)


def _request_vram_eviction(needed_bytes: int) -> None:
    """Ask comfy-env's parent ComfyUI process to evict cross-worker models
    so this transient-tensor node has VRAM headroom.

    Mechanics: every comfy-env worker subprocess gets a `comfy_worker` helper
    module injected into sys.modules at startup. Calling
    `comfy_worker.call_parent("request_vram_budget", total_size=N)` routes
    through the shim's `request_vram_budget` IPC channel (the same one used
    by the shimmed `load_models_gpu`). The parent ComfyUI process handles
    the request via `_handle_vram_budget(...)` which calls
    `comfy.model_management.free_memory(N * 1.1, device)` on its side --
    evicting any sibling-worker models (e.g. HYWM2's 10 GB DiT) since
    they're registered in the parent's `current_loaded_models`.

    This is the cross-worker equivalent of a native sampler's
    `mm.load_models_gpu([model], memory_required=N)`. For transient-tensor
    workloads (like PanoramaDepthMerge, which doesn't register a model)
    it's the only way to free VRAM held by other packs.

    No-ops gracefully outside a comfy-env worker (e.g. unit tests running
    in plain Python).
    """
    # Cross-worker eviction via comfy-env's IPC bridge.
    try:
        import comfy_worker  # noqa: F401 - injected by comfy-env at worker startup
        try:
            comfy_worker.call_parent(
                "request_vram_budget", total_size=int(needed_bytes)
            )
            _p(f"  -> requested {needed_bytes / 1e9:.2f} GB eviction via comfy_worker.call_parent")
        except Exception as e:
            _p(f"  -> comfy_worker.call_parent failed: {e}")
    except ImportError:
        _p("  -> comfy_worker module unavailable; local free_memory only")

    # Also run the standard intra-worker path. Harmless even if this
    # worker holds no models (PanoPack typically doesn't), and defensive
    # if PanoPack ever gains a model loader.
    try:
        import comfy.model_management as mm
        device = mm.get_torch_device()
        mm.free_memory(int(needed_bytes), device)
    except Exception as e:
        _p(f"  -> local mm.free_memory failed: {e}")

    # Drop any cached blocks the allocator is holding in THIS worker so the
    # next big alloc sees the freshly-evicted space contiguously.
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _build_debug_image(
    distance_maps, weights, extr_list, intr_list,
    depth_np, out_width: int, out_height: int,
):
    """Render a per-pixel disagreement viridis map + text caption.

    For each equirect pixel, reproject every face's input distance back to
    that pixel via the merger's same projection (`p_cam = R @ dir + t`,
    `K @ p_cam`, perspective divide). Pixels covered by ≥2 faces get a
    std across faces, normalized to a percentage of the merged depth.
    """
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    from ._vendor.worldgen.src.panorama_utils import spherical_uv_to_directions
    import utils3d
    import matplotlib.cm as mcm

    H, W = int(out_height), int(out_width)
    N = len(distance_maps)

    # Equirect uv grid → unit world directions (same as the GPU merger).
    uv = utils3d.np.uv_map((H, W))
    dirs = spherical_uv_to_directions(uv).astype(np.float32).reshape(-1, 3)  # (H*W, 3)

    # Disagreement is measured in LOG-DISTANCE space, the same space the
    # LSMR merger minimizes in. Rationale (Bregman / Fisher-Rao geometry on
    # ℝ⁺, plus monocular-depth noise being multiplicative): std(log d) is
    # scale-invariant and matches the solver's loss. Reported as a
    # multiplicative disagreement percentage via `100·(exp(std) − 1)`.
    sum_ld = np.zeros(H * W, dtype=np.float64)
    sum_ld2 = np.zeros(H * W, dtype=np.float64)
    n_cov = np.zeros(H * W, dtype=np.int32)

    for i in range(N):
        ext = extr_list[i]
        R = ext[:3, :3].astype(np.float32)
        t = ext[:3, 3].astype(np.float32)
        K = intr_list[i].astype(np.float32)
        d_face = np.asarray(distance_maps[i], dtype=np.float32)
        fh, fw = d_face.shape
        w_face = np.asarray(weights[i], dtype=np.float32)
        if w_face.shape != (fh, fw):
            w_face = (w_face > 0.5).astype(np.float32)

        # Project equirect rays into face camera, normalize, divide.
        p_cam = dirs @ R.T + t                                  # (H*W, 3)
        in_front = p_cam[:, 2] > 0
        p_proj = p_cam @ K.T                                    # (H*W, 3)
        safe_w = np.where(p_proj[:, 2] > 1e-12, p_proj[:, 2], 1.0)
        uv_face = p_proj[:, :2] / safe_w[:, None]               # (H*W, 2) in [0,1]
        in_face = (
            in_front
            & (uv_face[:, 0] >= 0) & (uv_face[:, 0] <= 1)
            & (uv_face[:, 1] >= 0) & (uv_face[:, 1] <= 1)
        )

        # cv2.remap for the actual sampling — bilinear, BORDER_REPLICATE.
        px = (uv_face[:, 0] * fw - 0.5).astype(np.float32).reshape(H, W)
        py = (uv_face[:, 1] * fh - 0.5).astype(np.float32).reshape(H, W)
        sampled = cv2.remap(d_face, px, py, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE).ravel()
        wsamp = cv2.remap(w_face, px, py, cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_REPLICATE).ravel()

        covered = in_face & (wsamp > 0.5) & (sampled > 0)
        log_sampled = np.log(np.maximum(sampled, 1e-6))
        sum_ld[covered] += log_sampled[covered]
        sum_ld2[covered] += log_sampled[covered] ** 2
        n_cov[covered] += 1

    has_overlap = n_cov >= 2
    n_safe = np.maximum(n_cov, 1).astype(np.float64)
    mean_ld = sum_ld / n_safe
    var_ld = np.maximum(sum_ld2 / n_safe - mean_ld ** 2, 0.0)
    std_ld = np.sqrt(var_ld).astype(np.float32)               # std(log d_i) per pixel
    std_ld[~has_overlap] = 0.0
    # Per-pixel multiplicative % disagreement: `100·(exp(σ_log) − 1)`. For
    # small σ this ≈ 100·σ ≈ coefficient of variation; for larger σ it
    # reflects the actual geometric spread of overlapping predictions.
    pct_per_pixel = 100.0 * (np.exp(std_ld) - 1.0).astype(np.float32)
    pct_per_pixel[~has_overlap] = 0.0

    # Solid-angle weights: dΩ = sin(φ) dφ dθ. v ∈ [0, H-1] maps to
    # φ = (v + 0.5) / H · π. Without sin(φ) the equirect pole rows
    # (which collapse to a single sphere point) would bias the mean.
    v_idx = np.arange(H, dtype=np.float64)
    sin_phi = np.sin((v_idx + 0.5) / float(H) * np.pi)
    w_flat = np.broadcast_to(sin_phi[:, None], (H, W)).reshape(H * W)
    overlap_w = (w_flat * has_overlap.astype(np.float64))
    overlap_w_sum = float(overlap_w.sum())

    global_median = float(np.median(depth_np))
    if overlap_w_sum > 0:
        # Sphere-uniform mean std(log d) → exp-1 → multiplicative %.
        mean_std_ld = float((std_ld * overlap_w).sum() / overlap_w_sum)
        mean_pct = 100.0 * (np.exp(mean_std_ld) - 1.0)
        # sin(φ)-weighted percentile for the colormap clip and the p99 stat.
        order = np.argsort(pct_per_pixel * has_overlap.astype(np.float32))
        sorted_pct = pct_per_pixel[order]
        sorted_w = overlap_w[order]
        cum_w = np.cumsum(sorted_w)
        if cum_w[-1] > 0:
            cum_w /= cum_w[-1]
            p99_idx = int(np.searchsorted(cum_w, 0.99))
            p99_pct = float(sorted_pct[min(p99_idx, len(sorted_pct) - 1)])
        else:
            p99_pct = 0.0
        max_pct = float(pct_per_pixel[has_overlap].max())
    else:
        mean_pct = 0.0
        p99_pct = 0.0
        max_pct = 0.0

    # Colormap normalization: clip to p99 so a handful of bright pixels don't
    # crush all the meaningful variation into the first few percent of the
    # viridis range.
    norm_denom = max(p99_pct, 1e-6)
    pct_2d = pct_per_pixel.reshape(H, W)
    pct_norm = np.clip(pct_2d / norm_denom, 0.0, 1.0)
    rgb = mcm.get_cmap('viridis')(pct_norm)[..., :3].astype(np.float32)  # (H, W, 3) in [0,1]

    # Caption strip below the colormap.
    cap_h = max(48, int(round(H * 0.05)))
    cap = Image.new('RGB', (W, cap_h), (24, 24, 24))
    draw = ImageDraw.Draw(cap)
    font = None
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            font = ImageFont.truetype(path, max(14, int(cap_h * 0.4)))
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    text = (
        f"multiplicative disagreement (log-d, sphere-uniform): "
        f"mean = {mean_pct:.2f}%,  p99 = {p99_pct:.2f}%,  "
        f"max = {max_pct:.2f}%  "
        f"(global median depth {global_median:.3f}m) "
        f"(colormap clipped at p99)"
    )
    # Vertical center
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        th = bbox[3] - bbox[1]
    except Exception:
        th = int(cap_h * 0.4)
    draw.text((10, max(2, (cap_h - th) // 2)), text, fill=(245, 245, 245), font=font)
    cap_np = np.asarray(cap, dtype=np.float32) / 255.0

    stacked = np.concatenate([rgb, cap_np], axis=0)  # (H + cap_h, W, 3)
    _p(f"debug_image: H={H}+{cap_h}, mult_disagreement (log-d, sphere-uniform): "
       f"mean={mean_pct:.2f}%, p99={p99_pct:.2f}%, max={max_pct:.2f}%; "
       f"overlap_pixels={int(has_overlap.sum())}")
    return torch.from_numpy(stacked).unsqueeze(0).contiguous().float()


def _render_xyz_views(
    depth_np: np.ndarray,
    mask_np: np.ndarray,
    panel_w: int = 720,
    panel_h: int = 720,
    max_points: int = 80_000,
) -> torch.Tensor:
    """Render orthographic XY (top-down), XZ (front), YZ (side) projections
    of the equirect-unprojected 3D point cloud via PyVista offscreen.

    Each panel shows:
      • the point cloud scattered (subsampled to `max_points` for speed)
      • an axes triad
      • a caption strip with bounding-box dimensions in metric units

    Returns a [1, H, panel_w × 3, 3] IMAGE tensor (3 panels side-by-side).
    Falls back to a small zero tensor + a stderr log if PyVista's
    offscreen rasterizer isn't available in the worker.
    """
    try:
        import pyvista as pv
    except ImportError:
        _p("xyz_views: pyvista not installed; returning empty image")
        return torch.zeros((1, 1, 1, 3), dtype=torch.float32)

    # --- Equirect → 3D point cloud (ray distance × spherical direction). ---
    H, W = depth_np.shape
    try:
        from utils3d.numpy.maps import uv_map as _uv_map
        from ._vendor.worldgen.src.panorama_utils import (
            spherical_uv_to_directions,
        )
    except Exception as e:
        _p(f"xyz_views: unprojection helpers unavailable ({e}); skipping")
        return torch.zeros((1, 1, 1, 3), dtype=torch.float32)

    uv = _uv_map(H, W)
    rays = spherical_uv_to_directions(uv).astype(np.float32)  # [H, W, 3]
    pts = rays * depth_np[..., None]                          # [H, W, 3] in meters
    valid = mask_np.astype(bool) & np.isfinite(pts).all(axis=-1)
    if not valid.any():
        _p("xyz_views: no valid points to render")
        return torch.zeros((1, 1, 1, 3), dtype=torch.float32)
    flat = pts[valid].reshape(-1, 3)

    # --- Subsample for speed. ---
    if flat.shape[0] > max_points:
        idx = np.random.default_rng(seed=42).choice(
            flat.shape[0], size=max_points, replace=False,
        )
        flat = flat[idx]

    # --- Bounding box stats. ---
    bb_min = flat.min(axis=0)
    bb_max = flat.max(axis=0)
    bb_size = bb_max - bb_min
    bb_center = (bb_min + bb_max) / 2.0
    _p(f"xyz_views: cloud bbox center=({bb_center[0]:+.2f}, "
       f"{bb_center[1]:+.2f}, {bb_center[2]:+.2f}) m, "
       f"size=({bb_size[0]:.2f} × {bb_size[1]:.2f} × {bb_size[2]:.2f}) m, "
       f"N={flat.shape[0]} (subsampled)")

    # --- Render the 3 orthographic views with PyVista. ---
    # off_screen=True uses VTK's OSMesa/EGL backend; works headless on
    # Linux as long as VTK was built with offscreen support (the
    # pixi/conda VTK 9.x build is).
    cloud = pv.PolyData(flat.astype(np.float32))

    views = [
        # (label, camera position offset from center, view-up axis, axis labels)
        ("TOP (XY, look -Y)",  np.array([0.0,  1.0, 0.0]), np.array([0.0, 0.0, 1.0]),  "X→ right, Z→ up"),
        ("FRONT (XZ, look -Z)", np.array([0.0, 0.0,  1.0]), np.array([0.0, 1.0, 0.0]),  "X→ right, Y→ up"),
        ("SIDE (YZ, look -X)",  np.array([1.0, 0.0,  0.0]), np.array([0.0, 1.0, 0.0]),  "Z→ right, Y→ up"),
    ]
    bg = (0.08, 0.08, 0.10)
    fg_pt = (0.95, 0.95, 0.95)
    panels: list[np.ndarray] = []
    for label, cam_dir, up_axis, axis_hint in views:
        try:
            p = pv.Plotter(off_screen=True, window_size=(panel_w, panel_h))
            p.background_color = bg
            p.add_mesh(cloud, color=fg_pt, point_size=1.5, render_points_as_spheres=False)
            # Orthographic camera. Distance is set to the bbox diagonal so
            # the whole cloud fits regardless of viewing axis. The view-up
            # picks the second-in-plane axis for each projection so the
            # labels line up.
            radius = float(np.linalg.norm(bb_size)) * 0.6 + 1e-3
            cam_pos = bb_center + cam_dir * (radius * 2.0)
            p.camera.position = tuple(cam_pos.tolist())
            p.camera.focal_point = tuple(bb_center.tolist())
            p.camera.up = tuple(up_axis.tolist())
            p.enable_parallel_projection()
            p.camera.parallel_scale = radius
            # Bounding box outline + grid for scale.
            p.show_bounds(
                grid="back", location="outer", color="white",
                font_size=10, n_xlabels=5, n_ylabels=5, n_zlabels=5,
                xtitle=f"X ({bb_size[0]:.2f} m)",
                ytitle=f"Y ({bb_size[1]:.2f} m)",
                ztitle=f"Z ({bb_size[2]:.2f} m)",
            )
            p.add_text(
                f"{label}\n{axis_hint}\n"
                f"bbox X={bb_size[0]:.2f}  Y={bb_size[1]:.2f}  Z={bb_size[2]:.2f} m\n"
                f"min=({bb_min[0]:+.2f},{bb_min[1]:+.2f},{bb_min[2]:+.2f}) "
                f"max=({bb_max[0]:+.2f},{bb_max[1]:+.2f},{bb_max[2]:+.2f})",
                position="upper_left", font_size=8, color="white",
                shadow=True,
            )
            img = p.screenshot(return_img=True)  # [H, W, 3] uint8
            p.close()
            if img is None:
                raise RuntimeError("screenshot returned None")
            # PyVista may return RGBA; trim to RGB.
            if img.ndim == 3 and img.shape[-1] == 4:
                img = img[..., :3]
            panels.append(img.astype(np.uint8))
        except Exception as e:
            _p(f"xyz_views: panel '{label}' failed ({e}); using placeholder")
            placeholder = np.full((panel_h, panel_w, 3), 32, dtype=np.uint8)
            panels.append(placeholder)

    composite = np.concatenate(panels, axis=1)              # [H, 3W, 3] uint8
    composite_f = composite.astype(np.float32) / 255.0
    return torch.from_numpy(composite_f).unsqueeze(0).contiguous()  # [1, H, 3W, 3]


class PanoramaDepthMerge(io.ComfyNode):
    """Per-face depth maps + per-face extrinsics/intrinsics → equirect depth IMAGE."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PanoramaDepthMerge",
            display_name="Panorama Depth Merge",
            category="PanoPack",
            description=(
                "Inverse of WorldNavPanoramaSplit: take the per-face depths "
                "produced by running a rectilinear depth model on the split "
                "views, and stitch them back into a single equirect distance "
                "map (the input PanoramaBuildMesh / WorldNavPerceive expect).\n\n"
                "Mirrors upstream HY-World's pred_pano_depth stitch step "
                "(Laplacian + gradient sparse linear solve in log-distance "
                "space)."
            ),
            inputs=[
                io.Image.Input(
                    "face_points",
                    tooltip="Per-face 3D POINT MAPS in camera space "
                            "(MoGe2Inference's `points_raw` output, run "
                            "on the face_images from WorldNavPanoramaSplit). "
                            "Shape (N, h, w, 3) with channels (X, Y, Z) in "
                            "the face's camera frame.\n\n"
                            "DO NOT wire MoGe2's `depth_raw` here — that's "
                            "planar Z (camera-axis depth), face-axis-"
                            "dependent and incompatible with the merger's "
                            "ray-distance assumption. This node computes "
                            "`||points||` internally — the rotation-"
                            "invariant Euclidean distance from camera "
                            "origin to surface, which is what the LSMR "
                            "merger needs. Matches upstream MoGe + HY-"
                            "World 2.0 convention."),
                io.Custom("EXTRINSICS").Input(
                    "extrinsics",
                    tooltip="From WorldNavPanoramaSplit. (N, 4, 4) per-face extrinsics."),
                io.Custom("INTRINSICS").Input(
                    "intrinsics",
                    tooltip="From WorldNavPanoramaSplit. (N, 3, 3) per-face intrinsics."),
                io.Mask.Input(
                    "face_valid_masks",
                    optional=True,
                    tooltip="Per-face BINARY valid masks (e.g. MoGe2 "
                            "Inference's `valid_mask` output). Shape "
                            "(N, h, w) MASK; thresholded at 0.5 internally. "
                            "If None, all pixels treated as valid."),
                io.Mask.Input(
                    "face_confidences",
                    optional=True,
                    tooltip="Per-face CONTINUOUS confidence weights in "
                            "[0, 1] (e.g. MoGe2 Inference's new "
                            "`confidence` output — the soft sigmoid of "
                            "the mask head, before the 0.5 binarization). "
                            "Shape (N, h, w) MASK. When provided, the "
                            "LSMR residuals get weighted by confidence so "
                            "low-confidence pixels contribute "
                            "proportionally less to the merged depth — "
                            "useful for handling soft transitions at "
                            "sky / silhouette boundaries.\n\n"
                            "If both face_valid_masks and "
                            "face_confidences are wired, the effective "
                            "weight per pixel is valid * confidence. If "
                            "only confidence is wired, it acts as both "
                            "validity (any pixel with weight > 1e-3 is "
                            "considered observed) and weight.\n\n"
                            "GPU path only — on use_gpu=False the "
                            "confidence is thresholded at 0.5 and treated "
                            "as a binary mask (continuous weighting is "
                            "lost). A warning is printed in that case."),
                io.Image.Input(
                    "face_normals",
                    optional=True,
                    tooltip="Per-face camera-space surface normals from "
                            "MoGe-2 Inference's `normal` output. Shape "
                            "(N, h, w, 3) — unit-length normals in each "
                            "face's local camera frame. When wired, "
                            "enables two normal-based per-face weight "
                            "modifiers: normal-jump edge-floater removal "
                            "(see `normal_edge_threshold_deg`) and "
                            "normal-consistency confidence boost (see "
                            "`normal_consistency_boost`). Both default to "
                            "off; this input is ignored unless at least "
                            "one is enabled."),
                io.Int.Input(
                    "out_width", default=1920, min=512, max=4096, step=64,
                    tooltip="Output equirect width. 2:1 aspect (height = width/2)."),
                io.Int.Input(
                    "out_height", default=960, min=256, max=2048, step=64,
                    tooltip="Output equirect height. Must be width/2 for 2:1 aspect."),
                io.Boolean.Input(
                    "scale_anchor", default=True,
                    optional=True,
                    tooltip=(
                        "Fix the LSMR's rank-1 scale ambiguity by "
                        "post-hoc-matching the solved depth's median to "
                        "the input-faces' median.\n\n"
                        "The gradient + Laplacian operators are "
                        "translation-invariant in log(d) — the solver "
                        "recovers shape but not absolute scale, and its "
                        "min-norm pick biases output toward d ≈ 1m. "
                        "This shift fixes that without touching the "
                        "shape fit.\n\n"
                        "On (default): median(log d) matched to input "
                        "median.\n"
                        "Off: legacy behavior (output scale drifts "
                        "toward 1m)."
                    )),
                io.Boolean.Input(
                    "use_gpu", default=True,
                    tooltip="GPU sparse LSMR via torch.sparse + scipy "
                            "LinearOperator (cuSPARSE under the hood). "
                            "Target ~20-50x speedup at 1920x960. Falls back "
                            "to CPU scipy if CUDA isn't available."),
                io.Int.Input(
                    "chunk_size", default=8, min=1, max=64, step=1,
                    optional=True,
                    tooltip=(
                        "How many faces to process simultaneously inside the "
                        "GPU merger. Lower = less peak GPU memory, slightly "
                        "more launch overhead. The face-axis reductions "
                        "(sum, any) are associative so chunking is "
                        "math-preserving — output is identical (modulo "
                        "fp32 sum-order round-off).\n\n"
                        "Default 8 keeps peak under ~3 GB at 4096×2048 "
                        "equirect with 42 faces. Bump to 16-32 on big GPUs "
                        "for marginally faster runs; drop to 2-4 if you're "
                        "still OOMing.\n\n"
                        "Only used on the use_gpu path."
                    )),
                io.Float.Input(
                    "center_weight_power", default=0.0, min=0.0, max=4.0, step=0.5,
                    optional=True,
                    tooltip=(
                        "Down-weight contributions from face corners "
                        "(where the camera ray is oblique to the face's "
                        "optical axis). Per-face per-pixel weight gets "
                        "multiplied by `cos(angle_from_axis)^power`.\n\n"
                        "  0.0 (default): off, every face pixel weighted "
                        "uniformly.\n"
                        "  1.0: cosine fall-off — at a 90° face corner, "
                        "weight ≈ 0.577.\n"
                        "  2.0: cos² fall-off (recommended) — corner ≈ "
                        "0.333. Matches the natural solid-angle weighting "
                        "and is a good trust-the-center prior for "
                        "monocular depth models like MoGe-2 which are "
                        "less accurate at large off-axis angles (training "
                        "distribution + rectilinear distortion).\n"
                        "  3.0-4.0: sharper falloff; effectively discards "
                        "corners.\n\n"
                        "Multiplies into the existing face_valid_masks × "
                        "face_confidences weight; all three combine "
                        "freely. GPU path only."
                    )),
                io.Float.Input(
                    "normal_edge_threshold_deg", default=0.0, min=0.0, max=180.0, step=5.0,
                    optional=True,
                    tooltip=(
                        "Drop pixels at depth discontinuities, detected "
                        "via SURFACE-NORMAL jumps. Requires `face_normals` "
                        "to be wired.\n\n"
                        "For each per-face pixel, compute the angle "
                        "between its normal and each of its 4 neighbors' "
                        "normals. If the MAX neighbor-angle exceeds the "
                        "threshold, that pixel sits on a discontinuity "
                        "(silhouette / depth edge / floater) and gets "
                        "weight=0 in the LSMR.\n\n"
                        "  0.0 (default): off.\n"
                        "  30-45°: aggressive; drops most edges, keeps "
                        "only flat regions. Good for clean CAD-like prep.\n"
                        "  60-75°: balanced. Drops sharp silhouettes but "
                        "preserves curved surfaces.\n"
                        "  90-120°: only drops the very sharpest jumps "
                        "(near-perpendicular adjacent surfaces).\n\n"
                        "Implements HY-World 2.0 paper's 'removing depth "
                        "discontinuities' filter via the more robust "
                        "normal-jump signal (vs depth-gradient quantile)."
                    )),
                io.Boolean.Input(
                    "normal_consistency_boost", default=False,
                    optional=True,
                    tooltip=(
                        "Multiply the per-pixel weight by the MEAN "
                        "cosine-similarity of the pixel's normal with "
                        "its 4 neighbors (clamped to [0, 1]). Requires "
                        "`face_normals` to be wired.\n\n"
                        "Strengthens the LSMR's trust in pixels lying on "
                        "smooth surfaces (where neighbor normals all "
                        "agree → boost ≈ 1.0) and weakens it on textured "
                        "/ noisy regions (boost < 1.0). Independent of "
                        "the normal_edge_threshold filter: the filter is "
                        "a HARD binary cutoff, this is a SOFT continuous "
                        "weighting. They combine multiplicatively if both "
                        "enabled.\n\n"
                        "False (default): off.\n"
                        "True: apply the soft consistency multiplier."
                    )),
            ],
            outputs=[
                io.Image.Output(display_name="depth"),
                io.Mask.Output(display_name="valid_mask"),
                io.Image.Output(
                    display_name="debug_image",
                    tooltip=(
                        "Per-pixel multiplicative-disagreement viridis "
                        "colormap on the equirect grid. For each pixel "
                        "covered by ≥2 faces, computes std of log-"
                        "distance across the reprojected per-face "
                        "predictions, then `100·(exp(σ_log) − 1)` — the "
                        "multiplicative percentage spread of overlapping "
                        "predictions at that direction. Log-distance "
                        "matches the LSMR merger's own objective (it "
                        "solves in log-d space), so this is the metric "
                        "the solver actually 'saw' the overlapping inputs "
                        "in.\n\nDark = strong consensus, bright = high "
                        "disagreement (typically silhouettes / depth "
                        "edges). The caption strip reports the sphere-"
                        "uniform mean, p99, and max — sin(φ)-weighted so "
                        "the pole rows of the equirect don't dominate. "
                        "Global median depth printed for unit context."
                    )),
                io.Image.Output(
                    display_name="xyz_views",
                    tooltip=(
                        "Three orthographic projections of the unprojected "
                        "depth cloud rendered side-by-side via PyVista: "
                        "TOP (XY, looking down -Y), FRONT (XZ, looking "
                        "down -Z), SIDE (YZ, looking down -X). Each panel "
                        "shows the point cloud with an axes triad, a "
                        "back-grid with metric tick labels (X/Y/Z in "
                        "meters), and an upper-left caption strip "
                        "reporting the bounding box dimensions and "
                        "min/max in world units. Useful to eyeball scene "
                        "scale and verify the LSMR-fused depth produces "
                        "a sensible 3D shape (e.g. a cubic room should "
                        "show ~equal bbox dims in TOP)."
                    )),
            ],
        )

    @classmethod
    def execute(cls, face_points, extrinsics, intrinsics,
                face_valid_masks=None, face_confidences=None,
                face_normals=None,
                out_width=1920, out_height=960,
                use_gpu=True, chunk_size=8, center_weight_power=0.0,
                normal_edge_threshold_deg=0.0,
                normal_consistency_boost=False,
                scale_anchor=True):
        from ._vendor.moge_panorama import merge_panorama_depth

        # --- face_points: (N, h, w, 3) point map → list of (h, w) ray distance.
        # Per-face Euclidean ||points|| is rotation-invariant: the same
        # world point projected into two overlapping faces gives the same
        # scalar, which is what the LSMR merger needs. Planar Z (=
        # points[..., 2]) would differ per face.
        p = face_points.detach().cpu().numpy() if isinstance(face_points, torch.Tensor) else np.asarray(face_points)
        if p.ndim != 4 or p.shape[-1] != 3:
            raise ValueError(
                f"PanoramaDepthMerge: face_points must be (N, h, w, 3) — the 3D point "
                f"map from MoGe2Inference's `points_raw` output. Got {p.shape}. "
                f"(If you wired MoGe-2's depth_raw by mistake, switch to points_raw — "
                f"depth_raw is planar Z and produces face-shaped seam artifacts here.)"
            )
        N, fh, fw, _ = p.shape
        p_f32 = p.astype(np.float32, copy=False)
        distance_maps = [np.linalg.norm(p_f32[i], axis=-1) for i in range(N)]

        # --- extrinsics / intrinsics: (N, 4, 4) and (N, 3, 3) ---
        ex = extrinsics.detach().cpu().numpy() if isinstance(extrinsics, torch.Tensor) else np.asarray(extrinsics)
        intr = intrinsics.detach().cpu().numpy() if isinstance(intrinsics, torch.Tensor) else np.asarray(intrinsics)
        if ex.shape != (N, 4, 4) or intr.shape != (N, 3, 3):
            raise ValueError(
                f"PanoramaDepthMerge: shape mismatch — extrinsics {ex.shape}, "
                f"intrinsics {intr.shape}, expected ({N},4,4) and ({N},3,3)"
            )
        extr_list = [ex[i].astype(np.float32) for i in range(N)]
        intr_list = [intr[i].astype(np.float32) for i in range(N)]

        # --- Combine face_valid_masks and face_confidences into a single
        # per-face weight tensor in [0, 1]. The merger now treats this as
        # a continuous weight (was bool-only before); for bool inputs
        # cast to 0.0/1.0 the math is bit-equivalent to the previous
        # bool-AND path. MASK socket convention is (N, h, w) float; if
        # someone wires an IMAGE accidentally we accept (N, h, w, C) too.
        def _mask_to_nhw(x, name):
            arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
            if arr.ndim == 4:
                arr = arr[..., 0]
            if arr.shape != (N, fh, fw):
                raise ValueError(
                    f"PanoramaDepthMerge: {name} shape {arr.shape} doesn't match "
                    f"face_points ({N}, {fh}, {fw})"
                )
            return arr.astype(np.float32)

        weights = np.ones((N, fh, fw), dtype=np.float32)
        if face_valid_masks is not None:
            v = _mask_to_nhw(face_valid_masks, "face_valid_masks")
            weights *= (v > 0.5).astype(np.float32)
        if face_confidences is not None:
            c = _mask_to_nhw(face_confidences, "face_confidences")
            np.clip(c, 0.0, 1.0, out=c)
            weights *= c

        # Normal-based per-face weight modifiers. Both require `face_normals`
        # wired and at least one of (edge filter, consistency boost) enabled.
        # The pre-computed neighbor cos-similarity field is shared between
        # the two modifiers.
        use_edge_filter = normal_edge_threshold_deg > 0.0
        use_consistency_boost = bool(normal_consistency_boost)
        if face_normals is not None and (use_edge_filter or use_consistency_boost):
            n = face_normals.detach().cpu().numpy() if isinstance(face_normals, torch.Tensor) else np.asarray(face_normals)
            if n.ndim != 4 or n.shape[-1] != 3 or n.shape[0] != N or n.shape[1] != fh or n.shape[2] != fw:
                raise ValueError(
                    f"PanoramaDepthMerge: face_normals shape {n.shape} doesn't match "
                    f"face_points (N={N}, h={fh}, w={fw}, 3). MoGe-2 emits normals at "
                    f"the same resolution as points."
                )
            n = n.astype(np.float32, copy=False)
            # Re-normalize defensively (MoGe-2's output is already unit but
            # any IMAGE-socket round-trip / clipping could nudge it slightly).
            n_len = np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
            n = n / n_len

            # 4-neighbor cos-similarity at each per-face pixel. Replicate-pad
            # the borders so edge pixels still get 4 (degenerate, self-aligned)
            # neighbors and don't get spuriously flagged.
            n_pad = np.pad(n, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='edge')   # (N, fh+2, fw+2, 3)
            center = n_pad[:, 1:-1, 1:-1, :]
            cos_up = (center * n_pad[:, :-2, 1:-1, :]).sum(axis=-1)
            cos_dn = (center * n_pad[:, 2:,   1:-1, :]).sum(axis=-1)
            cos_lf = (center * n_pad[:, 1:-1, :-2,  :]).sum(axis=-1)
            cos_rt = (center * n_pad[:, 1:-1, 2:,   :]).sum(axis=-1)

            if use_edge_filter:
                # Hard mask: drop pixels where the worst neighbor angle
                # exceeds the threshold. min(cos) corresponds to max(angle).
                cos_min = np.minimum(np.minimum(cos_up, cos_dn), np.minimum(cos_lf, cos_rt))
                cos_thresh = float(np.cos(np.deg2rad(float(normal_edge_threshold_deg))))
                edge_keep = (cos_min >= cos_thresh).astype(np.float32)
                dropped = int(N * fh * fw - edge_keep.sum())
                _p(f"normal-edge filter: threshold={normal_edge_threshold_deg:.1f}° "
                   f"(cos≥{cos_thresh:.3f}); dropped {dropped}/{N*fh*fw} pixels "
                   f"({100.0 * dropped / max(N*fh*fw, 1):.1f}%)")
                weights *= edge_keep

            if use_consistency_boost:
                # Soft multiplier: mean cos-similarity to 4 neighbors,
                # clamped to [0, 1]. On smooth surfaces all 4 cos ≈ 1 so
                # boost ≈ 1; on rough / textured regions boost drops.
                cos_mean = (cos_up + cos_dn + cos_lf + cos_rt) * 0.25
                consistency = np.clip(cos_mean, 0.0, 1.0).astype(np.float32)
                _p(f"normal-consistency boost: mean={float(consistency.mean()):.3f} "
                   f"min={float(consistency.min()):.3f} max={float(consistency.max()):.3f}")
                weights *= consistency

        # CPU path uses bool AND ops in `_vendor/moge_panorama.py`'s
        # `merge_panorama_depth`. Threshold the float weights back to
        # bool at 0.5 when on the CPU path. Warn if continuous
        # confidence was wired since the gradation is lost.
        if not use_gpu:
            if face_confidences is not None:
                _p("WARN: face_confidences wired but use_gpu=False — "
                   "continuous confidence is lost; thresholding at 0.5.")
            pred_masks = [(weights[i] > 0.5).astype(bool) for i in range(N)]
        else:
            pred_masks = [weights[i] for i in range(N)]

        _p(f"merging {N} faces @ {fh}×{fw} → equirect {out_height}×{out_width} "
           f"(use_gpu={use_gpu}); input = points (||·|| → ray distance)")

        try:
            import comfy.utils
            pbar = comfy.utils.ProgressBar(100)
        except Exception:
            class _Noop:
                def update_absolute(self, *_a, **_k): pass
            pbar = _Noop()
        pbar.update_absolute(1, 100)

        if use_gpu:
            # Ask ComfyUI's model manager (across all comfy-env worker
            # subprocesses, via comfy_worker.call_parent) to free VRAM
            # before we hit the top pyramid level. The top-level einsum
            # at large output resolutions (3840×1920 with N=42) needs
            # several GB of contiguous CUDA memory; if a sibling worker
            # is still holding a model (HYWM2's DiT, SHARP's UNet) the
            # allocation can OOM even when the totals would fit. Native
            # samplers do the same thing via `mm.load_models_gpu(...,
            # memory_required=N)`; ours is the transient-tensor analog.
            #
            # Peak estimate: per-chunk float32 working tensors ≈ 12
            # channels per output pixel per face in flight (p_cam +
            # p_proj + projected_uv + safe_w + sampled_log_dist +
            # sampled_pred_mask + grid xy + grad/lap intermediates),
            # summed over `chunk_size` faces. Plus a handful of
            # accumulators at full output resolution (six float + three
            # bool buffers ≈ 7 floats' worth). 10% headroom is added by
            # the parent's _handle_vram_budget. Estimate is approximate
            # but a generous overestimate helps -- worst case is we
            # evict more models than strictly needed; best case we
            # avoid the OOM entirely.
            N_f = len(distance_maps)
            Nc = max(1, min(int(chunk_size), N_f))
            per_chunk_bytes = Nc * int(out_height) * int(out_width) * 4 * 12
            accum_bytes = int(out_height) * int(out_width) * 4 * 7
            peak_estimate = per_chunk_bytes + accum_bytes
            _p(f"top-level peak VRAM estimate: "
               f"{peak_estimate / 1e9:.2f} GB ({out_width}×{out_height}, "
               f"N={N_f}, chunk_size={Nc})")
            _request_vram_eviction(peak_estimate)

            # GPU path: scipy LSMR with torch.sparse-backed matvecs (cuSPARSE
            # under the hood). The vendored merge_panorama_depth_gpu calls
            # solve_lsmr_gpu internally, which routes to the GPU operator when
            # CUDA is available and falls back to CPU scipy otherwise. The
            # GPU function doesn't accept a pbar — it's fast enough that
            # the progress bar just sits at 1% then jumps to 100% on completion.
            from ._vendor.worldgen.src.panorama_utils import merge_panorama_depth_gpu
            depth_np, mask_np = merge_panorama_depth_gpu(
                int(out_width), int(out_height),
                distance_maps, pred_masks,
                extr_list, intr_list,
                chunk_size=int(chunk_size),
                center_weight_power=float(center_weight_power),
                scale_anchor=bool(scale_anchor),
            )
        else:
            # CPU path: scipy LSMR end-to-end. Recursive LSMR (downsamples
            # first, refines up). pbar updates per recursion level, weighted
            # by pixels^1.5 so the bar advances slowly until the unwind
            # reaches the top level (which dominates total wall-time).
            depth_np, mask_np = merge_panorama_depth(
                int(out_width), int(out_height),
                distance_maps, pred_masks,
                extr_list, intr_list,
                pbar=pbar,
            )

        # Match upstream HY-World 2.0 post-merge processing
        # (panorama_utils.py:572-580). Two steps:
        #   1. Inpaint invalid pixels with the 99.9th-percentile (sky) depth
        #      instead of zero. Zero-depth puts those pixels right on the
        #      camera origin and breaks downstream point-cloud / mesh geometry.
        #   2. Smooth the south-pole strip (bottom 5%) to fix left-right
        #      seams the LSMR solver can't enforce due to the equirect
        #      parameterization stretching infinitely at the pole.
        depth_np = depth_np.astype(np.float32)
        valid = mask_np.astype(bool)
        if valid.any() and (~valid).any():
            sky_depth = float(np.nanquantile(depth_np[valid], 0.999))
            depth_np[~valid] = sky_depth
            _p(f"  inpainted {int((~valid).sum())} invalid pixels with sky_depth={sky_depth:.3f}")
        from ._vendor.worldgen.src.panorama_utils import smooth_south_pole_depth
        depth_np = smooth_south_pole_depth(depth_np, smooth_height_ratio=0.05)
        # Belt-and-braces: trap any residual non-finite values that escaped.
        depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
        _p(f"merged depth: shape {depth_np.shape}, min={float(depth_np.min()):.3f}, "
           f"median={float(np.median(depth_np)):.3f}, max={float(depth_np.max()):.3f}; "
           f"valid pixels: {int(mask_np.sum())} / {mask_np.size}")

        # IMAGE convention: (B, H, W, C). Broadcast depth across 3 channels so it
        # composes with regular depth-viz nodes.
        depth_img = torch.from_numpy(depth_np).unsqueeze(-1).expand(-1, -1, 3).unsqueeze(0).contiguous()
        valid_mask = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)

        # ----- Debug image: per-pixel disagreement among overlapping faces -----
        # Re-project each face's per-pixel ray distance to the equirect grid
        # using the same projection the LSMR uses internally, then compute the
        # std across the faces that cover each pixel. Strong consensus →
        # std ≈ 0 (dark); silhouette / depth-edge regions where the merger had
        # to compromise → std spikes (bright).
        debug_img = _build_debug_image(
            distance_maps=distance_maps,
            weights=weights,
            extr_list=extr_list,
            intr_list=intr_list,
            depth_np=depth_np,
            out_width=int(out_width),
            out_height=int(out_height),
        )

        # ----- xyz_views: 3 orthographic projections of the unprojected
        # depth cloud (TOP/FRONT/SIDE), rendered offscreen with PyVista
        # and concatenated into a single 3-panel IMAGE. -----
        try:
            xyz_views_img = _render_xyz_views(
                depth_np=depth_np,
                mask_np=mask_np,
            )
        except Exception as e:
            _p(f"xyz_views: render failed ({e}); returning empty image")
            xyz_views_img = torch.zeros((1, 1, 1, 3), dtype=torch.float32)

        # Release the merge function's GPU cache back to CUDA. The
        # caching allocator otherwise keeps cached blocks alive for the
        # lifetime of the comfy-env worker subprocess, starving the next
        # node in the workflow (CuMesh remesh, SHARP predict, etc.) of
        # contiguous VRAM. Also drop any local references the caller
        # would otherwise hold across the return boundary.
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pbar.update_absolute(100, 100)
        return io.NodeOutput(depth_img, valid_mask, debug_img, xyz_views_img)


NODE_CLASS_MAPPINGS = {"PanoramaDepthMerge": PanoramaDepthMerge}
NODE_DISPLAY_NAME_MAPPINGS = {"PanoramaDepthMerge": "Panorama Depth Merge"}
