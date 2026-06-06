"""PanoramaFacesPlaneSegment - per-face plane-instance segmentation via
Open3D RANSAC on a backprojected point cloud.

Algorithm (per face, independent):
  1. Backproject MoGe-2 depth -> 3D point cloud (or consume `points_raw`
     directly). Attach normals if provided so RANSAC scores inlier
     consistency by point-to-plane distance AND normal alignment.
  2. Iteratively call `pcd.segment_plane(distance_threshold, ransac_n=3,
     num_iterations)` until remaining-inlier fraction < threshold OR
     max planes hit.
  3. Optional normal-consistency filter: drop planes whose RANSAC
     normal disagrees with the median per-pixel normal of its inliers.
  4. DBSCAN cluster inliers to split disconnected coplanar pieces
     (e.g. two parallel walls behind / in front of the camera).
  5. Project per-cluster labels back to the image grid via the
     pixel->point-index map.

Output: per-face plane-instance MASK + RGB visualization.

Each face gets its own LOCAL plane IDs (1, 2, 3, ...; 0 = background /
no plane). Cross-face consistency is NOT enforced here - feed the
output into MeshSegmenter's `Lift2DTo3DLabels` (samesh algorithm) to
unify labels across views via a mesh.

Per-face inference: ~30-100 ms on CPU at 512x512.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from comfy_api.latest import io


def _p(msg: str) -> None:
    print(f"[PanoramaFacesPlaneSegment] {msg}", file=sys.stderr, flush=True)


def _label_to_color(label: int) -> tuple[int, int, int]:
    """Stable RGB color from an integer label via golden-ratio hue."""
    if label <= 0:
        return (0, 0, 0)
    # Golden-ratio hue scrambling for visually-distinct colors.
    h = (label * 0.61803398875) % 1.0
    s, v = 0.85, 0.95
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r, g, b = [
        (v, t, p), (q, v, p), (p, v, t),
        (p, q, v), (t, p, v), (v, p, q),
    ][i % 6]
    return (int(r * 255), int(g * 255), int(b * 255))


def _segment_planes_one_face(
    points: np.ndarray,
    normals: np.ndarray | None,
    valid: np.ndarray,
    distance_threshold: float,
    max_planes: int,
    min_inlier_fraction: float,
    ransac_iterations: int,
    dbscan_eps: float,
    dbscan_min_points: int,
    normal_consistency_threshold: float,
    face_tag: str = "",
    use_gpu: bool = True,
    ransac_iterations_per_batch: int = 200,
) -> tuple[np.ndarray, int]:
    """Run the iterative-RANSAC + DBSCAN-split pipeline on a single
    (H, W, 3) point map. Returns (label_map, n_planes_detected).

    `face_tag` is prepended to each diagnostic print so multi-face
    callers can identify which face the line belongs to (e.g. "5/42").

    `use_gpu=True` routes the RANSAC plane fit through
    `torch_ransac3d.plane.plane_fit` on CUDA (typical ~30-50x speedup
    on 2.4M-point faces). Falls back to Open3D's CPU `segment_plane`
    if `torch-ransac3d` isn't installed or CUDA isn't available.
    DBSCAN clustering stays on CPU regardless - only the plane fit
    moves to GPU (it's the dominant cost at 1536^2 resolution).
    """
    import torch
    import open3d as o3d

    tag = f"{face_tag} " if face_tag else ""
    H, W = points.shape[:2]
    label_map = np.zeros((H, W), dtype=np.int32)

    # Flatten + mask. Track pixel-index <-> point-index correspondence.
    pts_flat = points.reshape(-1, 3).astype(np.float64)
    valid_flat = valid.reshape(-1).astype(bool)
    pixel_indices = np.where(valid_flat)[0]
    if len(pixel_indices) < dbscan_min_points:
        _p(f"  {tag}skip: only {len(pixel_indices)} valid pixels "
           f"< dbscan_min_points={dbscan_min_points}")
        return label_map, 0

    valid_pts = pts_flat[pixel_indices]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_pts)
    if normals is not None:
        valid_normals = normals.reshape(-1, 3).astype(np.float64)[pixel_indices]
        # Normalize defensively.
        n_len = np.maximum(np.linalg.norm(valid_normals, axis=-1, keepdims=True), 1e-12)
        valid_normals = valid_normals / n_len
        pcd.normals = o3d.utility.Vector3dVector(valid_normals)
        normals_arr = valid_normals
    else:
        normals_arr = None

    # --- GPU RANSAC setup (best-effort; silent fallback to CPU). ---
    gpu_plane_fit = None
    pts_gpu = None
    if use_gpu:
        try:
            from torch_ransac3d.plane import plane_fit as _tr3d_plane_fit
            if torch.cuda.is_available():
                gpu_device = torch.device("cuda")
                pts_gpu = torch.from_numpy(valid_pts.astype(np.float32)).to(gpu_device)
                gpu_plane_fit = _tr3d_plane_fit
            else:
                _p(f"  {tag}GPU requested but CUDA unavailable; using Open3D CPU RANSAC")
        except ImportError:
            _p(f"  {tag}GPU requested but `torch-ransac3d` not installed; using Open3D CPU RANSAC")
    ransac_backend = "torch_ransac3d (CUDA)" if gpu_plane_fit is not None else "Open3D (CPU)"
    _p(f"  {tag}RANSAC backend: {ransac_backend}")

    n_total = len(valid_pts)
    remaining = np.arange(n_total)
    plane_id = 1
    n_skipped_normal = 0
    # GPU RANSAC batch - auto-shrunk on OOM (the (N x per_batch) distance tensor
    # is the VRAM hog on full-panorama point counts) before any CPU fallback.
    cur_per_batch = int(ransac_iterations_per_batch)
    attempt = 0
    exit_reason = "max_planes_per_face hit"
    min_remaining = max(int(min_inlier_fraction * n_total), dbscan_min_points)

    while plane_id <= max_planes and len(remaining) > min_remaining:
        attempt += 1

        if gpu_plane_fit is not None:
            # --- GPU path: torch_ransac3d.plane.plane_fit ---
            # Slice the remaining-points subset on GPU. `remaining` is a
            # numpy int64 array; gather via index_select for efficiency.
            rem_idx = torch.from_numpy(remaining).to(pts_gpu.device, dtype=torch.long)
            sub_pts = pts_gpu.index_select(0, rem_idx)
            try:
                result = gpu_plane_fit(
                    pts=sub_pts,
                    thresh=float(distance_threshold),
                    max_iterations=int(ransac_iterations),
                    iterations_per_batch=int(cur_per_batch),
                    epsilon=1e-8,
                    device=pts_gpu.device,
                )
            except RuntimeError as e:
                # OOM (or any CUDA runtime error) - common when a single face is
                # a full equirect panorama (~7M pts): the (N x per_batch) distance
                # tensor blows up VRAM. First shrink the batch (keeps the GPU
                # speedup); only fall back to Open3D CPU once it can't shrink.
                try:
                    del sub_pts, rem_idx
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if cur_per_batch > 10:
                    cur_per_batch = max(10, cur_per_batch // 4)
                    _p(f"  {tag}GPU RANSAC OOM ({e}); retrying on GPU with "
                       f"iterations_per_batch={cur_per_batch}")
                else:
                    _p(f"  {tag}GPU RANSAC still OOM at per_batch={cur_per_batch} "
                       f"({e}); falling back to Open3D CPU for remaining planes")
                    gpu_plane_fit = None
                    pts_gpu = None
                    ransac_backend = "Open3D (CPU, GPU fallback)"
                attempt -= 1   # this attempt didn't complete; retry it
                continue
            plane_eq = np.asarray(
                result.equation.detach().cpu().numpy() if hasattr(result.equation, "detach")
                else np.asarray(result.equation),
                dtype=np.float64,
            )
            # `result.inliers` are indices into `sub_pts` (i.e. into `remaining`).
            inl = result.inliers
            if hasattr(inl, "detach"):
                inl = inl.detach().cpu().numpy()
            inl_np = np.asarray(inl, dtype=np.int64)
            n_inliers_raw = int(len(inl_np))
        else:
            # --- CPU path: Open3D segment_plane ---
            sub = pcd.select_by_index(remaining.tolist())
            try:
                plane_eq, inliers = sub.segment_plane(
                    distance_threshold=float(distance_threshold),
                    ransac_n=3,
                    num_iterations=int(ransac_iterations),
                )
            except RuntimeError as e:
                exit_reason = f"segment_plane raised RuntimeError: {e}"
                break
            plane_eq = np.asarray(plane_eq, dtype=np.float64)
            inl_np = np.asarray(inliers, dtype=np.int64)
            n_inliers_raw = len(inl_np)

        a, b, c, d = (float(x) for x in plane_eq[:4])
        _p(f"  {tag}attempt #{attempt}: |a,b,c,d|=[{a:+.3f},{b:+.3f},{c:+.3f},{d:+.3f}], "
           f"inliers={n_inliers_raw}/{len(remaining)}")

        if n_inliers_raw < dbscan_min_points:
            exit_reason = f"last RANSAC fit gave only {n_inliers_raw} inliers < dbscan_min_points={dbscan_min_points}"
            break

        # Global indices (into valid_pts) of this plane's inliers.
        # `inl_np` is local indices into `remaining` for either backend.
        global_inliers = remaining[inl_np]

        # Optional normal-consistency check.
        if normals_arr is not None and normal_consistency_threshold > 0:
            plane_normal = np.asarray(plane_eq[:3], dtype=np.float64)
            plane_normal /= max(np.linalg.norm(plane_normal), 1e-12)
            inlier_normals = normals_arr[global_inliers]
            cos = np.abs(inlier_normals @ plane_normal)
            consistent = cos > float(normal_consistency_threshold)
            n_kept = int(consistent.sum())
            if n_kept < dbscan_min_points:
                _p(f"    {tag}normal-filter REJECT: only {n_kept}/{n_inliers_raw} "
                   f"inliers have |cos|>={normal_consistency_threshold:.2f} "
                   f"(< dbscan_min_points={dbscan_min_points})")
                # Plane doesn't pass the normal sanity check - strip those
                # inliers from contention and try the next one.
                remaining = np.setdiff1d(remaining, global_inliers, assume_unique=False)
                n_skipped_normal += 1
                continue
            _p(f"    {tag}normal-filter: kept {n_kept}/{n_inliers_raw} inliers "
               f"(|cos|>={normal_consistency_threshold:.2f})")
            # Keep only the normal-consistent subset.
            global_inliers = global_inliers[consistent]

        # DBSCAN-split disconnected coplanar pieces.
        inlier_pcd = pcd.select_by_index(global_inliers.tolist())
        clusters = np.asarray(
            inlier_pcd.cluster_dbscan(
                eps=float(dbscan_eps),
                min_points=int(dbscan_min_points),
                print_progress=False,
            ),
            dtype=np.int64,
        )

        # Assign labels.
        unique_clusters = np.unique(clusters)
        unique_clusters = unique_clusters[unique_clusters >= 0]  # drop noise (-1)
        cluster_sizes = [int((clusters == cid).sum()) for cid in unique_clusters]
        # Show top-5 sizes (the rest are usually small).
        size_preview = sorted(cluster_sizes, reverse=True)[:5]
        more = f" (+{len(cluster_sizes) - 5} more)" if len(cluster_sizes) > 5 else ""
        _p(f"    {tag}DBSCAN: {len(unique_clusters)} cluster(s), "
           f"top sizes={size_preview}{more}")

        any_accepted = False
        for cid in unique_clusters:
            cluster_mask = clusters == cid
            csize = int(cluster_mask.sum())
            if csize < dbscan_min_points:
                continue
            cluster_pts = global_inliers[cluster_mask]
            pix_idx = pixel_indices[cluster_pts]
            ys = pix_idx // W
            xs = pix_idx % W
            label_map[ys, xs] = plane_id
            _p(f"      {tag}-> label_id={plane_id}, {csize} px")
            plane_id += 1
            any_accepted = True
            if plane_id > max_planes:
                break

        # Remove this plane's inliers from contention regardless of
        # whether any cluster was kept - we don't want to refit them.
        remaining = np.setdiff1d(remaining, global_inliers, assume_unique=False)

        if not any_accepted:
            # All clusters too small; treat as ineffective attempt but
            # keep iterating until budgets run out.
            _p(f"    {tag}(no cluster >= dbscan_min_points; inliers retired)")

    else:
        # `while` exited normally on its condition - figure out which.
        if plane_id > max_planes:
            exit_reason = f"max_planes_per_face={max_planes} hit"
        elif len(remaining) <= min_remaining:
            exit_reason = (
                f"remaining={len(remaining)} <= min_remaining={min_remaining} "
                f"(min_inlier_fraction={min_inlier_fraction:.3f} of {n_total})"
            )

    n_planes = plane_id - 1
    _p(f"  {tag}RANSAC done ({exit_reason}); "
       f"{attempt} attempt(s), {n_skipped_normal} normal-filter rejects, "
       f"{n_planes} plane(s) accepted")
    return label_map, n_planes


class PanoramaFacesPlaneSegment(io.ComfyNode):
    """Per-face plane-instance segmentation via Open3D iterative RANSAC."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PanoramaFacesPlaneSegment",
            display_name="Panorama Faces Plane Segment (Open3D RANSAC)",
            category="PanoPack",
            description=(
                "Per-face planar instance segmentation. Takes a batch of "
                "perspective face crops' 3D point maps (e.g. MoGe-2's "
                "`points_raw` output, shape (B, H, W, 3)) and an optional "
                "normal map; returns per-face per-pixel plane-instance "
                "labels via iterative Open3D RANSAC + DBSCAN-split + "
                "optional normal-consistency filtering.\n\n"
                "Each face gets independent LOCAL plane IDs (1, 2, 3, ...; "
                "0 = no plane). Cross-face consistency is NOT enforced "
                "here - feed the output into MeshSegmenter's "
                "`Lift2DTo3DLabels` (samesh algorithm) to unify labels "
                "globally via the merged mesh."
            ),
            inputs=[
                io.Image.Input(
                    "face_points",
                    tooltip="Per-face 3D POINT MAPS in each face's camera "
                            "frame. Shape (B, H, W, 3). MoGe-2's "
                            "`points_raw` output, run on the face_images "
                            "from a panorama split."),
                io.Image.Input(
                    "face_normals", optional=True,
                    tooltip="Per-face surface normals (B, H, W, 3) - "
                            "MoGe-2's `normal` output. Used for the "
                            "normal-consistency filter: planes whose "
                            "RANSAC normal disagrees with the inlier "
                            "median normal get rejected. Highly "
                            "recommended; without it the filter is off."),
                io.Mask.Input(
                    "face_valid_masks", optional=True,
                    tooltip="Per-face validity masks (B, H, W). Pixels "
                            "with mask < 0.5 are excluded from RANSAC. "
                            "If unset, derived from `isfinite(z) & "
                            "(z > 0)` on the point map."),
                io.Boolean.Input(
                    "use_gpu", default=True, optional=True,
                    tooltip="Run the RANSAC plane fit on CUDA via "
                            "`torch-ransac3d.plane.plane_fit` instead of "
                            "Open3D's CPU `segment_plane`. ~30-50x faster "
                            "on 2.4M-point faces (1536^2) at typical "
                            "ransac_iterations=1000. Falls back to CPU "
                            "Open3D automatically if `torch-ransac3d` "
                            "isn't installed or CUDA isn't available "
                            "(prints a fallback line per face).\n\n"
                            "DBSCAN clustering stays on Open3D CPU "
                            "regardless - it's the smaller cost at this "
                            "scale."),
                io.Int.Input(
                    "ransac_iterations_per_batch", default=200, min=50,
                    max=5000, step=50, optional=True,
                    tooltip="GPU-only: number of RANSAC iterations "
                            "processed per parallel batch in "
                            "`torch_ransac3d`. The (N, K) distance "
                            "tensor scales as `points x batch`; with "
                            "200 batch x 2.4M points x 4B = ~1.9 GB "
                            "peak. Lower if you OOM, raise if you have "
                            "VRAM headroom (faster, fewer kernel "
                            "launches). Ignored when use_gpu=False."),
                io.Float.Input(
                    "distance_threshold", default=0.02, min=0.001, max=1.0,
                    step=0.001, optional=True,
                    tooltip="RANSAC inlier tolerance in METERS. A pixel "
                            "is on the candidate plane if its 3D point "
                            "is within this distance. 0.02 m = 2 cm is "
                            "a good starting point for room-scale indoor "
                            "scenes. Loosen to 0.05 for noisy depth, "
                            "tighten to 0.01 for clean depth."),
                io.Int.Input(
                    "max_planes_per_face", default=10, min=1, max=50,
                    step=1, optional=True,
                    tooltip="Stop after this many planes per face. 10 is "
                            "plenty for typical indoor rooms (4 walls + "
                            "floor + ceiling + a few large furniture "
                            "faces). Increase for cluttered scenes."),
                io.Float.Input(
                    "min_inlier_fraction", default=0.02, min=0.001,
                    max=0.5, step=0.001, optional=True,
                    tooltip="Stop iterating once remaining unsegmented "
                            "fraction falls below this. 0.02 = stop "
                            "when <= 2% of the point cloud is still "
                            "unsegmented."),
                io.Int.Input(
                    "ransac_iterations", default=1000, min=100, max=10000,
                    step=100, optional=True,
                    tooltip="Open3D RANSAC iterations per plane fit. "
                            "1000 is fast (~5-15 ms per plane); bump to "
                            "5000 for higher robustness on noisy point "
                            "clouds at a ~3-5x speed cost."),
                io.Float.Input(
                    "dbscan_eps", default=0.05, min=0.001, max=2.0,
                    step=0.001, optional=True,
                    tooltip="DBSCAN connectivity radius (METERS) for "
                            "splitting disconnected coplanar pieces. "
                            "Two coplanar walls behind / in front of "
                            "the camera get separate labels if they're "
                            "more than this apart. 0.05 m = 5 cm is a "
                            "good indoor default."),
                io.Int.Input(
                    "dbscan_min_points", default=100, min=10, max=10000,
                    step=10, optional=True,
                    tooltip="Minimum DBSCAN cluster size. Pieces smaller "
                            "than this are dropped as noise."),
                io.Float.Input(
                    "normal_consistency_threshold", default=0.85,
                    min=0.0, max=1.0, step=0.01, optional=True,
                    tooltip="Cosine-similarity threshold between the "
                            "RANSAC plane normal and per-pixel inlier "
                            "normals. Pixels with |n_pixel . n_plane| < "
                            "threshold are stripped from the plane's "
                            "inlier set. If too few survive, the plane "
                            "is rejected entirely.\n\n"
                            "0.85 ~= 30deg tolerance (reasonable default). "
                            "0.95 ~= 18deg (strict). 0.0 disables the "
                            "filter. Requires `face_normals` to be "
                            "wired; otherwise this parameter has no "
                            "effect."),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="plane_labels",
                    tooltip="(B, H, W) per-face plane-instance MASK. "
                            "Pixel value = plane ID (1, 2, 3, ...) within "
                            "the face. 0 = no plane / background. IDs "
                            "are LOCAL to each face - face 0 and face 1 "
                            "both use IDs 1..N, NOT shared."),
                io.Image.Output(
                    display_name="plane_visualization",
                    tooltip="(B, H, W, 3) RGB visualization of the "
                            "plane labels (golden-ratio hue per ID). "
                            "For previewing/debugging."),
                io.String.Output(
                    display_name="info",
                    tooltip="Per-face plane count + global stats as text."),
            ],
        )

    @classmethod
    def execute(
        cls,
        face_points: torch.Tensor,
        face_normals: torch.Tensor | None = None,
        face_valid_masks: torch.Tensor | None = None,
        use_gpu: bool = True,
        ransac_iterations_per_batch: int = 200,
        distance_threshold: float = 0.02,
        max_planes_per_face: int = 10,
        min_inlier_fraction: float = 0.02,
        ransac_iterations: int = 1000,
        dbscan_eps: float = 0.05,
        dbscan_min_points: int = 100,
        normal_consistency_threshold: float = 0.85,
    ):
        # --- Coerce inputs to numpy with consistent shapes. ---
        import torch
        p = (
            face_points.detach().cpu().numpy()
            if isinstance(face_points, torch.Tensor)
            else np.asarray(face_points)
        )
        if p.ndim != 4 or p.shape[-1] != 3:
            raise ValueError(
                f"face_points must be (B, H, W, 3); got {p.shape}. "
                f"Wire MoGe-2 Inference's `points_raw` output."
            )
        N, H, W, _ = p.shape
        p = p.astype(np.float32, copy=False)

        if face_normals is not None:
            n = (
                face_normals.detach().cpu().numpy()
                if isinstance(face_normals, torch.Tensor)
                else np.asarray(face_normals)
            )
            if n.ndim != 4 or n.shape != p.shape:
                raise ValueError(
                    f"face_normals shape {n.shape} doesn't match face_points {p.shape}."
                )
            n = n.astype(np.float32, copy=False)
            # Auto-detect normal encoding. MoGe-2's ComfyUI `normal` output
            # is RGB-encoded `(n + 1) / 2` so it can flow on the IMAGE
            # socket - range [0, 1], NOT [-1, +1]. Some other models emit
            # raw signed normals. Heuristic: if the global min >= -0.05,
            # treat as RGB-encoded and decode. Otherwise treat as raw.
            n_min = float(n.min())
            n_max = float(n.max())
            if n_min >= -0.05:
                _p(
                    f"normals: RGB-encoded detected (range "
                    f"[{n_min:.3f}, {n_max:.3f}]  subset of  [0, 1]) -> decoding via 2.x-1"
                )
                n = 2.0 * n - 1.0
            else:
                _p(
                    f"normals: signed range detected "
                    f"[{n_min:.3f}, {n_max:.3f}] -> using as-is"
                )
        else:
            n = None

        if face_valid_masks is not None:
            v = (
                face_valid_masks.detach().cpu().numpy()
                if isinstance(face_valid_masks, torch.Tensor)
                else np.asarray(face_valid_masks)
            )
            if v.ndim == 4:
                v = v[..., 0]
            if v.shape != (N, H, W):
                raise ValueError(
                    f"face_valid_masks shape {v.shape} doesn't match (N={N}, H={H}, W={W})."
                )
            v_arr = v > 0.5
        else:
            v_arr = None

        # --- Banner: state what we're about to do. ---
        normals_state = "wired" if n is not None else "None (filter disabled)"
        valid_state = "wired" if v_arr is not None else "auto (from z>0)"
        cuda_state = (
            f"CUDA available: {torch.cuda.is_available()}"
            if use_gpu else "CPU (use_gpu=False)"
        )
        _p(
            f"input: {N} faces @ {H}x{W}, normals={normals_state}, "
            f"valid_masks={valid_state}, RANSAC={cuda_state}"
        )
        _p(
            f"params: dist_thresh={distance_threshold:.3f}m, "
            f"max_planes={max_planes_per_face}, "
            f"min_inlier_frac={min_inlier_fraction:.3f}, "
            f"ransac_iter={ransac_iterations} "
            f"(per_batch={ransac_iterations_per_batch}), "
            f"dbscan_eps={dbscan_eps:.3f}m, "
            f"dbscan_min_pts={dbscan_min_points}, "
            f"normal_cos_thresh={normal_consistency_threshold:.2f}"
        )

        # ComfyUI progress bar (best-effort - not available in some test envs).
        try:
            import comfy.utils
            pbar = comfy.utils.ProgressBar(N)
        except Exception:
            class _NoopPbar:
                def update(self, *_a, **_k): pass
            pbar = _NoopPbar()

        # --- Run per-face segmentation. ---
        label_masks = np.zeros((N, H, W), dtype=np.int32)
        viz = np.zeros((N, H, W, 3), dtype=np.float32)
        per_face_counts: list[int] = []
        total_planes = 0
        t_start = time.perf_counter()

        # Precompute a color LUT large enough for any reasonable plane count.
        max_lut = max(max_planes_per_face * 4, 64)
        lut = np.zeros((max_lut + 1, 3), dtype=np.float32)
        for i in range(1, max_lut + 1):
            lut[i] = np.array(_label_to_color(i), dtype=np.float32) / 255.0

        for face_idx in range(N):
            face_pts = p[face_idx]
            face_norm = n[face_idx] if n is not None else None
            if v_arr is not None:
                face_valid = v_arr[face_idx]
            else:
                z = face_pts[..., 2]
                face_valid = np.isfinite(z) & (z > 1e-6)

            n_valid = int(face_valid.sum())
            face_total = H * W
            face_tag = f"face {face_idx + 1}/{N}"
            _p(
                f"{face_tag}: valid={n_valid}/{face_total} "
                f"({100.0 * n_valid / max(face_total, 1):.1f}%)"
            )

            t_face = time.perf_counter()
            label_map, n_planes = _segment_planes_one_face(
                face_pts, face_norm, face_valid,
                distance_threshold=float(distance_threshold),
                max_planes=int(max_planes_per_face),
                min_inlier_fraction=float(min_inlier_fraction),
                ransac_iterations=int(ransac_iterations),
                dbscan_eps=float(dbscan_eps),
                dbscan_min_points=int(dbscan_min_points),
                normal_consistency_threshold=float(normal_consistency_threshold),
                face_tag=face_tag,
                use_gpu=bool(use_gpu),
                ransac_iterations_per_batch=int(ransac_iterations_per_batch),
            )
            dt_face = time.perf_counter() - t_face
            label_masks[face_idx] = label_map
            per_face_counts.append(n_planes)
            total_planes += n_planes

            # Visualization: look up colors per label, indexed by min(label, max_lut).
            clamped = np.minimum(label_map, max_lut)
            viz[face_idx] = lut[clamped]

            labeled_px = int((label_map > 0).sum())
            cov_pct = 100.0 * labeled_px / max(face_total, 1)
            _p(
                f"{face_tag} done: {n_planes} plane(s), "
                f"{labeled_px}/{face_total} px labeled ({cov_pct:.1f}%), "
                f"{1000.0 * dt_face:.0f}ms"
            )

            pbar.update(1)

        dt_total = time.perf_counter() - t_start

        # --- Pack outputs. ---
        labels_mask = torch.from_numpy(label_masks.astype(np.float32))
        viz_img = torch.from_numpy(viz)

        if per_face_counts:
            cnt_arr = np.asarray(per_face_counts, dtype=np.int32)
            cnt_min = int(cnt_arr.min())
            cnt_med = int(np.median(cnt_arr))
            cnt_max = int(cnt_arr.max())
            cnt_zero = int((cnt_arr == 0).sum())
        else:
            cnt_min = cnt_med = cnt_max = cnt_zero = 0

        info_lines = [
            f"PanoramaFacesPlaneSegment: {N} faces processed in "
            f"{dt_total:.2f}s ({1000.0 * dt_total / max(N, 1):.0f} ms/face).",
            f"  total planes detected: {total_planes}",
            f"  per-face: min={cnt_min}, median={cnt_med}, max={cnt_max}, "
            f"zero-plane faces={cnt_zero}/{N}",
            f"  mean planes/face: {total_planes / max(N, 1):.2f}",
            f"  per-face counts: {per_face_counts}",
            f"  params: dist_thresh={distance_threshold:.3f}m, "
            f"max_planes={max_planes_per_face}, "
            f"dbscan_eps={dbscan_eps:.3f}m, "
            f"normal_cos_thresh={normal_consistency_threshold:.2f}",
        ]
        info = "\n".join(info_lines)

        _p(
            f"DONE: {N} faces -> {total_planes} total planes "
            f"in {dt_total:.2f}s "
            f"(min={cnt_min}/med={cnt_med}/max={cnt_max} planes/face, "
            f"zero-plane faces: {cnt_zero})"
        )

        return io.NodeOutput(labels_mask, viz_img, info)


NODE_CLASS_MAPPINGS = {"PanoramaFacesPlaneSegment": PanoramaFacesPlaneSegment}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PanoramaFacesPlaneSegment": "Panorama Faces Plane Segment (Open3D RANSAC)",
}
