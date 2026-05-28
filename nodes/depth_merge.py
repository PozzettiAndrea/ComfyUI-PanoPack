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

        pbar.update_absolute(100, 100)
        return io.NodeOutput(depth_img, valid_mask)


NODE_CLASS_MAPPINGS = {"PanoramaDepthMerge": PanoramaDepthMerge}
NODE_DISPLAY_NAME_MAPPINGS = {"PanoramaDepthMerge": "Panorama Depth Merge"}
