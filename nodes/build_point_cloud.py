"""PanoramaBuildPointCloud - panorama + depth -> trimesh.PointCloud.

Sibling of PanoramaBuildMesh. Same input contract (panorama + depth +
optional sky_mask + valid_mask + scene_type + contract), but emits a
TRIMESH point cloud instead of a triangle mesh.

Useful for:
- Feeding WorldStereo's per-frame point-cloud projection (the
  `render_video` + `render_mask` conditioning for the WorldStereo
  DiT) -- the cloud is the global-scene anchor.
- Gaussian-splat seeding / 3DGS pipelines that need a colored point
  cloud rather than a mesh.

Preprocessing mirrors PanoramaBuildMesh's `build_mesh` helper
(`_vendor/worldgen/pipeline.py:800`): equirect rays derived from depth
shape, sky + depth-edge + invalid masks unioned into a single
exclusion mask, distance clipped to the 99th percentile of valid
pixels, optional outdoor `contract` clip. We then dispatch to
`convert_rgbd2pcd_panorama` instead of the mesh converter.

NOT resized to mesh resolution (960x1920) the way BuildMesh does -
point clouds usually want fuller resolution since they have no face
budget to manage.
"""

from __future__ import annotations

import sys

import numpy as np
from PIL import Image
from comfy_api.latest import io

from .utils import PANORAMA_TYPE, unwrap_panorama_to_image


def _p(msg: str) -> None:
    print(f"[PanoramaBuildPointCloud] {msg}", file=sys.stderr, flush=True)


class PanoramaBuildPointCloud(io.ComfyNode):
    """Panorama + depth -> colored trimesh.PointCloud (TRIMESH)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PanoramaBuildPointCloud",
            display_name="Panorama Build Point Cloud",
            category="PanoPack",
            description=(
                "Companion to PanoramaBuildMesh. Same inputs (panorama + "
                "depth + sky_mask + valid_mask + scene_type + contract), "
                "but emits a TRIMESH point cloud instead of a triangle "
                "mesh. Dispatches to convert_rgbd2pcd_panorama with the "
                "same depth preprocessing as BuildMesh (edge + sky + "
                "invalid masks, q99 clip, optional outdoor contract).\n\n"
                "Wire into WorldStereoRenderPointCloud to produce the "
                "per-trajectory-frame point-cloud renders that feed the "
                "WorldStereo DiT."
            ),
            inputs=[
                io.Custom(PANORAMA_TYPE).Input(
                    "panorama",
                    tooltip="Equirectangular RGB panorama (2:1) - source of "
                            "per-point colors. PANORAMA = IMAGE wrapped in "
                            "PanoPack's typed socket. Wire from PanoramaWrap "
                            "or any node that emits a PANORAMA."),
                io.Image.Input(
                    "depth",
                    tooltip="Equirect distance map (from PanoramaDepthMerge "
                            "or any panoramic depth source)."),
                io.Mask.Input(
                    "sky_mask", optional=True,
                    tooltip="From WorldNavPerceive. Excluded from the cloud."),
                io.Mask.Input(
                    "valid_mask", optional=True,
                    tooltip="From MoGe2's valid_mask output. Invalid pixels "
                            "(typically sky / no-depth) are excluded."),
                io.Boolean.Input(
                    "contract", default=False,
                    tooltip="Enable depth contraction: clip far depth to "
                            "median x contract_beyond. Useful for outdoor "
                            "scenes to cap distant background."),
                io.Float.Input(
                    "contract_beyond",
                    default=8.0, min=1.0, max=64.0, step=0.5,
                    optional=True,
                    tooltip="Clip far depth to median x this value. Only "
                            "used when 'contract' is enabled. Default 8.0 "
                            "matches upstream HY-World."),
                io.Float.Input(
                    "edge_rtol", default=0.1, min=0.0, max=1.0, step=0.01,
                    optional=True,
                    tooltip="Relative-tolerance for depth-edge detection. "
                            "Smaller = stricter (more pixels excluded). "
                            "Matches BuildMesh."),
                io.Int.Input(
                    "max_size", default=4096, min=256, max=8192, step=16,
                    optional=True,
                    tooltip="Cap on the panorama dimension during cloud "
                            "construction. Larger = denser cloud + more "
                            "memory. Upstream default 4096."),
                io.Boolean.Input(
                    "dropout_pcd", default=False, optional=True,
                    tooltip="Random-thin to <= 1,000,000 points if the cloud "
                            "exceeds that. Off by default."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(
                    display_name="point_cloud",
                    tooltip="trimesh.PointCloud with vertices [N, 3] (world "
                            "space) + colors [N, 3]. Compatible with "
                            "ComfyUI-CADabra's TRIMESH socket. Wire into "
                            "WorldStereoRenderPointCloud."),
                io.Float.Output(
                    display_name="global_median_depth",
                    tooltip="Median distance over non-excluded pixels (same "
                            "as BuildMesh emits)."),
            ],
        )

    @classmethod
    def execute(cls, panorama, depth,
                sky_mask=None, valid_mask=None,
                contract: bool = False,
                contract_beyond: float = 8.0,
                edge_rtol: float = 0.1,
                max_size: int = 4096,
                dropout_pcd: bool = False):
        import torch
        import torch.nn.functional as F
        from utils3d.numpy.maps import uv_map as _uv_map, depth_map_edge as _depth_map_edge

        from ._vendor.worldgen.src.panorama_utils import (
            convert_rgbd2pcd_panorama,
            spherical_uv_to_directions,
        )

        # --- panorama -> PIL -> torch (H, W, 3) ---
        pano_t = unwrap_panorama_to_image(panorama)
        arr = pano_t.detach().cpu().numpy() if isinstance(pano_t, torch.Tensor) else np.asarray(pano_t)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.dtype != np.uint8:
            arr_u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr_u8 = arr
        pil = Image.fromarray(arr_u8)

        # --- depth -> np [H, W] float ---
        d = depth.detach().cpu().numpy() if isinstance(depth, torch.Tensor) else np.asarray(depth)
        if d.ndim == 4:
            d = d[0]
        if d.ndim == 3:
            d = d[..., 0]
        depth_np = d.astype(np.float32)
        H, W = depth_np.shape

        # --- sky_mask -> bool [H, W] (default: empty) ---
        if sky_mask is not None:
            sm = sky_mask.detach().cpu().numpy() if isinstance(sky_mask, torch.Tensor) else np.asarray(sky_mask)
            if sm.ndim == 3:
                sm = sm[0]
            sky_mask_np = (sm > 0.5).astype(bool)
        else:
            sky_mask_np = np.zeros((H, W), dtype=bool)

        # --- valid_mask -> invalid_init bool [H, W] ---
        if valid_mask is not None:
            vm = valid_mask.detach().cpu().numpy() if isinstance(valid_mask, torch.Tensor) else np.asarray(valid_mask)
            if vm.ndim == 3:
                vm = vm[0]
            invalid_init_np = ~(vm > 0.5).astype(bool)
        else:
            invalid_init_np = np.zeros((H, W), dtype=bool)

        # Resize sky_mask if shape mismatches depth (e.g. SAM3 ran at panorama
        # native resolution but depth came back at MoGe's resize_to).
        if sky_mask_np.shape != (H, W):
            sm_t = torch.from_numpy(sky_mask_np.astype(np.float32))[None, None]
            sky_mask_np = (F.interpolate(sm_t, size=(H, W), mode="nearest")[0, 0] > 0.5).numpy().astype(bool)
        if invalid_init_np.shape != (H, W):
            iv_t = torch.from_numpy(invalid_init_np.astype(np.float32))[None, None]
            invalid_init_np = (F.interpolate(iv_t, size=(H, W), mode="nearest")[0, 0] > 0.5).numpy().astype(bool)

        is_outdoor = bool(contract)
        _p(f"panorama {pil.size}, depth {depth_np.shape}, contract={contract}, "
           f"sky={int(sky_mask_np.sum())}, invalid={int(invalid_init_np.sum())}")

        # Equirect sanity: panorama is (W, H) via PIL; depth is (H, W) via
        # numpy. If they don't match, the user almost certainly wired
        # MoGe2's per-face `depth_raw` ([N, 512, 512, C]) directly here
        # instead of routing through PanoramaDepthMerge first.
        pano_w, pano_h = pil.size
        if (H, W) != (pano_h, pano_w):
            raise ValueError(
                f"PanoramaBuildPointCloud: depth shape ({H}, {W}) doesn't "
                f"match panorama ({pano_h}, {pano_w}). Wire your MoGe2 "
                f"per-face `depth_raw` through PanoramaDepthMerge first; "
                f"that produces an equirect distance map at the panorama's "
                f"resolution, which is what this node expects."
            )

        # --- Equirect rays from depth shape ---
        uv = _uv_map(H, W)
        rays_np = spherical_uv_to_directions(uv).astype(np.float32)

        # --- Depth post-process: edge + sky + invalid OR; q99 clip; optional outdoor contract ---
        edge_mask_np = _depth_map_edge(depth_np, rtol=float(edge_rtol)).astype(bool)
        full_mask_np = sky_mask_np | edge_mask_np | invalid_init_np
        _p(f"masks: edge={int(edge_mask_np.sum())}, "
           f"full(excluded)={int(full_mask_np.sum())}/{full_mask_np.size}")

        valid_depths = depth_np[~full_mask_np]
        if valid_depths.size == 0:
            raise ValueError(
                "PanoramaBuildPointCloud: every pixel is excluded "
                "(sky + edge + invalid masks cover the panorama). Check inputs."
            )
        max_d = float(np.quantile(valid_depths, 0.99))
        depth_np = np.clip(depth_np, 0, max_d)
        _p(f"q99 clip: distance clipped to [0, {max_d:.3f}]")

        if is_outdoor and contract_beyond > 0:
            median_d = float(np.median(depth_np[~full_mask_np]))
            contract_d = median_d * float(contract_beyond)
            depth_np = np.clip(depth_np, 0, contract_d)
            _p(f"contract: median={median_d:.3f} * {float(contract_beyond):.2f} = "
               f"{contract_d:.3f} -> clipping")

        global_median_depth = float(np.median(depth_np[~full_mask_np]))

        # --- RGBD -> point cloud ---
        rgb_t = torch.from_numpy(np.asarray(pil) / 255.0).float()

        # Color sanity: per-point colors are sampled straight from this panorama.
        # A (near-)constant panorama yields a flat-gray cloud -- almost always the
        # color socket being unconnected or fed a non-color image (depth/normal/mask).
        _p(f"panorama rgb: min={float(rgb_t.min()):.3f} max={float(rgb_t.max()):.3f} "
           f"mean={float(rgb_t.mean()):.3f} std={float(rgb_t.std()):.3f}")
        if float(rgb_t.std()) < 1e-3:
            _p(f"WARNING: panorama is (near-)constant {float(rgb_t.mean()):.3f} gray -- "
               f"the resulting point cloud will have no real color. The color socket "
               f"is likely unconnected or fed a non-color image (depth/normal/mask).")
        dist_t = torch.from_numpy(depth_np).float()
        rays_t = torch.from_numpy(rays_np).float()
        excluded_t = torch.from_numpy(full_mask_np).bool()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pcd = convert_rgbd2pcd_panorama(
            rgb=rgb_t,
            distance=dist_t,
            rays=rays_t,
            excluded_region_mask=excluded_t,
            max_size=int(max_size),
            device=device,
            dropout_pcd=bool(dropout_pcd),
        )
        n_pts = int(np.asarray(pcd.vertices).shape[0])
        _p(f"emit trimesh.PointCloud: {n_pts} vertices, "
           f"global_median_depth={global_median_depth:.3f}")
        return io.NodeOutput(pcd, global_median_depth)


NODE_CLASS_MAPPINGS = {"PanoramaBuildPointCloud": PanoramaBuildPointCloud}
NODE_DISPLAY_NAME_MAPPINGS = {"PanoramaBuildPointCloud": "Panorama Build Point Cloud"}
