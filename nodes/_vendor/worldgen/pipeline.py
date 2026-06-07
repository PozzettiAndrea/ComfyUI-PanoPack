"""Panorama RGBD -> mesh builder (vendored, trimmed for PanoPack).

Single entry point used by PanoPack's PanoramaBuildMesh node: `build_mesh(
panorama_pil, depth_np, ...)` -> live `trimesh.Trimesh`. It runs the depth
post-process (edge + sky + invalid masks, optional far-depth clip, optional
outdoor contract) and dispatches to `convert_rgbd2mesh_panorama`.

Originally part of the upstream HY-World WorldNav pipeline; the trajectory
planning, VLM captioning and SAM segmentation stages (and their navi_utils /
vlm_utils / camera_utils / seg_utils dependencies) are not used by PanoPack
and have been removed.
"""

from __future__ import annotations
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from glob import glob
from types import SimpleNamespace
from typing import Any, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from .src.general_utils import Timer, set_seed, rank0_log
from .src.panorama_utils import (
    convert_rgbd2pcd_panorama,
    convert_rgbd2mesh_panorama,
    spherical_uv_to_directions,
)
import utils3d

def _make_pbar(total: int = 100):
    """ComfyUI native progress bar, falling back to a no-op when not in a worker."""
    try:
        import comfy.utils
        pbar = comfy.utils.ProgressBar(total)
    except Exception:
        class _Noop:
            def update_absolute(self, *_a, **_k): pass
        pbar = _Noop()
    return pbar


def _normalize_panorama(panorama_pil: Image.Image) -> Image.Image:
    full_img = panorama_pil.convert("RGB")
    if full_img.size[1] > 1920:
        print(f"[WorldNav] panorama {full_img.size} > 1920 tall -> resizing to 3840x1920", flush=True)
        full_img = full_img.resize((3840, 1920), resample=Image.Resampling.BICUBIC)
    return full_img


def build_mesh(
    panorama_pil: Image.Image,
    depth_np: np.ndarray,
    *,
    sky_mask_np: Optional[np.ndarray] = None,
    valid_mask_np: Optional[np.ndarray] = None,
    scene_type: str = "indoor",
    contract: Optional[float] = 8.0,
    edge_rtol: float = 0.1,
    seed: int = 1024,
    device: str = "cuda",
    clip_quantile: float = 0.99,
):
    """Stage 2: depth post-process + RGBD -> 3D mesh. No models.

    Returns a live `trimesh.Trimesh` with WorldNav-specific metadata attached
    on `mesh.metadata`:
      - `global_median_depth`: float (used by trajectory planner)
      - `scene_type`: str pass-through ("indoor"/"outdoor")
      - `is_outdoor`: bool

    Compatible with ComfyUI-GeometryPack's TRIMESH socket convention - wire
    straight into `Preview Mesh (Three.js)` or any GeometryPack mesh op.
    """
    from utils3d.numpy.maps import uv_map as _uv_map, depth_map_edge as _depth_map_edge

    pbar = _make_pbar(100)
    def _stage(pct, msg):
        print(f"[WorldNavBuildMesh] [{pct:3d}%] {msg}", flush=True)
        pbar.update_absolute(pct, 100)

    set_seed(seed)
    timer = Timer()

    _stage(2, f"received depth {depth_np.shape}, sky_mask={'provided' if sky_mask_np is not None else 'None'}, "
              f"valid_mask={'provided' if valid_mask_np is not None else 'None'}, scene_type='{scene_type}'")
    full_img = _normalize_panorama(panorama_pil)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    is_outdoor = scene_type == "outdoor"

    # --- Equirect rays ---
    _stage(10, f"deriving equirect rays from depth shape {depth_np.shape}")
    H, W = depth_np.shape[:2]
    uv = _uv_map(H, W)
    rays_np = spherical_uv_to_directions(uv).astype(np.float32)
    d_min, d_med, d_max = float(np.nanmin(depth_np)), float(np.nanmedian(depth_np)), float(np.nanmax(depth_np))
    print(f"[WorldNavBuildMesh]   depth stats: min={d_min:.3f}, median={d_med:.3f}, max={d_max:.3f}", flush=True)

    full_depth = {
        "distance": torch.as_tensor(depth_np, dtype=torch.float32, device=dev),
        "rays":     torch.as_tensor(rays_np,  dtype=torch.float32, device=dev),
    }

    # --- Resolve sky mask + valid mask into a single 'invalid' mask ---
    if sky_mask_np is not None:
        sky_mask = torch.from_numpy(np.asarray(sky_mask_np, dtype=bool))
    else:
        sky_mask = torch.zeros((H, W), dtype=torch.bool)
    if valid_mask_np is not None:
        invalid_init = torch.as_tensor(~np.asarray(valid_mask_np, dtype=bool))
        print(f"[WorldNavBuildMesh]   external valid_mask: {valid_mask_np.sum()} valid / {valid_mask_np.size} total",
              flush=True)
    else:
        invalid_init = torch.zeros((H, W), dtype=torch.bool)

    # --- Depth post-process: edge mask + sky mask + quantile clip ---
    _stage(25, "depth post-process: edge mask + sky mask + quantile-99 clip")
    with timer.track("Depth post-process"):
        edge_mask = torch.from_numpy(
            _depth_map_edge(full_depth["distance"].cpu().numpy(), rtol=float(edge_rtol))
        ).bool()
        print(f"[WorldNavBuildMesh]   depth-edge mask: {edge_mask.sum().item()} edge pixels", flush=True)
        sky_mask_for_depth = sky_mask
        if sky_mask_for_depth.shape != edge_mask.shape:
            print(f"[WorldNavBuildMesh]   resizing sky mask {sky_mask_for_depth.shape} -> {edge_mask.shape}",
                  flush=True)
            sky_mask_for_depth = F.interpolate(
                sky_mask_for_depth[None, None].float(),
                size=edge_mask.shape, mode="nearest",
            )[0, 0].bool()
        full_mask = (sky_mask_for_depth | edge_mask | invalid_init.to(edge_mask.device)).to(dev)
        unmasked = full_depth["distance"][~full_mask]
        # Far-depth clip at the `clip_quantile` percentile. This caps hallucinated
        # far depth (windows / open doors) but ALSO clamps real room corners - the
        # farthest points in a room - flattening trihedral wall corners into a
        # spherical cap (a visible chamfer). Set clip_quantile >= 1.0 to disable
        # entirely (keeps corners crisp; only safe when there are no far outliers).
        if clip_quantile is not None and float(clip_quantile) < 1.0:
            q = float(clip_quantile)
            if unmasked.numel() == 0:
                # Degenerate: sky+edge+invalid covers every pixel (usually a
                # Perceive sky-mask misfire). Fall back to the quantile over the
                # full depth without exclusion.
                n_masked = int(full_mask.sum().item())
                print(f"[WorldNavBuildMesh]   WARN: sky+edge+invalid mask covers all "
                      f"{n_masked}/{full_mask.numel()} pixels (likely Perceive "
                      f"sky-mask misfire); falling back to q{q:.3f} over the full depth.",
                      flush=True)
                max_d = torch.quantile(full_depth["distance"].flatten(), q=q).item()
            else:
                max_d = torch.quantile(unmasked, q=q).item()
            print(f"[WorldNavBuildMesh]   q{q:.3f} depth = {max_d:.3f}, clipping distance to [0, {max_d:.3f}]", flush=True)
            full_depth["distance"] = torch.clip(full_depth["distance"], 0, max_d)
        else:
            print(f"[WorldNavBuildMesh]   far-depth clip DISABLED (clip_quantile={clip_quantile}); "
                  f"corners preserved", flush=True)

        # Outdoor-only `contract` step (mirrors upstream HY-World 2.0
        # traj_generate.py:316-317). When scene_type=="outdoor", clip far
        # depth to median * contract (default 8.0). Caps the sky / very
        # distant background so the mesh doesn't get crushed when
        # computing extents. No-op for indoor or when contract<=0.
        if is_outdoor and contract is not None and contract > 0:
            valid_depth = full_depth["distance"][~full_mask]
            if valid_depth.numel() > 0:
                median_depth = float(torch.median(valid_depth).item())
                contract_distance = median_depth * float(contract)
                print(f"[WorldNavBuildMesh]   outdoor contract: median={median_depth:.3f} "
                      f"x contract={float(contract):.2f} = {contract_distance:.3f} -> clipping", flush=True)
                full_depth["distance"] = torch.clip(full_depth["distance"], 0, contract_distance)

    # --- RGBD -> mesh (resized to mesh resolution 960x1920) ---
    _stage(45, "convert_rgbd2mesh_panorama (resizing to 960x1920 for mesh build)")
    with timer.track("RGBD -> mesh"):
        mesh_h, mesh_w = 960, 1920
        img_resized = full_img.resize((mesh_w, mesh_h), resample=Image.Resampling.BICUBIC)
        depth_resized = F.interpolate(
            full_depth["distance"][None, None], size=(mesh_h, mesh_w), mode="nearest"
        )[0, 0]
        rays_resized = F.interpolate(
            full_depth["rays"].permute(2, 0, 1)[None], size=(mesh_h, mesh_w), mode="bilinear"
        )[0].permute(1, 2, 0)
        sky_mask_resized = F.interpolate(
            sky_mask.float()[None, None].to(dev), size=(mesh_h, mesh_w), mode="nearest"
        )[0, 0].bool()
        print(f"[WorldNavBuildMesh]   resized to {mesh_h}x{mesh_w}; "
              f"sky pixels (mesh-res): {sky_mask_resized.sum().item()}", flush=True)

        rgb_t = torch.as_tensor(np.array(img_resized) / 255.0, dtype=torch.float32)
        mesh = convert_rgbd2mesh_panorama(
            rgb=rgb_t,
            distance=depth_resized.to(dev),
            rays=rays_resized.to(dev),
            excluded_region_mask=sky_mask_resized.to(dev),
            device=dev,
        )

    # --- Convert open3d mesh -> trimesh.Trimesh (preserve vertex colors) ---
    _stage(85, "converting open3d -> trimesh.Trimesh")
    import trimesh as _tm
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    global_median_depth = float(torch.median(full_depth["distance"][~full_mask]).item())
    # convert_rgbd2mesh_panorama builds the o3d mesh with vertex_colors set from
    # the panorama RGB (panorama_utils.py:1238). Forward those into the Trimesh
    # so GLB export carries them and downstream nodes (Preview3D, point cloud,
    # etc.) see actual scene colors instead of a uniform default.
    vertex_colors_rgba = None
    if mesh.has_vertex_colors():
        vc = np.asarray(mesh.vertex_colors)  # (V, 3) float in [0, 1]
        if vc.shape[0] == vertices.shape[0]:
            vc = np.clip(vc, 0.0, 1.0)
            vertex_colors_rgba = np.empty((vc.shape[0], 4), dtype=np.uint8)
            vertex_colors_rgba[:, :3] = (vc * 255.0 + 0.5).astype(np.uint8)
            vertex_colors_rgba[:, 3] = 255
        else:
            print(f"[WorldNavBuildMesh]   WARN: vertex_colors len {vc.shape[0]} "
                  f"!= vertices len {vertices.shape[0]} - dropping colors",
                  flush=True)
    tm_mesh = _tm.Trimesh(
        vertices=vertices,
        faces=triangles,
        vertex_colors=vertex_colors_rgba,
        process=False,
    )
    print(f"[WorldNavBuildMesh]   mesh: {len(vertices)} verts, {len(triangles)} faces; "
          f"global_median_depth = {global_median_depth:.3f}; "
          f"vertex_colors={'yes' if vertex_colors_rgba is not None else 'no'}", flush=True)

    _stage(100, "build_mesh DONE")
    # Return the mesh + the planning-specific scalars as separate values.
    # No hidden state on `mesh.metadata` - downstream sockets are explicit.
    return tm_mesh, global_median_depth, scene_type

    # (Unreachable - kept so the legacy dict shape is visible in the file.)
    _legacy = {
        "vertices": vertices,
        "triangles": triangles,
        "global_median_depth": global_median_depth,
        "scene_type": scene_type,
        "is_outdoor": is_outdoor,
    }
