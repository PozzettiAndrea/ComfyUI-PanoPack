"""PanoramaBuildMesh — depth + masks → 3D mesh (pure geometry, no models).

Stage 2 of WorldNav. Consumes panorama + depth + sky_mask + (optional) valid_mask
and produces an equirect-derived triangle mesh.

Bonus output: a `mesh_preview` IMAGE — three orthographic wireframe viewports
(top-down XZ, side XY, front ZY) so the user can sanity-check the mesh before
committing to the navmesh build.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from comfy_api.latest import io

from .utils import PANORAMA_TYPE, unwrap_panorama_to_image


def _p(msg: str) -> None:
    print(f"[PanoramaBuildMesh] {msg}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------
# Mesh preview rendering (pyvista — 3 axis-plane cross-sections)
# ----------------------------------------------------------------------

def _render_mesh_preview(
    vertices: np.ndarray,            # [V, 3]
    triangles: np.ndarray,           # [F, 3]
    *,
    width: int = 1536,
    height: int = 512,
) -> np.ndarray:
    """Render the mesh sliced through each axis plane (YZ at x=0, XZ at y=0,
    XY at z=0). For each cut we keep one half of the mesh and orient the
    camera on the discarded-half side, looking at the cut face. Combined
    into a single H×W×3 uint8 image with the three cross-sections side by
    side. Uses pyvista off-screen rendering (VTK OSMesa under the hood).
    """
    # Lazy import: pyvista is heavy and only this function needs it. Lets
    # the rest of the module import cleanly even if the env hasn't been
    # rebuilt yet after adding pyvista to comfy-env.toml.
    try:
        import os as _os
        _os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
        import pyvista as pv
    except Exception as e:
        _p(f"pyvista import failed: {e!r} — falling back to a placeholder preview")
        ph = Image.new("RGB", (width, height), (24, 24, 28))
        d = ImageDraw.Draw(ph)
        d.text((20, 20), f"pyvista not installed: {e}\nRun `cds install` to pick up "
               f"the new dep.", fill=(220, 220, 220))
        return np.array(ph, dtype=np.uint8)

    # trimesh / open3d-style (V, 3) verts + (F, 3) faces → pv.PolyData faces array
    # ([3, i0, i1, i2, 3, j0, j1, j2, ...]).
    F = int(triangles.shape[0])
    faces_flat = np.concatenate(
        [np.full((F, 1), 3, dtype=np.int64), triangles.astype(np.int64)],
        axis=1,
    ).flatten()
    poly = pv.PolyData(vertices.astype(np.float32), faces_flat)

    # Camera distance: a bit beyond the mesh extents so the cross-section
    # fills the viewport.
    bounds = poly.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    span = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4], 1e-6)
    R = span * 1.8

    # Three cuts: (title, normal, camera_pos, view_up).
    # For each: keep the half on the OPPOSITE side of `normal`, place camera
    # on the `+normal` side looking at the origin, so the camera sees the
    # cut face of the kept half.
    cuts = [
        ("YZ slice (x = 0)", ( 1.0, 0.0, 0.0), ( R, 0, 0), (0, 0, 1)),
        ("XZ slice (y = 0)", ( 0.0, 1.0, 0.0), (0,  R, 0), (0, 0, 1)),
        ("XY slice (z = 0)", ( 0.0, 0.0, 1.0), (0, 0,  R), (0, 1, 0)),
    ]

    plotter = pv.Plotter(
        off_screen=True,
        shape=(1, 3),
        window_size=(int(width), int(height)),
        border=True,
        border_color="gray",
    )
    plotter.set_background((0.09, 0.09, 0.11))

    for i, (title, normal, cam_pos, view_up) in enumerate(cuts):
        plotter.subplot(0, i)
        # pv.PolyData.clip(normal, origin, invert=True): keeps cells where
        # (p - origin) . normal < 0, i.e. the half OPPOSITE the camera (which
        # sits at +normal * R). That way the camera looks ACROSS the cut
        # plane at the cut face of the retained half — the juicy interior
        # cross-section — rather than at the outer surface of the half
        # nearest the camera.
        clipped = poly.clip(normal=normal, origin=(0.0, 0.0, 0.0), invert=True)
        if clipped.n_cells > 0:
            plotter.add_mesh(
                clipped,
                color=(0.55, 0.72, 0.95),
                show_edges=True,
                edge_color=(0.18, 0.28, 0.45),
                line_width=0.3,
                lighting=True,
                ambient=0.25,
                diffuse=0.7,
            )
        # Origin marker (where the camera was at capture time).
        plotter.add_mesh(pv.Sphere(radius=max(span * 0.01, 1e-3)),
                         color=(1.0, 0.85, 0.3))
        plotter.camera_position = [tuple(cam_pos), (0.0, 0.0, 0.0), tuple(view_up)]
        plotter.add_text(title, font_size=10, color="white", position="upper_left")

    img = plotter.screenshot(return_img=True)
    plotter.close()
    # Ensure shape (H, W, 3) uint8 with the requested dimensions.
    img = np.asarray(img, dtype=np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    if img.shape[:2] != (int(height), int(width)):
        img = np.array(
            Image.fromarray(img).resize((int(width), int(height)), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
    return img


# ----------------------------------------------------------------------
# Node
# ----------------------------------------------------------------------

class PanoramaBuildMesh(io.ComfyNode):
    """Stage 2: panorama + depth + sky_mask → 3D triangle mesh + preview IMAGE."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PanoramaBuildMesh",
            display_name="Panorama Build Mesh",
            category="PanoPack",
            description=(
                "Stage 2 of WorldNav. Pure geometry: takes panorama + depth + sky_mask "
                "and builds an Open3D triangle mesh (via convert_rgbd2mesh_panorama). "
                "Zero AI models loaded. Also emits a 3-viewport orthographic wireframe "
                "IMAGE for instant visual sanity-check before paying the navmesh cost."
            ),
            inputs=[
                io.Custom(PANORAMA_TYPE).Input(
                    "panorama",
                    tooltip="Equirectangular RGB panorama (2:1). PANORAMA = "
                            "IMAGE wrapped in PanoPack's typed socket. Wire "
                            "from PanoramaWrap or any node that emits a "
                            "PANORAMA."),
                io.Image.Input(
                    "depth",
                    tooltip="Equirect distance map (from MoGe panorama node)."),
                io.Mask.Input(
                    "sky_mask",
                    optional=True,
                    tooltip="From WorldNavPerceive. Excluded from mesh build."),
                io.Mask.Input(
                    "valid_mask",
                    optional=True,
                    tooltip="From MoGe2's valid_mask output. Untrusted pixels filtered out."),
                io.Boolean.Input(
                    "contract", default=False,
                    tooltip="Enable depth contraction: clip far depth to "
                            "median × contract_beyond. Useful for outdoor "
                            "scenes to cap distant background."),
                io.Float.Input(
                    "contract_beyond",
                    default=8.0, min=1.0, max=64.0, step=0.5,
                    optional=True,
                    tooltip="Clip far depth to median × this value. Only "
                            "used when 'contract' is enabled. Default 8.0 "
                            "matches upstream HY-World."),
                io.Float.Input(
                    "edge_rtol", default=0.1, min=0.0, max=1.0, step=0.01,
                    optional=True,
                    tooltip="Relative-tolerance for depth-edge detection in "
                            "the mesh build's depth post-process. Pixels "
                            "where the depth gradient exceeds rtol × local "
                            "depth get masked out (drops silhouettes / "
                            "depth discontinuities before triangulation, "
                            "preventing 'stretched' faces). Smaller = "
                            "stricter (more pixels excluded). Matches "
                            "BuildPointCloud's `edge_rtol`. Default 0.1 "
                            "preserves prior behavior."),
                io.Float.Input(
                    "clip_quantile", default=1.0, min=0.5, max=1.0, step=0.005,
                    optional=True,
                    tooltip="Far-depth clip percentile applied before "
                            "triangulation. Caps hallucinated far depth "
                            "(windows/open doors) — BUT also clamps real room "
                            "CORNERS (the farthest points), flattening "
                            "trihedral wall corners into a rounded cap (a "
                            "visible chamfer).\n\n"
                            "1.0 (default): DISABLED — corners preserved crisp. "
                            "Best for clean indoor scans with no far outliers.\n"
                            "0.99: legacy behavior — clips the farthest 1%, "
                            "smushes corners.\n"
                            "0.999: clip only extreme outliers, keep corners."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                io.Image.Output(display_name="mesh_preview"),
                io.Float.Output(display_name="global_median_depth"),
            ],
        )

    @classmethod
    def execute(cls, panorama, depth, sky_mask=None, valid_mask=None,
                contract=False, contract_beyond=8.0, edge_rtol=0.1,
                clip_quantile=1.0):
        # --- panorama → PIL ---
        pano_t = unwrap_panorama_to_image(panorama)
        arr = pano_t.detach().cpu().numpy() if isinstance(pano_t, torch.Tensor) else np.asarray(pano_t)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        pil = Image.fromarray(arr)

        # --- depth → [H, W] float ---
        d = depth.detach().cpu().numpy() if isinstance(depth, torch.Tensor) else np.asarray(depth)
        if d.ndim == 4:
            d = d[0]
        if d.ndim == 3:
            d = d[..., 0]
        depth_np = d.astype(np.float32)

        # --- sky_mask → [H, W] bool ---
        sky_mask_np = None
        if sky_mask is not None:
            sm = sky_mask.detach().cpu().numpy() if isinstance(sky_mask, torch.Tensor) else np.asarray(sky_mask)
            if sm.ndim == 3:
                sm = sm[0]
            sky_mask_np = (sm > 0.5).astype(bool)

        # --- valid_mask → [H, W] bool ---
        valid_mask_np = None
        if valid_mask is not None:
            vm = valid_mask.detach().cpu().numpy() if isinstance(valid_mask, torch.Tensor) else np.asarray(valid_mask)
            if vm.ndim == 3:
                vm = vm[0]
            valid_mask_np = (vm > 0.5).astype(bool)

        # Map contract toggle to scene_type for the underlying pipeline
        st = "outdoor" if contract else "indoor"
        contract_val = float(contract_beyond) if contract else None

        _p(f"panorama {pil.size} depth {depth_np.shape} contract={contract}")

        from ._vendor.worldgen.pipeline import build_mesh as _build_mesh

        mesh, global_median_depth, _scene_type_out = _build_mesh(
            pil,
            depth_np,
            sky_mask_np=sky_mask_np,
            valid_mask_np=valid_mask_np,
            scene_type=st,
            contract=contract_val,
            edge_rtol=float(edge_rtol),
            clip_quantile=float(clip_quantile),
        )

        # --- Mesh preview (PIL wireframe) ---
        _p("rendering mesh_preview wireframe…")
        preview_np = _render_mesh_preview(
            np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.int32),
            width=1536, height=512,
        )
        preview_t = torch.from_numpy(preview_np.astype(np.float32) / 255.0).unsqueeze(0)

        _p(f"output: mesh {len(mesh.vertices)} verts / {len(mesh.faces)} faces (TRIMESH); "
           f"global_median_depth={global_median_depth:.3f}; "
           f"preview {preview_t.shape}")
        return io.NodeOutput(mesh, preview_t, global_median_depth)


NODE_CLASS_MAPPINGS = {"PanoramaBuildMesh": PanoramaBuildMesh}
NODE_DISPLAY_NAME_MAPPINGS = {"PanoramaBuildMesh": "Panorama Build Mesh"}
