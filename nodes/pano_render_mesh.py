"""PanoRenderMesh — render a trimesh as an ERP panorama (color).
PanoRenderMeshDepth — same but outputs an ERP depth map.

Both use PyVista offscreen rendering of 6 cube faces stitched to equirect.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from comfy_api.latest import io

from .utils import PANORAMA_TYPE, wrap_image_as_panorama
from .utils.cube_to_equirect import cube_faces_to_equirect


def _p(msg: str) -> None:
    print(f"[PanoRenderMesh] {msg}", file=sys.stderr, flush=True)


# 6 cube face cameras from origin, 90° FOV each.
# (camera_position, focal_point, view_up)
_CUBE_CAMERAS = [
    # +X
    ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
    # -X
    ((0, 0, 0), (-1, 0, 0), (0, 1, 0)),
    # +Y
    ((0, 0, 0), (0, 1, 0), (0, 0, -1)),
    # -Y
    ((0, 0, 0), (0, -1, 0), (0, 0, 1)),
    # +Z
    ((0, 0, 0), (0, 0, 1), (0, 1, 0)),
    # -Z
    ((0, 0, 0), (0, 0, -1), (0, 1, 0)),
]


def _trimesh_to_pyvista(mesh):
    """Convert a trimesh.Trimesh or trimesh.PointCloud to pv.PolyData."""
    import pyvista as pv

    verts = np.asarray(mesh.vertices, dtype=np.float32)

    # Check if it's a mesh with faces or a point cloud
    if hasattr(mesh, "faces") and len(mesh.faces) > 0:
        faces = np.asarray(mesh.faces, dtype=np.int64)
        F = faces.shape[0]
        faces_flat = np.concatenate(
            [np.full((F, 1), 3, dtype=np.int64), faces], axis=1,
        ).flatten()
        poly = pv.PolyData(verts, faces_flat)
    else:
        # Point cloud — create as point set
        poly = pv.PolyData(verts)

    # Vertex colors if available
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors"):
        colors = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)
        if colors.shape[-1] >= 3:
            poly["RGB"] = colors[:, :3]
            poly.set_active_scalars("RGB")
    elif hasattr(mesh, "colors") and mesh.colors is not None:
        colors = np.asarray(mesh.colors, dtype=np.uint8)
        if colors.ndim == 2 and colors.shape[-1] >= 3:
            poly["RGB"] = colors[:, :3]
            poly.set_active_scalars("RGB")

    return poly


def _render_cube_faces_color(mesh, face_size, point_size=2.0):
    """Render 6 cube faces of a mesh/point cloud as color images."""
    import os
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import pyvista as pv

    poly = _trimesh_to_pyvista(mesh)
    has_colors = "RGB" in poly.array_names
    is_pointcloud = poly.n_cells == poly.n_points  # no faces = point cloud

    faces = []
    for cam_pos, focal, view_up in _CUBE_CAMERAS:
        plotter = pv.Plotter(off_screen=True, window_size=(face_size, face_size))
        plotter.set_background((0, 0, 0))

        if is_pointcloud:
            if has_colors:
                plotter.add_points(poly, scalars="RGB", rgb=True,
                                   point_size=point_size, render_points_as_spheres=True)
            else:
                plotter.add_points(poly, color=(0.7, 0.8, 0.95),
                                   point_size=point_size, render_points_as_spheres=True)
        else:
            if has_colors:
                plotter.add_mesh(poly, scalars="RGB", rgb=True,
                                 lighting=True, ambient=0.3, diffuse=0.7)
            else:
                plotter.add_mesh(poly, color=(0.7, 0.8, 0.95),
                                 lighting=True, ambient=0.3, diffuse=0.7)

        # Set up 90° FOV perspective camera at origin
        plotter.camera.position = cam_pos
        plotter.camera.focal_point = focal
        plotter.camera.up = view_up
        plotter.camera.view_angle = 90.0
        plotter.camera.clipping_range = (0.001, 1000.0)

        img = plotter.screenshot(return_img=True)
        plotter.close()
        faces.append(img.astype(np.float32) / 255.0)

    return faces


def _render_cube_faces_depth(mesh, face_size):
    """Render 6 cube faces of a mesh as depth images."""
    import os
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import pyvista as pv

    poly = _trimesh_to_pyvista(mesh)

    faces = []
    for cam_pos, focal, view_up in _CUBE_CAMERAS:
        plotter = pv.Plotter(off_screen=True, window_size=(face_size, face_size))
        plotter.set_background((0, 0, 0))

        plotter.add_mesh(poly, color="white", lighting=False)

        plotter.camera.position = cam_pos
        plotter.camera.focal_point = focal
        plotter.camera.up = view_up
        plotter.camera.view_angle = 90.0
        plotter.camera.clipping_range = (0.001, 1000.0)

        # Get the depth buffer from VTK
        plotter.render()
        zbuf = plotter.get_image_depth(fill_value=0.0)
        plotter.close()

        # zbuf is (H, W) float, distance from camera
        # Convert to 3-channel for cube_faces_to_equirect compatibility
        depth_3ch = np.stack([zbuf, zbuf, zbuf], axis=-1).astype(np.float32)
        faces.append(depth_3ch)

    return faces


class PanoRenderMesh(io.ComfyNode):
    """Render a trimesh/point cloud as an equirectangular panorama."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PanoRenderMesh",
            display_name="Pano Render Mesh",
            category="PanoPack",
            description=(
                "Render a TRIMESH (mesh or point cloud) as an equirectangular "
                "panorama from the world origin.\n\n"
                "Uses PyVista offscreen rendering of 6 cube faces "
                "stitched into a 2:1 equirect."
            ),
            inputs=[
                io.Custom("TRIMESH").Input(
                    "mesh",
                    tooltip="The mesh or point cloud to render."),
                io.Int.Input(
                    "width", default=2048, min=256, max=8192, step=64,
                    tooltip="Output panorama width. Height = width / 2."),
                io.Int.Input(
                    "face_resolution", default=1024, min=256, max=4096, step=64,
                    tooltip="Resolution of each cube face render."),
                io.Float.Input(
                    "point_size", default=2.0, min=0.5, max=20.0, step=0.5,
                    tooltip="Point size for point cloud rendering (ignored for meshes)."),
            ],
            outputs=[
                io.Custom(PANORAMA_TYPE).Output(display_name="panorama"),
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, mesh, width=2048, face_resolution=1024, point_size=2.0):
        import time
        t0 = time.perf_counter()

        erp_w = int(width)
        erp_h = erp_w // 2
        face_size = int(face_resolution)

        _p(f"rendering 6 cube faces @ {face_size}px (color)")
        faces = _render_cube_faces_color(mesh, face_size, point_size=float(point_size))

        _p(f"stitching to {erp_w}x{erp_h} equirect")
        erp = cube_faces_to_equirect(faces, erp_w, erp_h)
        erp = np.clip(erp, 0, 1).astype(np.float32)

        _p(f"done in {time.perf_counter() - t0:.2f}s")

        erp_t = torch.from_numpy(erp).unsqueeze(0)
        pano = wrap_image_as_panorama(erp_t)
        return io.NodeOutput(pano, erp_t)


class PanoRenderMeshDepth(io.ComfyNode):
    """Render a trimesh depth map as an equirectangular panorama."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PanoRenderMeshDepth",
            display_name="Pano Render Mesh Depth",
            category="PanoPack",
            description=(
                "Render a TRIMESH as an equirectangular depth panorama "
                "from the world origin.\n\n"
                "Output is a single-channel depth map (distance from origin) "
                "in the standard PANORAMA format (2:1 equirect)."
            ),
            inputs=[
                io.Custom("TRIMESH").Input(
                    "mesh",
                    tooltip="The mesh to render depth from."),
                io.Int.Input(
                    "width", default=2048, min=256, max=8192, step=64,
                    tooltip="Output panorama width. Height = width / 2."),
                io.Int.Input(
                    "face_resolution", default=1024, min=256, max=4096, step=64,
                    tooltip="Resolution of each cube face render."),
            ],
            outputs=[
                io.Custom(PANORAMA_TYPE).Output(display_name="depth_panorama"),
                io.Image.Output(
                    display_name="depth_image",
                    tooltip="Depth map normalized to [0, 1] for visualization. "
                            "The raw depth values are in the panorama output."),
                io.Float.Output(display_name="max_depth"),
            ],
        )

    @classmethod
    def execute(cls, mesh, width=2048, face_resolution=1024):
        import time
        t0 = time.perf_counter()

        erp_w = int(width)
        erp_h = erp_w // 2
        face_size = int(face_resolution)

        _p(f"rendering 6 cube faces @ {face_size}px (depth)")
        faces = _render_cube_faces_depth(mesh, face_size)

        _p(f"stitching to {erp_w}x{erp_h} equirect")
        erp = cube_faces_to_equirect(faces, erp_w, erp_h)

        # Extract single-channel depth
        depth = erp[..., 0]  # (H, W)
        max_depth = float(depth.max()) if depth.max() > 0 else 1.0

        # Normalized visualization (0 = near/black, 1 = far/white)
        depth_vis = depth / max(max_depth, 1e-6)
        depth_vis = np.clip(depth_vis, 0, 1).astype(np.float32)
        depth_vis_3ch = np.stack([depth_vis] * 3, axis=-1)

        _p(f"depth range: [0, {max_depth:.3f}], done in {time.perf_counter() - t0:.2f}s")

        # Panorama output: raw depth as 3-channel (for PANORAMA type compatibility)
        depth_pano_3ch = np.stack([depth, depth, depth], axis=-1).astype(np.float32)
        depth_pano_t = torch.from_numpy(depth_pano_3ch).unsqueeze(0)
        pano = wrap_image_as_panorama(depth_pano_t)

        depth_vis_t = torch.from_numpy(depth_vis_3ch).unsqueeze(0)
        return io.NodeOutput(pano, depth_vis_t, max_depth)


NODE_CLASS_MAPPINGS = {
    "PanoRenderMesh": PanoRenderMesh,
    "PanoRenderMeshDepth": PanoRenderMeshDepth,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PanoRenderMesh": "Pano Render Mesh",
    "PanoRenderMeshDepth": "Pano Render Mesh Depth",
}
