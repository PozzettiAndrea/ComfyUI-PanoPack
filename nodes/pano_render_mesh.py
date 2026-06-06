"""PanoRenderMesh — render a trimesh/pointcloud as an ERP panorama (color).
PanoRenderMeshDepth — same but outputs an ERP depth map.

Both use PyVista offscreen rendering of 6 cube faces stitched to equirect.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from comfy_api.latest import io

from .utils import PANORAMA_TYPE, wrap_image_as_panorama
from .utils.cube_to_equirect import FACE_FOV_DEG, cube_faces_to_equirect


def _p(msg: str) -> None:
    print(f"[PanoRenderMesh] {msg}", file=sys.stderr, flush=True)


def _log_face_stats(faces, label="faces") -> None:
    """Log per-face coverage/intensity so a black or empty render is obvious."""
    names = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
    for i, f in enumerate(faces):
        flat = f.reshape(-1, f.shape[-1])
        nz = int((flat.max(-1) > 1e-4).sum())
        _p(f"  {label} face {names[i]:>2} {f.shape}: nonblack={nz}/{flat.shape[0]} "
           f"({100.0 * nz / max(flat.shape[0], 1):.1f}%) "
           f"min={float(f.min()):.3f} mean={float(f.mean()):.3f} max={float(f.max()):.3f}")


def _log_mesh_info(mesh, label="mesh") -> None:
    """Log vertex/face counts, colors, and bbox of the incoming geometry."""
    verts = np.asarray(getattr(mesh, "vertices", []), dtype=np.float32)
    faces_attr = getattr(mesh, "faces", None)
    n_faces = len(faces_attr) if faces_attr is not None else 0
    has_vc = hasattr(mesh, "visual") and hasattr(getattr(mesh, "visual", None), "vertex_colors")
    has_c = getattr(mesh, "colors", None) is not None
    _p(f"  {label}: type={type(mesh).__name__} verts={len(verts)} faces={n_faces} "
       f"vertex_colors={has_vc} colors={has_c}")
    if len(verts):
        lo, hi = verts.min(0), verts.max(0)
        dist = np.linalg.norm(verts, axis=1)
        _p(f"  {label}: bbox min={lo.round(3).tolist()} max={hi.round(3).tolist()}")
        _p(f"  {label}: |v| from origin min={dist.min():.3f} med={np.median(dist):.3f} "
           f"max={dist.max():.3f}")


# 6 cube face cameras from origin, 90° FOV each.
# (camera_position, focal_point, view_up)
_CUBE_CAMERAS = [
    ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
    ((0, 0, 0), (-1, 0, 0), (0, 1, 0)),
    ((0, 0, 0), (0, 1, 0), (0, 0, -1)),
    ((0, 0, 0), (0, -1, 0), (0, 0, 1)),
    ((0, 0, 0), (0, 0, 1), (0, 1, 0)),
    ((0, 0, 0), (0, 0, -1), (0, 1, 0)),
]


def _apply_yz_flip(mesh):
    """Swap Y and Z axes on a trimesh object."""
    import trimesh
    verts = np.asarray(mesh.vertices, dtype=np.float32).copy()
    verts[:, [1, 2]] = verts[:, [2, 1]]

    if hasattr(mesh, "faces") and len(getattr(mesh, "faces", [])) > 0:
        new_mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(mesh.faces), process=False)
        if hasattr(mesh, "visual"):
            new_mesh.visual = mesh.visual
    else:
        new_mesh = trimesh.PointCloud(vertices=verts)
        if hasattr(mesh, "colors") and mesh.colors is not None:
            new_mesh.colors = mesh.colors
        elif hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors"):
            new_mesh.colors = mesh.visual.vertex_colors

    return new_mesh


def _trimesh_to_pyvista(mesh):
    """Convert a trimesh.Trimesh or trimesh.PointCloud to pv.PolyData."""
    import pyvista as pv

    verts = np.asarray(mesh.vertices, dtype=np.float32)

    if hasattr(mesh, "faces") and len(mesh.faces) > 0:
        faces = np.asarray(mesh.faces, dtype=np.int64)
        F = faces.shape[0]
        faces_flat = np.concatenate(
            [np.full((F, 1), 3, dtype=np.int64), faces], axis=1,
        ).flatten()
        poly = pv.PolyData(verts, faces_flat)
    else:
        poly = pv.PolyData(verts)

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


def _render_cube_faces_color(mesh, face_size, mode_params):
    """Render 6 cube faces as color images."""
    import os
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import pyvista as pv

    poly = _trimesh_to_pyvista(mesh)
    has_colors = "RGB" in poly.array_names
    mode = mode_params.get("mode", "mesh")
    _p(f"  pyvista poly: n_points={poly.n_points} n_cells={poly.n_cells} "
       f"has_RGB={has_colors} mode={mode}")

    # Screen-space point size is resolution-dependent: a 2px dot on a 1024px face
    # is nearly invisible. Scale the user's point_size by the face resolution
    # (relative to a 512px reference) so points stay visibly thick as you go up.
    base_pt = float(mode_params.get("point_size", 2.0))
    pt_size = max(base_pt * (face_size / 512.0), base_pt)
    if mode == "pointcloud":
        _p(f"  pointcloud: base_point_size={base_pt} -> effective={pt_size:.1f}px @ {face_size}px face")

    faces = []
    for idx, (cam_pos, focal, view_up) in enumerate(_CUBE_CAMERAS):
        plotter = pv.Plotter(off_screen=True, window_size=(face_size, face_size))
        plotter.set_background((0, 0, 0))

        if mode == "pointcloud":
            if has_colors:
                plotter.add_points(poly, scalars="RGB", rgb=True,
                                   point_size=pt_size,
                                   render_points_as_spheres=True)
            else:
                plotter.add_points(poly, color=(0.7, 0.8, 0.95),
                                   point_size=pt_size,
                                   render_points_as_spheres=True)
        else:
            kwargs = dict(lighting=True, ambient=0.3, diffuse=0.7)
            show_edges = str(mode_params.get("show_edges", "off")).lower() == "on"
            if show_edges:
                kwargs["show_edges"] = True
                kwargs["edge_color"] = (0.18, 0.28, 0.45)
                kwargs["line_width"] = float(mode_params.get("edge_thickness", 1.0))
            if has_colors:
                plotter.add_mesh(poly, scalars="RGB", rgb=True, **kwargs)
            else:
                plotter.add_mesh(poly, color=(0.7, 0.8, 0.95), **kwargs)

        plotter.camera.position = cam_pos
        plotter.camera.focal_point = focal
        plotter.camera.up = view_up
        plotter.camera.view_angle = FACE_FOV_DEG
        plotter.camera.clipping_range = (0.001, 1000.0)

        img = plotter.screenshot(return_img=True)
        plotter.close()
        face = img.astype(np.float32) / 255.0
        nz = int((face.reshape(-1, face.shape[-1]).max(-1) > 1e-4).sum())
        _p(f"  color face {idx} focal={focal}: nonblack={nz} max={float(face.max()):.3f}")
        faces.append(face)

    return faces


def _render_cube_faces_depth(mesh, face_size):
    """Render 6 cube faces as depth images."""
    import os
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import pyvista as pv

    poly = _trimesh_to_pyvista(mesh)
    _p(f"  depth pyvista poly: n_points={poly.n_points} n_cells={poly.n_cells}")

    # Per-pixel factor to turn camera-axis (perpendicular) depth into radial
    # distance from the origin: radial = z_perp * sqrt(1 + fx^2 + fy^2), where
    # fx,fy are tangent-space offsets (pixel - center) / focal. The cameras all
    # sit at the origin, so this yields true distance from the world origin.
    f_px = (face_size / 2.0) / np.tan(np.radians(FACE_FOV_DEG) / 2.0)
    ax = (np.arange(face_size, dtype=np.float32) + 0.5 - face_size / 2.0) / f_px
    fxx, fyy = np.meshgrid(ax, ax)
    radial_factor = np.sqrt(1.0 + fxx * fxx + fyy * fyy).astype(np.float32)

    faces = []
    for idx, (cam_pos, focal, view_up) in enumerate(_CUBE_CAMERAS):
        plotter = pv.Plotter(off_screen=True, window_size=(face_size, face_size))
        plotter.set_background((0, 0, 0))
        plotter.add_mesh(poly, color="white", lighting=False)

        plotter.camera.position = cam_pos
        plotter.camera.focal_point = focal
        plotter.camera.up = view_up
        plotter.camera.view_angle = FACE_FOV_DEG
        plotter.camera.clipping_range = (0.001, 1000.0)

        # off_screen render must go through show() to be marked rendered; a bare
        # render() leaves get_image_depth() raising "has not yet been rendered".
        plotter.show(auto_close=False)
        # get_image_depth returns negative camera-space depth (scene is in front),
        # NaN for background. Flip to positive distance and zero the background.
        zbuf = plotter.get_image_depth(fill_value=np.nan)
        plotter.close()

        zbuf = np.abs(np.asarray(zbuf, dtype=np.float32))
        zbuf[~np.isfinite(zbuf)] = 0.0
        # perpendicular depth -> radial distance from origin (background stays 0)
        zbuf = zbuf * radial_factor
        hit = int((zbuf > 0).sum())
        _p(f"  depth face {idx} focal={focal}: hit_px={hit} "
           f"range=[{float(zbuf.min()):.3f},{float(zbuf.max()):.3f}] (radial)")
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
                "Mode 'mesh': renders with faces, optional wireframe edges.\n"
                "Mode 'pointcloud': renders as point splats.\n\n"
                "Uses PyVista offscreen rendering of 6 cube faces "
                "stitched into a 2:1 equirect."
            ),
            inputs=[
                io.Custom("TRIMESH").Input(
                    "mesh",
                    tooltip="The mesh or point cloud to render."),
                io.DynamicCombo.Input("mode",
                    tooltip="Render mode.",
                    options=[
                        io.DynamicCombo.Option("mesh", [
                            io.DynamicCombo.Input("show_edges",
                                tooltip="Show triangle wireframe edges. "
                                        "Set 'on' to reveal edge thickness.",
                                options=[
                                    io.DynamicCombo.Option("off", []),
                                    io.DynamicCombo.Option("on", [
                                        io.Float.Input("edge_thickness",
                                            default=1.0, min=0.1, max=10.0, step=0.1,
                                            tooltip="Wireframe edge line width."),
                                    ]),
                                ]),
                        ]),
                        io.DynamicCombo.Option("pointcloud", [
                            io.Float.Input("point_size",
                                default=2.0, min=0.5, max=20.0, step=0.5,
                                tooltip="Screen-space point size in pixels."),
                            io.Float.Input("render_radius",
                                default=0.008, min=0.001, max=0.1, step=0.001,
                                display_mode="number",
                                tooltip="World-space splat radius "
                                        "(matches WorldStereo convention)."),
                        ]),
                    ]),
                io.Boolean.Input(
                    "yz_flip", default=True,
                    tooltip="Swap Y and Z axes. Enable when the input "
                            "uses Z-up convention (common in 3D scanning "
                            "and reconstruction)."),
                io.Int.Input(
                    "width", default=2048, min=256, max=8192, step=64,
                    tooltip="Output panorama width. Height = width / 2."),
                io.Int.Input(
                    "face_resolution", default=1024, min=256, max=4096, step=64,
                    tooltip="Resolution of each cube face render."),
            ],
            outputs=[
                io.Custom(PANORAMA_TYPE).Output(display_name="panorama"),
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, mesh, mode, yz_flip=True, width=2048, face_resolution=1024):
        import time
        t0 = time.perf_counter()

        _p(f"=== PanoRenderMesh: yz_flip={yz_flip} width={width} face_res={face_resolution} ===")
        _log_mesh_info(mesh, "input")
        if yz_flip:
            mesh = _apply_yz_flip(mesh)
            _log_mesh_info(mesh, "yz_flipped")

        erp_w = int(width)
        erp_h = erp_w // 2
        face_size = int(face_resolution)
        selected_mode = mode.get("mode", "mesh")

        _p(f"rendering 6 cube faces @ {face_size}px (color, mode={selected_mode})")
        faces = _render_cube_faces_color(mesh, face_size, mode_params=mode)
        _log_face_stats(faces, "color")

        _p(f"stitching to {erp_w}x{erp_h} equirect (FOV={FACE_FOV_DEG})")
        erp = cube_faces_to_equirect(faces, erp_w, erp_h)
        erp = np.clip(erp, 0, 1).astype(np.float32)
        nz = int((erp.reshape(-1, 3).max(-1) > 1e-4).sum())
        _p(f"  ERP: nonblack={nz}/{erp_w * erp_h} ({100.0 * nz / (erp_w * erp_h):.1f}%) "
           f"min={float(erp.min()):.3f} mean={float(erp.mean()):.3f} max={float(erp.max()):.3f}")

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
                io.Boolean.Input(
                    "yz_flip", default=True,
                    tooltip="Swap Y and Z axes. Enable when the input "
                            "uses Z-up convention."),
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
    def execute(cls, mesh, yz_flip=True, width=2048, face_resolution=1024):
        import time
        t0 = time.perf_counter()

        _p(f"=== PanoRenderMeshDepth: yz_flip={yz_flip} width={width} face_res={face_resolution} ===")
        _log_mesh_info(mesh, "input")
        if yz_flip:
            mesh = _apply_yz_flip(mesh)
            _log_mesh_info(mesh, "yz_flipped")

        erp_w = int(width)
        erp_h = erp_w // 2
        face_size = int(face_resolution)

        _p(f"rendering 6 cube faces @ {face_size}px (depth)")
        faces = _render_cube_faces_depth(mesh, face_size)
        _log_face_stats(faces, "depth")

        _p(f"stitching to {erp_w}x{erp_h} equirect (FOV={FACE_FOV_DEG})")
        erp = cube_faces_to_equirect(faces, erp_w, erp_h)

        depth = erp[..., 0]
        max_depth = float(depth.max()) if depth.max() > 0 else 1.0

        depth_vis = depth / max(max_depth, 1e-6)
        depth_vis = np.clip(depth_vis, 0, 1).astype(np.float32)
        depth_vis_3ch = np.stack([depth_vis] * 3, axis=-1)

        _p(f"depth range: [0, {max_depth:.3f}], done in {time.perf_counter() - t0:.2f}s")

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
