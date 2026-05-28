"""PanoramaCubeSplit - equirect panorama -> N perspective crops.

Mirrors upstream `traj_generate.py:464-501` which generates the `pano_bank/`
seed for PanoramaMemoryBank: 3 pitch levels (horizontal, +0.5z, -0.5z)
× 9 yaw steps (40 deg) at 120 deg horizontal × 90 deg vertical FOV, all
from world origin with +Z up.

Different from `ComfyUI-WorldNav.WorldNavPanoramaSplit`:
  - WorldNav splits an icosahedron at 90 deg FOV / 512x512 square --
    dense, square, for feeding rectilinear depth models (MoGe2) per face.
  - This node splits at fixed pitch/yaw grid at 120x90 FOV / 832x480 --
    sparser, anchor-aspect, for seeding the WorldStereo memory bank with
    the same crops the model was trained against.

Outputs:
  face_images          (IMAGE [N, H, W, 3])  -- wire to MoGe2Inference
                                                 (or any depth model)
  fov_x_deg            (FLOAT)                -- pass into MoGe2's fov_x widget
                                                 so it doesn't have to estimate
  entries              (MEMORY_BANK_ENTRIES)  -- has frames + ext + intr +
                                                 fnames + frame_idx + image_size.
                                                 depths=None by default; feed
                                                 straight into MemoryBankAdd
                                                 (HYWM2 / WorldMirror compute
                                                 their own depth at inference,
                                                 so an in-bank depth is
                                                 optional). To attach depth
                                                 anyway (e.g. for a downstream
                                                 prior-depth path), pipe
                                                 through WorldStereoMemoryBank-
                                                 EntriesAddDepth first.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from comfy_api.latest import io

from .utils import PANORAMA_TYPE, unwrap_panorama_to_image


def _p(msg: str) -> None:
    print(f"[PanoramaCubeSplit] {msg}", file=sys.stderr, flush=True)


class PanoramaCubeSplit(io.ComfyNode):
    """Equirect panorama -> N perspective crops at fixed pitch/yaw grid."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PanoramaCubeSplit",
            display_name="Panorama Cube Split",
            category="PanoPack",
            description=(
                "Sample N perspective crops from an equirect panorama at a fixed "
                "pitch/yaw grid (3 pitch x 9 yaw = 27 views by default). Same "
                "geometry as upstream traj_generate.py:464-501's pano_bank.\n\n"
                "Outputs:\n"
                "  face_images (IMAGE [N,H,W,3])  - wire to MoGe2Inference\n"
                "  fov_x_deg   (FLOAT)            - pass to MoGe2's fov_x widget\n"
                "  entries (MEMORY_BANK_ENTRIES, depths=None) -\n"
                "    feed straight into MemoryBankAdd (HYWM2 / WorldMirror\n"
                "    compute depth from frames+cameras at inference, so an\n"
                "    in-bank depth is optional). To attach depth anyway, pipe\n"
                "    through WorldStereoMemoryBankEntriesAddDepth first."
            ),
            inputs=[
                io.Custom(PANORAMA_TYPE).Input(
                    "panorama",
                    tooltip="Equirectangular RGB panorama (2:1). PANORAMA = "
                            "IMAGE wrapped in PanoPack's typed socket. Wire "
                            "from PanoramaWrap or any node that emits a "
                            "PANORAMA."),
                io.Int.Input(
                    "image_w", default=832, min=224, max=2048, step=16,
                    tooltip="Crop width. 832 matches WorldStereo's anchor width."),
                io.Int.Input(
                    "image_h", default=480, min=224, max=2048, step=16,
                    tooltip="Crop height. 480 matches WorldStereo's anchor height."),
                io.Float.Input(
                    "fov_x_deg", default=120.0, min=30.0, max=170.0, step=1.0,
                    tooltip="Horizontal FOV. 120 matches upstream."),
                io.Float.Input(
                    "fov_y_deg", default=90.0, min=30.0, max=170.0, step=1.0,
                    tooltip="Vertical FOV. 90 matches upstream."),
                io.Float.Input(
                    "rot_deg", default=40.0, min=10.0, max=120.0, step=5.0,
                    tooltip="Yaw step between adjacent views. 40 -> 9 views/pitch -> 27 total."),
                io.Float.Input(
                    "pitch_up", default=0.5, min=0.0, max=1.0, step=0.05,
                    tooltip="Z component of the upper-hemisphere look-at points."),
                io.Float.Input(
                    "pitch_down", default=-0.5, min=-1.0, max=0.0, step=0.05,
                    tooltip="Z component of the lower-hemisphere look-at points."),
                io.String.Input(
                    "fname", default="pano_bank", multiline=False,
                    tooltip="Provenance tag stored per entry (matches upstream's pano_bank label)."),
            ],
            outputs=[
                io.Image.Output(
                    display_name="face_images",
                    tooltip="N perspective crops [N, H, W, 3] in [0, 1]. Wire to "
                            "MoGe2Inference (or any depth model)."),
                io.Float.Output(
                    display_name="fov_x_deg",
                    tooltip="Horizontal FOV in degrees (passes through). Wire into "
                            "MoGe2Inference's fov_x_deg input so it doesn't have to estimate."),
                io.Custom("MEMORY_BANK_ENTRIES").Output(
                    display_name="entries",
                    tooltip="MEMORY_BANK_ENTRIES with frames + ext + intr + "
                            "fnames + frame_idx + image_size. depths=None by "
                            "default -- feed straight into MemoryBankAdd. "
                            "Optional: pipe through "
                            "WorldStereoMemoryBankEntriesAddDepth first to "
                            "attach depth (HYWM2 / WorldMirror compute their "
                            "own depth at inference, so this is rarely "
                            "needed)."),
                io.Custom("EXTRINSICS").Output(
                    display_name="extrinsics",
                    tooltip="Per-crop world-to-camera matrices [N, 4, 4] "
                            "(float32). Same N and ordering as face_images "
                            "(3 pitches × N_view yaws). All cameras share "
                            "the world origin (look-at from [0,0,0])."),
                io.Custom("INTRINSICS").Output(
                    display_name="intrinsics",
                    tooltip="Per-crop pixel-scale intrinsics [N, 3, 3] "
                            "(float32). cx = image_w/2, cy = image_h/2; fx, "
                            "fy derived from fov_x_deg / fov_y_deg. Matches "
                            "upstream pano_bank/cameras.json convention."),
            ],
        )

    @classmethod
    def execute(
        cls, panorama,
        image_w: int, image_h: int,
        fov_x_deg: float, fov_y_deg: float,
        rot_deg: float, pitch_up: float, pitch_down: float,
        fname: str,
    ):
        import cv2
        import utils3d
        from ._vendor.worldstereo_cube import split_panorama_image, rotate_around_z_axis

        # Panorama -> uint8 [H_pano, W_pano, 3]
        pano_t = unwrap_panorama_to_image(panorama)
        pano = pano_t.detach().cpu().numpy() if isinstance(pano_t, torch.Tensor) else np.asarray(pano_t)
        if pano.ndim == 4:
            pano = pano[0]
        if pano.dtype != np.uint8:
            pano = (np.clip(pano, 0, 1) * 255 + 0.5).astype(np.uint8)
        if pano.shape[-1] == 4:
            pano = pano[..., :3]

        # ----- Direction grid (verbatim from upstream traj_generate.py:464-475) -----
        start_points = [
            np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            np.array([-1.0, 0.0, float(pitch_up)], dtype=np.float32),
            np.array([-1.0, 0.0, float(pitch_down)], dtype=np.float32),
        ]
        direct_points = list(start_points)
        N_view = int(360 / max(1, int(rot_deg)))
        for start_point in start_points:
            for i in range(1, N_view):
                direct_points.append(
                    rotate_around_z_axis(start_point.reshape(1, 3), rot_deg * i)[0]
                )
        direct_points = np.stack(direct_points, axis=0).astype(np.float32)
        N_crops = direct_points.shape[0]

        # ----- Intrinsics + extrinsics (look-at from origin) -----
        K_norm = utils3d.numpy.intrinsics_from_fov(
            fov_x=np.deg2rad(fov_x_deg), fov_y=np.deg2rad(fov_y_deg),
        ).astype(np.float32)
        K_norm_list = [K_norm] * N_crops
        exts = utils3d.numpy.extrinsics_look_at(
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            direct_points,
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        ).astype(np.float32)  # [N, 4, 4]

        # ----- Sample perspective crops from the panorama -----
        crops = split_panorama_image(
            pano, exts, K_norm_list, h=int(image_h), w=int(image_w),
            interp=cv2.INTER_AREA,
        )  # list of [H, W, 3] uint8
        frames = np.stack(crops, axis=0).astype(np.float32) / 255.0  # [N, H, W, 3]

        # ----- Pixel-scale intrinsics (matches upstream pano_bank/cameras.json) -----
        K_pixel = K_norm.copy()
        K_pixel[0] *= float(image_w)
        K_pixel[1] *= float(image_h)
        Ks = np.broadcast_to(K_pixel, (N_crops, 3, 3)).astype(np.float32)

        frames_t = torch.from_numpy(frames).contiguous()
        exts_t = torch.from_numpy(exts).contiguous()
        Ks_t = torch.from_numpy(Ks.copy()).contiguous()
        entries = {
            "frames":     frames_t,
            "extrinsics": exts_t,
            "intrinsics": Ks_t,
            "depths":     None,    # filled later via WorldStereoMemoryBankEntriesAddDepth
            "fnames":     [str(fname or "pano_bank")] * N_crops,
            "frame_idx":  list(range(N_crops)),
            "image_size": [int(image_w), int(image_h)],
        }
        _p(
            f"split: {N_crops} crops at {image_w}x{image_h}, fov=({fov_x_deg:.0f},{fov_y_deg:.0f}), "
            f"rot={rot_deg:.0f}deg -> {N_view}/pitch x 3 pitches, fname={fname!r} "
            f"(depths=None; wire face_images through MoGe2Inference + AddDepth)"
        )
        # face_images output is the same tensor as entries["frames"] — exposed
        # separately so downstream depth nodes (MoGe2Inference) can wire in
        # without the entries dict in scope.
        return io.NodeOutput(frames_t, float(fov_x_deg), entries, exts_t, Ks_t)


NODE_CLASS_MAPPINGS = {"PanoramaCubeSplit": PanoramaCubeSplit}
NODE_DISPLAY_NAME_MAPPINGS = {"PanoramaCubeSplit": "Panorama Cube Split"}
