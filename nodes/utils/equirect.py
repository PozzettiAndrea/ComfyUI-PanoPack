"""Equirect / spherical helpers shared across PanoPack nodes.

The PANORAMA custom-socket payload is a plain `dict`:

    {
        "image": torch.Tensor of shape (B, H, W, C) with W == 2 * H,
        "meta":  {} (free-form, reserved for future per-pano metadata).
    }

We use a dict rather than a NamedTuple so the wire shape stays JSON-ish
and forward-compatible with extra fields. Downstream nodes should call
`unwrap_panorama_to_image()` to get the bare tensor.
"""

from __future__ import annotations

from typing import Any

import torch

# String tag used at ComfyUI socket-type level. Any node that wants to
# emit / accept a panorama should call `io.Custom(PANORAMA_TYPE).Input/Output(...)`.
PANORAMA_TYPE = "PANORAMA"


def is_two_to_one(image: torch.Tensor, tolerance: float = 0.01) -> bool:
    """Return True if the image tensor is a valid 2:1 equirect aspect.

    Accepts shapes [H, W, C], [B, H, W, C], or [H, W]. The aspect check
    compares W / H to 2.0 within `tolerance` (default 1%).
    """
    if image.dim() == 4:
        _, h, w, _ = image.shape
    elif image.dim() == 3:
        h, w, _ = image.shape
    elif image.dim() == 2:
        h, w = image.shape
    else:
        raise ValueError(f"unexpected image shape {tuple(image.shape)}")
    return abs((w / max(h, 1)) - 2.0) <= tolerance


def normalize_pano_tensor(image: torch.Tensor) -> torch.Tensor:
    """Coerce an input tensor to (B, H, W, C) float32 layout.

    - Adds a batch dim if input is (H, W, C) or (H, W).
    - Casts to float32.
    - Replicates a single-channel input to 3 channels (so downstream
      consumers can treat depth panoramas as IMAGE).
    """
    img = image
    if img.dim() == 2:
        img = img.unsqueeze(-1)
    if img.dim() == 3:
        img = img.unsqueeze(0)
    if img.dim() != 4:
        raise ValueError(f"unexpected image shape {tuple(image.shape)}")
    if img.dtype != torch.float32:
        img = img.float()
    if img.shape[-1] == 1:
        img = img.repeat(1, 1, 1, 3)
    return img.contiguous()


def wrap_image_as_panorama(
    image: torch.Tensor, meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """IMAGE → PANORAMA. Validates 2:1 aspect; raises otherwise."""
    img = normalize_pano_tensor(image)
    if not is_two_to_one(img):
        _, h, w, _ = img.shape
        raise ValueError(
            f"PANORAMA expects 2:1 equirect (W == 2·H), got {w}x{h} "
            f"(aspect {w/max(h,1):.3f}). Crop / pad the input to 2:1 first."
        )
    return {"image": img, "meta": dict(meta) if meta else {}}


def unwrap_panorama_to_image(pano: dict[str, Any] | torch.Tensor) -> torch.Tensor:
    """PANORAMA → IMAGE. Accepts either the dict payload or a raw tensor
    (passthrough convenience for upstream nodes that emit IMAGE directly).
    """
    if isinstance(pano, dict):
        img = pano.get("image")
        if not isinstance(img, torch.Tensor):
            raise ValueError(
                "PANORAMA payload missing 'image' tensor; got keys "
                f"{list(pano.keys())}"
            )
        return img
    if isinstance(pano, torch.Tensor):
        return pano
    raise TypeError(f"unsupported PANORAMA payload type {type(pano).__name__}")


def panorama_shift_horizontal(
    image: torch.Tensor, yaw_deg: float,
) -> torch.Tensor:
    """Rotate an equirect by `yaw_deg` degrees around the vertical axis.

    Implementation: integer-pixel `torch.roll` along the W axis, with the
    sub-pixel residual applied as a single bilinear remap (one
    `F.grid_sample` call) so the result is exact for integer-pixel shifts
    and smooth for fractional ones.

    Sign convention: positive yaw rotates the scene RIGHT (camera turns
    LEFT) — i.e. content at u_pixel=0 moves toward u_pixel=+shift_pixels.
    This matches the visual expectation when you say "rotate +90°".
    """
    import torch.nn.functional as F

    img = normalize_pano_tensor(image)
    _, h, w, _ = img.shape

    # Total shift in pixel units (positive = shift content right).
    shift_pixels = (float(yaw_deg) % 360.0) / 360.0 * w
    int_shift = int(round(shift_pixels))
    frac_shift = shift_pixels - int_shift

    # Integer-pixel wrap.
    if int_shift % w != 0:
        img = torch.roll(img, shifts=int_shift, dims=-2)

    if abs(frac_shift) < 1e-6:
        return img.contiguous()

    # Sub-pixel residual via grid_sample. Build a sampling grid that maps
    # output pixel (x, y) → input pixel (x − frac_shift, y), with x
    # wrapping around the seam via padding_mode='zeros' + an explicit
    # circular pad along W.
    # Wrap input along W by 1 pixel on each side so the bilinear sample
    # never reaches a zero-padded region for |frac_shift| < 1.
    img_wrap = torch.cat([img[..., -1:, :], img, img[..., :1, :]], dim=-2)
    # img_wrap shape: (B, H, W+2, C)

    device = img.device
    ys = torch.arange(h, dtype=torch.float32, device=device)
    xs = torch.arange(w, dtype=torch.float32, device=device)
    # In sampling-grid coords, +frac_shift content-right means we sample
    # from a position to the LEFT in the input → subtract.
    xs_sample = xs - frac_shift + 1.0  # +1 accounts for the leading wrap pixel
    # Normalize to [-1, +1] for grid_sample with align_corners=True over
    # the wrapped width of (W + 2) input pixels.
    norm_x = 2.0 * xs_sample / max(w + 2 - 1, 1) - 1.0
    norm_y = 2.0 * ys / max(h - 1, 1) - 1.0
    grid_x = norm_x.unsqueeze(0).expand(h, -1)
    grid_y = norm_y.unsqueeze(-1).expand(-1, w)
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    grid = grid.expand(img_wrap.shape[0], -1, -1, -1)

    img_nchw = img_wrap.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W+2)
    sampled = F.grid_sample(
        img_nchw, grid,
        mode="bilinear", padding_mode="border", align_corners=True,
    )
    out = sampled.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
    return out


def panorama_shift_pitch(
    image: torch.Tensor, pitch_deg: float,
) -> torch.Tensor:
    """Rotate an equirect by `pitch_deg` degrees around the horizontal axis.

    Implementation: spherical-coordinate remap via grid_sample.  Each output
    pixel (u, v) is mapped through spherical coordinates, the latitude is
    offset by the pitch angle, and we sample the input at the resulting
    position.

    Sign convention: positive pitch tilts the camera UP (scene moves down).
    """
    import torch.nn.functional as F
    import math

    if abs(pitch_deg) < 1e-6:
        return image

    img = normalize_pano_tensor(image)
    b, h, w, c = img.shape

    pitch_rad = math.radians(float(pitch_deg))

    device = img.device
    # Output pixel coordinates
    u = torch.linspace(0.5, w - 0.5, w, device=device)  # [0, W)
    v = torch.linspace(0.5, h - 0.5, h, device=device)  # [0, H)
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")  # (H, W)

    # Output pixel → spherical angles (longitude, latitude)
    lon = (grid_u / w) * 2.0 * math.pi - math.pi       # [-pi, pi]
    lat = (grid_v / h) * math.pi - (math.pi / 2.0)      # [-pi/2, pi/2] (top=-pi/2, bottom=pi/2)

    # Apply pitch offset to latitude
    lat_src = lat - pitch_rad

    # Convert (lon, lat_src) to 3D direction, then back to (lon_src, lat_src)
    # to handle poles correctly via the full spherical mapping.
    cos_lat = torch.cos(lat_src)
    x = cos_lat * torch.sin(lon)
    y = torch.sin(lat_src)
    z = cos_lat * torch.cos(lon)

    lon_src = torch.atan2(x, z)
    lat_src = torch.asin(y.clamp(-1.0, 1.0))

    # Spherical angles → input pixel coordinates, normalized to [-1, 1]
    u_src = (lon_src + math.pi) / (2.0 * math.pi)   # [0, 1]
    v_src = (lat_src + math.pi / 2.0) / math.pi      # [0, 1]
    norm_x = u_src * 2.0 - 1.0   # [-1, 1]
    norm_y = v_src * 2.0 - 1.0   # [-1, 1]

    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    grid = grid.expand(b, -1, -1, -1)

    img_nchw = img.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
    sampled = F.grid_sample(
        img_nchw, grid,
        mode="bilinear", padding_mode="border", align_corners=False,
    )
    return sampled.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)


def border_continuity_score(image: torch.Tensor) -> dict[str, float]:
    """Quantify how seamless an equirect is at its ±π wrap and ±π/2 poles.

    Returns a dict of per-axis scores in [0, 1] where 1.0 = perfectly
    continuous, 0 = fully discontinuous.

    - `seam`: similarity between leftmost and rightmost columns
      (E[1 − |L − R|] across all channels and rows). Penalises
      generative wraparound seams.
    - `north_pole_var`, `south_pole_var`: how uniform the top / bottom
      rows are. At a true pole, all pixels along a row image the SAME
      world point (a singularity); good panoramas have near-constant
      polar rows. Reported as 1 − clip(std, 0, 1).
    - `aspect_ratio`: the actual W/H of the input (informational).
    """
    img = normalize_pano_tensor(image)
    b, h, w, c = img.shape
    # Average across batch if more than 1; single-pano case is typical.
    arr = img.mean(dim=0)  # (H, W, C)

    left = arr[:, 0, :]
    right = arr[:, -1, :]
    seam_diff = (left - right).abs().mean().item()
    seam = float(max(0.0, 1.0 - seam_diff))

    north = arr[0, :, :]
    south = arr[-1, :, :]
    n_var = north.std(dim=0).mean().item()
    s_var = south.std(dim=0).mean().item()
    north_pole_var = float(max(0.0, 1.0 - min(n_var, 1.0)))
    south_pole_var = float(max(0.0, 1.0 - min(s_var, 1.0)))

    return {
        "seam": seam,
        "north_pole_var": north_pole_var,
        "south_pole_var": south_pole_var,
        "aspect_ratio": float(w / max(h, 1)),
    }
