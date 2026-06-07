"""Vendored batched-RANSAC plane fit from torch_ransac3d.

Adapted from harrydobbs/torch_ransac3d (MIT License, Copyright (c) 2024 Harry):
https://github.com/harrydobbs/torch_ransac3d

Why vendored instead of pip-installed: the upstream `torch-ransac3d==2.0.0`
py3-none-any wheel ships an archive member literally named `tests/*`, which is an
illegal filename on Windows (`os error 123` / ERROR_INVALID_NAME). uv/pixi cannot
extract it, so the whole isolated-env install aborts on Windows. The fitting code
itself is pure torch (no compiled CUDA extension -- the wheel is `none-any`), so we
copy the one function we use and avoid the broken wheel entirely. Works on Windows,
Linux and macOS, on GPU or CPU, with no external dependency.

Differences from upstream `plane.py`:
  * The `@torch.compile` decorator is dropped. We call this repeatedly with a
    shrinking point set (one call per peeled plane), so the input shape changes
    every call -- torch.compile would recompile each time (a net slowdown) and
    Triton is fragile on Windows. The numerics are identical without it.
  * The `numpy_to_torch` wrapper is dropped; the caller always passes a torch
    tensor already on the target device.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlaneFitResult:
    """Plane fit result: equation [a, b, c, d] for ax + by + cz + d = 0, plus inlier indices."""

    equation: "object"   # torch.Tensor (4,)
    inliers: "object"    # torch.Tensor (K,) long, indices into the input points


def plane_fit(
    pts,
    thresh: float = 0.05,
    max_iterations: int = 1000,
    iterations_per_batch: int = 1,
    epsilon: float = 1e-8,
    device=None,
) -> PlaneFitResult:
    """Find the best plane for a 3D point cloud via batched RANSAC.

    Processes `iterations_per_batch` RANSAC hypotheses in parallel for efficiency.

    :param pts: (N, 3) torch.Tensor point cloud.
    :param thresh: max point-to-plane distance to count a point as an inlier.
    :param max_iterations: total RANSAC hypotheses to try.
    :param iterations_per_batch: hypotheses evaluated in parallel per batch (VRAM knob;
        the (batch, N) distance tensor is the memory hog on large clouds).
    :param epsilon: small value to avoid division by zero when normalizing the normal.
    :param device: torch device to run on. Defaults to the input tensor's device.
    :return: PlaneFitResult(equation=(4,) tensor, inliers=(K,) long index tensor).
    """
    import torch

    if device is None:
        device = pts.device
    pts = pts.to(device).to(torch.float32)
    num_pts = pts.shape[0]

    best_inlier_indices = torch.tensor([], dtype=torch.long, device=device)
    best_inlier_count = 0
    best_eq = None

    with torch.no_grad():
        for start_idx in range(0, max_iterations, iterations_per_batch):
            end_idx = min(start_idx + iterations_per_batch, max_iterations)
            current_batch_size = end_idx - start_idx

            # Sample 3 random points for each hypothesis in the batch.
            rand_pt_idx = torch.randint(0, num_pts, (current_batch_size, 3), device=device)
            sampled_points = pts[rand_pt_idx]  # (batch, 3, 3)

            # Two in-plane edge vectors, then their cross product = plane normal.
            vec_A = sampled_points[:, 1, :] - sampled_points[:, 0, :]  # (batch, 3)
            vec_B = sampled_points[:, 2, :] - sampled_points[:, 0, :]  # (batch, 3)
            vec_C = torch.cross(vec_A, vec_B, dim=1)                   # (batch, 3)

            # Normalize the normal to a unit vector.
            vec_C = vec_C / (torch.norm(vec_C, dim=1, keepdim=True) + epsilon)

            # Constant term D for each plane: D = -(normal . point_on_plane).
            k = -torch.einsum("ij,ij->i", vec_C, sampled_points[:, 1, :])  # (batch,)

            # Plane coefficients [A, B, C, D].
            plane_eq = torch.cat([vec_C, k.unsqueeze(1)], dim=1)  # (batch, 4)

            # Signed point-to-plane distance of every point to every candidate plane.
            dist_pts = (
                plane_eq[:, 0:1] * pts[:, 0].unsqueeze(0)
                + plane_eq[:, 1:2] * pts[:, 1].unsqueeze(0)
                + plane_eq[:, 2:3] * pts[:, 2].unsqueeze(0)
                + plane_eq[:, 3:4]
            ) / torch.sqrt(
                plane_eq[:, 0] ** 2 + plane_eq[:, 1] ** 2 + plane_eq[:, 2] ** 2
            ).unsqueeze(1)

            # Inliers: |distance| <= threshold.
            inlier_mask = torch.abs(dist_pts) <= thresh   # (batch, num_pts)
            inlier_counts = inlier_mask.sum(dim=1)         # (batch,)

            # Best hypothesis in this batch.
            best_in_batch_idx = torch.argmax(inlier_counts)
            best_inlier_count_in_batch = inlier_counts[best_in_batch_idx].item()

            if best_inlier_count_in_batch > best_inlier_count:
                best_inlier_count = best_inlier_count_in_batch
                best_inlier_indices = torch.where(inlier_mask[best_in_batch_idx])[0]
                best_eq = plane_eq[best_in_batch_idx]

    return PlaneFitResult(equation=best_eq, inliers=best_inlier_indices)
