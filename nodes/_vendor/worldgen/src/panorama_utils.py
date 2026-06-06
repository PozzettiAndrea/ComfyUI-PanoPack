from collections import defaultdict
from typing import Optional, Literal, List

# cupy stripped - see solve_lsmr_gpu below (now CPU via scipy).
import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import trimesh
import utils3d
from PIL import Image
from scipy.sparse import csr_matrix as cp_csr_matrix
from scipy.sparse.linalg import lsmr as cp_lsmr
# `convolve`, `vstack`, `grad_equation`, `poisson_equation` are lazy-imported
# inside merge_panorama_depth_gpu - that function is the cube-face stitcher
# used by upstream's pred_pano_depth, which WorldNav doesn't call (MoGe is
# externalized to a sibling ComfyUI node). The moge dep stays optional.
# `List` was imported from moge.utils.panorama but is actually re-exported
# from typing - sourcing directly from typing above.
from scipy.sparse import csr_array
from tqdm import tqdm


def subdivide_icosahedron(subdivisions: int = 1) -> np.ndarray:
    """
    Subdivide an icosahedron to generate denser spherical sample points.

    Args:
        subdivisions: Number of subdivisions.
            - 0: 12 vertices
            - 1: 42 vertices
            - 2: 162 vertices
            - 3: 642 vertices
            Formula: V = 10 * 4^n + 2

    Returns:
        vertices: (N, 3) Subdivided vertex coordinates on the unit sphere.
    """
    # utils3d renamed `icosahedron()` -> `create_icosahedron_mesh()`.
    vertices, faces = utils3d.numpy.create_icosahedron_mesh()

    # Convert to a list so new vertices can be appended dynamically.
    vertices_list = [v for v in vertices]

    for _ in range(subdivisions):
        edge_midpoint_cache = {}  # Cache edge midpoint indices to avoid duplicates.
        new_faces = []

        def get_or_create_midpoint(idx1: int, idx2: int) -> int:
            """
            Get the midpoint index for the edge between two vertices.
            Create it if missing, otherwise return the cached index.
            """
            # Use the sorted tuple as the key so (a, b) and (b, a) are the same edge.
            edge_key = (min(idx1, idx2), max(idx1, idx2))

            if edge_key in edge_midpoint_cache:
                return edge_midpoint_cache[edge_key]

            # Create a new midpoint.
            v1 = vertices_list[idx1]
            v2 = vertices_list[idx2]
            midpoint = (v1 + v2) / 2.0

            # Project onto the unit sphere.
            midpoint = midpoint / np.linalg.norm(midpoint)

            # Add it to the vertex list.
            new_idx = len(vertices_list)
            vertices_list.append(midpoint)
            edge_midpoint_cache[edge_key] = new_idx

            return new_idx

        # Subdivide each triangle.
        for face in faces:
            v0, v1, v2 = face

            # Get the midpoints of the three edges.
            #       v0
            #      /  \
            #     a----c
            #    / \  / \
            #   v1--b----v2
            a = get_or_create_midpoint(v0, v1)
            b = get_or_create_midpoint(v1, v2)
            c = get_or_create_midpoint(v2, v0)

            # Split the original triangle into four smaller triangles.
            new_faces.append([v0, a, c])
            new_faces.append([a, v1, b])
            new_faces.append([c, b, v2])
            new_faces.append([a, b, c])

        faces = np.array(new_faces, dtype=np.int32)

    return np.array(vertices_list, dtype=np.float32)


def get_panorama_cameras_v2(subdivisions=0):
    vertices = subdivide_icosahedron(subdivisions=subdivisions)
    intrinsics = utils3d.numpy.intrinsics_from_fov(fov_x=np.deg2rad(90), fov_y=np.deg2rad(90))
    extrinsics = utils3d.numpy.extrinsics_look_at([0, 0, 0], vertices, [0, 0, 1]).astype(np.float32)
    return extrinsics, [intrinsics] * len(vertices)


def rotate_around_z_axis(points, angle_deg):
    """
    Rotate 3D points clockwise around the Z axis.

    Args:
        points: 3D point array with shape (N, 3), where each row is (x, y, z).
        angle_deg: Rotation angle in degrees. Positive values rotate clockwise.

    Returns:
        rotated_points: Rotated 3D point array with shape (N, 3).
    """
    # Convert the angle to radians.
    angle_rad = np.radians(angle_deg)

    # Build the clockwise rotation matrix around the Z axis.
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    # In a right-handed coordinate system, clockwise rotation equals negative counterclockwise rotation.
    rotation_matrix = np.array([
        [cos_theta, sin_theta, 0],  # X component
        [-sin_theta, cos_theta, 0],  # Y component
        [0, 0, 1]  # Z component, unchanged
    ])

    # Apply the rotation to each point by matrix multiplication.
    rotated_points = np.dot(points, rotation_matrix.T)

    return rotated_points


def directions_to_spherical_uv(directions: np.ndarray):
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    u = 1 - np.arctan2(directions[..., 1], directions[..., 0]) / (2 * np.pi) % 1.0
    v = np.arccos(directions[..., 2]) / np.pi
    return np.stack([u, v], axis=-1)


def split_panorama_image(image: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray, h: int, w: int, interp):
    height, width = image.shape[:2]
    safe_height = height // 2
    safe_width = int(round(safe_height / h * w))
    if interp == cv2.INTER_AREA: # remap does not support area downsampling; remap to a safe resolution first to avoid frequency artifacts.
        uv = utils3d.numpy.image_uv(width=safe_width, height=safe_height)
    else:
        uv = utils3d.numpy.image_uv(width=w, height=h)
    splitted_images = []
    for i in range(len(extrinsics)):
        spherical_uv = directions_to_spherical_uv(utils3d.numpy.unproject_cv(uv, extrinsics=extrinsics[i], intrinsics=intrinsics[i]))
        pixels = utils3d.numpy.uv_to_pixel(spherical_uv, width=width, height=height).astype(np.float32)
        if interp == cv2.INTER_AREA: # remap does not support area downsampling; remap to a safe resolution first to avoid frequency artifacts.
            splitted_image = cv2.remap(image, pixels[..., 0], pixels[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            splitted_image = cv2.resize(splitted_image, (w, h), interpolation=interp)
        else:
            splitted_image = cv2.remap(image, pixels[..., 0], pixels[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        splitted_images.append(splitted_image)
    return splitted_images


def split_panorama_depth(depth: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray, h: int, w: int, distance_to_depth=False):
    height, width = depth.shape[:2]
    depth = torch.tensor(depth, dtype=torch.float32)[None, None]
    uv = utils3d.numpy.image_uv(width=w, height=h)
    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    splitted_depths = []
    for i in range(len(extrinsics)):
        spherical_uv = directions_to_spherical_uv(utils3d.numpy.unproject_cv(uv, extrinsics=extrinsics[i], intrinsics=intrinsics[i]))
        pixels = utils3d.numpy.uv_to_pixel(spherical_uv, width=width, height=height).astype(np.float32)
        pixels = torch.tensor(pixels, dtype=torch.float32)[None, ...]  # [1,h,w,2]
        pixels[..., 0] /= width
        pixels[..., 1] /= height
        pixels = pixels * 2 - 1.0
        splitted_depth = F.grid_sample(depth, grid=pixels, mode="nearest", align_corners=True)

        if distance_to_depth:
            fx = intrinsics[i][0, 0] * w
            fy = intrinsics[i][1, 1] * h
            cx = intrinsics[i][0, 2] * w
            cy = intrinsics[i][1, 2] * h
            x_cam = (u_grid - cx) / fx
            y_cam = (v_grid - cy) / fy
            z_cam = np.ones_like(x_cam)
            rays_cam = np.stack([x_cam, y_cam, z_cam], axis=-1).astype(np.float32)  # (H, W, 3)
            ray_length = np.linalg.norm(rays_cam, axis=-1).astype(np.float32)
            splitted_depth = splitted_depth * (z_cam[None, None] / ray_length[None, None])

        splitted_depths.append(splitted_depth)
    return torch.cat(splitted_depths, dim=0).float()


def smooth_south_pole_depth(depth_map, smooth_height_ratio=0.03):
    """
    Smooth depth near the panorama south pole (bottom region) to fix left-right inconsistencies.

    Args:
        depth_map: Depth map (H, W).
        smooth_height_ratio: Height ratio of the smoothing region. The default 0.03 means the bottom 0.03 * H region.

    Returns:
        Smoothed depth map.
    """
    height, width = depth_map.shape
    smooth_height = int(height * smooth_height_ratio)

    if smooth_height == 0:
        return depth_map

    # Copy the depth map to avoid modifying the input.
    smoothed_depth = depth_map.copy()

    # Compute the reference depth from the last 3 rows when possible, otherwise from the bottom row.
    if smooth_height > 3:
        # Use the last 3 rows to compute the average depth.
        reference_rows = depth_map[-3:, :]
        reference_data = reference_rows.flatten()
    else:
        # Use the bottom row.
        reference_data = depth_map[-1, :]

    # Filter outliers, including invalid, overly large, or overly small depth values.
    valid_mask = np.isfinite(reference_data) & (reference_data > 0)

    if np.any(valid_mask):
        valid_depths = reference_data[valid_mask]

        # Use quantiles to filter extreme outliers.
        lower_bound, upper_bound = np.quantile(valid_depths, [0.1, 0.9])

        # Further remove overly large or small depth values.
        depth_filter_mask = (valid_depths >= lower_bound) & (valid_depths <= upper_bound)

        if np.any(depth_filter_mask):
            avg_depth = np.mean(valid_depths[depth_filter_mask])
        else:
            # Fall back to the median if all values are filtered out.
            avg_depth = np.median(valid_depths)
    else:
        avg_depth = np.nanmean(reference_data)

    # Set the bottom row to the average value.
    smoothed_depth[-1, :] = avg_depth

    # Smooth upward to the specified height.
    for i in range(1, smooth_height):
        y_idx = height - 1 - i  # Index moving upward from the bottom.
        if y_idx < 0:
            break

        # The closer to the bottom, the stronger the smoothing.
        weight = (smooth_height - i) / smooth_height

        # Smooth the current row.
        current_row = depth_map[y_idx, :]
        valid_mask = np.isfinite(current_row) & (current_row > 0)

        if np.any(valid_mask):
            valid_row_depths = current_row[valid_mask]

            # Apply outlier filtering to the current row as well.
            if len(valid_row_depths) > 1:
                q25, q75 = np.quantile(valid_row_depths, [0.25, 0.75])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                depth_filter_mask = (valid_row_depths >= lower_bound) & (valid_row_depths <= upper_bound)

                if np.any(depth_filter_mask):
                    row_avg = np.mean(valid_row_depths[depth_filter_mask])
                else:
                    row_avg = np.median(valid_row_depths)
            else:
                row_avg = valid_row_depths[0] if len(valid_row_depths) > 0 else avg_depth

            # Linearly interpolate between the original depth and the row average.
            smoothed_depth[y_idx, :] = (1 - weight) * current_row + weight * row_avg

    return smoothed_depth


def solve_lsmr_gpu(A, b, x0=None, atol=1e-5, btol=1e-5, conlim=0.0,
                   maxiter=None):
    """On-device LSMR for sparse least-squares ``min ||Ax - b||_2``.

    Same Golub-Kahan bidiagonalization + Givens-rotation update steps as
    scipy.sparse.linalg.lsmr (Fong & Saunders 2011), but the entire iteration
    runs through torch tensors - the big work vectors u, v, h, hbar, x live
    on the GPU; only the ~6 scalar Givens reductions per iteration come back
    to the host. Eliminates the per-iteration host<->device round-trip that
    the old `_TorchSparseLinearOperator` paid (matvec -> .cpu().numpy() ->
    scipy numpy ops -> as_tensor -> next matvec). For a 1920x960 panorama
    solve that's ~600 iterations x 4 transfers x 7 MB ~= 30 GB of avoided
    PCIe traffic.

    Numerical behaviour vs scipy fp64 CPU: SpMV runs in fp32 on cuSPARSE
    (same as the previous GPU path), all 1.84M-element vector ops in fp32;
    scalar reductions cast to fp64 on the host so the Givens math is
    bit-equivalent to scipy's. Same converged answer up to fp32 SpMV
    round-off, which is the existing baseline.

    Args:
        A:       scipy CSR-like sparse matrix (csr_array, csr_matrix, or
                 anything `csr_matrix(...)` accepts).
        b:       numpy 1-D vector, length A.shape[0].
        x0:      optional initial guess, length A.shape[1].
        atol,    stopping tolerances - same semantics as scipy.sparse.linalg.lsmr.
        btol:    `atol` controls ||A^T r||/(||A|| ||r||) test, `btol` the
                 ||r||/||b|| test.
        conlim:  if > 0, also stop when an estimate of cond(A) exceeds
                 conlim. Default 0 = disabled - the old GPU path with a
                 host-side LinearOperator triggered scipy's `condA` cast
                 to overflow at lsmr.py:407 because fp32 SpMV outputs
                 didn't match scipy's fp64 expectations; skipping the
                 tracking removes the warning AND saves a handful of host
                 ops per iteration. The same solution is returned (cond
                 estimate is an early-stop, never affects the iterate).
        maxiter: cap on iterations. Default min(m, n).

    Returns: numpy 1-D vector, length A.shape[1].
    """
    from math import sqrt, inf
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg._isolve.lsqr import _sym_ortho

    # --- CPU fallback: pure scipy lsmr. Same answer, slower. ---
    if not torch.cuda.is_available():
        A_csr = cp_csr_matrix(A)
        x, *_ = cp_lsmr(A_csr, b, atol=atol, btol=btol, x0=x0,
                        conlim=conlim if conlim > 0 else 1e8,
                        maxiter=maxiter)
        return x

    device = torch.device("cuda")
    # Normalize input to scipy CSR so we can read indptr / indices / data.
    A_csr = A if isinstance(A, csr_matrix) else csr_matrix(A)
    m, n = A_csr.shape
    if maxiter is None:
        maxiter = min(m, n)

    # --- Upload A and A^T as torch CSR (one-time per solve). ---
    # int32 indices are safe (m, n both < 2^31); halves SpMV index bandwidth
    # vs the int64 the old operator used.
    A_gpu = torch.sparse_csr_tensor(
        torch.from_numpy(A_csr.indptr.astype(np.int32)).to(device),
        torch.from_numpy(A_csr.indices.astype(np.int32)).to(device),
        torch.from_numpy(A_csr.data.astype(np.float32)).to(device),
        size=A_csr.shape,
    )
    AT_csr = A_csr.T.tocsr()
    AT_gpu = torch.sparse_csr_tensor(
        torch.from_numpy(AT_csr.indptr.astype(np.int32)).to(device),
        torch.from_numpy(AT_csr.indices.astype(np.int32)).to(device),
        torch.from_numpy(AT_csr.data.astype(np.float32)).to(device),
        size=AT_csr.shape,
    )

    def _matvec(vec):  # vec: (n,) -> (m,)
        return torch.sparse.mm(A_gpu, vec.unsqueeze(-1)).squeeze(-1)

    def _rmatvec(vec):  # vec: (m,) -> (n,)
        return torch.sparse.mm(AT_gpu, vec.unsqueeze(-1)).squeeze(-1)

    # --- Upload b (and x0 if given). ---
    b_gpu = torch.as_tensor(np.asarray(b, dtype=np.float32), device=device)
    u = b_gpu.clone()
    normb = float(torch.linalg.vector_norm(b_gpu).item())

    if x0 is None:
        x = torch.zeros(n, device=device, dtype=torch.float32)
        beta = normb
    else:
        x = torch.as_tensor(np.asarray(x0, dtype=np.float32),
                            device=device).clone()
        u.sub_(_matvec(x))
        beta = float(torch.linalg.vector_norm(u).item())

    if beta > 0.0:
        u.mul_(1.0 / beta)
        v = _rmatvec(u)
        alpha = float(torch.linalg.vector_norm(v).item())
    else:
        v = torch.zeros(n, device=device, dtype=torch.float32)
        alpha = 0.0

    if alpha > 0.0:
        v.mul_(1.0 / alpha)

    # --- Givens / convergence scalar state (fp64 on host). ---
    itn = 0
    zetabar = alpha * beta
    alphabar = alpha
    rho = 1.0
    rhobar = 1.0
    cbar = 1.0
    sbar = 0.0

    h = v.clone()
    hbar = torch.zeros(n, device=device, dtype=torch.float32)

    betadd = beta
    betad = 0.0
    rhodold = 1.0
    tautildeold = 0.0
    thetatilde = 0.0
    zeta = 0.0
    d = 0.0

    normA2 = alpha * alpha
    maxrbar = 0.0
    minrbar = 1e100
    normA = sqrt(normA2)
    condA = 1.0
    normx = 0.0
    damp = 0.0  # we never use damping in this codepath

    istop = 0
    ctol = (1.0 / conlim) if conlim > 0 else 0.0
    normr = beta
    normar = alpha * beta

    if normar == 0.0:
        return x.cpu().numpy()
    if normb == 0.0:
        x.zero_()
        return x.cpu().numpy()

    while itn < maxiter:
        itn += 1

        # Bidiagonalization step. u <- A v - alpha u; v <- A^T u - beta v.
        u.mul_(-alpha).add_(_matvec(v))
        beta = float(torch.linalg.vector_norm(u).item())
        if beta > 0.0:
            u.mul_(1.0 / beta)
            v.mul_(-beta).add_(_rmatvec(u))
            alpha = float(torch.linalg.vector_norm(v).item())
            if alpha > 0.0:
                v.mul_(1.0 / alpha)

        # Givens rotations (host scalar math, ~1us each).
        chat, shat, alphahat = _sym_ortho(alphabar, damp)
        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew = s * alpha
        alphabar = c * alpha

        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar

        # Update h, hbar, x - fused in-place AXPYs on GPU.
        hbar.mul_(-(thetabar * rho / (rhoold * rhobarold))).add_(h)
        x.add_(hbar, alpha=(zeta / (rho * rhobar)))
        h.mul_(-(thetanew / rho)).add_(v)

        # Estimate ||r||.
        betaacute = chat * betadd
        betacheck = -shat * betadd
        betahat = c * betaacute
        betadd = -s * betaacute

        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = sqrt(d + (betad - taud) ** 2 + betadd * betadd)

        # Estimate ||A||.
        normA2 = normA2 + beta * beta
        normA = sqrt(normA2)
        normA2 = normA2 + alpha * alpha

        # Estimate cond(A) - only if user asked (conlim > 0). Skipping the
        # whole block saves a handful of host ops AND avoids scipy's
        # `condA = max(...)/min(...)` fp32 cast overflow that polluted the
        # old GPU log with RuntimeWarnings.
        if conlim > 0.0:
            maxrbar = max(maxrbar, rhobarold)
            if itn > 1:
                minrbar = min(minrbar, rhobarold)
            condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)

        # Convergence tests.
        normar = abs(zetabar)
        normx = float(torch.linalg.vector_norm(x).item())

        test1 = normr / normb
        if (normA * normr) != 0.0:
            test2 = normar / (normA * normr)
        else:
            test2 = inf
        test3 = 1.0 / condA
        t1 = test1 / (1.0 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb

        if itn >= maxiter:
            istop = 7
        if 1.0 + test3 <= 1.0:
            istop = 6
        if 1.0 + test2 <= 1.0:
            istop = 5
        if 1.0 + t1 <= 1.0:
            istop = 4

        if conlim > 0.0 and test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1

        if istop > 0:
            break

    return x.cpu().numpy()


def merge_panorama_depth_gpu(width: int, height: int, distance_maps: List[np.ndarray], pred_masks: List[np.ndarray], extrinsics: List[np.ndarray], intrinsics: List[np.ndarray], chunk_size: int = 8, center_weight_power: float = 0.0, scale_anchor: bool = True):
    # `grad_equation` / `poisson_equation` defined in moge_panorama.py - pure
    # geometry sparse skeletons; cheap to rebuild per level (sub-0.4s at top
    # res). `vstack` stacks them after the per-face validity rows are
    # selected.
    import sys
    import time
    from scipy.sparse import vstack
    from ...moge_panorama import grad_equation, poisson_equation

    def _p(msg):
        print(f"[merge_panorama_depth_gpu {width:>4}x{height:<4}] {msg}",
              file=sys.stderr, flush=True)

    t_lvl_start = time.perf_counter()

    if max(width, height) > 256:
        panorama_depth_init, _ = merge_panorama_depth_gpu(width // 2, height // 2, distance_maps, pred_masks, extrinsics, intrinsics, chunk_size=chunk_size, center_weight_power=center_weight_power, scale_anchor=scale_anchor)
        panorama_depth_init = cv2.resize(panorama_depth_init, (width, height), cv2.INTER_LINEAR)
    else:
        panorama_depth_init = None

    # Force the caching allocator to return cached-but-unreferenced blocks
    # to CUDA *before* this level allocates its (potentially much larger)
    # working set. Without this, a 4x size jump from the recursive child
    # level to the parent level (e.g. 1920x960 -> 3840x1920, N=42) can OOM
    # on a 24 GB card even though Python has dropped every reference --
    # PyTorch's caching allocator holds the freed blocks at the child
    # level's size and the parent's bigger contiguous request fragments
    # the cache. Per-level empty_cache() lets the parent see the full
    # free pool. Cheap: empty_cache itself is a few-millisecond memcpy
    # of the allocator's metadata, not a real free of allocated memory.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Phase 2 (commit after 78380f4): the per-face cv2.remap loop +
    # scipy.ndimage.convolve calls + np.stack/np.sum reductions are now
    # one batched torch pipeline. All faces' projections, warps, gradients,
    # Laplacians, and mask reductions go through a single set of CUDA
    # kernels operating on (N, H, W) tensors. Math is fp32-equivalent to
    # the original numpy/cv2 path:
    # - grid_sample mode='bilinear' + padding_mode='border' matches
    #   cv2.INTER_LINEAR + BORDER_REPLICATE (verified by split helper).
    # - Gradient via x-wrap pad then forward-diff is bit-identical to the
    #   np.pad + slice pair.
    # - Laplacian via direct 5-point stencil (up + down + left + right -
    #   4*center) is the same kernel as scipy.ndimage.convolve with the
    #   symmetric [[0,1,0],[1,-4,1],[0,1,0]] kernel.
    # - Reductions (sum * mask / sum mask) reproduce the exact arithmetic
    #   of np.sum(maps * masks, axis=0) / np.sum(masks, axis=0).clip(1e-3).

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    N = len(distance_maps)
    fh, fw = distance_maps[0].shape

    # Sphere ray directions for the equirect grid. Tiny (~7 MB at 1920x960),
    # CPU build is fine - utils3d.np.uv_map + spherical_uv_to_directions is
    # vectorized numpy.
    uv = utils3d.np.uv_map((height, width))
    spherical_directions_np = spherical_uv_to_directions(uv).astype(np.float32)

    # --- Chunked per-face processing to bound peak memory. ---
    # Previously this block materialized ~10 simultaneous (N, H, W) fp32 /
    # bool tensors per pyramid level - at top-level 4096x2048 with N=42
    # that's ~14 GB of intermediates and OOMs on consumer GPUs. The
    # reductions consumed downstream are `(grad * mask).sum(0)`,
    # `mask.sum(0)`, and `mask.any(0)` - all ASSOCIATIVE across the face
    # dim - so we can chunk the face axis without changing the math.
    # Per-chunk peak is K*H*W*4B*~10 ~= chunk_size/N times the full-batch
    # peak. With default chunk_size=8 and N=42 that's a ~5x memory cut at
    # the cost of a few extra Python-level iterations.
    t_warp = time.perf_counter()

    dirs = torch.from_numpy(spherical_directions_np).to(device).reshape(-1, 3)         # (H*W, 3)
    # Per-face inputs uploaded once at full size - these are (N, fh, fw),
    # NOT (N, H, W), so they're not the dominant tensor here. At 42 x 1024^2
    # x 4B = 168 MB they fit comfortably.
    dist_stack = np.stack([d.astype(np.float32, copy=False) for d in distance_maps])    # (N, fh, fw)
    dist_gpu = torch.from_numpy(dist_stack).to(device)
    log_dist_gpu_all = torch.log(dist_gpu.clamp(min=1e-30))                            # (N, fh, fw)
    # `pred_masks` are now treated as continuous per-pixel WEIGHTS in
    # [0, 1] (was bool only). Bool inputs cast to 0.0/1.0 give bit-
    # equivalent results to the previous bool-AND path; continuous
    # confidence (e.g. MoGe-2's sigmoid mask_soft) gives a true
    # confidence-weighted residual where low-confidence pixels
    # contribute proportionally less to the LSMR objective.
    mask_stack = np.stack([np.asarray(m, dtype=np.float32) for m in pred_masks])        # (N, fh, fw)
    pred_mask_gpu_all = torch.from_numpy(mask_stack).to(device).clamp(0.0, 1.0)
    ext_gpu = torch.from_numpy(np.stack(extrinsics).astype(np.float32)).to(device)      # (N, 4, 4)
    intr_gpu = torch.from_numpy(np.stack(intrinsics).astype(np.float32)).to(device)     # (N, 3, 3)

    # Pre-allocate per-equirect-pixel accumulators (the only (H, W)-sized
    # tensors that persist across chunks). Float numerators / denominators
    # for the three weighted means; bool ORs for the validity unions.
    num_grad_x = torch.zeros(height, width, dtype=torch.float32, device=device)
    den_grad_x = torch.zeros(height, width, dtype=torch.float32, device=device)
    num_grad_y = torch.zeros(height - 1, width + 1, dtype=torch.float32, device=device)
    den_grad_y = torch.zeros(height - 1, width + 1, dtype=torch.float32, device=device)
    num_lap = torch.zeros(height, width, dtype=torch.float32, device=device)
    den_lap = torch.zeros(height, width, dtype=torch.float32, device=device)
    any_mask_x = torch.zeros(height, width, dtype=torch.bool, device=device)
    any_mask_y = torch.zeros(height - 1, width + 1, dtype=torch.bool, device=device)
    any_mask_lap = torch.zeros(height, width, dtype=torch.bool, device=device)
    pred_mask_union = torch.zeros(height, width, dtype=torch.bool, device=device)

    K = max(1, min(int(chunk_size), N))
    n_chunks = (N + K - 1) // K

    for chunk_start in range(0, N, K):
        chunk_end = min(chunk_start + K, N)
        Nc = chunk_end - chunk_start

        # --- Per-chunk slice of the uploaded inputs. ---
        log_dist_gpu = log_dist_gpu_all[chunk_start:chunk_end]        # (Nc, fh, fw)
        pred_mask_gpu = pred_mask_gpu_all[chunk_start:chunk_end]      # (Nc, fh, fw) float [0,1]
        R = ext_gpu[chunk_start:chunk_end, :3, :3]                    # (Nc, 3, 3)
        t = ext_gpu[chunk_start:chunk_end, :3, 3]                     # (Nc, 3)
        K_intr = intr_gpu[chunk_start:chunk_end]                       # (Nc, 3, 3)

        # --- Batched projection: equirect rays -> per-face image plane. ---
        # p_cam = R @ dirs + t. Then K @ p_cam, perspective divide.
        p_cam = torch.einsum('nij,pj->npi', R, dirs) + t.unsqueeze(1)  # (Nc, H*W, 3)
        p_proj = torch.einsum('nij,npj->npi', K_intr, p_cam)           # (Nc, H*W, 3)
        depth_c = p_cam[..., 2]                                        # (Nc, H*W)

        safe_w = torch.where(
            p_proj[..., 2:3] > 1e-12,
            p_proj[..., 2:3], torch.ones_like(p_proj[..., 2:3]),
        )
        projected_uv = p_proj[..., :2] / safe_w                        # (Nc, H*W, 2)
        projection_valid = (
            (depth_c > 0)
            & (projected_uv[..., 0] >= 0) & (projected_uv[..., 0] <= 1)
            & (projected_uv[..., 1] >= 0) & (projected_uv[..., 1] <= 1)
        ).reshape(Nc, height, width)                                   # (Nc, H, W) bool
        grid = (projected_uv.clamp(0.0, 1.0) * 2.0 - 1.0).reshape(Nc, height, width, 2)

        # Cosine of angle between equirect's world ray and the face's
        # optical axis: since `dirs` is unit and R is a rotation, p_cam
        # is unit, so p_cam[..., 2] == cos(angle from optical axis).
        # Used downstream for center-weighting (down-weight face corners
        # where MoGe-2 / monocular depth predictors are less reliable
        # due to rectilinear distortion + training-data center bias).
        if center_weight_power > 0:
            cos_axis = depth_c.clamp(min=0.0).reshape(Nc, height, width)  # (Nc, H, W) in [0, 1]
        else:
            cos_axis = None

        # Free the intermediates we don't need before grid_sample.
        del p_cam, p_proj, depth_c, safe_w, projected_uv

        # --- Bilinear sample of log-distance + nearest sample of weight. ---
        sampled_log_dist = F.grid_sample(
            log_dist_gpu.unsqueeze(1),
            grid,
            mode='bilinear', padding_mode='border', align_corners=False,
        ).squeeze(1)                                                   # (Nc, H, W)
        # Nearest-neighbour resample of the per-face weight - preserves
        # bool-input bit-equivalence (0/1 stay 0/1) AND keeps continuous
        # confidence values intact within a face. NO `> 0.5` threshold
        # here anymore - that was the path that destroyed the continuous
        # confidence signal.
        sampled_pred_mask = F.grid_sample(
            pred_mask_gpu.unsqueeze(1),
            grid,
            mode='nearest', padding_mode='border', align_corners=False,
        ).squeeze(1)                                                   # (Nc, H, W) float [0,1]
        del grid

        panorama_log_distance = torch.where(
            projection_valid, sampled_log_dist,
            torch.zeros_like(sampled_log_dist),
        )                                                              # (Nc, H, W)
        # Per-face panorama weight: geometric projection-validity (bool)
        # AND'd with the (continuous) per-face confidence weight. Encoded
        # as float multiplication so bool inputs (0/1) stay bit-equivalent
        # to the old bool-AND path.
        panorama_pred_mask_per_face = projection_valid.float() * sampled_pred_mask   # (Nc, H, W) float
        del sampled_log_dist, sampled_pred_mask, projection_valid

        # Apply optional center weighting - down-weight face corners
        # (high obliquity from optical axis) where MoGe-2 is less
        # reliable due to rectilinear distortion + training-data bias
        # toward image centers. cos_axis is already in [0, 1].
        # Power=1 -> cosine (corner ~0.577 at 90deg fov), power=2 -> cos^2
        # (~0.333 at corner), higher = sharper falloff.
        if cos_axis is not None:
            panorama_pred_mask_per_face = panorama_pred_mask_per_face * (cos_axis ** center_weight_power)
            del cos_axis

        # --- Gradient computation (wrap in x, no wrap in y). ---
        log_dist_xpad = torch.cat(
            [panorama_log_distance, panorama_log_distance[:, :, :1]], dim=-1
        )                                                              # (Nc, H, W+1)
        grad_x = log_dist_xpad[:, :, :-1] - log_dist_xpad[:, :, 1:]    # (Nc, H, W)
        grad_y = log_dist_xpad[:, :-1, :] - log_dist_xpad[:, 1:, :]    # (Nc, H-1, W+1)
        del log_dist_xpad

        mask_xpad = torch.cat(
            [panorama_pred_mask_per_face, panorama_pred_mask_per_face[:, :, :1]], dim=-1
        )                                                              # (Nc, H, W+1)
        # Bool AND replaced by float multiply: for {0, 1} weights this
        # is bit-equivalent to AND; for continuous weights it's the
        # product of two endpoint confidences (a gradient is only
        # trustworthy if both endpoints are confidently observed).
        mask_x = mask_xpad[:, :, :-1] * mask_xpad[:, :, 1:]            # (Nc, H, W)
        mask_y = mask_xpad[:, :-1, :] * mask_xpad[:, 1:, :]            # (Nc, H-1, W+1)
        del mask_xpad

        # --- 5-point Laplacian + 5-neighbour validity. ---
        log_dist_4d = panorama_log_distance.unsqueeze(1)               # (Nc, 1, H, W)
        log_dist_pad = F.pad(log_dist_4d, (0, 0, 1, 1), mode='replicate')
        log_dist_pad = F.pad(log_dist_pad, (1, 1, 0, 0), mode='circular')  # (Nc, 1, H+2, W+2)
        center = log_dist_pad[:, 0, 1:-1, 1:-1]
        up = log_dist_pad[:, 0, :-2, 1:-1]
        down = log_dist_pad[:, 0, 2:, 1:-1]
        left = log_dist_pad[:, 0, 1:-1, :-2]
        right = log_dist_pad[:, 0, 1:-1, 2:]
        laplacian = up + down + left + right - 4.0 * center            # (Nc, H, W)
        del log_dist_pad, center, up, down, left, right, log_dist_4d

        mask_4d = panorama_pred_mask_per_face.unsqueeze(1)             # (Nc, 1, H, W) float
        mask_pad = F.pad(mask_4d, (0, 0, 1, 1), mode='replicate')
        mask_pad = F.pad(mask_pad, (1, 1, 0, 0), mode='circular')      # (Nc, 1, H+2, W+2)
        # 5-neighbor weight: PRODUCT of all 5 (was bool AND of `>0.5`
        # thresholded values). For {0, 1} inputs this is bit-equivalent
        # to AND; for continuous confidence it down-weights pixels where
        # any neighbor is low-confidence (Laplacian only trusted when
        # all 5 surrounding pixels are well-observed).
        mc = mask_pad[:, 0, 1:-1, 1:-1]
        mu = mask_pad[:, 0, :-2, 1:-1]
        md = mask_pad[:, 0, 2:, 1:-1]
        ml = mask_pad[:, 0, 1:-1, :-2]
        mr = mask_pad[:, 0, 1:-1, 2:]
        mask_laplacian = mc * mu * md * ml * mr                        # (Nc, H, W) float
        del mask_pad, mc, mu, md, ml, mr, mask_4d

        # --- Accumulate weighted sums + validity unions into the (H, W)
        # buffers. Sum is associative: chunked partial sums combine
        # cleanly into the same final aggregate the old single-batch
        # path produced. mask_x / mask_y / mask_laplacian are already
        # float [0, 1] now - no `.float()` cast needed.
        num_grad_x.add_((grad_x * mask_x).sum(0))
        den_grad_x.add_(mask_x.sum(0))
        num_grad_y.add_((grad_y * mask_y).sum(0))
        den_grad_y.add_(mask_y.sum(0))
        num_lap.add_((laplacian * mask_laplacian).sum(0))
        den_lap.add_(mask_laplacian.sum(0))

        # Inclusion test for the LSMR row-selection masks: include a row
        # if any face contributed appreciable weight at that pixel.
        # `> 1e-3` matches the existing `clamp(min=1e-3)` floor used in
        # the divide step. For bool->float inputs (0/1) this is
        # bit-equivalent to the old `.any(0)`.
        any_mask_x.logical_or_((mask_x > 1e-3).any(0))
        any_mask_y.logical_or_((mask_y > 1e-3).any(0))
        any_mask_lap.logical_or_((mask_laplacian > 1e-3).any(0))
        pred_mask_union.logical_or_((panorama_pred_mask_per_face > 1e-3).any(0))

        # Drop the per-chunk tensors; next iteration's allocations can
        # reuse the freed memory.
        del (grad_x, grad_y, laplacian,
             mask_x, mask_y, mask_laplacian,
             panorama_log_distance, panorama_pred_mask_per_face)

    # Final divide once, on the accumulated (H, W) buffers.
    avg_grad_x = num_grad_x / den_grad_x.clamp(min=1e-3)
    avg_grad_y = num_grad_y / den_grad_y.clamp(min=1e-3)
    avg_lap = num_lap / den_lap.clamp(min=1e-3)

    # --- Single device->host transfer of the data we need on CPU for the
    # sparse-system build below. ---
    avg_grad_x_np = avg_grad_x.cpu().numpy()
    avg_grad_y_np = avg_grad_y.cpu().numpy()
    avg_lap_np = avg_lap.cpu().numpy()
    grad_x_mask_np = any_mask_x.cpu().numpy().reshape(-1)
    grad_y_mask_np = any_mask_y.cpu().numpy().reshape(-1)
    grad_mask_np = np.concatenate([grad_x_mask_np, grad_y_mask_np])
    laplacian_mask_np = any_mask_lap.cpu().numpy().reshape(-1)
    panorama_pred_mask_final_np = pred_mask_union.cpu().numpy()

    # Free GPU buffers we no longer need before LSMR allocates its
    # CSR + work vectors. The recursion holds these levels' tensors
    # alive otherwise.
    del (num_grad_x, den_grad_x, num_grad_y, den_grad_y, num_lap, den_lap,
         any_mask_x, any_mask_y, any_mask_lap, pred_mask_union,
         avg_grad_x, avg_grad_y, avg_lap,
         dist_gpu, log_dist_gpu_all, pred_mask_gpu_all,
         ext_gpu, intr_gpu, dirs)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    t_warp_done = time.perf_counter()

    # --- Solve overdetermined system (scipy CSR + on-device LSMR). ---
    t_build = time.perf_counter()
    A = vstack([
        grad_equation(width, height, wrap_x=True, wrap_y=False)[grad_mask_np],
        poisson_equation(width, height, wrap_x=True, wrap_y=False)[laplacian_mask_np],
    ])
    b = np.concatenate([
        avg_grad_x_np.reshape(-1)[grad_x_mask_np],
        avg_grad_y_np.reshape(-1)[grad_y_mask_np],
        avg_lap_np.reshape(-1)[laplacian_mask_np],
    ])
    t_build_done = time.perf_counter()

    t_solve = time.perf_counter()
    x = solve_lsmr_gpu(
        A, b,
        x0=np.log(panorama_depth_init).reshape(-1) if panorama_depth_init is not None else None,
    )
    t_solve_done = time.perf_counter()

    # Fix the LSMR rank-1 scale ambiguity (post-hoc shift). The gradient +
    # Laplacian operators are translation-invariant in log(d), so `1` lies
    # in null(A); LSMR's min-norm pick biases x toward 0 -> d ~= 1m. We
    # solve this only at the BOTTOM pyramid level (where x0=None and the
    # bias is set); upper levels inherit the corrected scale via x0.
    # Math is exact: the shift `c = median(log d_input) - median(x_solved)`
    # restores the input-scale median while leaving every gradient /
    # Laplacian residual untouched (because the operator nullspace IS the
    # 1-vector).
    scale_shift = 0.0
    if scale_anchor and panorama_depth_init is None:
        valid_logs = []
        for d_face, m_face in zip(distance_maps, pred_masks):
            d_arr = np.asarray(d_face, dtype=np.float32)
            m_arr = np.asarray(m_face, dtype=np.float32)
            keep = (m_arr > 1e-3) & np.isfinite(d_arr) & (d_arr > 1e-6)
            if keep.any():
                valid_logs.append(np.log(d_arr[keep]))
        if valid_logs:
            log_d_input_median = float(np.median(np.concatenate(valid_logs)))
            log_d_solved_median = float(np.median(x))
            scale_shift = log_d_input_median - log_d_solved_median
            x = x + scale_shift

    panorama_depth = np.exp(x).reshape(height, width).astype(np.float32)
    panorama_mask = panorama_pred_mask_final_np

    anchor_str = (
        f"scale_shift={scale_shift:+.4f} (exp~={float(np.exp(scale_shift)):.3f}); "
        if (scale_anchor and panorama_depth_init is None) else ""
    )
    _p(f"done {time.perf_counter() - t_lvl_start:.2f}s "
       f"(warp {t_warp_done - t_warp:.2f}, build {t_build_done - t_build:.2f}, "
       f"LSMR {t_solve_done - t_solve:.2f}); "
       f"N={N} K={K} chunks={n_chunks}; "
       f"{anchor_str}"
       f"depth min/med/max = {float(panorama_depth.min()):.3f}/"
       f"{float(np.median(panorama_depth)):.3f}/"
       f"{float(panorama_depth.max()):.3f}; "
       f"valid {int(panorama_mask.sum())}/{panorama_mask.size}")
    return panorama_depth, panorama_mask




# Panorama depth stitching based on normal constraints.
def compute_spherical_ray_derivatives(spherical_directions: np.ndarray):
    """
    Compute partial derivatives of panorama ray directions with respect to spherical coordinates.

    Args:
        spherical_directions: (H, W, 3) Ray direction for each panorama pixel in world coordinates as unit vectors.

    Returns:
        dray_dtheta: (H, W, 3) Partial derivative of rays with respect to azimuth theta.
        dray_dphi: (H, W, 3) Partial derivative of rays with respect to elevation phi.
    """
    H, W, _ = spherical_directions.shape

    # Recover spherical coordinates from ray directions.
    # Assumes x-right, y-up, z-forward coordinates; adjust if the actual convention differs.
    ray = spherical_directions

    # Azimuth theta and elevation phi.
    # ray = (cos(phi)sin(theta), sin(phi), cos(phi)cos(theta))
    # Or adapt to the coordinate-system definition in use.

    # Method 1: numerical computation, which is more robust.
    u = (np.arange(W) + 0.5) / W  # [0, 1]
    v = (np.arange(H) + 0.5) / H  # [0, 1]
    u_grid, v_grid = np.meshgrid(u, v)

    theta = (u_grid - 0.5) * 2 * np.pi  # [-pi, pi]
    phi = (0.5 - v_grid) * np.pi  # [pi/2, -pi/2]

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # dray/dtheta, assuming standard equirectangular mapping.
    # Adjust according to how spherical_directions is actually computed.
    dray_dtheta = np.stack([
        cos_phi * cos_theta,
        np.zeros_like(theta),
        -cos_phi * sin_theta
    ], axis=-1)

    # dray/dphi
    dray_dphi = np.stack([
        -sin_phi * sin_theta,
        cos_phi,
        -sin_phi * cos_theta
    ], axis=-1)

    return dray_dtheta, dray_dphi


def normal_to_log_distance_gradient(
        panorama_normal: np.ndarray,
        spherical_directions: np.ndarray,
        width: int,
        height: int
) -> tuple:
    """
    Compute log-distance gradients from panorama normals.

    Derivation:
        Surface point P = d * ray
        Tangent vector dP/dtheta = (dd/dtheta) * ray + d * (dray/dtheta)
        The normal is perpendicular to the tangent: n . dP/dtheta = 0
        => d(log d)/dtheta = -(n . dray/dtheta) / (n . ray)

    Args:
        panorama_normal: (H, W, 3) Panorama normal map in world coordinates.
        spherical_directions: (H, W, 3) Ray directions in world coordinates.
        width, height: Panorama dimensions.

    Returns:
        grad_x: (H, W) x-direction gradient corresponding to log_d[j] - log_d[j+1].
        grad_y: (H-1, W) y-direction gradient corresponding to log_d[i] - log_d[i+1].
        valid_mask: (H, W) Valid region.
    """
    H, W = height, width

    # Get the ray-direction partial derivatives.
    dray_dtheta, dray_dphi = compute_spherical_ray_derivatives(spherical_directions)

    # Compute dot products.
    n_dot_ray = np.sum(panorama_normal * spherical_directions, axis=-1)  # (H, W)
    n_dot_dray_dtheta = np.sum(panorama_normal * dray_dtheta, axis=-1)  # (H, W)
    n_dot_dray_dphi = np.sum(panorama_normal * dray_dphi, axis=-1)  # (H, W)

    # Validity check: normals cannot be perpendicular to the view direction.
    eps = 1e-4
    valid_mask = np.abs(n_dot_ray) > eps

    # Safe division.
    n_dot_ray_safe = np.where(valid_mask, n_dot_ray, 1.0)

    # Continuous-space log(d) gradients with respect to angles.
    # d(log d)/dtheta = -(n . dray/dtheta) / (n . ray)
    dlogd_dtheta = -n_dot_dray_dtheta / n_dot_ray_safe  # (H, W)
    dlogd_dphi = -n_dot_dray_dphi / n_dot_ray_safe  # (H, W)

    # Convert to discrete pixel gradients.
    # In this code, grad_x = log_d[j] - log_d[j+1] = -d(log d)/dtheta * Deltatheta.
    # Deltatheta = 2pi / W, the theta change per pixel.
    # grad_y = log_d[i] - log_d[i+1] = -d(log d)/dphi * Deltaphi = d(log d)/dphi * (pi/H)

    delta_theta = 2 * np.pi / W
    delta_phi = np.pi / H

    # Pixel-scale continuous gradients.
    pixel_dlogd_dx = -dlogd_dtheta * delta_theta  # Corresponds to log_d[j] - log_d[j+1].
    pixel_dlogd_dy = dlogd_dphi * delta_phi  # Corresponds to log_d[i] - log_d[i+1].

    # Discrete gradients, averaging both sides at boundaries.
    # X direction (wrap).
    padded_grad_x = np.pad(pixel_dlogd_dx, ((0, 0), (0, 1)), mode='wrap')
    grad_x = (padded_grad_x[:, :-1] + padded_grad_x[:, 1:]) / 2  # (H, W)

    # Y direction (no wrap).
    grad_y = (pixel_dlogd_dy[:-1, :] + pixel_dlogd_dy[1:, :]) / 2  # (H-1, W)

    # Valid masks for gradients
    padded_valid = np.pad(valid_mask, ((0, 0), (0, 1)), mode='wrap')
    mask_x = padded_valid[:, :-1] & padded_valid[:, 1:]  # (H, W)
    mask_y = valid_mask[:-1, :] & valid_mask[1:, :]  # (H-1, W)

    return grad_x, grad_y, mask_x, mask_y, valid_mask


def grad_equation_separate(width: int, height: int, wrap_x: bool = False, wrap_y: bool = False):
    """
    Return gradient equation matrices for the x and y directions separately.

    Returns:
        grad_eq_x: x-direction gradient matrix, with H * W rows if wrap_x, otherwise H * (W - 1).
        grad_eq_y: y-direction gradient matrix, with (H - 1) * W rows if wrap_y is false.
    """
    grid_index = np.arange(width * height).reshape(height, width)

    # X direction.
    if wrap_x:
        grid_x = np.pad(grid_index, ((0, 0), (0, 1)), mode='wrap')
    else:
        grid_x = grid_index

    n_grad_x = grid_x.shape[0] * (grid_x.shape[1] - 1)
    data_x = np.concatenate([
        np.ones((grid_x.shape[0], grid_x.shape[1] - 1), dtype=np.float32).reshape(-1, 1),
        -np.ones((grid_x.shape[0], grid_x.shape[1] - 1), dtype=np.float32).reshape(-1, 1),
    ], axis=1).reshape(-1)
    indices_x = np.concatenate([
        grid_x[:, :-1].reshape(-1, 1),
        grid_x[:, 1:].reshape(-1, 1),
    ], axis=1).reshape(-1)
    indptr_x = np.arange(0, n_grad_x * 2 + 1, 2)
    grad_eq_x = csr_array((data_x, indices_x, indptr_x), shape=(n_grad_x, height * width))

    # Y direction, usually without wrapping.
    if wrap_y:
        grid_y = np.pad(grid_index, ((0, 1), (0, 0)), mode='wrap')
    else:
        grid_y = grid_index

    n_grad_y = (grid_y.shape[0] - 1) * grid_y.shape[1]
    data_y = np.concatenate([
        np.ones((grid_y.shape[0] - 1, grid_y.shape[1]), dtype=np.float32).reshape(-1, 1),
        -np.ones((grid_y.shape[0] - 1, grid_y.shape[1]), dtype=np.float32).reshape(-1, 1),
    ], axis=1).reshape(-1)
    indices_y = np.concatenate([
        grid_y[:-1, :].reshape(-1, 1),
        grid_y[1:, :].reshape(-1, 1),
    ], axis=1).reshape(-1)
    indptr_y = np.arange(0, n_grad_y * 2 + 1, 2)
    grad_eq_y = csr_array((data_y, indices_y, indptr_y), shape=(n_grad_y, height * width))

    return grad_eq_x, grad_eq_y










def spherical_uv_to_directions(uv: np.ndarray):
    theta, phi = (1 - uv[..., 0]) * (2 * np.pi), uv[..., 1] * np.pi
    directions = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=-1)
    return directions


def convert_rgbd2pcd_panorama(
        rgb: torch.Tensor,  # (H, W, 3) RGB image, values [0, 1]
        distance: torch.Tensor,  # (H, W) Distance map
        rays: torch.Tensor,  # (H, W, 3) Ray directions (unit vectors ideally)
        excluded_region_mask: Optional[torch.Tensor] = None,  # (H, W) Optional boolean mask
        max_size: int = 4096,  # Max dimension for resizing
        device: Literal["cuda", "cpu"] = "cuda",  # Computation device
        dropout_pcd=False
):
    """
    Converts panoramic RGBD data (image, distance, rays) into an Open3D mesh.

    Args:
        image: Input RGB image tensor (H, W, 3), uint8 or float [0, 255].
        distance: Input distance map tensor (H, W).
        rays: Input ray directions tensor (H, W, 3). Assumed to originate from (0,0,0).
        excluded_region_mask: Optional boolean mask tensor (H, W). True values indicate regions to potentially exclude.
        max_size: Maximum size (height or width) to resize inputs to.
        device: The torch device ('cuda' or 'cpu') to use for computations.

    Returns:
        An Open3D TriangleMesh object.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, "Image must be HxWx3"
    assert distance.ndim == 2, "Distance must be HxW"
    assert rays.ndim == 3 and rays.shape[2] == 3, "Rays must be HxWx3"
    assert (
            rgb.shape[:2] == distance.shape[:2] == rays.shape[:2]
    ), "Input shapes must match"

    mask = excluded_region_mask

    if mask is not None:
        assert (
                mask.ndim == 2 and mask.shape[:2] == rgb.shape[:2]
        ), "Mask shape must match"
        assert mask.dtype == torch.bool, "Mask must be a boolean tensor"

    rgb = rgb.to(device)
    distance = distance.to(device)
    rays = rays.to(device)
    if mask is not None:
        mask = mask.to(device)

    H, W = distance.shape
    if max(H, W) > max_size:
        scale = max_size / max(H, W)
    else:
        scale = 1.0

    rgb_nchw = rgb.permute(2, 0, 1).unsqueeze(0)
    distance_nchw = distance.unsqueeze(0).unsqueeze(0)
    rays_nchw = rays.permute(2, 0, 1).unsqueeze(0)

    rgb_resized = (
        F.interpolate(
            rgb_nchw,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )
        .squeeze(0)
        .permute(1, 2, 0)
    )

    distance_resized = (
        F.interpolate(
            distance_nchw,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )
        .squeeze(0)
        .squeeze(0)
    )

    rays_resized_nchw = F.interpolate(
        rays_nchw,
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False,
    )

    # IMPORTANT: Renormalize ray directions after interpolation
    rays_resized = rays_resized_nchw.squeeze(0).permute(1, 2, 0)
    rays_norm = torch.linalg.norm(rays_resized, dim=-1, keepdim=True)
    rays_resized = rays_resized / (rays_norm + 1e-8)

    if mask is not None:
        mask_resized = (
            F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),  # Needs float for interpolation
                scale_factor=scale,
                mode="nearest",  # Or 'nearest' if sharp boundaries are critical
                # align_corners=False,
                recompute_scale_factor=False,
            )
            .squeeze(0)
            .squeeze(0)
        )
        mask_resized = mask_resized > 0.5  # Convert back to boolean
    else:
        mask_resized = None

    # --- Calculate 3D Vertices ---
    # Vertex position = origin + distance * ray_direction
    # Assuming origin is (0, 0, 0)
    distance_flat = distance_resized.reshape(-1, 1)  # (H*W, 1)
    rays_flat = rays_resized.reshape(-1, 3)  # (H*W, 3)
    vertices = distance_flat * rays_flat  # (H*W, 3)
    vertex_colors = rgb_resized.reshape(-1, 3)  # (H*W, 3)
    if mask_resized is not None:
        mask_resized = mask_resized.reshape(-1, )
        vertices = vertices[~mask_resized]
        vertex_colors = vertex_colors[~mask_resized]

    # downsample
    if dropout_pcd and vertices.shape[0] > 1_000_000:
        rdx = np.arange(vertices.shape[0])
        np.random.shuffle(rdx)
        rdx = rdx[:1_000_000]
        vertices = vertices[rdx]
        vertex_colors = vertex_colors[rdx]

    pcd = trimesh.PointCloud(vertices=vertices.cpu().numpy(), colors=vertex_colors.cpu().numpy())

    return pcd


def convert_rgbd2pcd_multi_scale_panorama(
        rgb: torch.Tensor,  # (H, W, 3) RGB image, values [0, 1]
        distance: torch.Tensor,  # (H, W) Distance map
        rays: torch.Tensor,  # (H, W, 3) Ray directions (unit vectors ideally)
        excluded_region_mask: Optional[torch.Tensor] = None,  # (H, W) Optional boolean mask
        device: Literal["cuda", "cpu"] = "cuda",  # Computation device
        depth_intervals=[0, 1, 2, 4, 8]
):
    """
    Converts panoramic RGBD data (image, distance, rays) into an Open3D mesh.

    Args:
        image: Input RGB image tensor (H, W, 3), uint8 or float [0, 255].
        distance: Input distance map tensor (H, W).
        rays: Input ray directions tensor (H, W, 3). Assumed to originate from (0,0,0).
        excluded_region_mask: Optional boolean mask tensor (H, W). True values indicate regions to potentially exclude.
        device: The torch device ('cuda' or 'cpu') to use for computations.

    Returns:
        An Open3D TriangleMesh object.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, "Image must be HxWx3"
    assert distance.ndim == 2, "Distance must be HxW"
    assert rays.ndim == 3 and rays.shape[2] == 3, "Rays must be HxWx3"
    assert (
            rgb.shape[:2] == distance.shape[:2] == rays.shape[:2]
    ), "Input shapes must match"

    mask = excluded_region_mask

    if mask is not None:
        assert (
                mask.ndim == 2 and mask.shape[:2] == rgb.shape[:2]
        ), "Mask shape must match"
        assert mask.dtype == torch.bool, "Mask must be a boolean tensor"

    rgb = rgb.to(device)
    distance = distance.to(device)
    rays = rays.to(device)
    if mask is not None:
        mask = mask.to(device)

    rgb_nchw = rgb.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    distance_nchw = distance.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    rays_nchw = rays.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    median_distance = torch.median(distance).item()

    total_points = []
    total_colors = []

    for i in tqdm(range(1, len(depth_intervals)), desc="Processing depth intervals"):
        if i == len(depth_intervals) - 1:
            interval_mask = distance_nchw > (median_distance * depth_intervals[i - 1])
        else:
            interval_mask = ((median_distance * depth_intervals[i - 1]) < distance_nchw) & (distance_nchw <= (median_distance * depth_intervals[i]))

        # pointclouds number ~ depth^2
        resize_scale = depth_intervals[i]
        if interval_mask.sum() == 0:
            continue

        rgb_resized = (
            F.interpolate(
                rgb_nchw,
                scale_factor=resize_scale,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
            .squeeze(0)
            .permute(1, 2, 0)
        )

        distance_resized = (
            F.interpolate(
                distance_nchw,
                scale_factor=resize_scale,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

        rays_resized_nchw = F.interpolate(
            rays_nchw,
            scale_factor=resize_scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )

        interval_mask_resized = F.interpolate(interval_mask.float(),
                                              scale_factor=resize_scale,
                                              mode="nearest",
                                              recompute_scale_factor=False).bool().squeeze(0).squeeze(0)

        # IMPORTANT: Renormalize ray directions after interpolation
        rays_resized = rays_resized_nchw.squeeze(0).permute(1, 2, 0)
        rays_norm = torch.linalg.norm(rays_resized, dim=-1, keepdim=True)
        rays_resized = rays_resized / (rays_norm + 1e-8)

        if mask is not None:
            mask_resized = (
                F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),  # Needs float for interpolation
                    scale_factor=resize_scale,
                    mode="nearest",  # Or 'nearest' if sharp boundaries are critical
                    # align_corners=False,
                    recompute_scale_factor=False,
                )
                .squeeze(0)
                .squeeze(0)
            )
            mask_resized = mask_resized > 0.5  # Convert back to boolean
        else:
            mask_resized = None

        mask_resized = mask_resized & interval_mask_resized

        # --- Calculate 3D Vertices ---
        # Vertex position = origin + distance * ray_direction
        # Assuming origin is (0, 0, 0)
        distance_flat = distance_resized.reshape(-1, 1)  # (H*W, 1)
        rays_flat = rays_resized.reshape(-1, 3)  # (H*W, 3)
        vertices = distance_flat * rays_flat  # (H*W, 3)
        vertex_colors = rgb_resized.reshape(-1, 3)  # (H*W, 3)
        if mask_resized is not None:
            mask_resized = mask_resized.reshape(-1, )
            vertices = vertices[~mask_resized]
            vertex_colors = vertex_colors[~mask_resized]

        total_points.append(vertices)
        total_colors.append(vertex_colors)
        print(f"Depth interval: {depth_intervals[i]}, Number of points: {vertices.shape[0]}")

    vertices = torch.cat(total_points, dim=0)
    vertex_colors = torch.cat(total_colors, dim=0)
    pcd = trimesh.PointCloud(vertices=vertices.cpu().numpy(), colors=vertex_colors.cpu().numpy())

    return pcd


def convert_rgbd2pcd_panorama_da360(depth, rgb, mask=None, dropout_pcd=True):
    h, w = depth.shape
    Theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2
    Theta = np.repeat(Theta, w, axis=1)
    Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
    Phi = -np.repeat(Phi, h, axis=0)

    X = depth * np.sin(Theta) * np.sin(Phi)
    Y = depth * np.cos(Theta)
    Z = depth * np.sin(Theta) * np.cos(Phi)

    if mask is None:
        X = X.flatten()
        Y = Y.flatten()
        Z = Z.flatten()
        R = rgb[:, :, 0].flatten()
        G = rgb[:, :, 1].flatten()
        B = rgb[:, :, 2].flatten()
    else:
        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]
        R = rgb[:, :, 0][mask]
        G = rgb[:, :, 1][mask]
        B = rgb[:, :, 2][mask]

    XYZ = np.stack([X, Y, Z], axis=1)
    RGB = np.stack([R, G, B], axis=1)

    # downsample
    if dropout_pcd and XYZ.shape[0] > 1_000_000:
        rdx = np.arange(XYZ.shape[0])
        np.random.shuffle(rdx)
        rdx = rdx[:1_000_000]
        XYZ = XYZ[rdx]
        RGB = RGB[rdx]

    pcd = trimesh.PointCloud(vertices=XYZ, colors=RGB)
    return pcd


def _generate_faces_numpy(H: int, W: int, mask: Optional[torch.Tensor]) -> np.ndarray:
    """
    Pure NumPy implementation, 2-3x faster than the PyTorch version.
    """
    # Precompute all vertex indices.
    idx = np.arange(H * W, dtype=np.int32).reshape(H, W)

    # Four corners of each quad, with horizontal wrapping.
    tl = idx[:-1, :]  # top-left
    tr = idx[:-1, :].copy()
    tr[:, :-1] = idx[:-1, 1:]
    tr[:, -1] = idx[:-1, 0]  # wrap
    bl = idx[1:, :]  # bottom-left
    br = idx[1:, :].copy()
    br[:, :-1] = idx[1:, 1:]
    br[:, -1] = idx[1:, 0]  # wrap

    # Apply mask.
    if mask is not None:
        mask_np = mask.cpu().numpy()
        # Check whether any of the four corners is masked.
        m_tl = mask_np[:-1, :]
        m_tr = np.roll(mask_np[:-1, :], -1, axis=1)
        m_bl = mask_np[1:, :]
        m_br = np.roll(mask_np[1:, :], -1, axis=1)

        keep = ~(m_tl | m_tr | m_bl | m_br)

        tl, tr, bl, br = tl[keep], tr[keep], bl[keep], br[keep]
    else:
        tl, tr, bl, br = tl.ravel(), tr.ravel(), bl.ravel(), br.ravel()

    # Build triangles, two per quad.
    n = len(tl)
    faces = np.empty((2 * n, 3), dtype=np.int32)
    faces[0::2] = np.column_stack([tl, tr, bl])
    faces[1::2] = np.column_stack([tr, br, bl])

    return faces


def convert_rgbd2mesh_panorama(
        rgb: torch.Tensor,  # (H, W, 3) RGB image, values [0, 1]
        distance: torch.Tensor,  # (H, W) Distance map
        rays: torch.Tensor,  # (H, W, 3) Ray directions (unit vectors ideally)
        excluded_region_mask: Optional[torch.Tensor] = None,  # (H, W) Optional boolean mask
        max_size: int = 4096,  # Max dimension for resizing
        device: Literal["cuda", "cpu"] = "cuda",  # Computation device
        connect_boundary_max_dist: Optional[float] = 0.5,  # Max distance to bridge boundary vertices
        connect_boundary_repeat_times: int = 2
) -> o3d.geometry.TriangleMesh:
    """
    Converts panoramic RGBD data (image, distance, rays) into an Open3D mesh.

    Args:
        image: Input RGB image tensor (H, W, 3), uint8 or float [0, 255].
        distance: Input distance map tensor (H, W).
        rays: Input ray directions tensor (H, W, 3). Assumed to originate from (0,0,0).
        excluded_region_mask: Optional boolean mask tensor (H, W). True values indicate regions to potentially exclude.
        max_size: Maximum size (height or width) to resize inputs to.
        device: The torch device ('cuda' or 'cpu') to use for computations.

    Returns:
        An Open3D TriangleMesh object.
    """
    """Optimized version: about 3-5x faster."""
    H, W = distance.shape
    scale = min(1.0, max_size / max(H, W))
    need_resize = scale < 1.0

    # ========== 1. Data preparation with asynchronous transfers. ==========
    rgb = rgb.to(device, non_blocking=True)
    distance = distance.to(device, non_blocking=True)
    rays = rays.to(device, non_blocking=True)
    if excluded_region_mask is not None:
        mask = excluded_region_mask.to(device, non_blocking=True)
    else:
        mask = None

    # ========== 2. Scaling. ==========
    if need_resize:
        H_new, W_new = int(H * scale), int(W * scale)

        # Combine rgb and rays to reduce interpolation calls.
        combined = torch.cat([rgb, rays], dim=-1).permute(2, 0, 1).unsqueeze(0)
        combined_resized = F.interpolate(combined, size=(H_new, W_new), mode='bilinear', align_corners=False)
        combined_resized = combined_resized.squeeze(0).permute(1, 2, 0)

        rgb_resized = combined_resized[..., :3]
        rays_resized = F.normalize(combined_resized[..., 3:], dim=-1)

        distance_resized = F.interpolate(
            distance[None, None], size=(H_new, W_new), mode='bilinear', align_corners=False
        ).squeeze()

        if mask is not None:
            mask_resized = F.max_pool2d(
                mask[None, None].float(),
                kernel_size=int(1 / scale),
                stride=int(1 / scale)
            ).squeeze().bool()
            # Ensure the size matches.
            if mask_resized.shape != (H_new, W_new):
                mask_resized = F.interpolate(
                    mask[None, None].float(), size=(H_new, W_new), mode='nearest'
                ).squeeze().bool()
        else:
            mask_resized = None
    else:
        H_new, W_new = H, W
        rgb_resized = rgb
        rays_resized = F.normalize(rays, dim=-1)
        distance_resized = distance
        mask_resized = mask

    # ========== 3. Compute vertices on GPU. ==========
    vertices = (distance_resized.unsqueeze(-1) * rays_resized).reshape(-1, 3)
    vertex_colors = rgb_resized.reshape(-1, 3)

    # Synchronize and convert to NumPy.
    torch.cuda.synchronize() if device == 'cuda' else None
    vertices_np = vertices.cpu().numpy().astype(np.float64)
    colors_np = vertex_colors.cpu().numpy().astype(np.float64)

    # ========== 4. Generate faces on CPU/NumPy, which is faster. ==========
    faces_np = _generate_faces_numpy(H_new, W_new, mask_resized)

    # ========== 5. Create Open3D Mesh. ==========
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
    mesh.triangles = o3d.utility.Vector3iVector(faces_np)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors_np)

    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()

    # ========== 6. Boundary handling. ==========
    if connect_boundary_max_dist is not None and connect_boundary_max_dist > 0:
        mesh = _fill_small_boundary_spikes(mesh, connect_boundary_max_dist, connect_boundary_repeat_times)
        # Recompute normals after potential modification, if mesh still valid
        if mesh.has_triangles() and mesh.has_vertices():
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()  # Also computes triangle normals if vertex normals are computed

    return mesh


def _fill_small_boundary_spikes(
        mesh: o3d.geometry.TriangleMesh,
        max_bridge_dist: float,
        repeat_times: int = 3
) -> o3d.geometry.TriangleMesh:
    print(f"\t - DEBUG: Filling small boundary spikes with max_bridge_dist: {max_bridge_dist} and repeat_times: {repeat_times}")
    for iteration in range(repeat_times):
        if not mesh.has_triangles() or not mesh.has_vertices():
            return mesh

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        # 1. Identify boundary edges
        edge_to_triangle_count = defaultdict(int)

        for tri_idx, tri in enumerate(triangles):
            for i in range(3):
                v1_idx, v2_idx = tri[i], tri[(i + 1) % 3]
                edge = tuple(sorted((v1_idx, v2_idx)))
                edge_to_triangle_count[edge] += 1

        boundary_edges = [edge for edge, count in edge_to_triangle_count.items() if count == 1]

        if not boundary_edges:
            return mesh

        # 2. Create an adjacency list for boundary vertices using only boundary edges
        boundary_adj = defaultdict(list)
        for v1_idx, v2_idx in boundary_edges:
            boundary_adj[v1_idx].append(v2_idx)
            boundary_adj[v2_idx].append(v1_idx)

        # 3. Process boundary vertices with new smooth filling algorithm
        new_triangles_list = []
        edge_added = defaultdict(bool)

        # print(f"DEBUG: Found {len(boundary_edges)} boundary edges.")
        # print(f"DEBUG: Max bridge distance set to: {max_bridge_dist}")

        new_triangles_added_count = 0

        for v_curr_idx, neighbors in boundary_adj.items():
            if len(neighbors) != 2:  # Only process vertices with exactly 2 boundary neighbors
                continue

            v_a_idx, v_b_idx = neighbors[0], neighbors[1]

            # Skip if these vertices already form a triangle
            potential_edge = tuple(sorted((v_a_idx, v_b_idx)))
            if edge_to_triangle_count[potential_edge] > 0 or edge_added[potential_edge]:
                continue

            # Calculate distances
            v_curr_coord = vertices[v_curr_idx]
            v_a_coord = vertices[v_a_idx]
            v_b_coord = vertices[v_b_idx]

            dist_a_b = np.linalg.norm(v_a_coord - v_b_coord)

            # Skip if distance exceeds threshold
            if dist_a_b > max_bridge_dist:
                continue

            # Create simple triangle (v_a, v_b, v_curr)
            new_triangles_list.append([v_a_idx, v_b_idx, v_curr_idx])
            new_triangles_added_count += 1
            edge_added[potential_edge] = True

            # Mark edges as processed
            edge_added[tuple(sorted((v_curr_idx, v_a_idx)))] = True
            edge_added[tuple(sorted((v_curr_idx, v_b_idx)))] = True

        # 4. Now process multi-step connections for better smoothing
        # First build boundary chains for multi-step connections
        boundary_loops = []
        visited_vertices = set()

        # Find boundary vertices with exactly 2 neighbors (part of continuous chains)
        chain_starts = [v for v in boundary_adj if len(boundary_adj[v]) == 2 and v not in visited_vertices]

        for start_vertex in chain_starts:
            if start_vertex in visited_vertices:
                continue

            chain = []
            curr_vertex = start_vertex

            # Follow the chain in one direction
            while curr_vertex not in visited_vertices:
                visited_vertices.add(curr_vertex)
                chain.append(curr_vertex)

                next_candidates = [n for n in boundary_adj[curr_vertex] if n not in visited_vertices]
                if not next_candidates:
                    break

                curr_vertex = next_candidates[0]

            if len(chain) >= 3:
                boundary_loops.append(chain)

        # print(f"DEBUG: Found {len(boundary_loops)} boundary chains for smoothing.")

        # Process each boundary chain for multi-step smoothing
        for chain in boundary_loops:
            chain_length = len(chain)

            # Skip very small chains
            if chain_length < 3:
                continue

            # Compute multi-step connections
            max_step = min(8, chain_length - 1)

            for i in range(chain_length):
                anchor_idx = chain[i]
                anchor_coord = vertices[anchor_idx]

                for step in range(3, max_step + 1):
                    if i + step >= chain_length:
                        break

                    far_idx = chain[i + step]
                    far_coord = vertices[far_idx]

                    # Check distance criteria
                    dist_anchor_far = np.linalg.norm(anchor_coord - far_coord)
                    if dist_anchor_far > max_bridge_dist * step:
                        continue

                    # Check if anchor and far are already connected
                    edge_anchor_far = tuple(sorted((anchor_idx, far_idx)))
                    if edge_to_triangle_count[edge_anchor_far] > 0 or edge_added[edge_anchor_far]:
                        continue

                    # Create fan triangles
                    fan_valid = True
                    fan_triangles = []

                    prev_mid_idx = anchor_idx

                    for j in range(1, step):
                        mid_idx = chain[i + j]

                        if prev_mid_idx != anchor_idx:
                            tri_edge1 = tuple(sorted((anchor_idx, mid_idx)))
                            tri_edge2 = tuple(sorted((prev_mid_idx, mid_idx)))

                            # Check if edges already exist (not created by our fan)
                            if (edge_to_triangle_count[tri_edge1] > 0 and not edge_added[tri_edge1]) or \
                                    (edge_to_triangle_count[tri_edge2] > 0 and not edge_added[tri_edge2]):
                                fan_valid = False
                                break

                            fan_triangles.append([anchor_idx, prev_mid_idx, mid_idx])

                        prev_mid_idx = mid_idx

                    # Add final triangle to connect to far_idx
                    if fan_valid:
                        fan_triangles.append([anchor_idx, prev_mid_idx, far_idx])

                    # Add all fan triangles if valid
                    if fan_valid and fan_triangles:
                        for triangle in fan_triangles:
                            v_a, v_b, v_c = triangle
                            edge_ab = tuple(sorted((v_a, v_b)))
                            edge_bc = tuple(sorted((v_b, v_c)))
                            edge_ac = tuple(sorted((v_a, v_c)))

                            new_triangles_list.append(triangle)
                            new_triangles_added_count += 1

                            edge_added[edge_ab] = True
                            edge_added[edge_bc] = True
                            edge_added[edge_ac] = True

                        # Once we've added a fan, move to the next anchor
                        break

        # print(f"DEBUG: Total new triangles added in iteration {iteration}: {new_triangles_added_count}")

        if new_triangles_added_count == 0:
            break

        # Update the mesh with new triangles
        if new_triangles_list:
            all_triangles_np = np.vstack((triangles, np.array(new_triangles_list, dtype=np.int32)))

            final_mesh = o3d.geometry.TriangleMesh()
            final_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            final_mesh.triangles = o3d.utility.Vector3iVector(all_triangles_np)

            if mesh.has_vertex_colors():
                final_mesh.vertex_colors = mesh.vertex_colors

            # Clean up the mesh
            final_mesh.remove_degenerate_triangles()
            final_mesh.remove_unreferenced_vertices()
            mesh = final_mesh

    return mesh


def get_view_point_from_panorama_point(global_pcd, w2c, K, image_h, image_w):
    # Get valid points corresponding to the current view.
    projected_uv, projected_depth = utils3d.numpy.project_cv(global_pcd.vertices, extrinsics=w2c, intrinsics=K)
    projection_valid_mask = (projected_depth > 0) & (projected_uv > 0).all(axis=-1) & (projected_uv < 1).all(axis=-1)
    projected_uv = projected_uv[projection_valid_mask]
    projected_uv[:, 0] = (projected_uv[:, 0] * image_w).round()
    projected_uv[:, 1] = (projected_uv[:, 1] * image_h).round()
    projected_uv = projected_uv.astype(np.int64)
    projected_uv[:, 0] = np.clip(projected_uv[:, 0], 0, image_w - 1)
    projected_uv[:, 1] = np.clip(projected_uv[:, 1], 0, image_h - 1)
    projected_depth = projected_depth[projection_valid_mask]

    projected_uv_1d = projected_uv[:, 1] * image_w + projected_uv[:, 0]  # Convert coordinates to 1D.
    original_indices = np.arange(projected_uv_1d.shape[0])
    # Double sort: for equal 1D coordinates, sort by depth ascending so only the nearest point is kept for each u, v.
    sorted_indices = np.lexsort((-projected_depth, projected_uv_1d))
    projected_uv_1d_sorted = projected_uv_1d[sorted_indices]
    sub_uv_1d = projected_uv_1d_sorted[:-1] - projected_uv_1d_sorted[1:]  # Offset subtraction; nonzero entries have the minimum depth.
    final_valid_indices = original_indices[sorted_indices][:-1][(sub_uv_1d != 0)]

    projected_points = global_pcd.vertices[projection_valid_mask][final_valid_indices]
    projected_uv = projected_uv[final_valid_indices]
    projected_colors = global_pcd.colors[projection_valid_mask][final_valid_indices, :3]

    return projected_points, projected_colors, projected_uv


def smooth_sky_depth_boundary(
        depth: torch.Tensor,
        sky_mask: torch.Tensor,
        transition_width: int = 50,
        depth_max: float = None,
        method: str = 'mean'  # 'mean', 'median', 'gaussian'
) -> torch.Tensor:
    """
    Smooth the depth transition between sky and foreground.

    Args:
        depth: Depth map, with sky regions already set to depth_max.
        sky_mask: Sky mask, where True indicates sky.
        transition_width: Transition-region width in pixels.
        depth_max: Sky depth value.
        method: Boundary diffusion method.
            - 'mean': Weighted mean diffusion, recommended.
            - 'median': Boundary median.
            - 'gaussian': Gaussian-blur diffusion.

    Returns:
        Depth map after boundary smoothing.
    """
    # 1. Dimension handling.
    original_dim = depth.dim()
    if original_dim == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
        sky_mask = sky_mask.unsqueeze(0).unsqueeze(0)
    elif original_dim == 3:
        depth = depth.unsqueeze(0)
        sky_mask = sky_mask.unsqueeze(0)

    device = depth.device
    dtype = depth.dtype

    if depth_max is None:
        depth_max = depth.max().item()

    sky_mask_np = sky_mask.squeeze().cpu().numpy().astype(np.uint8)
    depth_np = depth.squeeze().cpu().numpy()

    if sky_mask_np.sum() == 0 or sky_mask_np.sum() == sky_mask_np.size:
        return _restore_dim(depth, original_dim)

    # ---------------------------------------------------------
    # 2. Compute the distance from each sky pixel to the boundary.
    # ---------------------------------------------------------
    dist_to_foreground = cv2.distanceTransform(sky_mask_np, cv2.DIST_L2, 5)

    # ---------------------------------------------------------
    # 3. Get boundary depth values according to the selected method.
    # ---------------------------------------------------------
    sky_mask_tensor = sky_mask.bool()

    if method == 'mean':
        boundary_depth = _mean_diffusion(depth, sky_mask_tensor, transition_width)
    elif method == 'median':
        boundary_depth = _median_boundary(depth_np, sky_mask_np, depth_max, device, dtype, depth.shape)
    elif method == 'gaussian':
        boundary_depth = _gaussian_diffusion(depth, sky_mask_tensor, depth_max, transition_width)
    else:
        raise ValueError(f"Unknown method: {method}")

    # ---------------------------------------------------------
    # 4. Linear interpolation: use diffused depth at the boundary and depth_max deeper in the sky.
    # ---------------------------------------------------------
    t = np.clip(dist_to_foreground / max(transition_width, 1), 0, 1)
    t = t * t * (3 - 2 * t)  # smoothstep
    t = torch.from_numpy(t).to(device=device, dtype=dtype).view_as(depth)

    interpolated = boundary_depth * (1 - t) + depth_max * t

    # ---------------------------------------------------------
    # 5. Clamp the range and modify only sky regions.
    # ---------------------------------------------------------
    interpolated = torch.clamp(interpolated, max=depth_max)
    result = torch.where(sky_mask_tensor, interpolated, depth)

    return _restore_dim(result, original_dim)


def _mean_diffusion(depth: torch.Tensor, sky_mask: torch.Tensor, transition_width: int) -> torch.Tensor:
    """
    Weighted mean diffusion, recommended.
    Preserves local depth characteristics while avoiding extreme values.
    """
    device = depth.device
    dtype = depth.dtype

    # Valid-region mask, where non-sky equals 1.
    valid_mask = (~sky_mask).float()

    # Initialize sky regions to 0.
    flood_depth = depth.clone()
    flood_depth[sky_mask] = 0

    kernel_size = 31
    pad = kernel_size // 2
    max_iterations = (transition_width // pad) + 10

    for _ in range(max_iterations):
        # Compute the sum of neighboring depths.
        depth_sum = F.avg_pool2d(flood_depth, kernel_size, stride=1, padding=pad) * (kernel_size ** 2)
        # Compute the number of valid neighboring pixels.
        valid_count = F.avg_pool2d(valid_mask, kernel_size, stride=1, padding=pad) * (kernel_size ** 2)

        # Mean = total sum / valid count.
        mean_depth = depth_sum / (valid_count + 1e-8)

        # Update only sky pixels whose neighborhoods contain valid pixels.
        can_update = sky_mask & (valid_count > 0)

        # Check whether any pixels still need updates.
        newly_filled = can_update & (valid_mask < 0.5)
        if not newly_filled.any():
            break

        # Update.
        flood_depth = torch.where(can_update, mean_depth, flood_depth)
        valid_mask = torch.where(can_update, torch.ones_like(valid_mask), valid_mask)

    # Fallback: fill any remaining zero regions with the global foreground mean.
    still_zero = (flood_depth == 0) & sky_mask
    if still_zero.any():
        fg_mean = depth[~sky_mask].mean()
        flood_depth = torch.where(still_zero, fg_mean, flood_depth)

    return flood_depth


def _median_boundary(depth_np: np.ndarray, sky_mask_np: np.ndarray,
                     depth_max: float, device, dtype, shape) -> torch.Tensor:
    """
    Boundary median method.
    Extract boundary depths and use their median as the uniform transition value.
    """
    # Extract the boundary: non-sky pixels adjacent to sky.
    kernel = np.ones((3, 3), np.uint8)
    dilated_sky = cv2.dilate(sky_mask_np, kernel)
    boundary_mask = (dilated_sky == 1) & (sky_mask_np == 0)

    boundary_depths = depth_np[boundary_mask]

    if len(boundary_depths) > 0:
        # Filter out values that are already depth_max.
        valid_depths = boundary_depths[boundary_depths < depth_max * 0.99]
        if len(valid_depths) > 0:
            median_depth = np.median(valid_depths)
        else:
            median_depth = np.median(boundary_depths)
    else:
        median_depth = depth_np[sky_mask_np == 0].mean()

    return torch.full(shape, median_depth, device=device, dtype=dtype)


def _gaussian_diffusion(depth: torch.Tensor, sky_mask: torch.Tensor,
                        depth_max: float, transition_width: int) -> torch.Tensor:
    """
    Gaussian-blur diffusion.
    Blur foreground depth first, then diffuse into sky regions.
    """
    device = depth.device
    dtype = depth.dtype

    # Fill sky with the foreground mean to avoid depth_max affecting the blur.
    fg_mean = depth[~sky_mask].mean()
    temp_depth = depth.clone()
    temp_depth[sky_mask] = fg_mean

    # Gaussian blur.
    blur_size = max(transition_width // 2, 3)
    if blur_size % 2 == 0:
        blur_size += 1

    from torchvision.transforms.functional import gaussian_blur
    blurred = gaussian_blur(temp_depth, kernel_size=[blur_size, blur_size], sigma=[blur_size / 4])

    # Iterative diffusion using min-pooling on blurred values, after extreme values have been smoothed.
    flood_depth = blurred.clone()
    flood_depth[sky_mask] = float('inf')

    kernel_size = 31
    pad = kernel_size // 2

    for _ in range(transition_width // pad + 5):
        if not torch.isinf(flood_depth[sky_mask]).any():
            break
        next_step = -F.max_pool2d(-flood_depth, kernel_size, stride=1, padding=pad)
        is_inf = torch.isinf(flood_depth)
        flood_depth = torch.where(is_inf, next_step, flood_depth)

    flood_depth = torch.where(torch.isinf(flood_depth), fg_mean, flood_depth)

    return flood_depth


def _restore_dim(tensor: torch.Tensor, original_dim: int) -> torch.Tensor:
    if original_dim == 2:
        return tensor.squeeze(0).squeeze(0)
    elif original_dim == 3:
        return tensor.squeeze(0)
    return tensor


def erp_distance_ray_to_normal(distance_map, ray_directions,
                               smooth_sigma=0.0,
                               facing_camera=True):
    """
    Convert an ERP distance map and ray directions to a world-coordinate normal map.

    Args:
        distance_map: (H, W) Distance map.
        ray_directions: (H, W, 3) Ray direction for each pixel, as unit vectors emitted from the origin.
        smooth_sigma: Smoothing sigma. 0 means no smoothing.
        facing_camera: True makes normals face the camera (origin); False makes them face outward.

    Returns:
        normal_map: (H, W, 3) World-coordinate normals in the range [-1, 1].
        normal_rgb: (H, W, 3) RGB visualization in the range [0, 255].

    Coordinate system (OpenCV):
        X: right
        Y: down
        Z: forward
    """
    # 0. Optional: smooth the distance map.
    if smooth_sigma > 0:
        distance_map = cv2.GaussianBlur(
            distance_map.astype(np.float32), (0, 0), smooth_sigma
        )

    # 1. Compute 3D point coordinates: P = ray * distance.
    d = distance_map.astype(np.float64)
    points = ray_directions * d[..., np.newaxis]  # (H, W, 3)

    # Split x, y, and z.
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]

    # 2. Compute tangent vectors.
    # Horizontal direction Tu uses cyclic boundaries because ERP wraps left to right.
    x_right = np.roll(x, -1, axis=1)
    y_right = np.roll(y, -1, axis=1)
    z_right = np.roll(z, -1, axis=1)

    x_left = np.roll(x, 1, axis=1)
    y_left = np.roll(y, 1, axis=1)
    z_left = np.roll(z, 1, axis=1)

    Tu_x = x_right - x_left
    Tu_y = y_right - y_left
    Tu_z = z_right - z_left

    # Vertical direction Tv does not wrap, so boundaries need special handling.
    x_down = np.roll(x, -1, axis=0)
    y_down = np.roll(y, -1, axis=0)
    z_down = np.roll(z, -1, axis=0)

    x_up = np.roll(x, 1, axis=0)
    y_up = np.roll(y, 1, axis=0)
    z_up = np.roll(z, 1, axis=0)

    Tv_x = x_down - x_up
    Tv_y = y_down - y_up
    Tv_z = z_down - z_up

    # Handle top and bottom boundaries near the poles.
    # Top (v=0): one-sided difference.
    Tv_x[0, :] = x[1, :] - x[0, :]
    Tv_y[0, :] = y[1, :] - y[0, :]
    Tv_z[0, :] = z[1, :] - z[0, :]

    # Bottom (v=H-1): one-sided difference.
    Tv_x[-1, :] = x[-1, :] - x[-2, :]
    Tv_y[-1, :] = y[-1, :] - y[-2, :]
    Tv_z[-1, :] = z[-1, :] - z[-2, :]

    # 3. Compute normals with a cross product: N = Tu x Tv.
    normal_x = Tu_y * Tv_z - Tu_z * Tv_y
    normal_y = Tu_z * Tv_x - Tu_x * Tv_z
    normal_z = Tu_x * Tv_y - Tu_y * Tv_x

    # 4. Normalize.
    norm = np.sqrt(normal_x ** 2 + normal_y ** 2 + normal_z ** 2)
    norm = np.where(norm > 1e-10, norm, 1e-10)

    normal_x = normal_x / norm
    normal_y = normal_y / norm
    normal_z = normal_z / norm

    # 5. Ensure normals face the correct direction.
    if facing_camera:
        # Normals should face the camera (origin).
        # Check dot(normal, -ray) > 0, equivalent to dot(normal, ray) < 0.
        dot_product = (normal_x * ray_directions[..., 0] +
                       normal_y * ray_directions[..., 1] +
                       normal_z * ray_directions[..., 2])

        # If dot > 0, flip the normals.
        flip_mask = dot_product > 0
        normal_x = np.where(flip_mask, -normal_x, normal_x)
        normal_y = np.where(flip_mask, -normal_y, normal_y)
        normal_z = np.where(flip_mask, -normal_z, normal_z)

    # 6. Stack into (H, W, 3).
    normal_map = np.stack([normal_x, normal_y, normal_z], axis=-1)

    # 7. Convert to an RGB visualization.
    # [-1, 1] -> [0, 255]
    normal_rgb = ((normal_map + 1.0) / 2.0 * 255).astype(np.uint8)

    return normal_map, normal_rgb
