"""WorldNav pipeline orchestrator.

Single entry point: `plan_trajectories(panorama_pil, **kwargs)` that runs the
full upstream HY-World WorldNav loop on a single panorama and returns the
generated camera trajectories as numpy arrays compatible with HYWM2's
`EXTRINSICS`/`INTRINSICS` CameraPack convention.

This file lives alongside the verbatim-vendored upstream sources in
`_vendor/worldgen/`. Upstream's `traj_generate.py` is kept untouched (still
runnable as a script). Here we re-import the helpers it uses and arrange
them into a single in-process function that:
  - replaces the vLLM/OpenAI client with our `QwenVLClient` (HF transformers
    Qwen3-VL-2B-Instruct inline),
  - replaces ZIM+Grounding-DINO sky segmentation with SAM3 + "sky." prompt,
  - writes intermediates to a temp directory (the upstream pipeline expects
    a `{scene_dir}/render_results/...` layout), and
  - reads the per-trajectory `camera.json` outputs back as tensors.

Models are loaded lazily on first call and cached at module level — the
caller (the WorldNavPlan ComfyUI node) lives inside a long-running
comfy-env worker subprocess, so we pay model-load cost exactly once.
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

# Upstream helpers (vendored verbatim except for the surgical import
# fixes documented in commit `8b6a5a2`).
from .src.general_utils import Timer, set_seed, rank0_log
from .src.panorama_utils import (
    convert_rgbd2pcd_panorama,
    convert_rgbd2mesh_panorama,
    spherical_uv_to_directions,
)
# navi_utils pulls in `recast` (RecastNavigation bindings), which only the
# trajectory-planning path needs. PanoramaBuildMesh + PanoramaBuildPointCloud
# don't, so we lazy-import these helpers inside the functions that use them —
# keeps `from .pipeline import build_mesh` working in environments without
# pyrecast.
from .src.vlm_utils import QwenVLClient, get_qwen_caption_format
import utils3d


# ----------------------------------------------------------------------
# Module-level model cache (one-time load inside the comfy-env worker)
# ----------------------------------------------------------------------

_MODELS: Optional["_Models"] = None


@dataclass
class _Models:
    """Lazy-loaded model bundle.

    All three models are externalized (sibling ComfyUI nodes hand us either
    weights paths or config dicts; we lazy-build on first use):
      - depth: arrives as IMAGE input from a MoGe panorama node
      - sam3:  built from SAM3_MODEL_CONFIG dict via vendored
               `sam3_pkg.get_or_build_model`
      - vlm:   built from QWEN_VL_MODEL config dict
    """

    device: torch.device
    sam3: Any = None                     # SAM3UnifiedModel (vendored sam3_pkg)
    vlm: Optional[QwenVLClient] = None


def _load_models(
    device: str = "cuda",
    sam3_model_config: Optional[dict] = None,
    vlm_config: Optional[dict] = None,
) -> _Models:
    """Build (or reuse cached) SAM3 + Qwen-VL instances inside this worker.

    Args:
        sam3_model_config: dict emitted by ComfyUI-SAM3's LoadSAM3Model.
            Required keys: checkpoint_path, bpe_path, precision, dtype, compile.
            Required for sky-mask + unique-object segmentation.
        vlm_config: dict emitted by DownloadAndLoadQwen3VL.
            Required keys: model_id, precision, attn_impl. None is allowed
            (scene_type classification + unique-object labeling get skipped).
    """
    global _MODELS
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    if _MODELS is None:
        _MODELS = _Models(device=dev)

    # SAM3 — required.
    if _MODELS.sam3 is None:
        if sam3_model_config is None:
            raise RuntimeError(
                "WorldNavPlan: sam3_model_config is None. Wire ComfyUI-SAM3's "
                "`(Down)Load SAM3 Model` node into WorldNavPlan.sam3_model_config."
            )
        print("[WorldNav] Building SAM3 via vendored sam3_pkg…", flush=True)
        from ..sam3_pkg import get_or_build_model
        _MODELS.sam3 = get_or_build_model(sam3_model_config)
        print("[WorldNav] SAM3 ready", flush=True)

    # Qwen-VL — optional.
    if _MODELS.vlm is None and vlm_config is not None:
        print(f"[WorldNav] Building QwenVLClient ({vlm_config.get('model_id')})…", flush=True)
        precision = vlm_config.get("precision", "auto")
        if precision == "auto":
            dtype = None  # let QwenVLClient pick based on device
        else:
            dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
            dtype = dtype_map.get(precision, None)
        _MODELS.vlm = QwenVLClient(
            model_id=vlm_config.get("model_id", "Qwen/Qwen3-VL-2B-Instruct"),
            device=dev,
            dtype=dtype,
            attn_impl=vlm_config.get("attn_impl", "sdpa"),
        )
        print("[WorldNav] QwenVLClient ready", flush=True)

    return _MODELS


# ----------------------------------------------------------------------
# Stage helpers
# ----------------------------------------------------------------------

def _classify_scene_type(vlm: QwenVLClient, panorama_pil: Image.Image) -> str:
    """Indoor / outdoor classification via Qwen3-VL.

    Mirrors traj_generate.py's main() block at lines ~252-272 byte-identically
    on the prompt side; only the client backend differs.
    """
    from .src.navi_utils import pil_image_to_base64
    base64_image = pil_image_to_base64(panorama_pil)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": get_qwen_caption_format("env_cls")},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
        ]},
    ]
    response = vlm.chat.completions.create(
        model=vlm.model_id, messages=messages,
        max_tokens=1024, temperature=0.0, seed=1024,
    )
    raw = response.choices[0].message.content.strip()
    clean = (raw.replace("[", "").replace("]", "").replace('"', "")
                .replace("'", "").replace("```json", "").replace("```", "")
                .strip().lower())
    if "outdoor" in clean:
        return "outdoor"
    return "indoor"


def _sam3_text_mask(
    panorama_pil: Image.Image, text_prompt: str, sam3_model,
    confidence_threshold: float = 0.3,
) -> np.ndarray:
    """SAM3 text-prompted segmentation using the vendored sam3_pkg API.

    `sam3_model` is the SAM3UnifiedModel instance returned by
    sam3_pkg.get_or_build_model. Returns a binary HxW uint8 mask (union of
    all instance masks for the prompt).
    """
    import comfy.model_management
    comfy.model_management.load_models_gpu([sam3_model])

    processor = sam3_model.processor
    if hasattr(processor, "sync_device_with_model"):
        processor.sync_device_with_model()

    processor.set_confidence_threshold(confidence_threshold)
    state = processor.set_image(panorama_pil)
    if text_prompt and text_prompt.strip():
        state = processor.set_text_prompt(text_prompt.strip(), state)

    # DIAG: snapshot of what SAM3 actually returned. The user is debugging
    # why every label produces an empty mask. Surface:
    #   - threshold being applied
    #   - the keys/attrs present on state
    #   - masks shape + dtype (or "missing")
    #   - score tensor stats if present (max, count above threshold)
    state_keys = list(state.keys()) if isinstance(state, dict) else [
        k for k in dir(state) if not k.startswith("_")
    ]
    keys_head = ", ".join(state_keys[:8]) + ("…" if len(state_keys) > 8 else "")

    def _maybe(key):
        if isinstance(state, dict):
            return state.get(key)
        return getattr(state, key, None)

    masks_t = _maybe("masks")
    scores_t = _maybe("scores")
    boxes_t = _maybe("boxes")
    logits_t = _maybe("masks_logits")
    masks_shape = (tuple(masks_t.shape) if hasattr(masks_t, "shape")
                   else (len(masks_t) if hasattr(masks_t, "__len__") else "None"))
    # masks_logits tells us whether SAM3 produced any RAW detections before
    # threshold-filtering. If logits shape is also (0, ...) the model genuinely
    # sees nothing in this image; if logits has N > 0 candidates but masks
    # came back (0, ...), then it's purely a confidence_threshold issue.
    if hasattr(logits_t, "shape"):
        try:
            l_arr = logits_t.detach().cpu().float().numpy() if hasattr(logits_t, "detach") else np.asarray(logits_t, dtype=np.float32)
            if l_arr.size > 0:
                logit_info = (f"masks_logits shape={tuple(logits_t.shape)}, "
                              f"raw max={float(l_arr.max()):.3f}, min={float(l_arr.min()):.3f}")
            else:
                logit_info = f"masks_logits shape={tuple(logits_t.shape)} (empty)"
        except Exception as _e:
            logit_info = f"masks_logits shape={tuple(logits_t.shape)} (summary failed: {_e})"
    else:
        logit_info = "masks_logits: None"
    if scores_t is not None and hasattr(scores_t, "__len__") and len(scores_t) > 0:
        try:
            s_arr = scores_t.detach().cpu().numpy() if hasattr(scores_t, "detach") else np.asarray(scores_t)
            s_max = float(s_arr.max())
            s_min = float(s_arr.min())
            n_pass = int((s_arr >= confidence_threshold).sum())
            score_info = (f"scores: n={s_arr.size}, min={s_min:.3f}, max={s_max:.3f}, "
                          f"≥thr({confidence_threshold:.2f})={n_pass}")
        except Exception as _e:
            score_info = f"scores present but failed to summarize: {_e}"
    else:
        score_info = "scores: None"
    print(f"[WorldNavPerceive]     SAM3 '{text_prompt}' thr={confidence_threshold:.2f} "
          f"→ state keys=[{keys_head}], masks shape={masks_shape}, {logit_info}, {score_info}, "
          f"boxes={('shape=' + str(tuple(boxes_t.shape))) if hasattr(boxes_t, 'shape') else 'None'}",
          flush=True)

    # Pull masks out of the predictor state — see ComfyUI-SAM3's
    # SAM3TextSegmentation._segment_grounding for the canonical extraction.
    H, W = panorama_pil.size[::-1]
    if masks_t is None or (hasattr(masks_t, "__len__") and len(masks_t) == 0):
        return np.zeros((H, W), dtype=np.uint8)

    masks_np = masks_t.detach().cpu().numpy() if hasattr(masks_t, "detach") else np.asarray(masks_t)
    # Could be [N, H, W] or [N, 1, H, W]; flatten the channel axis.
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0]
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    union = np.any(masks_np > 0.5, axis=0).astype(np.uint8)
    if not union.any():
        print(f"[WorldNavPerceive]     SAM3 returned {masks_np.shape[0]} mask(s) "
              f"but all-zero after >0.5 union (raw max={float(masks_np.max()):.3f})",
              flush=True)
    return union


# ----------------------------------------------------------------------
# Multi-view SAM3 — required because SAM3 was trained on rectilinear
# photos. Feeding a 1920×960 equirectangular panorama (squashed by
# Sam3Processor's resize to 1008×1008) produces zero detections for
# every prompt. The fix is to slice the panorama into rectilinear cube
# faces, run SAM3 per-face, and project the per-face masks back to
# equirect coords. This matches SAM3's training distribution.
# ----------------------------------------------------------------------

def _build_cube_face_views(
    panorama_pil: Image.Image,
    n_views: int = 6,
    view_resolution: int = 768,
    view_fov_deg: float = 90.0,
) -> tuple:
    """Build N rectilinear cube-face views from a panorama.

    Returns (view_pil_list, extrinsics, intrinsics) where:
      - view_pil_list: list of length n_views of PIL Images (RGB,
        view_resolution × view_resolution)
      - extrinsics:    (n_views, 4, 4) float32 world-to-camera matrices
      - intrinsics:    (n_views, 3, 3) float32 normalized intrinsics
                        (utils3d convention: K maps to [0, 1] uv)

    Forward directions are the six cube faces in this order:
    +X, -X, +Y, -Y, +Z, -Z. Uses the same look-at + pole-vertex up-swap
    pattern as WorldNavPanoramaSplit.
    """
    from ..moge_panorama import split_panorama_image_gpu

    cube_forwards = np.array([
        [+1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        [0.0, +1.0, 0.0], [0.0, -1.0, 0.0],
        [0.0, 0.0, +1.0], [0.0, 0.0, -1.0],
    ], dtype=np.float32)[:n_views]

    eye = np.zeros(3, dtype=np.float32)
    UP_DEFAULT = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    UP_FALLBACK = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    PARALLEL_COS = 0.999

    n = len(cube_forwards)
    extrinsics = np.empty((n, 4, 4), dtype=np.float32)
    for i, fwd in enumerate(cube_forwards):
        up = UP_FALLBACK if abs(float(fwd @ UP_DEFAULT)) > PARALLEL_COS else UP_DEFAULT
        extrinsics[i] = utils3d.np.extrinsics_look_at(eye, fwd, up).astype(np.float32)

    intrinsics_one = utils3d.np.intrinsics_from_fov(
        fov_x=np.deg2rad(view_fov_deg), fov_y=np.deg2rad(view_fov_deg),
    )
    intrinsics = np.stack([intrinsics_one] * n, axis=0).astype(np.float32)

    arr = np.asarray(panorama_pil.convert("RGB"))
    view_arrays = split_panorama_image_gpu(arr, extrinsics, intrinsics, view_resolution)
    view_pil_list = [Image.fromarray(a) for a in view_arrays]
    return view_pil_list, extrinsics, intrinsics


def _project_view_mask_to_equirect(
    view_mask: np.ndarray,
    extrinsics_one: np.ndarray,
    intrinsics_one: np.ndarray,
    eq_height: int,
    eq_width: int,
) -> np.ndarray:
    """Inverse-warp a rectilinear-view mask back onto the equirect.

    For each equirect pixel, compute its 3D ray direction, project
    through the view camera (extrinsics + intrinsics), and sample the
    view mask at the resulting (u, v). Out-of-view or behind-camera
    pixels are set to 0.

    view_mask: (fh, fw) bool/uint8.
    Returns: (eq_height, eq_width) uint8.
    """
    fh, fw = view_mask.shape
    uv_eq = utils3d.np.uv_map((eq_height, eq_width)).astype(np.float32)  # (H, W, 2)
    dirs = spherical_uv_to_directions(uv_eq).astype(np.float32)           # (H, W, 3)

    R = extrinsics_one[:3, :3]
    t = extrinsics_one[:3, 3]
    p_cam = dirs @ R.T + t                                                # (H, W, 3)
    p_proj = p_cam @ intrinsics_one.T                                     # (H, W, 3)
    safe_w = np.where(p_proj[..., 2:3] > 1e-12,
                      p_proj[..., 2:3], np.ones_like(p_proj[..., 2:3]))
    view_uv = p_proj[..., :2] / safe_w                                    # (H, W, 2)
    depth = p_cam[..., 2]
    valid = (
        (depth > 0)
        & (view_uv[..., 0] >= 0.0) & (view_uv[..., 0] <= 1.0)
        & (view_uv[..., 1] >= 0.0) & (view_uv[..., 1] <= 1.0)
    )

    # Sample view_mask at (uv_x * fw, uv_y * fh). Nearest-pixel index
    # (clamp to in-bounds for safety; valid mask handles edge cases).
    pix_x = np.clip((view_uv[..., 0] * fw).astype(np.int32), 0, fw - 1)
    pix_y = np.clip((view_uv[..., 1] * fh).astype(np.int32), 0, fh - 1)
    sampled = view_mask[pix_y, pix_x] > 0
    return (valid & sampled).astype(np.uint8)


def _sam3_text_mask_multiview(
    panorama_pil: Image.Image,
    text_prompt: str,
    sam3_model,
    *,
    confidence_threshold: float = 0.2,
    view_cache: Optional[dict] = None,
) -> np.ndarray:
    """SAM3 text-prompted segmentation via multi-view aggregation.

    Builds (or reuses) 6 rectilinear cube-face views of the panorama,
    runs SAM3 on each with the given text prompt, and projects the
    per-view masks back to equirect coords (union across views).
    Returns a binary HxW uint8 mask.

    `view_cache` is a dict that callers can construct once outside a
    per-label loop to avoid re-encoding the 6 view images every time.
    Mutate-in-place:
      - First call (cache empty): builds views, encodes each into a SAM3
        state, fills the cache.
      - Subsequent calls: reuses cached image-encoded SAM3 states.
    """
    import comfy.model_management
    comfy.model_management.load_models_gpu([sam3_model])

    processor = sam3_model.processor
    if hasattr(processor, "sync_device_with_model"):
        processor.sync_device_with_model()
    processor.set_confidence_threshold(confidence_threshold)

    H = panorama_pil.size[1]
    W = panorama_pil.size[0]

    # Build / look up the per-view SAM3 cache.
    if view_cache is None or "states" not in view_cache:
        view_pil_list, extrinsics, intrinsics = _build_cube_face_views(panorama_pil)
        if view_cache is None:
            view_cache = {}
        view_cache["views"] = view_pil_list
        view_cache["extrinsics"] = extrinsics
        view_cache["intrinsics"] = intrinsics
        view_cache["states"] = []
        for v_pil in view_pil_list:
            view_cache["states"].append(processor.set_image(v_pil))
        print(f"[WorldNavPerceive]     SAM3 multiview cache built: "
              f"{len(view_pil_list)} views @ {view_pil_list[0].size}", flush=True)

    views = view_cache["views"]
    extrinsics = view_cache["extrinsics"]
    intrinsics = view_cache["intrinsics"]
    states = view_cache["states"]

    # Confidence threshold can change between labels — update for each call.
    processor.set_confidence_threshold(confidence_threshold)

    eq_union = np.zeros((H, W), dtype=np.uint8)
    per_view_counts = []
    for i, state in enumerate(states):
        # Reuses backbone_out cached in state; only re-encodes the text.
        state = processor.set_text_prompt(text_prompt.strip(), state)
        # In-place mutation, but reassign for clarity.
        view_cache["states"][i] = state

        masks_t = state.get("masks") if isinstance(state, dict) else getattr(state, "masks", None)
        if masks_t is None or (hasattr(masks_t, "__len__") and len(masks_t) == 0):
            per_view_counts.append(0)
            continue
        masks_np = masks_t.detach().cpu().numpy() if hasattr(masks_t, "detach") else np.asarray(masks_t)
        if masks_np.ndim == 4 and masks_np.shape[1] == 1:
            masks_np = masks_np[:, 0]
        if masks_np.ndim == 2:
            masks_np = masks_np[None, ...]
        view_union = np.any(masks_np > 0.5, axis=0).astype(np.uint8)
        if not view_union.any():
            per_view_counts.append(0)
            continue
        eq_mask = _project_view_mask_to_equirect(
            view_union, extrinsics[i], intrinsics[i], H, W,
        )
        eq_union |= eq_mask
        per_view_counts.append(int(masks_np.shape[0]))

    print(f"[WorldNavPerceive]     SAM3 '{text_prompt}' multiview thr={confidence_threshold:.2f} "
          f"→ per-view mask counts {per_view_counts}, equirect union "
          f"area={int(eq_union.sum())}/{H * W}", flush=True)
    return eq_union


def _default_args() -> SimpleNamespace:
    """Replicate traj_generate.py's argparse defaults."""
    return SimpleNamespace(
        fov_x=120, fov_y=90, seed=1024,
        split_view_num=3, splitted_resolution=480, nframe=21,
        distance_threshold=0.1, obs_iteration_limit=3,
        rotation_deg=120, rotation_up=45, up_right=60, obs_decay=2 / 3,
        contract=8.0, skip_exist=False,
        apply_nav_traj=True, wonder_topk=3, recon_topk=5,
        move_dist=8.0, radius_threshold=4.0,
        min_angle_threshold=40.0, traj_sim_threshold=0.7, traj_sim_threshold_recon=0.7,
        apply_up_route=True, apply_recon_iteration=True,
        eloop_dist=0.25, force_vlm=False,
        cellSize=0.1, cellHeight=0.1, agentHeight=0.2,
        agentRadius=0.1, agentMaxClimb=0.1, maxSlope=30.0,
        roof_height_threshold=0.1,
        node_rank=0, node_size=1,
    )


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------

@dataclass
class TrajectoryBundle:
    extrinsics: np.ndarray         # [K * N, 4, 4] world-to-camera
    intrinsics: np.ndarray         # [K * N, 3, 3] pinhole K
    num_trajectories: int          # K
    num_frames: int                # N (per trajectory)
    metadata: List[dict] = field(default_factory=list)
    # Pass-through of the input trimesh. process_single_scene mutates its
    # own o3d copy (Z-up -> Y-up for Recast -> Z-up via save_artifacts),
    # never touches this object. Returning it lets downstream nodes verify
    # the mesh is byte-equal to what came in (the user can wire mesh_out
    # alongside the original mesh socket and diff vertex arrays).
    mesh: Any = None


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
        print(f"[WorldNav] panorama {full_img.size} > 1920 tall → resizing to 3840×1920", flush=True)
        full_img = full_img.resize((3840, 1920), resample=Image.Resampling.BICUBIC)
    return full_img


# ----------------------------------------------------------------------
# Stage 1 — Perception (Qwen3-VL + SAM3)
# ----------------------------------------------------------------------

def _label_objects_via_vlm(vlm: QwenVLClient, panorama_pil: Image.Image) -> List[str]:
    """Ask Qwen3-VL for a comma- or JSON-list of objects visible in the panorama.

    Uses upstream's `get_navigation_instruction` prompt verbatim. Returns a
    deduplicated list of short object names ("chair", "door", "lamp", ...).
    """
    from .src.navi_utils import (
        get_navigation_instruction, deduplicate_ordered, pil_image_to_base64,
    )
    base64_image = pil_image_to_base64(panorama_pil)
    instruction = get_navigation_instruction(force_vlm=False)
    messages = [
        {"role": "system", "content": "You are a robot navigation assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
        ]},
    ]
    # DIAG: show the prompt being sent so we can sanity-check it matches
    # upstream's get_navigation_instruction and isn't being mangled by some
    # prompt-template substitution.
    print(f"[WorldNavPerceive]   VLM system='{messages[0]['content']}'",
          flush=True)
    instr_head = instruction.replace('\n', ' ')[:280]
    print(f"[WorldNavPerceive]   VLM user prompt (len={len(instruction)}): "
          f"{instr_head}{'…' if len(instruction) > 280 else ''}",
          flush=True)
    print(f"[WorldNavPerceive]   VLM image: PIL {panorama_pil.size} mode={panorama_pil.mode}",
          flush=True)

    # 256 tokens fits ~30-50 short comma-separated object names. The previous
    # 1024 ceiling let Qwen3-VL fall into a repetition loop (latest log:
    # `len=2576`, same 16 labels repeated 10+ times, even produced a
    # truncated 'cur' at the tail). 256 cuts the runaway cleanly; the parser
    # already handles short responses.
    response = vlm.chat.completions.create(
        model=vlm.model_id, messages=messages,
        max_tokens=256, temperature=0.0, seed=1024,
    )
    raw = response.choices[0].message.content.strip()
    # DIAG: show what came back BEFORE any parsing — the user wants to see
    # the raw Qwen3-VL response so they can debug parser issues separately
    # from model-quality issues.
    raw_head = raw.replace('\n', ' ')[:500]
    print(f"[WorldNavPerceive]   VLM raw response (len={len(raw)}): "
          f"{raw_head}{'…' if len(raw) > 500 else ''}", flush=True)

    # Upstream's parser: strip brackets/quotes/code-fences, split on commas.
    clean = (raw.replace("[", "").replace("]", "").replace('"', "")
                .replace("'", "").replace("```json", "").replace("```", "")
                .replace("-", "_"))
    names = deduplicate_ordered([s.strip() for s in clean.split(",") if s.strip()])
    n_before_filter = len(names)
    # Drop unhelpfully-long phrases + the sun.
    names = [n for n in names if len(n.split()) < 8 and n.lower() not in ("sun",)]
    if len(names) != n_before_filter:
        print(f"[WorldNavPerceive]   filtered {n_before_filter} → {len(names)} "
              f"names (dropped >7-word phrases and 'sun')", flush=True)
    return names


def _project_mask_to_3d(
    mask_np: np.ndarray,       # (H, W) bool
    depth_np: np.ndarray,       # (H, W) float
    rays_np: np.ndarray,        # (H, W, 3) unit dirs
) -> tuple:
    """Back-project a mask's pixels to 3D and return (center_point_3d, scale_3d, depth_distance).

    center_point_3d: median 3D position of in-mask pixels.
    scale_3d:        rough size proxy — RMS distance from center to in-mask points.
    depth_distance:  median depth at in-mask pixels.
    """
    ys, xs = np.where(mask_np)
    if len(ys) == 0:
        return np.zeros(3, dtype=np.float32), 0.0, 0.0
    d = depth_np[ys, xs]                            # (K,)
    r = rays_np[ys, xs]                             # (K, 3)
    pts = r * d[:, None]                            # (K, 3) world-space 3D points (Z-up)
    center = np.median(pts, axis=0).astype(np.float32)
    scale = float(np.sqrt(np.mean(np.sum((pts - center) ** 2, axis=-1))))
    depth_dist = float(np.median(d))
    return center, scale, depth_dist


def _unique_object_loop(
    panorama_pil: Image.Image,
    depth_np: np.ndarray,
    rays_np: np.ndarray,
    vlm: QwenVLClient,
    sam3,                                # vendored SAM3UnifiedModel
    *,
    # SAM3's score in this wrapper is sigmoid(class_logit) × sigmoid(presence).
    # On panoramic input the presence gate collapses (model trained on
    # rectilinear photos), so 0.4 filters everything. The ComfyUI-SAM3
    # reference (segmentation.py:57) uses 0.2 as its documented default
    # with the tooltip "Lower threshold (0.2) works better with SAM3's
    # presence scoring". Match that here.
    confidence_threshold: float = 0.2,
    max_mask_top_frac: float = 0.4,      # drop masks whose top is in upper 40%
    min_mask_bottom_frac: float = 0.6,   # drop masks whose bottom is in lower 40%
    max_mask_width_frac: float = 0.75,   # drop masks wider than 75% of panorama
) -> List[dict]:
    """Run upstream's unique-object loop in our worker.

    Per-object: SAM3 text-segment → filter (too-high/too-low/too-wide) → back-project.
    Returns list of dicts with {mask, label, score, center_point_3d, scale_3d,
    depth_distance, area} ready for process_single_scene.
    """
    print("[WorldNavPerceive]   step 1: Qwen3-VL labeling objects via get_navigation_instruction…", flush=True)
    names = _label_objects_via_vlm(vlm, panorama_pil)
    print(f"[WorldNavPerceive]   VLM returned {len(names)} unique objects: {names}", flush=True)
    if not names:
        return []

    H = panorama_pil.size[1]
    W = panorama_pil.size[0]
    seg: List[dict] = []
    # Shared view cache for multi-view SAM3: built lazily on first call,
    # reused across all labels. Six SAM3 image encodings up front, then
    # cheap text-only re-encoding per label.
    view_cache: dict = {}
    for obj_name in names:
        print(f"[WorldNavPerceive]   step 2: SAM3 segment '{obj_name}' (multiview)…", flush=True)
        mask_np = _sam3_text_mask_multiview(
            panorama_pil, obj_name, sam3,
            confidence_threshold=confidence_threshold,
            view_cache=view_cache,
        ).astype(bool)
        if not mask_np.any():
            print(f"[WorldNavPerceive]     no mask returned for '{obj_name}', skipping", flush=True)
            continue
        ys, xs = np.where(mask_np)
        top, bot = ys.min(), ys.max()
        left, right = xs.min(), xs.max()
        if top > H * max_mask_top_frac:
            print(f"[WorldNavPerceive]     '{obj_name}' mask too low ({top}/{H}), skipping", flush=True)
            continue
        if bot < H * min_mask_bottom_frac:
            print(f"[WorldNavPerceive]     '{obj_name}' mask too high ({bot}/{H}), skipping", flush=True)
            continue
        if (right - left) > W * max_mask_width_frac:
            print(f"[WorldNavPerceive]     '{obj_name}' mask too wide ({right - left}/{W}), skipping", flush=True)
            continue
        center, scale, depth_dist = _project_mask_to_3d(mask_np, depth_np, rays_np)
        seg.append({
            "mask":            mask_np,
            "label":           obj_name,
            "score":           1.0,           # vendored SAM3 doesn't surface per-mask score here
            "center_point_3d": center,
            "scale_3d":        scale,
            "depth_distance":  depth_dist,
            "area":            int(mask_np.sum()),
        })
        print(f"[WorldNavPerceive]     '{obj_name}' → 3D center {center.tolist()}, "
              f"scale={scale:.2f}, depth={depth_dist:.2f}, area={int(mask_np.sum())}", flush=True)
    print(f"[WorldNavPerceive]   unique-object loop produced {len(seg)} objects after filtering", flush=True)
    return seg


def perceive_panorama(
    panorama_pil: Image.Image,
    sam3_model_config: dict,
    *,
    depth_np: Optional[np.ndarray] = None,
    vlm_config: Optional[dict] = None,
    scene_type: str = "auto",
    seed: int = 1024,
    device: str = "cuda",
    use_unique_object_loop: bool = False,
    sam3_confidence_threshold: float = 0.2,
) -> dict:
    """Stage 1: turn a panorama into structured perception (no geometry).

    Args:
        panorama_pil: equirect RGB panorama (≤1920 tall expected).
        sam3_model_config: dict from `(Down)Load SAM3 Model`.
        depth_np: equirect distance map [H, W] float. Required for the
            unique-object loop (used to back-project mask centroids to 3D).
            Optional otherwise.
        vlm_config: dict from `(Down)Load Qwen3-VL Model`. Required for
            scene_type='auto' and for the unique-object loop.
        scene_type: "auto" runs Qwen-VL; "indoor"/"outdoor" skips that step.
        use_unique_object_loop: if True, run Qwen3-VL → SAM3 unique-object
            discovery. Requires depth_np + vlm_config. Adds 10-30s per call.

    Returns a dict with:
      - `panorama_pil`: normalized PIL image
      - `scene_type`: "indoor" / "outdoor"
      - `is_outdoor`: bool
      - `sky_mask_np`: [H, W] bool (all-False on indoor)
      - `segmentation_data`: list-of-dicts. Empty when use_unique_object_loop is False.
    """
    pbar = _make_pbar(100)
    def _stage(pct, msg):
        print(f"[WorldNavPerceive] [{pct:3d}%] {msg}", flush=True)
        pbar.update_absolute(pct, 100)

    set_seed(seed)
    timer = Timer()

    _stage(2, f"received panorama PIL {panorama_pil.size}, scene_type='{scene_type}', seed={seed}")
    full_img = _normalize_panorama(panorama_pil)
    H, W = full_img.size[1], full_img.size[0]
    print(f"[WorldNavPerceive] normalized panorama: {full_img.size}", flush=True)

    _stage(10, "loading SAM3 + (optional) Qwen3-VL inside this worker…")
    models = _load_models(
        device=device,
        sam3_model_config=sam3_model_config,
        vlm_config=vlm_config,
    )
    dev = models.device
    print(f"[WorldNavPerceive] device={dev}, sam3=ready, "
          f"vlm={'ready' if models.vlm is not None else 'none'}", flush=True)

    # --- Scene classification ---
    if scene_type == "auto":
        if models.vlm is None:
            raise RuntimeError(
                "WorldNavPerceive: scene_type='auto' needs a VLM. "
                "Either wire DownloadAndLoadQwen3VL to vlm, or set scene_type='indoor'/'outdoor'."
            )
        _stage(30, "Qwen3-VL: classifying scene as indoor/outdoor…")
        with timer.track("VLM scene classify"):
            scene_type = _classify_scene_type(models.vlm, full_img)
    else:
        _stage(30, f"scene_type fixed at '{scene_type}', skipping VLM classify")
    is_outdoor = scene_type == "outdoor"
    print(f"[WorldNavPerceive] scene_type={scene_type} (is_outdoor={is_outdoor})", flush=True)

    # --- Sky mask (SAM3, outdoor only) ---
    if is_outdoor:
        _stage(60, "SAM3: text-prompted sky segmentation ('sky.')")
        with timer.track("Sky mask"):
            sky_mask_np = _sam3_text_mask(full_img, "sky.", models.sam3).astype(bool)
            frac = float(sky_mask_np.mean())
            print(f"[WorldNavPerceive] sky pixels: {frac * 100:.1f}% of panorama", flush=True)
            if frac > 0.9:
                print("[WorldNavPerceive]   sky_frac > 0.9, zeroing (treat all as non-sky)", flush=True)
                sky_mask_np = np.zeros_like(sky_mask_np)
    else:
        _stage(60, "indoor scene → empty sky mask (all False)")
        sky_mask_np = np.zeros((H, W), dtype=bool)

    # --- Unique-object segmentation loop (gated on toggle) ---
    segmentation_data: List[dict] = []
    if use_unique_object_loop:
        if depth_np is None:
            raise RuntimeError(
                "WorldNavPerceive: use_unique_object_loop=True needs a `depth` input "
                "(equirect distance map). Wire MoGe2Inference's depth_raw into the "
                "depth socket, or turn the toggle off."
            )
        if models.vlm is None:
            raise RuntimeError(
                "WorldNavPerceive: use_unique_object_loop=True needs a VLM. "
                "Wire DownloadAndLoadQwen3VL into vlm, or turn the toggle off."
            )
        _stage(80, "running unique-object loop (Qwen3-VL labels → SAM3 segment → 3D back-project)")
        with timer.track("Unique-object SAM3 loop"):
            from utils3d.numpy.maps import uv_map as _uv_map
            dH, dW = depth_np.shape[:2]
            uv = _uv_map(dH, dW)
            rays_np = spherical_uv_to_directions(uv).astype(np.float32)
            segmentation_data = _unique_object_loop(
                full_img, depth_np.astype(np.float32), rays_np, models.vlm, models.sam3,
                confidence_threshold=sam3_confidence_threshold,
            )
    else:
        _stage(80, "unique-object loop OFF (toggle disabled) → segmentation_data=[]")
        print("[WorldNavPerceive]   without unique-object SAM3 loop, downstream trajectories "
              "will be exploration-only (8 directions). Flip the toggle ON in WorldNavPerceive "
              "to enable target/surround/reconstruct paths.", flush=True)

    _stage(100, f"perceive DONE: scene_type={scene_type}, sky_frac={float(sky_mask_np.mean()):.3f}, "
                f"objects={len(segmentation_data)}")
    return {
        "panorama_pil": full_img,
        "scene_type": scene_type,
        "is_outdoor": is_outdoor,
        "sky_mask_np": sky_mask_np,
        "segmentation_data": segmentation_data,
    }


# ----------------------------------------------------------------------
# Stage 2 — Mesh build (pure geometry, zero models)
# ----------------------------------------------------------------------

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
    """Stage 2: depth post-process + RGBD → 3D mesh. No models.

    Returns a live `trimesh.Trimesh` with WorldNav-specific metadata attached
    on `mesh.metadata`:
      - `global_median_depth`: float (used by trajectory planner)
      - `scene_type`: str pass-through ("indoor"/"outdoor")
      - `is_outdoor`: bool

    Compatible with ComfyUI-GeometryPack's TRIMESH socket convention — wire
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
            print(f"[WorldNavBuildMesh]   resizing sky mask {sky_mask_for_depth.shape} → {edge_mask.shape}",
                  flush=True)
            sky_mask_for_depth = F.interpolate(
                sky_mask_for_depth[None, None].float(),
                size=edge_mask.shape, mode="nearest",
            )[0, 0].bool()
        full_mask = (sky_mask_for_depth | edge_mask | invalid_init.to(edge_mask.device)).to(dev)
        unmasked = full_depth["distance"][~full_mask]
        # Far-depth clip at the `clip_quantile` percentile. This caps hallucinated
        # far depth (windows / open doors) but ALSO clamps real room corners — the
        # farthest points in a room — flattening trihedral wall corners into a
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

    # --- RGBD → mesh (resized to mesh resolution 960×1920) ---
    _stage(45, "convert_rgbd2mesh_panorama (resizing to 960×1920 for mesh build)")
    with timer.track("RGBD → mesh"):
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
        print(f"[WorldNavBuildMesh]   resized to {mesh_h}×{mesh_w}; "
              f"sky pixels (mesh-res): {sky_mask_resized.sum().item()}", flush=True)

        rgb_t = torch.as_tensor(np.array(img_resized) / 255.0, dtype=torch.float32)
        mesh = convert_rgbd2mesh_panorama(
            rgb=rgb_t,
            distance=depth_resized.to(dev),
            rays=rays_resized.to(dev),
            excluded_region_mask=sky_mask_resized.to(dev),
            device=dev,
        )

    # --- Convert open3d mesh → trimesh.Trimesh (preserve vertex colors) ---
    _stage(85, "converting open3d → trimesh.Trimesh")
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
                  f"!= vertices len {vertices.shape[0]} — dropping colors",
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
    # No hidden state on `mesh.metadata` — downstream sockets are explicit.
    return tm_mesh, global_median_depth, scene_type

    # (Unreachable — kept so the legacy dict shape is visible in the file.)
    _legacy = {
        "vertices": vertices,
        "triangles": triangles,
        "global_median_depth": global_median_depth,
        "scene_type": scene_type,
        "is_outdoor": is_outdoor,
    }


# ----------------------------------------------------------------------
# Stage 3 — Plan trajectories on the mesh (recast + Dijkstra + B-spline)
# ----------------------------------------------------------------------

def plan_from_mesh(
    mesh,                                          # trimesh.Trimesh
    *,
    global_median_depth: float,
    scene_type: str = "indoor",
    segmentation_data: Optional[List[dict]] = None,
    num_frames: int = 21,
    seed: int = 1024,
    render_width: int = 832,
    render_height: int = 480,
    render_fov_x_deg: float = 120.0,
    render_fov_y_deg: float = 90.0,
    work_dir: Optional[str] = None,
    cleanup: bool = True,
) -> TrajectoryBundle:
    """Stage 3: navmesh build + Dijkstra + spline-fit trajectory poses.

    Args:
        mesh: trimesh.Trimesh from `build_mesh`. Pure geometry, no hidden
            metadata — pipeline-specific scalars come in as separate args.
        global_median_depth: scalar from `build_mesh`. Calibrates navmesh
            agent dims to scene scale.
        scene_type: "indoor" or "outdoor" pass-through from `perceive`.
            Drives the outdoor `contract` step + sky-mask logic in
            `process_single_scene`.
        segmentation_data: list-of-dicts per detected object (output of
            `perceive_panorama`). Empty/None → only exploration trajectories.
    """
    import open3d as o3d
    import trimesh as _tm

    pbar = _make_pbar(100)
    def _stage(pct, msg):
        print(f"[WorldNavPlanTraj] [{pct:3d}%] {msg}", flush=True)
        pbar.update_absolute(pct, 100)

    set_seed(seed)
    timer = Timer()

    if not isinstance(mesh, _tm.Trimesh):
        raise TypeError(f"plan_from_mesh: `mesh` must be a trimesh.Trimesh, got {type(mesh).__name__}")
    is_outdoor = scene_type == "outdoor"
    _stage(2, f"received trimesh with {len(mesh.vertices)} verts, {len(mesh.faces)} faces; "
              f"scene_type={scene_type}, global_median_depth={global_median_depth:.3f}, "
              f"segmentation_data={0 if not segmentation_data else len(segmentation_data)} entries")

    # --- Convert trimesh → open3d TriangleMesh (process_single_scene wants o3d) ---
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
    print(f"[WorldNavPlanTraj] converted trimesh → o3d.TriangleMesh", flush=True)

    # --- Scratch dir ---
    tmp_ctx = tempfile.TemporaryDirectory(prefix="worldnav_") if work_dir is None else None
    scene_dir = work_dir or tmp_ctx.name
    print(f"[WorldNavPlanTraj] scratch dir: {scene_dir}", flush=True)

    try:
        os.makedirs(f"{scene_dir}/render_results", exist_ok=True)
        # process_single_scene needs meta_info.json
        with open(f"{scene_dir}/meta_info.json", "w") as f:
            json.dump({"scene_type": scene_type}, f)

        # --- NavMesh + trajectory planning ---
        _stage(10, "process_single_scene: recast navmesh build + Dijkstra + spline-fit")
        if not segmentation_data:
            print("[WorldNavPlanTraj]   no segmentation_data → exploration-only trajectories "
                  "(target/surround/reconstruct paths skipped).", flush=True)
        from .src.navi_utils import process_single_scene
        args = _default_args()
        args.target_path = scene_dir
        with timer.track("NavMesh + trajectories"):
            process_single_scene(
                scene_dir=scene_dir,
                scene_name=os.path.basename(scene_dir),
                mesh=o3d_mesh,
                args=args,
                segmentation_data=segmentation_data,
                global_median_depth=global_median_depth,
                is_outdoor=is_outdoor,
                timer=timer,
            )

        # --- Convert paths.json → camera poses via process_trajectories ---
        # process_single_scene writes per-task paths.json files (raw 3D waypoint
        # lists) but does NOT emit camera.json. Upstream's traj_generate.py
        # closes that gap by calling process_trajectories(paths, D, X) after
        # process_single_scene; we replicate that step here, in-memory.
        from .src.navi_utils import process_trajectories

        _stage(85, f"reading paths.json under {scene_dir}/navmesh/*/paths.json")
        path_jsons = sorted(glob(f"{scene_dir}/navmesh/*/paths.json"))
        print(f"[WorldNavPlanTraj]   found {len(path_jsons)} paths.json files", flush=True)
        if not path_jsons:
            raise RuntimeError(
                f"WorldNavPlanTraj: process_single_scene produced no paths.json in {scene_dir}. "
                "Check stderr above for navmesh build errors."
            )

        # Build a constant pinhole K (pixel-space) for all output frames.
        # The HYWM2 CameraPack convention is pixel-space K. We use a sensible
        # default — the user can swap to whatever their downstream node wants
        # via re-wiring the intrinsics (or we can expose fov_x/y/w/h widgets
        # on WorldNavPlanTrajectories — follow-up).
        out_w, out_h = render_width, render_height
        fov_x_rad = np.deg2rad(render_fov_x_deg)
        fov_y_rad = np.deg2rad(render_fov_y_deg)
        fx_px = 0.5 * out_w / np.tan(fov_x_rad / 2)
        fy_px = 0.5 * out_h / np.tan(fov_y_rad / 2)
        K = np.array([
            [fx_px, 0.0,    out_w / 2.0],
            [0.0,   fy_px,  out_h / 2.0],
            [0.0,   0.0,    1.0],
        ], dtype=np.float32)
        print(f"[WorldNavPlanTraj]   intrinsics: {out_w}x{out_h} pixel-space K "
              f"(fov_x={render_fov_x_deg}°, fov_y={render_fov_y_deg}°, "
              f"fx={fx_px:.1f}, fy={fy_px:.1f})", flush=True)

        move_threshold_base = 8.0 * global_median_depth  # upstream args.move_dist=8.0

        all_extr, all_intr, metadata = [], [], []
        _diag_printed_paths = False
        for pj in path_jsons:
            task_name = os.path.basename(os.path.dirname(pj))
            with open(pj) as f:
                raw_paths = json.load(f)
            paths = [np.asarray(p, dtype=np.float32) for p in raw_paths if p is not None and len(p) >= 4]
            if not paths:
                print(f"[WorldNavPlanTraj]     {task_name}: 0 paths, skipping", flush=True)
                continue

            # ---- DIAG: raw paths from paths.json (Y-up Recast frame).
            # First task only so we don't drown the log. ----
            if not _diag_printed_paths:
                try:
                    p0 = paths[0]
                    print(f"[DIAG plan_from_mesh] task='{task_name}' "
                          f"len(paths)={len(paths)} paths[0].shape={p0.shape}",
                          flush=True)
                    print(f"[DIAG plan_from_mesh]   paths[0] axis-min: "
                          f"{p0.min(0).tolist()}", flush=True)
                    print(f"[DIAG plan_from_mesh]   paths[0] axis-max: "
                          f"{p0.max(0).tolist()}", flush=True)
                    print(f"[DIAG plan_from_mesh]   paths[0][ 0]: {p0[0].tolist()}", flush=True)
                    print(f"[DIAG plan_from_mesh]   paths[0][-1]: {p0[-1].tolist()}", flush=True)
                    print(f"[DIAG plan_from_mesh]   paths[0] start->end "
                          f"|delta|={float(np.linalg.norm(p0[-1] - p0[0])):.4f}", flush=True)
                except Exception as _e:
                    print(f"[DIAG plan_from_mesh] (paths print failed: {_e})", flush=True)
                _diag_printed_paths = True

            # Clamp Z >= 0 BEFORE per-task trims + process_trajectories.
            # Upstream traj_generate.py:986: `current_path_flat[:, 2] =
            # np.maximum(current_path_flat[:, 2], 0)`. Keeps the agent on or
            # above the floor plane (Z-up). Without this, a navmesh vertex
            # that ended up slightly below origin (depth noise + the
            # AABB-fallback floor distance) drags the spline tangent
            # downward at frame 0, making the camera basis ill-conditioned
            # (cross(near-vertical-tangent, [0,0,1]) ≈ 0).
            for p in paths:
                np.maximum(p[:, 2], 0.0, out=p[:, 2])

            # Upstream's per-task tweaks.
            if task_name == "exploration":
                move_thr = move_threshold_base
                smoothing = 0.5
                paths = [p[:-1] for p in paths if len(p) > 4]  # trim final point
            else:
                move_thr = move_threshold_base * 1.5
                smoothing = 0.2 if task_name == "reconstruct" else 0.5

            print(f"[WorldNavPlanTraj]     {task_name}: {len(paths)} paths → process_trajectories "
                  f"(D={move_thr:.2f}, X={num_frames}, smoothing={smoothing})", flush=True)

            try:
                c2ws = process_trajectories(
                    paths, move_thr, num_frames,
                    smoothing=smoothing, world_up=np.array([0, 0, 1]),
                    look_at_target=None,
                    is_recon=(task_name == "reconstruct"),
                )
            except Exception as e:
                print(f"[WorldNavPlanTraj]     {task_name}: process_trajectories FAILED ({e}), skipping",
                      flush=True)
                continue

            c2ws = np.asarray(c2ws, dtype=np.float32)
            if c2ws.size == 0 or c2ws.ndim != 4:
                print(f"[WorldNavPlanTraj]     {task_name}: empty c2ws (shape {c2ws.shape}), skipping",
                      flush=True)
                continue
            N, X = c2ws.shape[:2]

            # Pin frame 0 of each trajectory to world origin (the panorama
            # capture point). Upstream traj_generate.py:1083: `c2ws[0, :3, 3]
            # = 0`. The B-spline can overshoot a few cm from the first
            # waypoint depending on the smoothing factor; zeroing the first
            # translation guarantees the anchor frame samples the panorama
            # at the actual capture point — anything else produces a
            # parallax mismatch between anchor and traj[0].
            c2ws[:, 0, :3, 3] = 0.0

            # CameraPack convention is world-to-camera. Invert each pose.
            w2cs = np.linalg.inv(c2ws.reshape(-1, 4, 4)).reshape(N, X, 4, 4).astype(np.float32)

            print(f"[WorldNavPlanTraj]     {task_name}: built {N} trajectories × {X} frames",
                  flush=True)

            # ---- DIAG: pose[0]'s camera center. Z is the height axis.
            try:
                if N > 0 and X > 0:
                    Rk = w2cs[0, 0, :3, :3]
                    tk = w2cs[0, 0, :3, 3]
                    cam_center_zup = -Rk.T @ tk
                    print(f"[DIAG plan_from_mesh] task='{task_name}' "
                          f"traj 0 frame 0 cam_center={cam_center_zup.tolist()}",
                          flush=True)
            except Exception as _e:
                print(f"[DIAG plan_from_mesh] (cam_center print failed: {_e})",
                      flush=True)

            # ---- DIAG: pose0/mid/last for trajectory 0, both c2w + w2c ----
            try:
                if N > 0 and X > 0:
                    mid_idx = X // 2
                    print(f"[DIAG plan_from_mesh] task='{task_name}' traj 0 (of {N}): "
                          f"c2w positions (XYZ):", flush=True)
                    print(f"[DIAG plan_from_mesh]   c2w[ 0 ].t = {c2ws[0, 0, :3, 3].tolist()}", flush=True)
                    print(f"[DIAG plan_from_mesh]   c2w[{mid_idx:>3}].t = "
                          f"{c2ws[0, mid_idx, :3, 3].tolist()}", flush=True)
                    print(f"[DIAG plan_from_mesh]   c2w[{X - 1:>3}].t = "
                          f"{c2ws[0, X - 1, :3, 3].tolist()}", flush=True)
                    delta = c2ws[0, X - 1, :3, 3] - c2ws[0, 0, :3, 3]
                    print(f"[DIAG plan_from_mesh]   c2w start->end |delta|="
                          f"{float(np.linalg.norm(delta)):.4f}", flush=True)
                    # And derived camera centers from w2c (should match c2w.t):
                    for k, lab in [(0, " 0"), (mid_idx, f"{mid_idx:>3}"), (X - 1, f"{X - 1:>3}")]:
                        Rk = w2cs[0, k, :3, :3]
                        tk = w2cs[0, k, :3, 3]
                        cam_center = -Rk.T @ tk
                        print(f"[DIAG plan_from_mesh]   w2c[{lab}] -> cam_center="
                              f"{cam_center.tolist()}", flush=True)
                    # Camera basis vectors per pose (each column of c2w[:3, :3]):
                    # [:3, 0] = right (X-in-world)
                    # [:3, 1] = up    (Y-in-world)
                    # [:3, 2] = forward(Z-in-world, look direction in OpenCV)
                    for k, lab in [(0, " 0"), (mid_idx, f"{mid_idx:>3}"), (X - 1, f"{X - 1:>3}")]:
                        rgt = c2ws[0, k, :3, 0]
                        upv = c2ws[0, k, :3, 1]
                        fwd = c2ws[0, k, :3, 2]
                        print(f"[DIAG plan_from_mesh]   c2w[{lab}] basis: "
                              f"right={[round(v, 3) for v in rgt.tolist()]} "
                              f"up={[round(v, 3) for v in upv.tolist()]} "
                              f"fwd={[round(v, 3) for v in fwd.tolist()]}", flush=True)
            except Exception as _e:
                print(f"[DIAG plan_from_mesh] (pose-diag print failed: {_e})", flush=True)

            for i in range(N):
                all_extr.append(w2cs[i])                                       # (X, 4, 4)
                all_intr.append(np.broadcast_to(K, (X, 3, 3)).copy())          # (X, 3, 3)
                metadata.append({
                    "type":     task_name,
                    "index":    i,
                    "n_frames": int(X),
                    "width":    int(out_w),
                    "height":   int(out_h),
                })

        if not all_extr:
            raise RuntimeError(
                f"WorldNavPlanTraj: process_trajectories returned no usable poses for any task in {scene_dir}."
            )

        extrinsics = np.concatenate(all_extr, axis=0)
        intrinsics = np.concatenate(all_intr, axis=0)

        _stage(100, f"plan_from_mesh DONE — {len(metadata)} trajectories × "
                    f"~{extrinsics.shape[0] // max(len(metadata), 1)} frames; total {extrinsics.shape[0]} poses")
        return TrajectoryBundle(
            mesh=mesh,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            num_trajectories=len(metadata),
            num_frames=num_frames,
            metadata=metadata,
        )

    finally:
        if cleanup and tmp_ctx is not None:
            try:
                tmp_ctx.cleanup()
            except Exception as e:
                print(f"[WorldNavPlanTraj] tempdir cleanup warning: {e}", flush=True)


# ----------------------------------------------------------------------
# Backward-compat wrapper: the original monolithic plan_trajectories
# ----------------------------------------------------------------------

def plan_trajectories(
    panorama_pil: Image.Image,
    depth_np: np.ndarray,
    sam3_model_config: dict,
    valid_mask_np: Optional[np.ndarray] = None,
    *,
    vlm_config: Optional[dict] = None,
    scene_type: str = "auto",
    num_frames: int = 21,
    seed: int = 1024,
    device: str = "cuda",
    work_dir: Optional[str] = None,
    cleanup: bool = True,
) -> TrajectoryBundle:
    """Backward-compat monolith — calls perceive → build_mesh → plan_from_mesh."""
    perc = perceive_panorama(
        panorama_pil,
        sam3_model_config,
        vlm_config=vlm_config,
        scene_type=scene_type,
        seed=seed,
        device=device,
    )
    mesh, global_median_depth, scene_type = build_mesh(
        perc["panorama_pil"],
        depth_np,
        sky_mask_np=perc["sky_mask_np"],
        valid_mask_np=valid_mask_np,
        scene_type=perc["scene_type"],
        seed=seed,
        device=device,
    )
    return plan_from_mesh(
        mesh,
        global_median_depth=global_median_depth,
        scene_type=scene_type,
        segmentation_data=perc["segmentation_data"],
        num_frames=num_frames,
        seed=seed,
        work_dir=work_dir,
        cleanup=cleanup,
    )


# ----------------------------------------------------------------------
# Old monolithic body — preserved below the wrapper as a reference. The
# real entry points above already cover everything it does.
# ----------------------------------------------------------------------
