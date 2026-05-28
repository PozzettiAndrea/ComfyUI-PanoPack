# ComfyUI-PanoPack

Shared utilities for 360° equirectangular panoramas in ComfyUI. Provides:

- **`PANORAMA` custom socket type** — a tagged 2:1 equirect tensor, distinct
  from generic `IMAGE` so accidental wires to perspective-image consumers
  fail at graph-link time instead of silently producing wrong geometry.
- **Wrap / Unwrap** converters — bidirectional `IMAGE ↔ PANORAMA` with
  aspect-ratio validation and an optional snap-pad.
- **Panorama Info** — reports resolution, aspect, dynamic range, and a
  border-continuity score (how seamless is the ±π wraparound).
- **Panorama Shift** — rotate the equirect by a yaw angle in degrees,
  preserving the 2:1 aspect (pure horizontal wrap, no resampling
  artifacts).
- **Panorama Viewer** (output node) — saves the equirect to disk and
  shows it in ComfyUI's preview lane; web-side enhancement to a
  drag-rotate 360° viewer is planned for a follow-up.

Designed to slot in alongside the depth/geometry packs in your environment
(ComfyUI-MoGe2, ComfyUI-Sharp, ComfyUI-WorldNav, ComfyUI-HYWM2) — anywhere
a panorama is produced or consumed, this pack gives you a clean typed
handoff plus diagnostic / manipulation tools.

## Install

Clone into your ComfyUI `custom_nodes/` directory:

```bash
cd /path/to/ComfyUI/custom_nodes/
git clone https://github.com/PozzettiAndrea/ComfyUI-PanoPack.git
```

Restart ComfyUI; nodes appear under the `PanoPack` category.

## Convention

`PANORAMA` is always:

- **2:1 aspect** (W = 2·H), enforced at the Wrap boundary.
- **Equirectangular** — longitude on x (yaw ∈ [−π, +π]), latitude on y
  (pitch ∈ [+π/2, −π/2]).
- **Float [0, 1]** RGB or single-channel depth (the type doesn't care
  what's in the channels; just the projection convention).

Conversion convention follows MoGe / WorldStereo / HY-World (`world up = +Z`
for WorldNav-side consumers; `world up = +Y` for HYWM2-side consumers —
see each downstream pack's docs).

## License

MIT. See [LICENSE](LICENSE).
