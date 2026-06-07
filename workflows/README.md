# PanoPack — example workflows

Five demonstration workflows, one per node (plus a combined pipeline).
Each loads `pano.png` — copy `assets/pano.png` into your ComfyUI
`input/` directory first:

```bash
cp ../assets/pano.png /path/to/ComfyUI/input/pano.png
```

Then drag any `.json` from this directory onto the ComfyUI canvas (or
`Workflow → Load`) to import.

## Workflows

| # | File | Demonstrates |
|---|---|---|
| 01 | `01_wrap_unwrap.json` | `PanoramaWrap` + `PanoramaUnwrap` — IMAGE ↔ PANORAMA type round-trip with 2:1 aspect validation. Output should be visually identical to the input. |
| 02 | `02_panorama_info.json` | `PanoramaInfo` — text + numeric quality scores (seam continuity at ±π, polar row uniformity). Useful diagnostic for generated panoramas. |
| 03 | `03_panorama_shift.json` | `PanoramaShift` — rotates the panorama 90° around the vertical axis. Demonstrates clean ±π seam wrap. |
| 04 | `04_panorama_viewer.json` | `PanoramaViewer` — saves the panorama to `output/` and surfaces it in ComfyUI's preview lane. (Future: drag-rotate 360° viewer; v0.1 ships a static preview.) |
| 05 | `05_full_pipeline.json` | Everything together: load, wrap, branch into Info + Shift(180°), unwrap, preview, viewer. |

## Convention

All workflows follow the canonical pattern:

```
LoadImage(pano.png) → PanoramaWrap → <PanoPack nodes> → PanoramaUnwrap → PreviewImage
```

`PanoramaWrap` validates the 2:1 aspect ratio at the type boundary and
errors clearly if the input isn't equirectangular.
