

https://github.com/user-attachments/assets/8ee9ab85-59e9-4990-82fd-a76a69b48735


https://github.com/user-attachments/assets/5dd4310e-cfb6-49e8-883b-0d5f641dc8d3

> [!WARNING]
> Warning, uses experimental package `comfy-env` to attempt a one click isolated install. Will download and use pixi package manager.

# ComfyUI-PanoPack

## Installation

Three options, in order of speed → reliability:

1. **ComfyUI Manager (recommended)** — search for `PanoPack` in the Manager and click Install from the highest version displayed. If that doesn't work, try nightly.
2. **Manager via Git URL** — in ComfyUI Manager: "Install via Git URL" with `https://github.com/PozzettiAndrea/ComfyUI-PanoPack.git`.
3. **Manual (most reliable)**:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/PozzettiAndrea/ComfyUI-PanoPack.git
   cd ComfyUI-PanoPack
   pip install -r requirements.txt --upgrade
   python install.py
   ```

> **Please report any problems** you hit during installation or use of my nodes — open a [Discussion](https://github.com/PozzettiAndrea/ComfyUI-PanoPack/discussions) or [Issue](https://github.com/PozzettiAndrea/ComfyUI-PanoPack/issues). Very grateful for your help! 🙏

---


<div align="center">
<a href="https://pozzettiandrea.github.io/ComfyUI-PanoPack/">
<img src="https://pozzettiandrea.github.io/ComfyUI-PanoPack/gallery-preview.png" alt="Workflow Test Gallery" width="800">
</a>
<br>
<b><a href="https://pozzettiandrea.github.io/ComfyUI-PanoPack/">View Live Test Gallery →</a></b>
</div>

Shared utilities for 360° equirectangular panoramas in ComfyUI: wrap / crop / shift, depth + normal merging, mesh / point-cloud / gaussian rendering, plane segmentation, and an interactive 360° panorama viewer. Nodes appear under the `PanoPack` category.

## Convention

`PANORAMA` is always:

- **2:1 aspect** (W = 2·H), enforced at the Wrap boundary.
- **Equirectangular** — longitude on x (yaw ∈ [−π, +π]), latitude on y
  (pitch ∈ [+π/2, −π/2]).

## License

MIT. See [LICENSE](LICENSE).
