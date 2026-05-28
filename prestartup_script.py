"""ComfyUI-PanoPack prestartup script.

Runs before ComfyUI imports the pack's nodes. Two jobs:

  1. `setup_env()` — boot the Pixi-managed isolated env this pack runs
     under (handled by `comfy-env`).
  2. `copy_files()` — copy bundled test assets into ComfyUI's `input/`
     directory so the example workflows in `workflows/` can find them
     via `LoadImage(filename="pano.png")` out of the box.
"""

from pathlib import Path

from comfy_env import copy_files, setup_env

setup_env()

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

# Copy bundled equirectangular test panorama into ComfyUI/input/ so the
# `workflows/*.json` examples load it without manual user setup.
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input", "**/*")
