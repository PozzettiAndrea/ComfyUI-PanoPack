"""ComfyUI-PanoPack prestartup script.

Runs before ComfyUI imports the pack's nodes. Three jobs:

  1. `setup_env()` — boot the Pixi-managed isolated env this pack runs
     under (handled by `comfy-env`).
  2. `copy_viewer()` — copy shared viewer widgets (text_report) from
     `comfy-3d-viewers` into our web/ directory.
  3. `copy_files()` — copy bundled test assets into ComfyUI's input dirs.
"""

from pathlib import Path

from comfy_env import copy_files, setup_env
from comfy_3d_viewers import copy_viewer

setup_env()

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

# Copy shared viewers from comfy-3d-viewers
viewers = [
    "text_report",
]
for viewer in viewers:
    try:
        copy_viewer(viewer, SCRIPT_DIR / "web")
    except Exception as e:
        import logging
        logging.getLogger("panopack").warning("Failed to copy viewer %s: %s", viewer, e)

# Copy bundled test images into ComfyUI/input/
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input", "**/*.png")
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input", "**/*.jpg")

# Copy 3D assets (.ply, .vtp) into ComfyUI/input/3d/
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input" / "3d", "**/*.ply")
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input" / "3d", "**/*.vtp")
