"""ComfyUI-PanoPack — toplevel entry point.

Re-exports NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS so ComfyUI's
custom-node loader picks the nodes up automatically. Also tells ComfyUI
where to find the JS assets for the eventual 360-viewer frontend.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
