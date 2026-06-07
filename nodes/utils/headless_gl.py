"""Configure headless software OpenGL for VTK/pyvista off-screen rendering.

On a headless host with no GPU and no X display, VTK 9.4+ tries its render
backends in order (GLX -> EGL -> OSMesa). GLX needs an X server; OSMesa needs
libOSMesa, which Mesa removed upstream (and conda-forge `mesalib` dropped as of
25.1.9, Feb 2026). The only remaining software path is EGL backed by Mesa's
llvmpipe rasterizer. The env vars below steer VTK/Mesa onto that path.

All are set with os.environ.setdefault, so a real GPU or X display (local dev,
GPU CI) is never overridden -- the software path only kicks in when nothing
better is configured.

Runtime requirement: libEGL + the llvmpipe gallium driver must be present. These
come from the conda-forge `mesalib` package (see nodes/comfy-env.toml
[dependencies]).
"""

from __future__ import annotations

import os


def setup_headless_software_gl() -> None:
    """Point VTK/Mesa at surfaceless EGL + llvmpipe software rendering.

    Call once before creating a pyvista/VTK render window (pv.Plotter). Cheap and
    idempotent; safe to call from every render entry point.
    """
    # pyvista: never try to open an interactive window.
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    # Mesa: use the llvmpipe software rasterizer instead of probing for a GPU.
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")
    os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
    # EGL: get a GL context with no X server / display via the surfaceless platform.
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
