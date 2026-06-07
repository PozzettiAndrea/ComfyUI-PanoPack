"""Configure headless software OpenGL for VTK/pyvista off-screen rendering.

On a headless host with no GPU and no X display, VTK 9.4+ tries its render
backends in order: GLX (needs an X server), EGL (needs an enumerable GPU/EGL
device), then OSMesa (pure software, no display/device). On CI only the OSMesa
path can work -- and the CI logs confirm VTK reaches it ("vtkOSOpenGLRenderWindow
... libOSMesa not found"); it only failed for lack of the library.

So the fix is simply to provide libOSMesa.so and point VTK straight at the OSMesa
window. libOSMesa (with llvmpipe software GL compiled in) ships in conda-forge
`mesalib` up to 25.0.5 (later versions dropped it); see nodes/comfy-env.toml,
which pins that range and keeps vtk on conda so its RPATH finds the lib.

This module sets, before any pv.Plotter is created:
  * VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow -- skip GLX/EGL, go to OSMesa.
  * PYVISTA_OFF_SCREEN=true                            -- never open a window.
  * $CONDA_PREFIX/lib prepended to LD_LIBRARY_PATH     -- belt-and-suspenders for
    VTK's runtime dlopen of libOSMesa.so (RPATH already covers conda vtk).
All via setdefault, so a real GPU / X display (local dev, GPU CI) is untouched.
"""

from __future__ import annotations

import os
import sys


def _p(msg: str) -> None:
    print(f"[headless_gl] {msg}", file=sys.stderr, flush=True)


def setup_headless_software_gl() -> None:
    """Steer VTK onto its OSMesa (software, off-screen) render window.

    Call once before creating a pyvista/VTK render window (pv.Plotter). Cheap and
    idempotent; safe to call from every render entry point.
    """
    import glob

    # pyvista: never try to open an interactive window.
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    # VTK: use the OSMesa off-screen window class directly (skip the GLX/EGL probes
    # that abort the worker on a headless, GPU-less, displayless runner).
    os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkOSOpenGLRenderWindow")
    # Mesa: software rasterizer (harmless if OSMesa already implies llvmpipe).
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")

    # Make conda's libOSMesa.so findable for VTK's runtime dlopen. conda vtk's
    # RPATH already includes $CONDA_PREFIX/lib; this is defensive.
    prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    libdir = os.path.join(prefix, "lib")
    ldp = os.environ.get("LD_LIBRARY_PATH", "")
    if os.path.isdir(libdir) and libdir not in ldp.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = libdir + (os.pathsep + ldp if ldp else "")

    # One-time diagnostic: confirm libOSMesa is actually present in the env.
    if not os.environ.get("_HEADLESS_GL_DIAG_DONE"):
        os.environ["_HEADLESS_GL_DIAG_DONE"] = "1"
        try:
            # libOSMesa.so on Linux (lib/); osmesa.dll on Windows (Library/bin/).
            osmesa = sorted(os.path.basename(p) for p in (
                glob.glob(os.path.join(libdir, "libOSMesa*"))
                + glob.glob(os.path.join(prefix, "Library", "bin", "osmesa*"))
                + glob.glob(os.path.join(prefix, "Library", "bin", "OSMesa*"))
            ))
            _p(f"prefix={prefix}")
            _p(f"OSMesa: {osmesa or 'MISSING'}")
            _p(
                "env: "
                f"VTK_DEFAULT_OPENGL_WINDOW={os.environ.get('VTK_DEFAULT_OPENGL_WINDOW')} "
                f"PYVISTA_OFF_SCREEN={os.environ.get('PYVISTA_OFF_SCREEN')}"
            )
        except Exception as e:
            _p(f"diagnostics failed: {e!r}")
