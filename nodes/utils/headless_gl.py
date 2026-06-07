"""Configure headless software OpenGL for VTK/pyvista off-screen rendering.

On a headless host with no GPU and no X display, VTK 9.4+ tries its render
backends in order (GLX -> EGL -> OSMesa). GLX needs an X server; OSMesa needs
libOSMesa, which Mesa removed upstream (and conda-forge `mesalib` dropped as of
25.1.9, Feb 2026). The remaining software path is EGL backed by Mesa's llvmpipe
rasterizer.

The catch: VTK's vtkEGLRenderWindow enumerates EGL *devices* via
eglQueryDevicesEXT and picks one. That only returns Mesa's software (llvmpipe)
device if the GLVND dispatcher (libEGL.so.1) can find Mesa's EGL *vendor*
library + its ICD JSON. On a runner whose system Mesa never installed (the CI
apt step fails on a stale `libegl1-mesa` package name), the system GLVND has no
vendor registered, so eglQueryDevicesEXT returns 0 devices and VTK reports
"Could not initialize a device."

So in addition to the software-rendering env vars, this points GLVND at the
conda-forge Mesa EGL vendor (shipped by `mesalib` / `libglvnd`, see
nodes/comfy-env.toml [dependencies]) via __EGL_VENDOR_LIBRARY_DIRS /
__EGL_VENDOR_LIBRARY_FILENAMES. With the vendor found, Mesa's llvmpipe software
device becomes enumerable and VTK's EGL backend initializes.

All env vars use setdefault, so a real GPU or X display (local dev, GPU CI) is
never overridden -- the software path only kicks in when nothing better is set.
"""

from __future__ import annotations

import os
import sys


def _p(msg: str) -> None:
    print(f"[headless_gl] {msg}", file=sys.stderr, flush=True)


def setup_headless_software_gl() -> None:
    """Steer VTK/Mesa onto EGL + llvmpipe software rendering, headless.

    Call once before creating a pyvista/VTK render window (pv.Plotter). Cheap and
    idempotent; safe to call from every render entry point.
    """
    import glob

    # pyvista: never try to open an interactive window.
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    # Mesa: use the llvmpipe software rasterizer instead of probing for a GPU.
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")
    os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
    # EGL: get a GL context with no X server / display via the surfaceless platform.
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")

    # --- Point GLVND at conda's Mesa EGL vendor so a software device enumerates. ---
    prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    libdir = os.path.join(prefix, "lib")
    vendor_dir = os.path.join(prefix, "share", "glvnd", "egl_vendor.d")

    if os.path.isdir(vendor_dir):
        os.environ.setdefault("__EGL_VENDOR_LIBRARY_DIRS", vendor_dir)

    # Belt-and-suspenders: pin the Mesa EGL vendor lib directly if we can find it.
    if "__EGL_VENDOR_LIBRARY_FILENAMES" not in os.environ:
        cands = sorted(glob.glob(os.path.join(libdir, "libEGL_mesa.so*")))
        if cands:
            os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = cands[0]

    # Make sure conda's GL/EGL libs win the loader search (mesalib's libEGL_mesa
    # pulls in libgallium/llvmpipe from the same prefix).
    ldp = os.environ.get("LD_LIBRARY_PATH", "")
    if libdir not in ldp.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = libdir + (os.pathsep + ldp if ldp else "")

    # --- Diagnostics: dump what EGL/GL bits actually exist in the prefix. ---
    # (One-time per process; the next CI log tells us the real prefix layout.)
    if not os.environ.get("_HEADLESS_GL_DIAG_DONE"):
        os.environ["_HEADLESS_GL_DIAG_DONE"] = "1"
        try:
            egl_libs = sorted(
                os.path.basename(p) for p in glob.glob(os.path.join(libdir, "libEGL*"))
            )
            gl_libs = sorted(
                os.path.basename(p)
                for p in glob.glob(os.path.join(libdir, "libGL*"))
                + glob.glob(os.path.join(libdir, "libgallium*"))
                + glob.glob(os.path.join(libdir, "libOSMesa*"))
            )
            vendors = (
                sorted(os.listdir(vendor_dir)) if os.path.isdir(vendor_dir) else "MISSING"
            )
            _p(f"prefix={prefix}")
            _p(f"libEGL*: {egl_libs}")
            _p(f"libGL*/gallium/OSMesa: {gl_libs}")
            _p(f"egl_vendor.d: {vendors}")
            _p(
                "env: "
                f"__EGL_VENDOR_LIBRARY_DIRS={os.environ.get('__EGL_VENDOR_LIBRARY_DIRS')} "
                f"__EGL_VENDOR_LIBRARY_FILENAMES={os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES')} "
                f"EGL_PLATFORM={os.environ.get('EGL_PLATFORM')} "
                f"LIBGL_ALWAYS_SOFTWARE={os.environ.get('LIBGL_ALWAYS_SOFTWARE')}"
            )
        except Exception as e:
            _p(f"diagnostics failed: {e!r}")
