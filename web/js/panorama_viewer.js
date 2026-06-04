/**
 * ComfyUI PanoPack - Interactive 360 Panorama Viewer
 * Three.js equirectangular viewer with mouse drag, arrow keys / WASD,
 * preset view buttons (top-right overlay), and live UV coordinate display.
 */

import { app } from "../../../scripts/app.js";
import * as THREE from "./lib/three.module.min.js";

const PRESET_VIEWS = [
    { label: "Front",  lon: 0,    lat: 0   },
    { label: "Back",   lon: 180,  lat: 0   },
    { label: "Left",   lon: -90,  lat: 0   },
    { label: "Right",  lon: 90,   lat: 0   },
    { label: "Top",    lon: 0,    lat: 90  },
    { label: "Bottom", lon: 0,    lat: -90 },
];

const KEY_ROTATE_SPEED = 2;
const VIEWER_MIN_HEIGHT = 300;
const INFO_BAR_HEIGHT = 22;

class PanoViewer {
    constructor(container, infoLabel) {
        this.container = container;
        this.infoLabel = infoLabel;
        this.lon = 0;
        this.lat = 0;
        this.isUserInteracting = false;
        this.fov = 75;
        this.keysDown = new Set();
        this.panoWidth = 0;
        this.panoHeight = 0;

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(this.fov, 1, 0.1, 1100);
        this.camera.position.set(0, 0, 0);

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(this.renderer.domElement);

        // Mouse
        container.addEventListener("mousedown", this._onMouseDown.bind(this));
        container.addEventListener("mousemove", this._onMouseMove.bind(this));
        container.addEventListener("mouseup", this._onMouseUp.bind(this));
        container.addEventListener("mouseleave", this._onMouseUp.bind(this));
        container.addEventListener("wheel", this._onWheel.bind(this), { passive: false });

        // Keyboard
        container.setAttribute("tabindex", "0");
        container.style.outline = "none";
        container.addEventListener("keydown", this._onKeyDown.bind(this));
        container.addEventListener("keyup", this._onKeyUp.bind(this));

        this._animate = this._animate.bind(this);
        this._animating = false;

        this.resizeView = () => {
            const w = container.clientWidth;
            const h = container.clientHeight;
            if (w > 0 && h > 0) {
                this.camera.aspect = w / h;
                this.camera.updateProjectionMatrix();
                this.renderer.setSize(w, h);
            }
        };
    }

    loadFromUrl(url) {
        const loader = new THREE.TextureLoader();
        loader.load(
            url,
            (texture) => {
                texture.colorSpace = THREE.SRGBColorSpace;
                texture.mapping = THREE.EquirectangularReflectionMapping;
                texture.minFilter = THREE.LinearFilter;
                texture.magFilter = THREE.LinearFilter;
                this.scene.background = texture;
                this.resizeView();
                if (!this._animating) {
                    this._animating = true;
                    this._animate();
                }
            },
            undefined,
            (err) => console.error("PanoPack: texture load error", err),
        );
    }

    setView(lon, lat) {
        this.lon = lon;
        this.lat = lat;
    }

    _onMouseDown(e) {
        this.isUserInteracting = true;
        this._pointerX = e.clientX;
        this._pointerY = e.clientY;
        this._startLon = this.lon;
        this._startLat = this.lat;
        this.container.focus();
        e.preventDefault();
    }

    _onMouseMove(e) {
        if (!this.isUserInteracting) return;
        this.lon = (this._pointerX - e.clientX) * 0.15 + this._startLon;
        this.lat = (this._pointerY - e.clientY) * 0.15 + this._startLat;
        e.preventDefault();
    }

    _onMouseUp() {
        this.isUserInteracting = false;
    }

    _onWheel(e) {
        this.fov = Math.max(30, Math.min(100, this.fov + e.deltaY * 0.05));
        this.camera.fov = this.fov;
        this.camera.updateProjectionMatrix();
        e.preventDefault();
    }

    _onKeyDown(e) {
        const nav = ["ArrowUp","ArrowDown","ArrowLeft","ArrowRight","w","a","s","d","W","A","S","D"];
        if (nav.includes(e.key)) {
            this.keysDown.add(e.key.toLowerCase());
            e.preventDefault();
            e.stopPropagation();
        }
    }

    _onKeyUp(e) {
        this.keysDown.delete(e.key.toLowerCase());
    }

    _animate() {
        requestAnimationFrame(this._animate);

        if (this.keysDown.has("arrowleft")  || this.keysDown.has("a")) this.lon -= KEY_ROTATE_SPEED;
        if (this.keysDown.has("arrowright") || this.keysDown.has("d")) this.lon += KEY_ROTATE_SPEED;
        if (this.keysDown.has("arrowup")    || this.keysDown.has("w")) this.lat += KEY_ROTATE_SPEED;
        if (this.keysDown.has("arrowdown")  || this.keysDown.has("s")) this.lat -= KEY_ROTATE_SPEED;

        this.lat = Math.max(-85, Math.min(85, this.lat));

        const phi   = THREE.MathUtils.degToRad(90 - this.lat);
        const theta = THREE.MathUtils.degToRad(this.lon);

        const target = new THREE.Vector3(
            Math.sin(phi) * Math.cos(theta),
            Math.cos(phi),
            Math.sin(phi) * Math.sin(theta),
        );
        this.camera.lookAt(target);
        this.renderer.render(this.scene, this.camera);

        // Update live UV info
        this._updateInfo();
    }

    _updateInfo() {
        // Normalize lon to [0, 360), lat is [-85, 85]
        let lonNorm = ((this.lon % 360) + 360) % 360;
        // U: 0..1 across the panorama width (lon 0 = center = u 0.5)
        const u = lonNorm / 360.0;
        // V: 0..1 from top to bottom (lat +90 = top = v 0, lat -90 = bottom = v 1)
        const v = 0.5 - (this.lat / 180.0);

        // Pixel coords if panorama dimensions are known
        let pixelStr = "";
        if (this.panoWidth > 0 && this.panoHeight > 0) {
            const px = Math.round(u * this.panoWidth) % this.panoWidth;
            const py = Math.min(Math.round(v * this.panoHeight), this.panoHeight - 1);
            pixelStr = `  |  px: (${px}, ${py})`;
        }

        const lonDisp = ((lonNorm + 180) % 360 - 180);  // show as [-180, 180]
        this.infoLabel.textContent =
            `lon: ${lonDisp.toFixed(1)}°  lat: ${this.lat.toFixed(1)}°` +
            `  |  u: ${u.toFixed(4)}  v: ${v.toFixed(4)}` +
            `  |  fov: ${this.fov.toFixed(0)}°` +
            pixelStr;
    }
}

// --- Preset buttons (overlay, top-right) ---
function buildButtonOverlay(viewer) {
    const bar = document.createElement("div");
    bar.style.cssText =
        "position:absolute; top:6px; right:6px; display:flex; gap:3px; flex-wrap:wrap; " +
        "z-index:10; pointer-events:auto;";

    for (const preset of PRESET_VIEWS) {
        const btn = document.createElement("button");
        btn.textContent = preset.label;
        btn.style.cssText =
            "padding:2px 6px; font-size:9px; border:1px solid #555; border-radius:3px; " +
            "background:rgba(30,30,50,0.8); color:#ccc; cursor:pointer; line-height:1.3; " +
            "backdrop-filter:blur(4px);";
        btn.addEventListener("mouseenter", () => { btn.style.background = "rgba(60,60,90,0.9)"; });
        btn.addEventListener("mouseleave", () => { btn.style.background = "rgba(30,30,50,0.8)"; });
        btn.addEventListener("click", (e) => {
            viewer.setView(preset.lon, preset.lat);
            e.stopPropagation();
        });
        bar.appendChild(btn);
    }
    return bar;
}

// --- ComfyUI extension ---
app.registerExtension({
    name: "panopack.panorama_viewer",

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== "PanoramaViewer") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;

            // Outer wrapper
            const wrapper = document.createElement("div");
            wrapper.style.cssText =
                "background:#000; border:1px solid #333; border-radius:6px; overflow:hidden; " +
                "display:flex; flex-direction:column; height:100%;";

            // Viewport area (canvas + button overlay)
            const viewport = document.createElement("div");
            viewport.style.cssText =
                "flex:1; min-height:0; position:relative;";
            wrapper.appendChild(viewport);

            // Info bar below the viewer (live UV coordinates)
            const infoBar = document.createElement("div");
            infoBar.style.cssText =
                `height:${INFO_BAR_HEIGHT}px; padding:2px 8px; font-size:10px; ` +
                "font-family:monospace; color:#aaa; background:#111; border-top:1px solid #333; " +
                "display:flex; align-items:center; flex-shrink:0; white-space:nowrap; overflow:hidden;";
            infoBar.textContent = "lon: --  lat: --  |  u: --  v: --";
            wrapper.appendChild(infoBar);

            // Viewer
            const viewer = new PanoViewer(viewport, infoBar);
            this._panoViewer = viewer;

            // Preset buttons (overlay, top-right of viewport)
            const buttonOverlay = buildButtonOverlay(viewer);
            viewport.appendChild(buttonOverlay);

            // Track desired viewer size
            let vWidth = VIEWER_MIN_HEIGHT;
            let vHeight = VIEWER_MIN_HEIGHT;
            this._setViewerSize = (w, h) => {
                vWidth = Math.max(VIEWER_MIN_HEIGHT, w);
                vHeight = Math.max(VIEWER_MIN_HEIGHT, h);
                node.setSize([vWidth + 30, node.computeSize()[1]]);
                node.setDirtyCanvas(true, true);
                viewer.resizeView();
            };

            // DOM widget
            const widget = this.addDOMWidget("panorama_viewer_360", "PANO_VIEWER", wrapper, {
                serialize: false,
                hideOnZoom: false,
                getMinHeight: () => vHeight + INFO_BAR_HEIGHT,
                getHeight: () => vHeight + INFO_BAR_HEIGHT,
            });

            widget.computeSize = function (width) {
                const total = vHeight + INFO_BAR_HEIGHT;
                this.computedHeight = total + 10;
                return [width, total];
            };

            widget.options.afterResize = viewer.resizeView;
            requestAnimationFrame(() => {
                viewer.resizeView();
                node.setSize([node.size[0], node.computeSize()[1]]);
                node.setDirtyCanvas(true, true);
            });

            return r;
        };

        // On execution — load the panorama image
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            if (message?.panorama && message.panorama.length > 0) {
                const pano = message.panorama[0];
                const params = new URLSearchParams({
                    filename: pano.filename,
                    subfolder: pano.subfolder || "",
                    type: pano.type || "output",
                });
                this._panoViewer?.loadFromUrl(`/view?${params.toString()}`);

                // Pass panorama pixel dimensions for UV→pixel display
                if (pano.width && pano.height) {
                    this._panoViewer.panoWidth = pano.width;
                    this._panoViewer.panoHeight = pano.height;
                }

                // Apply viewer widget size from node inputs
                if (pano.viewer_width && pano.viewer_height) {
                    this._setViewerSize?.(pano.viewer_width, pano.viewer_height);
                }

                // Set initial view direction
                if (pano.initial_yaw !== undefined && pano.initial_pitch !== undefined) {
                    this._panoViewer.setView(pano.initial_yaw, pano.initial_pitch);
                }
            }
            else if (message?.images && message.images.length > 0) {
                const img = message.images[0];
                const params = new URLSearchParams({
                    filename: img.filename,
                    subfolder: img.subfolder || "",
                    type: img.type || "output",
                });
                this._panoViewer?.loadFromUrl(`/view?${params.toString()}`);
            }
        };
    },
});
