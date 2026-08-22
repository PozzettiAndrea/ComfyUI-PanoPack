/**
 * ComfyUI PanoPack - Interactive 360 Panorama Viewer (iframe-based)
 * Three.js runs inside an iframe to avoid conflicts with other extensions.
 */

import { app } from "../../../scripts/app.js";

// Resolve the viewer HTML relative to this module so the served path
// (".../js/viewer/panorama_viewer.html") is always correct.
const VIEWER_HTML_URL = new URL("./viewer/panorama_viewer.html", import.meta.url).href;

const VIEWER_MIN_HEIGHT = 300;

app.registerExtension({
    name: "panopack.panorama_viewer",

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== "PanoramaViewer") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;

            // Create iframe for the panorama viewer
            const iframe = document.createElement("iframe");
            iframe.style.width = "100%";
            iframe.style.height = "100%";
            iframe.style.border = "none";
            iframe.style.backgroundColor = "#000";
            iframe.style.borderRadius = "6px";
            // Let the in-iframe Fullscreen button fill the whole screen.
            iframe.allowFullscreen = true;
            iframe.setAttribute("allow", "fullscreen");
            iframe.src = `${VIEWER_HTML_URL}?v=${Date.now()}`;

            this._panoIframe = iframe;

            // Sync the viewer's live camera orientation back into the node's
            // widgets so re-running the node yields a crop matching the view.
            const setWidget = (name, val) => {
                const w = node.widgets?.find((x) => x.name === name);
                if (!w || typeof val !== "number" || Number.isNaN(val)) return;
                let v = val;
                if (w.options?.min !== undefined) v = Math.max(w.options.min, v);
                if (w.options?.max !== undefined) v = Math.min(w.options.max, v);
                w.value = v;
            };
            const onMessage = (e) => {
                if (e.source !== iframe.contentWindow) return;  // only this node's viewer
                const d = e.data;
                if (!d || d.type !== "camera_state") return;
                setWidget("initial_yaw", d.yaw);
                setWidget("initial_pitch", d.pitch);
                setWidget("crop_fov", d.fov);
                node.setDirtyCanvas(true, true);
            };
            window.addEventListener("message", onMessage);
            this._panoOnMessage = onMessage;

            // Track desired viewer size
            let vWidth = VIEWER_MIN_HEIGHT;
            let vHeight = VIEWER_MIN_HEIGHT;
            this._setViewerSize = (w, h) => {
                vWidth = Math.max(VIEWER_MIN_HEIGHT, w);
                vHeight = Math.max(VIEWER_MIN_HEIGHT, h);
                node.setSize([vWidth + 30, node.computeSize()[1]]);
                node.setDirtyCanvas(true, true);
            };

            // DOM widget
            const widget = this.addDOMWidget("panorama_viewer_360", "PANO_VIEWER", iframe, {
                serialize: false,
                hideOnZoom: false,
                getMinHeight: () => vHeight,
                getHeight: () => vHeight,
            });

            widget.computeSize = function (width) {
                this.computedHeight = vHeight + 10;
                return [width, vHeight];
            };

            requestAnimationFrame(() => {
                node.setSize([node.size[0], node.computeSize()[1]]);
                node.setDirtyCanvas(true, true);
            });

            return r;
        };

        // On execution — send panorama data to iframe via postMessage
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            const iframe = this._panoIframe;
            if (!iframe?.contentWindow) return;

            const sendToViewer = (msg) => {
                // Retry a few times in case iframe isn't ready yet
                let attempts = 0;
                const trySend = () => {
                    if (iframe.contentWindow) {
                        iframe.contentWindow.postMessage(msg, "*");
                    } else if (attempts < 5) {
                        attempts++;
                        setTimeout(trySend, 200);
                    }
                };
                trySend();
            };

            if (message?.panorama && message.panorama.length > 0) {
                const pano = message.panorama[0];
                const params = new URLSearchParams({
                    filename: pano.filename,
                    subfolder: pano.subfolder || "",
                    type: pano.type || "output",
                });

                sendToViewer({
                    type: "load_panorama",
                    url: `/view?${params.toString()}`,
                    width: pano.width || 0,
                    height: pano.height || 0,
                    initial_yaw: pano.initial_yaw ?? 0,
                    initial_pitch: pano.initial_pitch ?? 0,
                    crop_fov: pano.crop_fov ?? 90,
                });

                if (pano.viewer_width && pano.viewer_height) {
                    this._setViewerSize?.(pano.viewer_width, pano.viewer_height);
                }
            }
            else if (message?.images && message.images.length > 0) {
                const img = message.images[0];
                const params = new URLSearchParams({
                    filename: img.filename,
                    subfolder: img.subfolder || "",
                    type: img.type || "output",
                });
                sendToViewer({
                    type: "load_panorama",
                    url: `/view?${params.toString()}`,
                });
            }
        };

        // Remove the camera-state listener when the node is deleted.
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            if (this._panoOnMessage) {
                window.removeEventListener("message", this._panoOnMessage);
                this._panoOnMessage = null;
            }
            return onRemoved?.apply(this, arguments);
        };
    },
});
