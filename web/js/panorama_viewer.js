/**
 * ComfyUI PanoPack - Interactive 360 Panorama Viewer (iframe-based)
 * Three.js runs inside an iframe to avoid conflicts with other extensions.
 */

import { app } from "../../../scripts/app.js";

const EXTENSION_FOLDER = (() => {
    const url = import.meta.url;
    const match = url.match(/\/extensions\/([^/]+)\//);
    return match ? match[1] : "ComfyUI-PanoPack";
})();

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
            iframe.src = `/extensions/${EXTENSION_FOLDER}/viewer/panorama_viewer.html?v=${Date.now()}`;

            this._panoIframe = iframe;

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
    },
});
