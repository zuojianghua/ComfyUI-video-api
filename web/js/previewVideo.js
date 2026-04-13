import { app, ANIM_PREVIEW_WIDGET } from "../../../scripts/app.js";
import { createImageHost } from "../../../scripts/ui/imagePreview.js";

const BASE_SIZE = 512;

function resizeVideoAspectRatio(videoEl, maxW, maxH) {
  const ratio = videoEl.videoWidth / videoEl.videoHeight;
  let w, h;
  if (videoEl.videoWidth / maxW > videoEl.videoHeight / maxH) {
    w = maxW;
    h = w / ratio;
  } else {
    h = maxH;
    w = h * ratio;
  }
  videoEl.style.width = `${w}px`;
  videoEl.style.height = `${h}px`;
}

function createVideoElement(url) {
  return new Promise((resolve) => {
    const el = document.createElement("video");
    el.addEventListener("loadedmetadata", () => {
      el.controls = true;
      el.loop = true;
      el.muted = true;
      resizeVideoAspectRatio(el, BASE_SIZE, BASE_SIZE);
      resolve(el);
    });
    el.addEventListener("error", () => resolve(null));
    el.src = url;
  });
}

function addVideoPreview(nodeType) {
  const origOnExecuted = nodeType.prototype.onExecuted;

  nodeType.prototype.onExecuted = function (message) {
    if (origOnExecuted) origOnExecuted.apply(this, arguments);

    if (!message?.video_url?.length) return;

    this.images = message.video_url;
    this.setDirtyCanvas(true);
  };

  nodeType.prototype.onDrawBackground = function (ctx) {
    if (this.flags.collapsed) return;

    const urls = this.images ?? [];
    const key = JSON.stringify(urls);

    if (this._prevUrlKey === key) return;
    this._prevUrlKey = key;

    if (!urls.length) {
      this.imgs = null;
      this.animatedImages = false;
      return;
    }

    Promise.all(urls.map(createVideoElement)).then((elements) => {
      this.imgs = elements.filter(Boolean);
      if (!this.imgs.length) return;

      this.animatedImages = true;
      const idx = this.widgets?.findIndex(
        (w) => w.name === ANIM_PREVIEW_WIDGET
      );

      if (idx > -1) {
        this.widgets[idx].options.host.updateImages(this.imgs);
      } else {
        const host = createImageHost(this);
        const widget = this.addDOMWidget(
          ANIM_PREVIEW_WIDGET,
          "img",
          host.el,
          {
            host,
            getHeight: host.getHeight,
            onDraw: host.onDraw,
            hideOnZoom: false,
          }
        );
        widget.serializeValue = () => ({ height: BASE_SIZE });
        widget.options.host.updateImages(this.imgs);
      }

      this.imgs.forEach((el) => {
        if (el instanceof HTMLVideoElement) {
          el.muted = true;
          el.autoplay = true;
          el.play();
        }
      });

      this.setDirtyCanvas(true, true);
    });
  };
}

app.registerExtension({
  name: "ComfyUI-video-api.PreviewVideo",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "PreviewVideoFromURL") return;
    addVideoPreview(nodeType);
  },
});
