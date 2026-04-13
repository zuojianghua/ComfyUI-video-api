"""Preview and Save Video nodes for ComfyUI.

PreviewVideo: passes video_url to the UI for in-browser playback.
SaveVideo: downloads the video from URL and saves to ComfyUI output directory.
"""

import os
import re

import requests

from .utils import get_output_video_path


LOG_PREFIX = "[ComfyUI-video-api]"


class PreviewVideo:
    CATEGORY = "video_generation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "video_url": ("STRING", {"forceInput": True}),
            },
        }

    def run(self, video_url: str):
        if not video_url:
            print(f"{LOG_PREFIX} PreviewVideo: empty video_url, nothing to preview")
            return {"ui": {"text": ["No video URL provided"]}, "result": ("",)}

        return {"ui": {"video_url": [video_url]}, "result": (video_url,)}


class SaveVideo:
    CATEGORY = "video_generation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "video_url": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "filename_prefix": (
                    "STRING",
                    {"default": "video"},
                ),
            },
        }

    def run(self, video_url: str, filename_prefix: str = "video"):
        if not video_url:
            print(f"{LOG_PREFIX} SaveVideo: empty video_url, nothing to save")
            return ("",)

        ext = ".mp4"
        save_path = get_output_video_path(prefix=filename_prefix, ext=ext)

        print(f"{LOG_PREFIX} SaveVideo: downloading to {save_path}")
        try:
            resp = requests.get(video_url, timeout=120, stream=True)
            resp.raise_for_status()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            print(f"{LOG_PREFIX} SaveVideo download failed: {e}")
            return ("",)

        abs_path = os.path.abspath(save_path)
        print(f"{LOG_PREFIX} SaveVideo: saved to {abs_path}")
        return (abs_path,)


NODE_CLASS_MAPPINGS = {
    "PreviewVideoFromURL": PreviewVideo,
    "SaveVideoFromURL": SaveVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreviewVideoFromURL": "Preview Video (URL)",
    "SaveVideoFromURL": "Save Video (URL)",
}
