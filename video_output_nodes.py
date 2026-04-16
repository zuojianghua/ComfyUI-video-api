"""Preview and Save Video nodes for ComfyUI.

PreviewVideo: downloads video to temp dir, passes to UI for in-browser playback.
SaveVideo: downloads the video from URL and saves to ComfyUI output directory.
"""

import os
import random
import re

import requests

from .utils import download_video, get_output_video_path, make_video_ui_result


LOG_PREFIX = "[ComfyUI-video-api]"


def _get_temp_directory() -> str:
    try:
        import folder_paths  # type: ignore[import-untyped]
        return folder_paths.get_temp_directory()
    except Exception:
        d = os.path.join(os.path.dirname(__file__), "temp")
        os.makedirs(d, exist_ok=True)
        return d


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
            return {"ui": {}, "result": ("",)}

        temp_dir = _get_temp_directory()
        suffix = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(8))
        filename = f"preview_{suffix}.mp4"
        save_path = os.path.join(temp_dir, filename)

        try:
            download_video(video_url, save_path)
        except Exception as e:
            print(f"{LOG_PREFIX} PreviewVideo download failed: {e}")
            return {"ui": {}, "result": ("",)}

        return {
            "ui": {
                "images": [{"filename": filename, "subfolder": "", "type": "temp"}],
                "animated": (True,),
            },
            "result": (video_url,),
        }


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
            return {"ui": {}, "result": ("",)}

        save_path = get_output_video_path(prefix=filename_prefix, ext=".mp4")

        print(f"{LOG_PREFIX} SaveVideo: downloading to {save_path}")
        try:
            download_video(video_url, save_path)
        except Exception as e:
            print(f"{LOG_PREFIX} SaveVideo download failed: {e}")
            return {"ui": {}, "result": ("",)}

        abs_path = os.path.abspath(save_path)
        print(f"{LOG_PREFIX} SaveVideo: saved to {abs_path}")

        return {
            "ui": make_video_ui_result(abs_path),
            "result": (abs_path,),
        }


NODE_CLASS_MAPPINGS = {
    "PreviewVideoFromURL": PreviewVideo,
    "SaveVideoFromURL": SaveVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreviewVideoFromURL": "Preview Video (URL)",
    "SaveVideoFromURL": "Save Video (URL)",
}
