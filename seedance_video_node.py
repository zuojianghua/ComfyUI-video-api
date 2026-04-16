"""Seedance (字节 Seedance) Image-to-Video node for ComfyUI.

Integrates with ByteDance Volcengine / BytePlus API to generate video from
a single image or first+last frame pair.  Authentication uses Bearer token.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import requests
import torch
from dotenv import load_dotenv

from .utils import (
    download_video,
    get_output_video_path,
    get_provider_config,
    make_video_ui_result,
    pil_to_base64_data_url,
    poll_until_complete,
    tensor_to_pils,
)

DOTENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)

LOG_PREFIX = "[ComfyUI-Seedance]"

_cfg = get_provider_config("seedance")
MODELS = [m["id"] for m in _cfg.get("models", [])] or ["doubao-seedance-1-5-pro-251215"]
RESOLUTIONS = _cfg.get("resolutions", ["480p", "720p", "1080p"])
RATIOS = _cfg.get("ratios", ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "adaptive"])
_defaults = _cfg.get("defaults", {})

POLL_INTERVAL = 5.0
POLL_TIMEOUT = 300.0


def _env(key: str) -> Optional[str]:
    return os.getenv(key) or None


def _headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _build_content(
    prompt: str,
    first_frame_b64: str,
    last_frame_b64: Optional[str] = None,
) -> list:
    content: list[dict] = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": first_frame_b64},
            "role": "first_frame",
        },
    ]
    if last_frame_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": last_frame_b64},
                "role": "last_frame",
            }
        )
    return content


def _create_task(
    base_url: str,
    api_key: str,
    content: list,
    model: str,
    duration: int = 5,
    resolution: str = "720p",
    ratio: str = "16:9",
    seed: int = -1,
) -> str:
    body: dict = {
        "model": model,
        "content": content,
        "duration": duration,
        "resolution": resolution,
        "ratio": ratio,
    }
    if seed >= 0:
        body["seed"] = seed

    url = f"{base_url.rstrip('/')}/contents/generations/tasks"
    print(f"{LOG_PREFIX} Creating task: model={model} duration={duration}s resolution={resolution}")
    resp = requests.post(url, json=body, headers=_headers(api_key), timeout=60)

    if resp.status_code != 200:
        try:
            err_body = resp.json()
        except Exception:
            err_body = resp.text
        raise RuntimeError(
            f"{LOG_PREFIX} HTTP {resp.status_code}: {err_body}"
        )

    data = resp.json()

    task_id = data.get("id")
    if not task_id:
        raise RuntimeError(f"{LOG_PREFIX} No task id in response: {data}")

    print(f"{LOG_PREFIX} Task created: {task_id}")
    return task_id


def _poll_task(base_url: str, api_key: str, task_id: str) -> str:
    url = f"{base_url.rstrip('/')}/contents/generations/tasks/{task_id}"

    def _fetch() -> dict:
        r = requests.get(url, headers=_headers(api_key), timeout=30)
        r.raise_for_status()
        return r.json()

    result = poll_until_complete(
        poll_fn=_fetch,
        is_done=lambda d: d.get("status") == "succeeded",
        is_failed=lambda d: d.get("status") == "failed",
        extract_error=lambda d: d.get("error", "unknown"),
        interval=POLL_INTERVAL,
        timeout=POLL_TIMEOUT,
        log_prefix=LOG_PREFIX,
    )

    video_url = result.get("content", {}).get("video_url")
    if not video_url:
        raise RuntimeError(f"{LOG_PREFIX} No video_url in result")
    return video_url


class SeedanceImageToVideo:
    """ComfyUI node – Seedance image-to-video (supports optional end frame)."""

    CATEGORY = "video_generation"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "file_path", "status")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
                "model": (MODELS, {"default": _defaults.get("model", MODELS[0])}),
            },
            "optional": {
                "image_tail": ("IMAGE",),
                "duration": (
                    "INT",
                    {"default": _defaults.get("duration", 5), "min": 3, "max": 12, "step": 1},
                ),
                "resolution": (RESOLUTIONS, {"default": _defaults.get("resolution", "720p")}),
                "ratio": (RATIOS, {"default": _defaults.get("ratio", "16:9")}),
                "seed": (
                    "INT",
                    {"default": _defaults.get("seed", -1), "min": -1, "max": 2147483647},
                ),
            },
        }

    def generate(
        self,
        image: torch.Tensor,
        prompt: str,
        model: str,
        image_tail: Optional[torch.Tensor] = None,
        duration: int = 5,
        resolution: str = "720p",
        ratio: str = "16:9",
        seed: int = -1,
    ) -> Tuple[str, str, str]:
        api_key = _env("SEEDANCE_API_KEY")
        base_url = _env("SEEDANCE_BASE_URL")
        if not api_key:
            return ("", "", f"{LOG_PREFIX} Error: SEEDANCE_API_KEY not set in .env")
        if not base_url:
            base_url = "https://ark.cn-beijing.volces.com/api/v3"

        start_pil = tensor_to_pils(image)[0]
        first_frame_b64 = pil_to_base64_data_url(start_pil, fmt="png")

        last_frame_b64: Optional[str] = None
        if image_tail is not None:
            tail_pil = tensor_to_pils(image_tail)[0]
            last_frame_b64 = pil_to_base64_data_url(tail_pil, fmt="png")

        content = _build_content(prompt, first_frame_b64, last_frame_b64)

        try:
            task_id = _create_task(
                base_url=base_url,
                api_key=api_key,
                content=content,
                model=model,
                duration=duration,
                resolution=resolution,
                ratio=ratio,
                seed=seed,
            )
            video_url = _poll_task(base_url, api_key, task_id)
        except Exception as e:
            print(f"{LOG_PREFIX} Error: {e}")
            return ("", "", f"{LOG_PREFIX} Error: {e}")

        file_path = get_output_video_path(prefix="seedance")
        try:
            download_video(video_url, file_path)
        except Exception as e:
            print(f"{LOG_PREFIX} Video ready but download failed: {e}")
            return (video_url, "", f"{LOG_PREFIX} Video ready but download failed: {e}")

        return {
            "ui": make_video_ui_result(file_path),
            "result": (video_url, file_path, "Video generated successfully"),
        }


NODE_CLASS_MAPPINGS = {
    "SeedanceImageToVideo": SeedanceImageToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedanceImageToVideo": "Seedance Image to Video",
}
