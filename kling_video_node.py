"""Kling (可灵) Image-to-Video node for ComfyUI.

Integrates with Kuaishou Kling API to generate video from a single image
or first+last frame pair.  Authentication uses JWT (HS256).
"""

import os
import time
from pathlib import Path
from typing import Optional, Tuple

import jwt
import requests
import torch
from dotenv import load_dotenv

from .utils import (
    download_video,
    get_output_video_path,
    pil_to_base64_data_url,
    poll_until_complete,
    tensor_to_pils,
)

DOTENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)

LOG_PREFIX = "[ComfyUI-Kling]"

MODELS = [
    "kling-v2-6",
    "kling-v2-6-pro",
    "kling-v3",
]
MODES = ["std", "pro"]
DURATIONS = ["5", "10"]
ASPECT_RATIOS = ["16:9", "9:16", "1:1"]

POLL_INTERVAL = 5.0
POLL_TIMEOUT = 300.0


def _env(key: str) -> Optional[str]:
    return os.getenv(key) or None


def _generate_jwt(access_key: str, secret_key: str, expire: int = 1800) -> str:
    now = int(time.time())
    payload = {
        "iss": access_key,
        "exp": now + expire,
        "nbf": now - 5,
    }
    return jwt.encode(payload, secret_key, algorithm="HS256", headers={"typ": "JWT"})


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _create_task(
    base_url: str,
    token: str,
    image_b64: str,
    prompt: str,
    model_name: str,
    mode: str,
    duration: str,
    negative_prompt: str = "",
    cfg_scale: float = 0.5,
    aspect_ratio: str = "16:9",
    image_tail_b64: Optional[str] = None,
) -> str:
    body: dict = {
        "model_name": model_name,
        "image": image_b64,
        "prompt": prompt,
        "mode": mode,
        "duration": duration,
        "cfg_scale": cfg_scale,
        "aspect_ratio": aspect_ratio,
    }
    if negative_prompt:
        body["negative_prompt"] = negative_prompt
    if image_tail_b64:
        body["image_tail"] = image_tail_b64

    url = f"{base_url.rstrip('/')}/videos/image2video"
    print(f"{LOG_PREFIX} Creating task: model={model_name} mode={mode} duration={duration}s")
    resp = requests.post(url, json=body, headers=_headers(token), timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if data.get("code") != 0:
        raise RuntimeError(f"{LOG_PREFIX} API error: {data.get('message', data)}")

    task_id = data["data"]["task_id"]
    print(f"{LOG_PREFIX} Task created: {task_id}")
    return task_id


def _poll_task(base_url: str, token: str, task_id: str) -> str:
    url = f"{base_url.rstrip('/')}/videos/image2video/{task_id}"

    def _fetch() -> dict:
        r = requests.get(url, headers=_headers(token), timeout=30)
        r.raise_for_status()
        body = r.json()
        if body.get("code") != 0:
            raise RuntimeError(f"{LOG_PREFIX} Poll error: {body.get('message', body)}")
        return body["data"]

    result = poll_until_complete(
        poll_fn=_fetch,
        is_done=lambda d: d.get("task_status") == "succeed",
        is_failed=lambda d: d.get("task_status") == "failed",
        extract_error=lambda d: d.get("task_status_msg", "unknown"),
        interval=POLL_INTERVAL,
        timeout=POLL_TIMEOUT,
        log_prefix=LOG_PREFIX,
    )

    videos = result.get("task_result", {}).get("videos", [])
    if not videos:
        raise RuntimeError(f"{LOG_PREFIX} No videos in result")
    return videos[0]["url"]


class KlingImageToVideo:
    """ComfyUI node – Kling image-to-video (supports optional end frame)."""

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
                "model_name": (MODELS, {"default": MODELS[0]}),
                "mode": (MODES, {"default": "pro"}),
                "duration": (DURATIONS, {"default": "5"}),
            },
            "optional": {
                "image_tail": ("IMAGE",),
                "negative_prompt": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
                "cfg_scale": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9"}),
            },
        }

    def generate(
        self,
        image: torch.Tensor,
        prompt: str,
        model_name: str,
        mode: str,
        duration: str,
        image_tail: Optional[torch.Tensor] = None,
        negative_prompt: str = "",
        cfg_scale: float = 0.5,
        aspect_ratio: str = "16:9",
    ) -> Tuple[str, str, str]:
        access_key = _env("KLING_ACCESS_KEY")
        secret_key = _env("KLING_SECRET_KEY")
        base_url = _env("KLING_BASE_URL")
        if not access_key or not secret_key:
            return ("", "", f"{LOG_PREFIX} Error: KLING_ACCESS_KEY / KLING_SECRET_KEY not set in .env")
        if not base_url:
            base_url = "https://api-beijing.klingai.com/v1"

        try:
            token = _generate_jwt(access_key, secret_key)
        except Exception as e:
            return ("", "", f"{LOG_PREFIX} JWT generation failed: {e}")

        start_pil = tensor_to_pils(image)[0]
        image_b64 = pil_to_base64_data_url(start_pil)

        image_tail_b64: Optional[str] = None
        if image_tail is not None:
            tail_pil = tensor_to_pils(image_tail)[0]
            image_tail_b64 = pil_to_base64_data_url(tail_pil)

        try:
            task_id = _create_task(
                base_url=base_url,
                token=token,
                image_b64=image_b64,
                prompt=prompt,
                model_name=model_name,
                mode=mode,
                duration=duration,
                negative_prompt=negative_prompt,
                cfg_scale=cfg_scale,
                aspect_ratio=aspect_ratio,
                image_tail_b64=image_tail_b64,
            )
            video_url = _poll_task(base_url, token, task_id)
        except Exception as e:
            return ("", "", f"{LOG_PREFIX} Error: {e}")

        file_path = get_output_video_path(prefix="kling")
        try:
            download_video(video_url, file_path)
        except Exception as e:
            return (video_url, "", f"{LOG_PREFIX} Video ready but download failed: {e}")

        return (video_url, file_path, "Video generated successfully")


NODE_CLASS_MAPPINGS = {
    "KlingImageToVideo": KlingImageToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KlingImageToVideo": "Kling Image to Video",
}
