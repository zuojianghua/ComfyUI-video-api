"""Kling (可灵) Image-to-Video node for ComfyUI.

使用可灵新版 API（kling-2.6 / kling-3.0）：Bearer API Key 鉴权，支持首帧图生视频与
首尾帧图生视频。协议细节与能力校验见 kling_api.py。
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import requests
import torch
from dotenv import load_dotenv

from .jizhu_reporting import begin_video, report_video_completed, report_video_failed
from .kling_api import (
    DEFAULT_BASE_URL,
    MODEL_RULES,
    build_image_to_video_request,
    check_image_size,
    extract_video_url,
    parse_create_response,
    parse_task_item,
)
from .utils import (
    download_video,
    get_output_video_path,
    get_provider_config,
    make_video_ui_result,
    pil_to_base64,
    poll_until_complete,
    tensor_to_pils,
)

DOTENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)

LOG_PREFIX = "[ComfyUI-Kling]"

_cfg = get_provider_config("kling")
MODELS = [m["id"] for m in _cfg.get("models", [])] or list(MODEL_RULES)
# 下拉框是各模型能力的并集，具体组合在提交前按所选模型校验（如 2.6 仅 5/10 秒、无 4k）
RESOLUTIONS = _cfg.get("resolutions", ["720p", "1080p", "4k"])
DURATIONS = _cfg.get("durations", [str(d) for d in range(3, 16)])
_defaults = _cfg.get("defaults", {})

_RESOLUTION_ORDER = ["720p", "1080p", "4k"]
#: 下发给前端 web/js/klingOptions.js：切换模型时把「时长 / 清晰度」下拉收敛到该模型支持的值
KLING_UI_RULES = {
    model: {
        "durations": [str(d) for d in sorted(MODEL_RULES[model]["durations"])],
        "resolutions": [r for r in _RESOLUTION_ORDER if r in MODEL_RULES[model]["resolutions"]],
    }
    for model in MODELS
    if model in MODEL_RULES
}

POLL_INTERVAL = 5.0
# 增加等待时长
POLL_TIMEOUT = 3600.0


def _headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _create_task(base_url: str, api_key: str, path: str, body: dict) -> str:
    print(f"{LOG_PREFIX} Creating task: {path} settings={body['settings']}")
    resp = requests.post(f"{base_url}{path}", json=body, headers=_headers(api_key), timeout=60)
    try:
        data = resp.json()
    except Exception:
        data = None
    task_id = parse_create_response(resp.status_code, data)
    print(f"{LOG_PREFIX} Task created: {task_id}")
    return task_id


def _poll_task(base_url: str, api_key: str, task_id: str) -> str:
    def _fetch() -> dict:
        r = requests.get(
            f"{base_url}/tasks",
            params={"task_ids": task_id},
            headers=_headers(api_key),
            timeout=30,
        )
        r.raise_for_status()
        return parse_task_item(r.json(), task_id)

    result = poll_until_complete(
        poll_fn=_fetch,
        is_done=lambda d: d.get("status") == "succeeded",
        is_failed=lambda d: d.get("status") == "failed",
        extract_error=lambda d: d.get("message") or "unknown",
        interval=POLL_INTERVAL,
        timeout=POLL_TIMEOUT,
        log_prefix=LOG_PREFIX,
    )
    return extract_video_url(result)


def _image_to_base64(image: torch.Tensor, label: str) -> str:
    pil = tensor_to_pils(image)[0]
    check_image_size(pil.width, pil.height, label)
    return pil_to_base64(pil)


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
                "model_name": (
                    MODELS,
                    {"default": _defaults.get("model_name", MODELS[0]), "kling_rules": KLING_UI_RULES},
                ),
                "resolution": (RESOLUTIONS, {"default": _defaults.get("resolution", "1080p")}),
                "duration": (DURATIONS, {"default": _defaults.get("duration", "5")}),
                "audio": ("BOOLEAN", {"default": _defaults.get("audio", False)}),
            },
            "optional": {
                "image_tail": ("IMAGE",),
                # 新版接口无 seed 参数；保留该输入用于改值后强制重新执行节点
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
        model_name: str,
        resolution: str,
        duration: str,
        audio: bool,
        image_tail: Optional[torch.Tensor] = None,
        seed: int = -1,
    ) -> Tuple[str, str, str]:
        # 校验不通过、可灵拒绝或任务失败都直接抛出：ComfyUI 会把节点标红并弹窗显示原因，
        # 而不是只写进未连接的 status 输出、让用户以为什么都没发生。
        api_key = os.getenv("KLING_API_KEY")
        if not api_key:
            raise RuntimeError(f"{LOG_PREFIX} 未配置 KLING_API_KEY（节点目录 .env）")
        base_url = (os.getenv("KLING_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")

        first_frame = _image_to_base64(image, "首帧")
        last_frame = _image_to_base64(image_tail, "尾帧") if image_tail is not None else None
        path, body = build_image_to_video_request(
            model=model_name,
            prompt=prompt,
            first_frame=first_frame,
            last_frame=last_frame,
            resolution=resolution,
            duration=int(duration),
            audio=audio,
        )
        # 参数与图片校验通过后再校验机杼额度，避免无效参数占用额度校验
        jizhu_client, execution, error = begin_video(
            model=model_name,
            provider="kling",
            duration=duration,
        )
        if error:
            raise RuntimeError(f"{LOG_PREFIX} {error}")

        try:
            task_id = _create_task(base_url, api_key, path, body)
            video_url = _poll_task(base_url, api_key, task_id)
        except Exception:
            report_video_failed(jizhu_client, execution, model_name, "kling", duration)
            raise

        # 视频已生成但下载失败：仍返回链接，避免丢失已扣费的结果
        file_path = get_output_video_path(prefix="kling")
        try:
            download_video(video_url, file_path)
        except Exception as e:
            report_video_failed(jizhu_client, execution, model_name, "kling", duration)
            print(f"{LOG_PREFIX} Video ready but download failed: {e}")
            return (video_url, "", f"{LOG_PREFIX} Video ready but download failed: {e}")

        try:
            report_video_completed(
                jizhu_client,
                execution,
                model_name,
                "kling",
                duration,
                file_path,
            )
        except Exception as e:
            print(f"{LOG_PREFIX} Result reporting failed: {e}")
            return (video_url, "", f"{LOG_PREFIX} Result reporting failed: {e}")

        return {
            "ui": make_video_ui_result(file_path),
            "result": (video_url, file_path, "Video generated successfully"),
        }


NODE_CLASS_MAPPINGS = {
    "KlingImageToVideo": KlingImageToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KlingImageToVideo": "Kling Image to Video",
}
