"""可灵 Kling 新版 API（kling-2.6 / kling-3.0）的纯协议逻辑：能力校验、请求体构造、响应解析。

不依赖 torch / ComfyUI，可直接单测（见 tests/test_kling_api.py）。

新版接口（Bearer API Key 鉴权，国内域名 https://api-beijing.klingai.com）：
- 图生视频：POST /image-to-video/{model}，contents = prompt / first_frame / last_frame，
  first_frame / last_frame 的 url 支持公网 URL 或 Base64；
- 查询：GET /tasks?task_ids=...，状态 submitted / processing / succeeded / failed，
  视频在 outputs[type=video].url（保留 30 天）。
"""

from typing import Any, Dict, Optional, Tuple

DEFAULT_BASE_URL = "https://api-beijing.klingai.com"

#: 各模型官方能力（与机杼 vtryon 的 Kling adapter 一致；3.0 节点额外开放 4k）。
MODEL_RULES: Dict[str, Dict[str, Any]] = {
    "kling-3.0": {
        "durations": set(range(3, 16)),
        "resolutions": {"720p", "1080p", "4k"},
        # 3.0 的 multi_shot 默认 true，会把单条视频拆成多镜头，固定关闭。
        "multi_shot": False,
        "audio_requires_1080p": False,
        "last_frame_requires_1080p": False,
    },
    "kling-2.6": {
        "durations": {5, 10},
        "resolutions": {"720p", "1080p"},
        "multi_shot": None,  # 2.6 无此参数，不发送
        "audio_requires_1080p": True,
        "last_frame_requires_1080p": True,
    },
}

#: 官方首尾帧图片限制：边长 ≥ 300px，宽高比 1:2.5 – 2.5:1。
MIN_IMAGE_SIDE = 300
MIN_ASPECT_RATIO = 1 / 2.5
MAX_ASPECT_RATIO = 2.5


def check_image_size(width: int, height: int, label: str) -> None:
    """提交前检查首/尾帧尺寸，不合规直接报中文原因，避免白跑一次接口。"""
    if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE:
        raise ValueError(f"{label}尺寸为 {width}×{height}px，宽和高均需不小于 {MIN_IMAGE_SIDE}px")
    ratio = width / height
    if not MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO:
        raise ValueError(f"{label}宽高比为 {ratio:.2f}，需在 1:2.5 到 2.5:1 之间")


def build_image_to_video_request(
    *,
    model: str,
    prompt: str,
    first_frame: str,
    last_frame: Optional[str],
    resolution: str,
    duration: int,
    audio: bool,
) -> Tuple[str, Dict[str, Any]]:
    """校验能力组合并构造（路径, 请求体）。first_frame / last_frame 为 URL 或 Base64。"""
    rules = MODEL_RULES.get(model)
    if rules is None:
        raise ValueError(f"不支持的模型 {model}，可选：{', '.join(MODEL_RULES)}")
    if not first_frame:
        raise ValueError("必须提供首帧图片")
    if duration not in rules["durations"]:
        allowed = "、".join(str(d) for d in sorted(rules["durations"]))
        raise ValueError(f"{model} 时长仅支持 {allowed} 秒，当前为 {duration} 秒")
    if resolution not in rules["resolutions"]:
        raise ValueError(f"{model} 不支持清晰度 {resolution}，可选：{', '.join(sorted(rules['resolutions']))}")
    if audio and rules["audio_requires_1080p"] and resolution != "1080p":
        raise ValueError(f"{model} 生成有声视频时仅支持 1080p")
    if last_frame and rules["last_frame_requires_1080p"] and resolution != "1080p":
        raise ValueError(f"{model} 使用尾帧时仅支持 1080p")

    contents = []
    if prompt.strip():
        contents.append({"type": "prompt", "text": prompt.strip()})
    contents.append({"type": "first_frame", "url": first_frame})
    if last_frame:
        contents.append({"type": "last_frame", "url": last_frame})

    settings: Dict[str, Any] = {
        "resolution": resolution,
        "duration": duration,
        "audio": "native" if audio else "off",
    }
    if rules["multi_shot"] is not None:
        settings["multi_shot"] = rules["multi_shot"]
    body = {
        "contents": contents,
        "settings": settings,
        "options": {"watermark_info": {"enabled": False}},
    }
    return f"/image-to-video/{model}", body


def parse_create_response(status_code: int, data: Any) -> str:
    """解析创建任务响应，返回任务 ID；失败抛出带业务码与原因的错误。"""
    if not isinstance(data, dict):
        raise RuntimeError(f"创建任务失败 HTTP {status_code}：响应无法解析")
    if status_code != 200 or data.get("code") not in (0, "0"):
        raise RuntimeError(f"创建任务失败 HTTP {status_code}：{data.get('code')} {data.get('message', '')}".strip())
    task = data.get("data") or {}
    task_id = task.get("id") if isinstance(task, dict) else None
    if not task_id:
        raise RuntimeError("创建任务失败：响应缺少任务 ID")
    return str(task_id)


def parse_task_item(data: Any, task_id: str) -> Dict[str, Any]:
    """从 GET /tasks 响应中取出指定任务。"""
    if not isinstance(data, dict) or data.get("code") not in (0, "0"):
        message = data.get("message", "") if isinstance(data, dict) else ""
        raise RuntimeError(f"查询任务失败：{message or '响应异常'}")
    for item in data.get("data") or []:
        if isinstance(item, dict) and str(item.get("id")) == task_id:
            return item
    raise RuntimeError(f"查询不到任务 {task_id}")


def extract_video_url(item: Dict[str, Any]) -> str:
    for output in item.get("outputs") or []:
        if isinstance(output, dict) and output.get("type") == "video" and output.get("url"):
            return str(output["url"])
    raise RuntimeError("任务成功但结果中没有视频")
