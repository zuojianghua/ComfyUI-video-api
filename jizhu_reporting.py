"""视频节点复用机杼共享客户端，按提交时长上报单个最终成品。"""

import mimetypes
from pathlib import Path

from jizhu_comfy_client import JizhuComfyClient, QuotaExhaustedError


def begin_video(model, provider, duration):
    """在提交模型任务前校验额度，并创建本次调用的稳定执行标识。"""
    client = JizhuComfyClient()
    execution = client.start_execution()
    try:
        client.check_quota("video", int(duration), model)
    except QuotaExhaustedError as exc:
        return None, None, f"额度不足：{exc}"
    except Exception as exc:
        return None, None, f"机杼额度校验失败：{exc}"
    return client, execution, None


def report_video_failed(client, execution, model, provider, duration):
    """尽力写入失败事实，不能遮蔽模型侧原始异常。"""
    try:
        client.report_failed(
            execution,
            media_type="video",
            requested_units=int(duration),
            model=model,
            provider=provider,
        )
    except Exception:
        pass


def report_video_completed(client, execution, model, provider, duration, file_path):
    """读取下载后的成品文件，由机杼上传 OSS 并写入成品室和消耗。"""
    path = Path(file_path)
    content_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
    client.report_completed(
        execution,
        media_type="video",
        requested_units=int(duration),
        model=model,
        provider=provider,
        filename=path.name,
        content_type=content_type,
        binary=path.read_bytes(),
    )
