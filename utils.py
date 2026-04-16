"""Shared utilities for ComfyUI-video-api.

Provides:
- ComfyUI tensor <-> PIL Image conversion
- PIL Image <-> Base64 encoding
- Video download helper
- ComfyUI output path generation
- Generic async task polling framework
"""

import base64
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Callable, List, Optional, TypeVar

import numpy as np
import requests
import torch
from PIL import Image

T = TypeVar("T")

_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
_config_cache: Optional[dict] = None


def load_config() -> dict:
    global _config_cache
    if _config_cache is None:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            _config_cache = json.load(f)
    return _config_cache


def get_provider_config(provider: str) -> dict:
    return load_config().get(provider, {})

# ---------------------------------------------------------------------------
# Image conversion
# ---------------------------------------------------------------------------


def tensor_to_pils(tensor: torch.Tensor) -> List[Image.Image]:
    """Convert ComfyUI image tensor (B,H,W,C) float 0-1 to list of PIL Images."""
    tensor = tensor.cpu().detach()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    images: list[Image.Image] = []
    for i in range(tensor.shape[0]):
        arr = (tensor[i].numpy() * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)
    return images


def pils_to_tensor(images: List[Image.Image]) -> torch.Tensor:
    """Convert list of PIL Images to ComfyUI tensor (B,H,W,3) float 0-1."""
    tensors = []
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(arr))
    return torch.stack(tensors, dim=0)


# ---------------------------------------------------------------------------
# Base64 helpers
# ---------------------------------------------------------------------------


def pil_to_base64(pil_img: Image.Image, fmt: str = "jpeg") -> str:
    """Return raw base64 string (no data-url prefix)."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    fmt = fmt.lower()
    if fmt not in ("jpeg", "png"):
        fmt = "jpeg"
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt.upper(), quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def pil_to_base64_data_url(pil_img: Image.Image, fmt: str = "jpeg") -> str:
    """Return ``data:image/<mime>;base64,...`` string."""
    b64 = pil_to_base64(pil_img, fmt=fmt)
    mime = "image/jpeg" if fmt.lower() == "jpeg" else "image/png"
    return f"data:{mime};base64,{b64}"


def base64_to_pil(b64_string: str) -> Image.Image:
    """Decode a base64 (or data-url) string into a PIL Image."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    raw = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(raw))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ---------------------------------------------------------------------------
# Video download
# ---------------------------------------------------------------------------


def download_video(url: str, save_path: str, timeout: int = 120) -> str:
    """Download a video from *url* and write it to *save_path*.

    Returns the absolute path of the saved file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    resp = requests.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return os.path.abspath(save_path)


# ---------------------------------------------------------------------------
# ComfyUI output path helpers
# ---------------------------------------------------------------------------


def make_video_ui_result(file_path: str) -> dict:
    """Build the ``ui`` dict so a saved video appears in ComfyUI history.

    Returns the same format as built-in ``SaveImage`` / ``PreviewVideo``::

        {"images": [{"filename": ..., "subfolder": ..., "type": "output"}],
         "animated": (True,)}

    Note: the key is ``"images"`` (not ``"videos"``) — this is a ComfyUI
    convention that applies to all visual media including video.
    """
    try:
        import folder_paths  # type: ignore[import-untyped]

        output_dir = folder_paths.get_output_directory()
    except Exception:
        output_dir = os.path.join(os.path.dirname(__file__), "output")

    abs_path = os.path.abspath(file_path)
    rel = os.path.relpath(abs_path, output_dir)
    subfolder = os.path.dirname(rel)
    filename = os.path.basename(rel)

    return {
        "images": [{"filename": filename, "subfolder": subfolder, "type": "output"}],
        "animated": (True,),
    }


def get_output_video_path(prefix: str = "video", ext: str = ".mp4") -> str:
    """Generate a unique file path under ComfyUI's output directory.

    Uses ``folder_paths`` when available (running inside ComfyUI), otherwise
    falls back to a local ``output/`` directory for standalone testing.
    """
    try:
        import folder_paths  # type: ignore[import-untyped]

        output_dir = folder_paths.get_output_directory()
    except Exception:
        output_dir = os.path.join(os.path.dirname(__file__), "output")

    if "/" in prefix or "\\" in prefix:
        sub_dir = os.path.dirname(prefix)
        base_prefix = os.path.basename(prefix)
        scan_dir = os.path.join(output_dir, sub_dir)
    else:
        sub_dir = ""
        base_prefix = prefix
        scan_dir = output_dir

    os.makedirs(scan_dir, exist_ok=True)

    pattern = re.compile(rf"{re.escape(base_prefix)}_(\d+)\..+", re.IGNORECASE)
    max_counter = 0
    for name in os.listdir(scan_dir):
        m = pattern.fullmatch(name)
        if m:
            c = int(m.group(1))
            if c > max_counter:
                max_counter = c

    filename = f"{base_prefix}_{max_counter + 1:05d}{ext}"
    if sub_dir:
        return os.path.join(output_dir, sub_dir, filename)
    return os.path.join(output_dir, filename)


# ---------------------------------------------------------------------------
# Generic async polling
# ---------------------------------------------------------------------------


def poll_until_complete(
    poll_fn: Callable[[], T],
    is_done: Callable[[T], bool],
    is_failed: Callable[[T], bool],
    extract_error: Callable[[T], str],
    interval: float = 5.0,
    timeout: float = 300.0,
    log_prefix: str = "[poll]",
) -> T:
    """Call *poll_fn* repeatedly until *is_done* or *is_failed* returns True.

    Args:
        poll_fn: Zero-arg callable that returns the latest status object.
        is_done: Predicate – return True when the task has succeeded.
        is_failed: Predicate – return True when the task has failed.
        extract_error: Extract an error message string from a failed result.
        interval: Seconds between polls.
        timeout: Maximum total wait time in seconds.
        log_prefix: Prefix for progress log lines.

    Returns:
        The final status object (on success).

    Raises:
        RuntimeError: On failure or timeout.
    """
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            raise RuntimeError(f"{log_prefix} Timeout after {elapsed:.0f}s")

        result = poll_fn()

        if is_done(result):
            print(f"{log_prefix} Completed in {elapsed:.0f}s")
            return result

        if is_failed(result):
            msg = extract_error(result)
            raise RuntimeError(f"{log_prefix} Failed: {msg}")

        print(f"{log_prefix} Waiting... ({elapsed:.0f}s elapsed)")
        time.sleep(interval)
