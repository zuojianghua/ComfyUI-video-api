try:
    from .kling_video_node import KlingImageToVideo
except ImportError as e:
    print(f"[ComfyUI-video-api] Warning: Could not import KlingImageToVideo: {e}")
    KlingImageToVideo = None

try:
    from .seedance_video_node import SeedanceImageToVideo
except ImportError as e:
    print(f"[ComfyUI-video-api] Warning: Could not import SeedanceImageToVideo: {e}")
    SeedanceImageToVideo = None

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if KlingImageToVideo is not None:
    NODE_CLASS_MAPPINGS["KlingImageToVideo"] = KlingImageToVideo
    NODE_DISPLAY_NAME_MAPPINGS["KlingImageToVideo"] = "Kling Image to Video"

if SeedanceImageToVideo is not None:
    NODE_CLASS_MAPPINGS["SeedanceImageToVideo"] = SeedanceImageToVideo
    NODE_DISPLAY_NAME_MAPPINGS["SeedanceImageToVideo"] = "Seedance Image to Video"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
