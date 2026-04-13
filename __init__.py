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

try:
    from .video_output_nodes import PreviewVideo, SaveVideo
except ImportError as e:
    print(f"[ComfyUI-video-api] Warning: Could not import video output nodes: {e}")
    PreviewVideo = None
    SaveVideo = None

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if KlingImageToVideo is not None:
    NODE_CLASS_MAPPINGS["KlingImageToVideo"] = KlingImageToVideo
    NODE_DISPLAY_NAME_MAPPINGS["KlingImageToVideo"] = "Kling Image to Video"

if SeedanceImageToVideo is not None:
    NODE_CLASS_MAPPINGS["SeedanceImageToVideo"] = SeedanceImageToVideo
    NODE_DISPLAY_NAME_MAPPINGS["SeedanceImageToVideo"] = "Seedance Image to Video"

if PreviewVideo is not None:
    NODE_CLASS_MAPPINGS["PreviewVideoFromURL"] = PreviewVideo
    NODE_DISPLAY_NAME_MAPPINGS["PreviewVideoFromURL"] = "Preview Video (URL)"

if SaveVideo is not None:
    NODE_CLASS_MAPPINGS["SaveVideoFromURL"] = SaveVideo
    NODE_DISPLAY_NAME_MAPPINGS["SaveVideoFromURL"] = "Save Video (URL)"

WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
