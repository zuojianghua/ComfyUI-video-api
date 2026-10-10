"""kling_api 纯协议逻辑测试：python -m unittest tests.test_kling_api（在仓库根目录执行，无需 torch）。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kling_api as k  # noqa: E402


def _build(**overrides):
    values = dict(
        model="kling-3.0", prompt="模特转身", first_frame="BASE64FIRST", last_frame=None,
        resolution="1080p", duration=5, audio=False,
    )
    values.update(overrides)
    return k.build_image_to_video_request(**values)


class BuildRequestTests(unittest.TestCase):
    def test_kling_30_snapshot(self):
        path, body = _build(last_frame="BASE64LAST", audio=True, resolution="4k", duration=12)
        self.assertEqual(path, "/image-to-video/kling-3.0")
        self.assertEqual(body, {
            "contents": [
                {"type": "prompt", "text": "模特转身"},
                {"type": "first_frame", "url": "BASE64FIRST"},
                {"type": "last_frame", "url": "BASE64LAST"},
            ],
            "settings": {"resolution": "4k", "duration": 12, "audio": "native", "multi_shot": False},
            "options": {"watermark_info": {"enabled": False}},
        })

    def test_kling_26_has_no_multi_shot_and_empty_prompt_is_omitted(self):
        path, body = _build(model="kling-2.6", prompt="  ", duration=10)
        self.assertEqual(path, "/image-to-video/kling-2.6")
        self.assertNotIn("multi_shot", body["settings"])
        self.assertEqual(body["contents"], [{"type": "first_frame", "url": "BASE64FIRST"}])

    def test_capability_rules(self):
        bad = [
            dict(model="kling-v3"),                                    # 旧版模型名
            dict(first_frame=""),
            dict(duration=2), dict(duration=16),
            dict(model="kling-2.6", duration=7),                       # 2.6 仅 5/10
            dict(model="kling-2.6", resolution="4k"),
            dict(model="kling-2.6", resolution="720p", audio=True),    # 有声仅 1080p
            dict(model="kling-2.6", resolution="720p", last_frame="X"),  # 尾帧仅 1080p
        ]
        for overrides in bad:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                _build(**overrides)
        # 合法组合
        _build(model="kling-2.6", resolution="1080p", audio=True, last_frame="X", duration=10)
        _build(model="kling-3.0", resolution="720p", audio=True, last_frame="X", duration=3)


class ImageSizeTests(unittest.TestCase):
    def test_limits(self):
        k.check_image_size(300, 300, "首帧")
        k.check_image_size(750, 300, "首帧")   # 2.5:1 边界
        for w, h in ((177, 119), (300, 299), (751, 300), (300, 751)):
            with self.subTest(size=(w, h)), self.assertRaises(ValueError):
                k.check_image_size(w, h, "首帧")


class ResponseParsingTests(unittest.TestCase):
    def test_create_response(self):
        self.assertEqual(k.parse_create_response(200, {"code": 0, "data": {"id": "937"}}), "937")
        for status, data in ((429, {"code": 1303, "message": "并发超限"}), (200, {"code": 1201, "message": "bad"}),
                             (200, {"code": 0, "data": {}}), (500, None)):
            with self.subTest(status=status, data=data), self.assertRaises(RuntimeError):
                k.parse_create_response(status, data)

    def test_task_item_and_video_url(self):
        resp = {"code": 0, "data": [{"id": "937", "status": "succeeded",
                                     "outputs": [{"type": "video", "url": "https://kling.example/v.mp4"}]}]}
        item = k.parse_task_item(resp, "937")
        self.assertEqual(k.extract_video_url(item), "https://kling.example/v.mp4")
        with self.assertRaises(RuntimeError):
            k.parse_task_item({"code": 0, "data": []}, "937")
        with self.assertRaises(RuntimeError):
            k.extract_video_url({"outputs": []})


if __name__ == "__main__":
    unittest.main()
