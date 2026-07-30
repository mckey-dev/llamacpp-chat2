"""multimodal（画像長辺リサイズ）のユニットテスト。"""

from __future__ import annotations

import base64
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from core.multimodal import (  # noqa: E402
    _image_bytes_for_api,
    file_to_data_url,
)


def _save_rgb(path: Path, size: tuple[int, int]) -> None:
    Image.new("RGB", size, color=(20, 40, 60)).save(path, format="PNG")


class ImageResizeTests(unittest.TestCase):
    """長い辺リサイズのテスト。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_large_image_resized_to_1024(self) -> None:
        path = self.dir / "large.png"
        _save_rgb(path, (2000, 1000))
        raw, mime = _image_bytes_for_api(str(path), max_long_edge=1024)
        self.assertEqual("image/jpeg", mime)
        with Image.open(BytesIO(raw)) as im:
            self.assertEqual(1024, max(im.size))
            self.assertEqual(512, min(im.size))

    def test_small_image_unchanged_bytes(self) -> None:
        path = self.dir / "small.png"
        _save_rgb(path, (800, 600))
        original = path.read_bytes()
        raw, mime = _image_bytes_for_api(str(path), max_long_edge=1024)
        self.assertEqual(original, raw)
        self.assertEqual("image/png", mime)

    def test_max_long_edge_zero_skips_resize(self) -> None:
        path = self.dir / "huge.png"
        _save_rgb(path, (2000, 1500))
        original = path.read_bytes()
        raw, _mime = _image_bytes_for_api(str(path), max_long_edge=0)
        self.assertEqual(original, raw)

    def test_data_url_prefix(self) -> None:
        path = self.dir / "wide.png"
        _save_rgb(path, (1600, 400))
        url = file_to_data_url(str(path), max_long_edge=1024)
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        b64 = url.split(",", 1)[1]
        data = base64.b64decode(b64)
        with Image.open(BytesIO(data)) as im:
            self.assertEqual(1024, im.size[0])
            self.assertEqual(256, im.size[1])

    def test_rgba_resized_stays_png(self) -> None:
        path = self.dir / "alpha.png"
        Image.new("RGBA", (1500, 1500), color=(10, 20, 30, 128)).save(
            path, format="PNG"
        )
        raw, mime = _image_bytes_for_api(str(path), max_long_edge=1024)
        self.assertEqual("image/png", mime)
        with Image.open(BytesIO(raw)) as im:
            self.assertEqual((1024, 1024), im.size)
            self.assertIn(im.mode, ("RGBA", "P"))


if __name__ == "__main__":
    unittest.main()
