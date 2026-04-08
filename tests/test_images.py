from __future__ import annotations

import unittest

from PIL import Image

from shadowgen_ml_service.utils.images import prepare_working_crop


class ImageUtilityTests(unittest.TestCase):
    def test_prepare_working_crop_keeps_outer_margin(self) -> None:
        image = Image.new("RGBA", (200, 200), (255, 0, 0, 255))
        crop = prepare_working_crop(image, (20, 20, 180, 180), 128, content_scale=0.8)
        self.assertEqual(crop.size, (128, 128))
        self.assertEqual(crop.getpixel((0, 0))[3], 0)
        self.assertEqual(crop.getpixel((64, 64))[3], 255)


if __name__ == "__main__":
    unittest.main()
