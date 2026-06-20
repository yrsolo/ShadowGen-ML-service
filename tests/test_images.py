from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from shadowgen_ml_service.utils.images import normals_from_depth, prepare_working_crop


class ImageUtilityTests(unittest.TestCase):
    def test_prepare_working_crop_keeps_outer_margin(self) -> None:
        image = Image.new("RGBA", (200, 200), (255, 0, 0, 255))
        crop = prepare_working_crop(image, (20, 20, 180, 180), 128, content_scale=0.8)
        self.assertEqual(crop.size, (128, 128))
        self.assertEqual(crop.getpixel((0, 0))[3], 0)
        self.assertEqual(crop.getpixel((64, 64))[3], 255)

    def test_prepare_working_crop_uses_scale_as_max_content_fraction(self) -> None:
        image = Image.new("RGBA", (200, 100), (255, 0, 0, 255))
        crop = prepare_working_crop(image, (0, 0, 200, 100), 100, content_scale=0.68)
        alpha_bbox = crop.getchannel("A").getbbox()

        self.assertEqual(alpha_bbox, (16, 33, 84, 67))

    def test_normals_from_depth_suppresses_silhouette_edges_and_keeps_internal_variation(self) -> None:
        depth = np.zeros((64, 64), dtype=np.uint8)
        yy, xx = np.indices(depth.shape)
        mask = ((xx - 32) ** 2 + (yy - 32) ** 2) <= 20 ** 2
        bump = 180.0 - np.sqrt((xx - 32) ** 2 + (yy - 32) ** 2) * 5.0
        depth[mask] = np.clip(bump[mask], 0, 255).astype(np.uint8)
        normals = normals_from_depth(Image.fromarray(depth, mode="L"))
        arr = np.asarray(normals, dtype=np.uint8)

        outside = arr[4, 4]
        boundary = arr[12, 32]
        interior_a = arr[24, 32]
        interior_b = arr[32, 24]

        self.assertTrue(np.all(np.abs(outside.astype(np.int16) - np.array([127, 127, 255])) <= 12))
        self.assertTrue(np.all(np.abs(boundary.astype(np.int16) - np.array([127, 127, 255])) <= 50))
        self.assertGreater(np.abs(interior_a.astype(np.int16) - interior_b.astype(np.int16)).sum(), 8)


if __name__ == "__main__":
    unittest.main()
