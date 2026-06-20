from __future__ import annotations

import unittest

from PIL import Image

from shadowgen_ml_service.infrastructure.stages.foreground_refinement.fast_foreground_estimation import FastForegroundColorEstimator
from shadowgen_ml_service.utils.images import asset_to_pil


class ForegroundRefinerTests(unittest.TestCase):
    def test_fast_foreground_estimator_preserves_alpha_and_size(self) -> None:
        image = Image.new("RGBA", (64, 64), (24, 80, 180, 255))
        alpha = Image.new("L", (64, 64), 0)
        for x in range(16, 48):
            for y in range(12, 52):
                alpha.putpixel((x, y), 160 if x < 24 else 255)

        estimator = FastForegroundColorEstimator(coarse_radius=9, refine_radius=3)
        result = estimator.refine(image, alpha)
        cutout = asset_to_pil(result.cutout_rgba)

        self.assertEqual(cutout.size, image.size)
        self.assertEqual(cutout.getchannel("A").getextrema(), alpha.getextrema())


if __name__ == "__main__":
    unittest.main()
