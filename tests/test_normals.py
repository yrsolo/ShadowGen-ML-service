from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from shadowgen_ml_service.adapters.real import StableNormalEstimator


class FakeHub:
    def __init__(self) -> None:
        self.last_load = None

    def load(self, repo, variant, **kwargs):
        self.last_load = (repo, variant, kwargs)

        class Predictor:
            def __call__(self, image, **call_kwargs):
                self.last_image_size = image.size
                self.last_call_kwargs = call_kwargs
                return Image.new("RGB", image.size, (128, 180, 255))

        return Predictor()


class FakeTorch:
    class cuda:
        @staticmethod
        def is_available():
            return True

    def __init__(self) -> None:
        self.hub = FakeHub()


class StableNormalTests(unittest.TestCase):
    def test_stable_normal_estimator_uses_torch_hub_predictor(self) -> None:
        fake_torch = FakeTorch()
        with tempfile.TemporaryDirectory() as tmp_dir:
            estimator = StableNormalEstimator(
                torch_module=fake_torch,
                model_variant="StableNormal_turbo",
                target_device="cuda:0",
                cache_dir=Path(tmp_dir),
                resolution=768,
            )
            result = estimator.estimate(Image.new("RGBA", (96, 80), (255, 255, 255, 255)))

        self.assertEqual(result.normal_map.size, (96, 80))
        self.assertEqual(estimator.device_label, "cuda:0")
        repo, variant, kwargs = fake_torch.hub.last_load
        self.assertEqual(repo, "Stable-X/StableNormal")
        self.assertEqual(variant, "StableNormal_turbo")
        self.assertEqual(kwargs["device"], "cuda:0")
        self.assertIn("local_cache_dir", kwargs)


if __name__ == "__main__":
    unittest.main()
