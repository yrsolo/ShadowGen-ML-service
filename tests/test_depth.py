from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from shadowgen_ml_service.adapters.real import RealDepthEstimator
from shadowgen_ml_service.core.stage_io import DepthInput


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTorch:
    class cuda:
        @staticmethod
        def is_available():
            return True

    @staticmethod
    def no_grad():
        return FakeNoGrad()


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def to(self, device):
        self.device = device
        return self


class FakeBatch(dict):
    def to(self, device):
        return FakeBatch({key: value.to(device) if hasattr(value, "to") else value for key, value in self.items()})


class FakeProcessor:
    last_from_pretrained = None

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.last_from_pretrained = (model_id, kwargs)
        return cls()

    def __call__(self, images, return_tensors):
        return FakeBatch({"pixel_values": FakeTensor(np.ones((1, 3, 16, 16), dtype=np.float32))})


class FakeDepthOutput:
    def __init__(self):
        self.predicted_depth = FakeTensor(np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(1, 16, 16))


class FakeDepthModel:
    last_from_pretrained = None

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.last_from_pretrained = (model_id, kwargs)
        return cls()

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def __call__(self, **inputs):
        self.last_inputs = inputs
        return FakeDepthOutput()


class FakeTransformersDepthModule:
    AutoImageProcessor = FakeProcessor
    AutoModelForDepthEstimation = FakeDepthModel


class DepthTests(unittest.TestCase):
    def test_real_depth_estimator_returns_resized_depth_map(self) -> None:
        estimator = RealDepthEstimator(
            transformers_module=FakeTransformersDepthModule,
            torch_module=FakeTorch(),
            model_id="depth-anything/Depth-Anything-V2-Small-hf",
            target_device="cuda:0",
        )
        image = Image.new("RGBA", (96, 80), (255, 255, 255, 255))
        mask = Image.new("L", image.size, 255)
        result = estimator.estimate(DepthInput(image=image, mask=mask))
        self.assertEqual(result.depth_map.size, image.size)
        self.assertEqual(estimator.device_label, "cuda:0")
        self.assertEqual(estimator._model.device, "cuda:0")


if __name__ == "__main__":
    unittest.main()
