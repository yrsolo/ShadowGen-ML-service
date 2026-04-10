from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from shadowgen_ml_service.adapters.real import RealDetector, select_primary_detection
from shadowgen_ml_service.core.stage_io import DetectionInput


class FakeTensor:
    def __init__(self, values):
        self._values = values
        self.last_device = None

    def reshape(self, *_shape):
        return self

    def tolist(self):
        return list(self._values)

    def __iter__(self):
        return iter(self._values)

    def __array__(self):
        return np.asarray(self._values, dtype=np.float32)

    def to(self, device):
        self.last_device = device
        return self


class FakeProcessor:
    last_from_pretrained = None

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.last_from_pretrained = (model_id, kwargs)
        return cls()

    def __call__(self, images, text, return_tensors):
        self.last_call = {"text": text, "return_tensors": return_tensors, "size": images.size}
        return {"input_ids": FakeTensor([1, 2, 3]), "pixel_values": FakeTensor([4, 5, 6])}

    def post_process_grounded_object_detection(self, outputs, input_ids, threshold, text_threshold, target_sizes):
        self.last_post_process = {
            "outputs": outputs,
            "input_ids": input_ids,
            "threshold": threshold,
            "text_threshold": text_threshold,
            "target_sizes": target_sizes,
        }
        return [
            {
                "boxes": [
                    FakeTensor([20, 30, 100, 200]),
                    FakeTensor([18, 28, 110, 220]),
                ],
                "scores": [0.90, 0.88],
            }
        ]


class FakeModel:
    last_from_pretrained = None

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.last_from_pretrained = (model_id, kwargs)
        return cls()

    def __call__(self, **inputs):
        self.last_inputs = inputs
        return {"logits": "fake"}

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self


class FakeTransformersModule:
    GroundingDinoProcessor = FakeProcessor
    GroundingDinoForObjectDetection = FakeModel


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


class DetectorTests(unittest.TestCase):
    def test_select_primary_detection_prefers_larger_area_when_scores_are_close(self) -> None:
        bbox, score = select_primary_detection(
            [
                ((10, 10, 50, 50), 0.90),
                ((12, 12, 80, 100), 0.88),
            ]
        )
        self.assertEqual(bbox, (12, 12, 80, 100))
        self.assertEqual(score, 0.88)

    def test_real_detector_returns_primary_bbox(self) -> None:
        detector = RealDetector(
            transformers_module=FakeTransformersModule,
            torch_module=FakeTorch(),
            model_id="IDEA-Research/grounding-dino-base",
            prompt="object.",
            box_threshold=0.25,
            text_threshold=0.2,
            target_device="cuda:0",
        )
        result = detector.detect(DetectionInput(image=Image.new("RGB", (320, 240), "white"), padding_px=100))
        self.assertEqual(result.bbox, (18, 28, 110, 220))
        self.assertAlmostEqual(result.confidence, 0.88, places=3)
        self.assertEqual(FakeProcessor.last_from_pretrained[0], "IDEA-Research/grounding-dino-base")
        self.assertEqual(FakeModel.last_from_pretrained[0], "IDEA-Research/grounding-dino-base")
        self.assertEqual(detector.device_label, "cuda:0")
        self.assertEqual(detector._model.device, "cuda:0")


if __name__ == "__main__":
    unittest.main()
