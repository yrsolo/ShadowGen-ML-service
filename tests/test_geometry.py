from __future__ import annotations

import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from shadowgen_ml_service.adapters.real import RealGeometryEstimator


class FakeScalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class FakeGeoCalibModel:
    def __init__(self, result) -> None:
        self.result = result

    def load_image(self, image):
        return image

    def calibrate(self, image):
        return self.result


class FakeGeoCalibModule:
    def __init__(self, result) -> None:
        self._result = result

    def GeoCalib(self):
        return FakeGeoCalibModel(self._result)


class GeometryEstimatorTests(unittest.TestCase):
    def test_real_geometry_estimator_parses_result(self) -> None:
        result = {
            "camera": SimpleNamespace(vfov=FakeScalar(math.radians(51.0))),
            "gravity": SimpleNamespace(pitch=FakeScalar(math.radians(-6.0)), roll=FakeScalar(math.radians(3.0))),
            "confidence": FakeScalar(0.82),
        }
        estimator = RealGeometryEstimator(geocalib_module=FakeGeoCalibModule(result))
        geometry = estimator.estimate(Image.new("RGB", (64, 64), "white"))
        self.assertAlmostEqual(geometry.camera_fov, 51.0, places=2)
        self.assertAlmostEqual(geometry.camera_pitch, -6.0, places=2)
        self.assertAlmostEqual(geometry.camera_roll, 3.0, places=2)
        self.assertAlmostEqual(geometry.confidence, 0.82, places=2)

    def test_real_geometry_estimator_uses_proxy_confidence(self) -> None:
        result = {
            "camera": SimpleNamespace(vfov=FakeScalar(math.radians(60.0))),
            "gravity": SimpleNamespace(pitch=FakeScalar(math.radians(-2.0)), roll=FakeScalar(math.radians(1.5))),
        }
        estimator = RealGeometryEstimator(geocalib_module=FakeGeoCalibModule(result))
        geometry = estimator.estimate(Image.new("RGB", (64, 64), "white"))
        self.assertGreaterEqual(geometry.confidence, 0.05)
        self.assertLessEqual(geometry.confidence, 0.95)

    def test_real_geometry_estimator_bubbles_runtime_errors(self) -> None:
        with patch("shadowgen_ml_service.adapters.real._import_module", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                RealGeometryEstimator()


if __name__ == "__main__":
    unittest.main()
