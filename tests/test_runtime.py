from __future__ import annotations

import unittest
from unittest.mock import patch

from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.pipeline.runtime import build_runtime


class RuntimeTests(unittest.TestCase):
    def test_auto_runtime_uses_fallback_mode(self) -> None:
        runtime = build_runtime(Settings(runtime_mode="auto"))
        self.assertTrue(runtime.descriptor.degraded)
        self.assertIn("fallback", runtime.descriptor.mode)

    def test_mock_runtime_is_not_degraded(self) -> None:
        runtime = build_runtime(Settings(runtime_mode="mock"))
        self.assertFalse(runtime.descriptor.degraded)
        self.assertEqual(runtime.descriptor.mode, "mock")

    def test_auto_runtime_uses_real_geometry_when_available(self) -> None:
        with patch("shadowgen_ml_service.pipeline.runtime.probe_geocalib") as probe_geocalib:
            with patch("shadowgen_ml_service.pipeline.runtime.RealGeometryEstimator") as real_geometry:
                probe_geocalib.return_value.model_name = "GeoCalib"
                probe_geocalib.return_value.model_version = "bootstrap-probe"
                probe_geocalib.return_value.available = True
                runtime = build_runtime(Settings(runtime_mode="auto"))
                self.assertFalse(next(item for item in runtime.descriptor.components if item.name == "geometry_estimator").using_mock)
                real_geometry.assert_called_once()

    def test_real_runtime_falls_back_to_mock_geometry_when_init_fails(self) -> None:
        with patch("shadowgen_ml_service.pipeline.runtime.probe_geocalib") as probe_geocalib:
            with patch("shadowgen_ml_service.pipeline.runtime.RealGeometryEstimator", side_effect=RuntimeError("init failed")):
                probe_geocalib.return_value.model_name = "GeoCalib"
                probe_geocalib.return_value.model_version = "bootstrap-probe"
                probe_geocalib.return_value.available = True
                runtime = build_runtime(Settings(runtime_mode="real"))
                geometry_component = next(item for item in runtime.descriptor.components if item.name == "geometry_estimator")
                self.assertTrue(geometry_component.using_mock)
                self.assertEqual(geometry_component.implementation, "mock-fallback")

    def test_auto_runtime_uses_real_detector_when_available(self) -> None:
        with patch("shadowgen_ml_service.pipeline.runtime.probe_grounding_dino") as probe_grounding:
            with patch("shadowgen_ml_service.pipeline.runtime.RealDetector") as real_detector:
                probe_grounding.return_value.model_name = "IDEA-Research/grounding-dino-base"
                probe_grounding.return_value.model_version = "bootstrap-probe"
                probe_grounding.return_value.available = True
                runtime = build_runtime(Settings(runtime_mode="auto"))
                detector_component = next(item for item in runtime.descriptor.components if item.name == "detector")
                self.assertFalse(detector_component.using_mock)
                self.assertEqual(detector_component.implementation, "real")
                real_detector.assert_called_once()

    def test_real_runtime_falls_back_to_mock_detector_when_init_fails(self) -> None:
        with patch("shadowgen_ml_service.pipeline.runtime.probe_grounding_dino") as probe_grounding:
            with patch("shadowgen_ml_service.pipeline.runtime.RealDetector", side_effect=RuntimeError("init failed")):
                probe_grounding.return_value.model_name = "IDEA-Research/grounding-dino-base"
                probe_grounding.return_value.model_version = "bootstrap-probe"
                probe_grounding.return_value.available = True
                runtime = build_runtime(Settings(runtime_mode="real"))
                detector_component = next(item for item in runtime.descriptor.components if item.name == "detector")
                self.assertTrue(detector_component.using_mock)
                self.assertEqual(detector_component.implementation, "mock-fallback")

    def test_auto_runtime_uses_real_segmenter_when_available(self) -> None:
        with patch("shadowgen_ml_service.pipeline.runtime.probe_birefnet") as probe_birefnet:
            with patch("shadowgen_ml_service.pipeline.runtime.RealSegmenter") as real_segmenter:
                probe_birefnet.return_value.model_name = "ZhengPeng7/BiRefNet_lite-matting"
                probe_birefnet.return_value.model_version = "bootstrap-probe"
                probe_birefnet.return_value.available = True
                runtime = build_runtime(Settings(runtime_mode="auto"))
                segmenter_component = next(item for item in runtime.descriptor.components if item.name == "segmenter")
                self.assertFalse(segmenter_component.using_mock)
                self.assertEqual(segmenter_component.implementation, "real")
                real_segmenter.assert_called_once()

    def test_real_runtime_falls_back_to_mock_segmenter_when_init_fails(self) -> None:
        with patch("shadowgen_ml_service.pipeline.runtime.probe_birefnet") as probe_birefnet:
            with patch("shadowgen_ml_service.pipeline.runtime.RealSegmenter", side_effect=RuntimeError("init failed")):
                probe_birefnet.return_value.model_name = "ZhengPeng7/BiRefNet_lite-matting"
                probe_birefnet.return_value.model_version = "bootstrap-probe"
                probe_birefnet.return_value.available = True
                runtime = build_runtime(Settings(runtime_mode="real"))
                segmenter_component = next(item for item in runtime.descriptor.components if item.name == "segmenter")
                self.assertTrue(segmenter_component.using_mock)
                self.assertEqual(segmenter_component.implementation, "mock-fallback")


if __name__ == "__main__":
    unittest.main()
