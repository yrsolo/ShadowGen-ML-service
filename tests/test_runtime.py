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

    def test_auto_runtime_uses_real_depth_when_available(self) -> None:
        with patch("shadowgen_ml_service.pipeline.runtime.probe_depth_anything") as probe_depth_anything:
            with patch("shadowgen_ml_service.pipeline.runtime.RealDepthEstimator") as real_depth:
                probe_depth_anything.return_value.model_name = "depth-anything/Depth-Anything-V2-Small-hf"
                probe_depth_anything.return_value.model_version = "bootstrap-probe"
                probe_depth_anything.return_value.available = True
                runtime = build_runtime(Settings(runtime_mode="auto"))
                depth_component = next(item for item in runtime.descriptor.components if item.name == "depth_estimator")
                self.assertFalse(depth_component.using_mock)
                self.assertEqual(depth_component.implementation, "real")
                real_depth.assert_called_once()

    def test_real_runtime_falls_back_to_mock_depth_when_init_fails(self) -> None:
        with patch("shadowgen_ml_service.pipeline.runtime.probe_depth_anything") as probe_depth_anything:
            with patch("shadowgen_ml_service.pipeline.runtime.RealDepthEstimator", side_effect=RuntimeError("init failed")):
                probe_depth_anything.return_value.model_name = "depth-anything/Depth-Anything-V2-Small-hf"
                probe_depth_anything.return_value.model_version = "bootstrap-probe"
                probe_depth_anything.return_value.available = True
                runtime = build_runtime(Settings(runtime_mode="real"))
                depth_component = next(item for item in runtime.descriptor.components if item.name == "depth_estimator")
                self.assertTrue(depth_component.using_mock)
                self.assertEqual(depth_component.implementation, "mock-fallback")

    def test_auto_runtime_uses_stable_normal_when_available(self) -> None:
        with patch("shadowgen_ml_service.pipeline.runtime.probe_stable_normal") as probe_stable_normal:
            with patch("shadowgen_ml_service.pipeline.runtime.StableNormalEstimator") as stable_normal:
                probe_stable_normal.return_value.model_name = "Stable-X/StableNormal"
                probe_stable_normal.return_value.model_version = "bootstrap-probe"
                probe_stable_normal.return_value.available = True
                probe_stable_normal.return_value.detail = "CUDA runtime detected"
                runtime = build_runtime(Settings(runtime_mode="auto"))
                normal_component = next(item for item in runtime.descriptor.components if item.name == "normal_estimator")
                self.assertFalse(normal_component.using_mock)
                self.assertEqual(normal_component.model_name, "Stable-X/StableNormal")
                stable_normal.assert_called_once()

    def test_real_runtime_falls_back_to_depth_normals_when_stable_normal_init_fails(self) -> None:
        with patch("shadowgen_ml_service.pipeline.runtime.probe_stable_normal") as probe_stable_normal:
            with patch("shadowgen_ml_service.pipeline.runtime.StableNormalEstimator", side_effect=RuntimeError("init failed")):
                probe_stable_normal.return_value.model_name = "Stable-X/StableNormal"
                probe_stable_normal.return_value.model_version = "bootstrap-probe"
                probe_stable_normal.return_value.available = True
                probe_stable_normal.return_value.detail = "CUDA runtime detected"
                runtime = build_runtime(Settings(runtime_mode="real"))
                normal_component = next(item for item in runtime.descriptor.components if item.name == "normal_estimator")
                self.assertFalse(normal_component.using_mock)
                self.assertEqual(normal_component.model_name, "normal-map-from-depth")
                self.assertIn("StableNormal init failed", normal_component.detail)

    def test_auto_runtime_uses_real_shadow_when_weights_exist(self) -> None:
        with patch("shadowgen_ml_service.pipeline.runtime.probe_shadow_pix2pix") as probe_shadow:
            with patch("shadowgen_ml_service.pipeline.runtime.Pix2PixShadowGenerator") as real_shadow:
                probe_shadow.return_value.model_name = "legacy-shadow-pix2pix"
                probe_shadow.return_value.model_version = "bootstrap-probe"
                probe_shadow.return_value.available = True
                probe_shadow.return_value.detail = "CUDA runtime detected"
                runtime = build_runtime(Settings(runtime_mode="auto"))
                shadow_component = next(item for item in runtime.descriptor.components if item.name == "shadow_generator")
                self.assertFalse(shadow_component.using_mock)
                self.assertEqual(shadow_component.implementation, "real")
                real_shadow.assert_called_once()

    def test_real_runtime_falls_back_to_stub_shadow_when_init_fails(self) -> None:
        with patch("shadowgen_ml_service.pipeline.runtime.probe_shadow_pix2pix") as probe_shadow:
            with patch("shadowgen_ml_service.pipeline.runtime.Pix2PixShadowGenerator", side_effect=RuntimeError("init failed")):
                probe_shadow.return_value.model_name = "legacy-shadow-pix2pix"
                probe_shadow.return_value.model_version = "bootstrap-probe"
                probe_shadow.return_value.available = True
                probe_shadow.return_value.detail = "CUDA runtime detected"
                runtime = build_runtime(Settings(runtime_mode="real"))
                shadow_component = next(item for item in runtime.descriptor.components if item.name == "shadow_generator")
                self.assertTrue(shadow_component.using_mock)
                self.assertEqual(shadow_component.implementation, "mock-fallback")


if __name__ == "__main__":
    unittest.main()
