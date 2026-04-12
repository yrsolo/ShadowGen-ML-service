from __future__ import annotations

import unittest
from unittest.mock import patch

from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.pipeline.runtime import build_runtime


class RuntimeTests(unittest.TestCase):
    def test_auto_runtime_uses_local_fallback_mode(self) -> None:
        runtime = build_runtime(Settings(runtime_mode="auto"))
        self.assertIn(runtime.descriptor.mode, {"local", "local-fallback"})
        self.assertEqual(runtime.descriptor.execution_default_backend, "local")

    def test_mock_runtime_is_mock_mode(self) -> None:
        runtime = build_runtime(Settings(runtime_mode="mock"))
        self.assertEqual(runtime.descriptor.mode, "mock")
        self.assertFalse(runtime.descriptor.degraded)

    def test_auto_runtime_registers_local_detector_when_available(self) -> None:
        with patch("shadowgen_ml_service.bootstrap.container.probe_grounding_dino") as probe_grounding:
            with patch("shadowgen_ml_service.bootstrap.container.RealDetector") as real_detector:
                probe_grounding.return_value.model_name = "IDEA-Research/grounding-dino-base"
                probe_grounding.return_value.model_version = "bootstrap-probe"
                probe_grounding.return_value.available = True
                probe_grounding.return_value.detail = None
                runtime = build_runtime(Settings(runtime_mode="auto"))
                detector_component = next(item for item in runtime.descriptor.components if item.name == "detector")
                self.assertFalse(detector_component.using_mock)
                self.assertEqual(detector_component.backend_kind, "local")
                real_detector.assert_called_once()

    def test_real_runtime_falls_back_to_mock_detector_when_local_init_fails(self) -> None:
        with patch("shadowgen_ml_service.bootstrap.container.probe_grounding_dino") as probe_grounding:
            with patch("shadowgen_ml_service.bootstrap.container.RealDetector", side_effect=RuntimeError("init failed")):
                probe_grounding.return_value.model_name = "IDEA-Research/grounding-dino-base"
                probe_grounding.return_value.model_version = "bootstrap-probe"
                probe_grounding.return_value.available = True
                probe_grounding.return_value.detail = None
                runtime = build_runtime(Settings(runtime_mode="real"))
                detector_component = next(item for item in runtime.descriptor.components if item.name == "detector")
                self.assertTrue(detector_component.using_mock)
                self.assertEqual(detector_component.implementation, "mock-fallback")

    def test_auto_runtime_uses_local_geometry_when_available(self) -> None:
        with patch("shadowgen_ml_service.bootstrap.container.probe_geocalib") as probe_geocalib:
            with patch("shadowgen_ml_service.bootstrap.container.RealGeometryEstimator") as real_geometry:
                probe_geocalib.return_value.model_name = "GeoCalib"
                probe_geocalib.return_value.model_version = "bootstrap-probe"
                probe_geocalib.return_value.available = True
                probe_geocalib.return_value.detail = None
                runtime = build_runtime(Settings(runtime_mode="auto"))
                geometry_component = next(item for item in runtime.descriptor.components if item.name == "geometry_estimator")
                self.assertEqual(geometry_component.backend_kind, "local")
                real_geometry.assert_called_once()

    def test_auto_runtime_uses_local_segmenter_when_available(self) -> None:
        with patch("shadowgen_ml_service.bootstrap.container.probe_birefnet") as probe_birefnet:
            with patch("shadowgen_ml_service.bootstrap.container.RealSegmenter") as real_segmenter:
                probe_birefnet.return_value.model_name = "ZhengPeng7/BiRefNet_lite-matting"
                probe_birefnet.return_value.model_version = "bootstrap-probe"
                probe_birefnet.return_value.available = True
                probe_birefnet.return_value.detail = None
                runtime = build_runtime(Settings(runtime_mode="auto"))
                segmenter_component = next(item for item in runtime.descriptor.components if item.name == "segmenter")
                self.assertEqual(segmenter_component.backend_kind, "local")
                real_segmenter.assert_called_once()

    def test_auto_runtime_uses_local_depth_when_available(self) -> None:
        with patch("shadowgen_ml_service.bootstrap.container.probe_depth_anything") as probe_depth_anything:
            with patch("shadowgen_ml_service.bootstrap.container.RealDepthEstimator") as real_depth:
                probe_depth_anything.return_value.model_name = "depth-anything/Depth-Anything-V2-Small-hf"
                probe_depth_anything.return_value.model_version = "bootstrap-probe"
                probe_depth_anything.return_value.available = True
                probe_depth_anything.return_value.detail = None
                runtime = build_runtime(Settings(runtime_mode="auto"))
                depth_component = next(item for item in runtime.descriptor.components if item.name == "depth_estimator")
                self.assertEqual(depth_component.backend_kind, "local")
                real_depth.assert_called_once()

    def test_normals_use_depth_fallback_when_stable_normal_unavailable(self) -> None:
        with patch("shadowgen_ml_service.bootstrap.container.probe_stable_normal") as probe_stable_normal:
            probe_stable_normal.return_value.model_name = "Stable-X/StableNormal"
            probe_stable_normal.return_value.model_version = "bootstrap-probe"
            probe_stable_normal.return_value.available = False
            probe_stable_normal.return_value.detail = "CUDA unavailable"
            runtime = build_runtime(Settings(runtime_mode="auto"))
            normal_component = next(item for item in runtime.descriptor.components if item.name == "normal_estimator")
            self.assertFalse(normal_component.using_mock)
            self.assertEqual(normal_component.backend_kind, "local")
            self.assertEqual(normal_component.model_name, "normal-map-from-depth")
            self.assertEqual(normal_component.implementation, "local-fallback")

    def test_auto_runtime_uses_local_shadow_when_weights_exist(self) -> None:
        with patch("shadowgen_ml_service.bootstrap.container.probe_shadow_pix2pix") as probe_shadow:
            with patch("shadowgen_ml_service.bootstrap.container.Pix2PixShadowGenerator") as real_shadow:
                probe_shadow.return_value.model_name = "V1-GAN"
                probe_shadow.return_value.model_version = "bootstrap-probe"
                probe_shadow.return_value.available = True
                probe_shadow.return_value.detail = "CUDA runtime detected"
                runtime = build_runtime(Settings(runtime_mode="auto"))
                shadow_component = next(item for item in runtime.descriptor.components if item.name == "shadow_generator")
                self.assertFalse(shadow_component.using_mock)
                self.assertEqual(shadow_component.backend_kind, "local")
                self.assertEqual(shadow_component.model_variant, "v1-gan")
                real_shadow.assert_called_once()

    def test_capabilities_include_triton_shadow_backend_descriptor(self) -> None:
        runtime = build_runtime(Settings(runtime_mode="auto"))
        shadow_component = next(item for item in runtime.descriptor.components if item.name == "shadow_generator")
        triton_descriptors = [item for item in shadow_component.backends if item.backend_kind == "triton"]
        self.assertEqual(len(triton_descriptors), 1)
        self.assertEqual(triton_descriptors[0].model_variant, "v2-diff")

    def test_segmenter_triton_descriptor_is_only_available_when_binding_probe_passes(self) -> None:
        with patch("shadowgen_ml_service.bootstrap.container.TritonInferenceClient.ping", return_value=True):
            with patch(
                "shadowgen_ml_service.bootstrap.container.TritonInferenceClient.probe_binding",
                side_effect=lambda *args, **_kwargs: (
                    args[-1].stage_key == "segmenter",
                    "ok" if args[-1].stage_key == "segmenter" else "missing",
                ),
            ):
                runtime = build_runtime(Settings(runtime_mode="auto", triton_url="http://triton.local", segmenter_backend_kind="triton"))

        segmenter_component = next(item for item in runtime.descriptor.components if item.name == "segmenter")
        self.assertEqual(segmenter_component.backend_kind, "triton")
        self.assertEqual(segmenter_component.endpoint, "http://triton.local")
        depth_component = next(item for item in runtime.descriptor.components if item.name == "depth_estimator")
        self.assertNotEqual(depth_component.backend_kind, "triton")


if __name__ == "__main__":
    unittest.main()
