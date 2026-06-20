from __future__ import annotations

import unittest

from shadowgen_ml_service.application.services.backend_policy import fallback_candidate_ids, preferred_variant
from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.core.models import StageBackendId


class BackendPolicyTests(unittest.TestCase):
    def test_fallback_candidates_for_detector_triton_variant(self) -> None:
        self.assertEqual(
            fallback_candidate_ids("detector", "triton", "grounding-dino-onnx"),
            [
                StageBackendId("detector", "triton", "grounding-dino-onnx"),
                StageBackendId("detector", "triton", "grounding-dino"),
                StageBackendId("detector", "local", "grounding-dino"),
                StageBackendId("detector", "mock", "mock-v1"),
            ],
        )

    def test_fallback_candidates_for_segmenter_triton_variant(self) -> None:
        self.assertEqual(
            fallback_candidate_ids("segmenter", "triton", "rmbg-2.0"),
            [
                StageBackendId("segmenter", "triton", "rmbg-2.0"),
                StageBackendId("segmenter", "triton", "birefnet"),
                StageBackendId("segmenter", "local", "birefnet"),
                StageBackendId("segmenter", "mock", "mock-v1"),
            ],
        )

    def test_fallback_candidates_for_normals_prefer_from_depth(self) -> None:
        self.assertEqual(
            fallback_candidate_ids("normal_estimator", "local", "from-depth-v2"),
            [
                StageBackendId("normal_estimator", "local", "from-depth-v2"),
                StageBackendId("normal_estimator", "mock", "mock-v1"),
            ],
        )

    def test_fallback_candidates_for_shadow_v2_diff(self) -> None:
        self.assertEqual(
            fallback_candidate_ids("shadow_generator", "local", "v2-diff"),
            [
                StageBackendId("shadow_generator", "local", "v2-diff"),
                StageBackendId("shadow_generator", "local", "v1-gan"),
                StageBackendId("shadow_generator", "mock", "mock"),
            ],
        )

    def test_preferred_variant_reads_settings_without_duplicate_segmenter_branch(self) -> None:
        settings = Settings(segmenter_model_variant="rmbg-2.0", detector_model_variant="grounding-dino-onnx")

        self.assertEqual(preferred_variant("segmenter", settings), "rmbg-2.0")
        self.assertEqual(preferred_variant("detector", settings), "grounding-dino-onnx")


if __name__ == "__main__":
    unittest.main()
