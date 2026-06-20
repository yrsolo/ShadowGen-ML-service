from __future__ import annotations

import unittest

from shadowgen_ml_service.application.dependencies import PipelineBackendRegistry, PipelineRuntime
from shadowgen_ml_service.application.services.backend_selector import BackendSelector
from shadowgen_ml_service.core.models import ComponentStatus, RuntimeDescriptor, StageBackendDescriptor, StageBackendId


class BackendSelectorTests(unittest.TestCase):
    def test_debug_fallback_reason_includes_unavailable_backend_detail(self) -> None:
        registry = PipelineBackendRegistry()
        registry.register(
            StageBackendDescriptor(
                backend_id=StageBackendId("segmenter", "triton", "birefnet"),
                model_name="shadowgen_segmenter",
                model_version="triton-managed",
                available=False,
                detail="Triton endpoint is unavailable",
                device="triton",
            ),
            handler=None,
        )
        local_descriptor = StageBackendDescriptor(
            backend_id=StageBackendId("segmenter", "local", "birefnet"),
            model_name="BiRefNet",
            model_version="test",
            available=True,
            device="cpu",
        )
        registry.register(local_descriptor, handler=object())
        registry.set_default("segmenter", local_descriptor.backend_id)
        runtime = PipelineRuntime(
            registry=registry,
            encoder=None,
            cache=None,
            previews=None,
            descriptor=RuntimeDescriptor(
                mode="local-fallback",
                degraded=True,
                components=[
                    ComponentStatus(
                        name="segmenter",
                        implementation="local",
                        model_name="BiRefNet",
                        model_version="test",
                        available=True,
                        using_mock=False,
                        backend_kind="local",
                        model_variant="birefnet",
                    )
                ],
                model_version="test",
            ),
        )

        selection = BackendSelector(runtime).select_for_debug("segmenter", "triton", "birefnet")

        self.assertEqual(selection.actual_backend_kind, "local")
        self.assertEqual(selection.actual_mode, "local-fallback")
        self.assertEqual(
            selection.fallback_reason,
            "triton:birefnet unavailable for segmenter: Triton endpoint is unavailable",
        )


if __name__ == "__main__":
    unittest.main()
