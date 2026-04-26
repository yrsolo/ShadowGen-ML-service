from __future__ import annotations

import unittest

from shadowgen_ml_service.application.models import ExecutionSelection, PipelineContext
from shadowgen_ml_service.application.services.stage_runner import StageRunner
from shadowgen_ml_service.core.commands import BackgroundSpec, OutputSpec, RenderCommand, ShadowSpec, SourceImage
from shadowgen_ml_service.core.errors import ProcessingFailedServiceError
from shadowgen_ml_service.core.models import StageBackendDescriptor, StageBackendId


def _command() -> RenderCommand:
    return RenderCommand(
        request_id="runner-test",
        pipeline_version="ml-shadowgen-v1",
        source=SourceImage(mime_type="image/png", image_base64=""),
        padding_px=100,
        shadow=ShadowSpec(model=None, angle_deg=45.0, elevation_deg=30.0, softness=0.5, opacity=0.5, reflection=0.0),
        background=BackgroundSpec(mode="solid", color_hex="#ffffff"),
        output=OutputSpec(format="png", width=None, height=None, return_debug=False),
    )


class StageRunnerTests(unittest.TestCase):
    def test_capture_errors_returns_failed_stage_execution(self) -> None:
        runner = StageRunner()
        context = PipelineContext(command=_command())
        selection = ExecutionSelection(
            stage_key="segmenter",
            backend_id=StageBackendId("segmenter", "triton", "birefnet"),
            requested_backend_kind="triton",
            actual_backend_kind="triton",
            requested_variant="birefnet",
            actual_variant="birefnet",
            actual_mode="triton",
            descriptor=StageBackendDescriptor(
                backend_id=StageBackendId("segmenter", "triton", "birefnet"),
                model_name="segmenter",
                model_version="test",
                available=True,
                endpoint="http://triton.local",
                device="triton",
            ),
        )

        value, execution = runner.execute(
            stage_key="segmenter",
            selection=selection,
            context=context,
            backend=object(),
            invocation=lambda _: (_ for _ in ()).throw(RuntimeError("boom")),
            capture_errors=True,
        )

        self.assertIsNone(value)
        self.assertEqual(execution.status, "failed")
        self.assertIn("segmenter execution failed", execution.error)

    def test_public_execution_raises_service_error(self) -> None:
        runner = StageRunner()
        context = PipelineContext(command=_command())
        selection = ExecutionSelection(
            stage_key="depth_estimator",
            backend_id=StageBackendId("depth_estimator", "local", "depth-anything-v2-small"),
            requested_backend_kind="local",
            actual_backend_kind="local",
            requested_variant="depth-anything-v2-small",
            actual_variant="depth-anything-v2-small",
            actual_mode="local",
            descriptor=StageBackendDescriptor(
                backend_id=StageBackendId("depth_estimator", "local", "depth-anything-v2-small"),
                model_name="depth",
                model_version="test",
                available=True,
                device="cuda:0",
            ),
        )

        with self.assertRaises(ProcessingFailedServiceError):
            runner.execute(
                stage_key="depth_estimator",
                selection=selection,
                context=context,
                backend=object(),
                invocation=lambda _: (_ for _ in ()).throw(RuntimeError("depth broke")),
                capture_errors=False,
            )


if __name__ == "__main__":
    unittest.main()
