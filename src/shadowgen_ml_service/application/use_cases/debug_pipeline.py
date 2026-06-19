from __future__ import annotations

from shadowgen_ml_service.application.dependencies import PipelineRuntime
from shadowgen_ml_service.application.models import DebugPipelineOutcome, PipelineContext, StageExecution
from shadowgen_ml_service.application.services.backend_selector import BackendSelector
from shadowgen_ml_service.application.services.pipeline_stage_executor import PipelineStageExecutor
from shadowgen_ml_service.application.services.stage_catalog import get_stage_definition
from shadowgen_ml_service.application.services.stage_runner import StageRunner
from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.core.commands import DebugPipelineCommand
from shadowgen_ml_service.core.errors import UnsupportedInputServiceError, ValidationServiceError
from shadowgen_ml_service.utils.images import decode_image


class DebugPipelineUseCase:
    def __init__(self, settings: Settings, runtime: PipelineRuntime) -> None:
        self.settings = settings
        self.runtime = runtime
        self.selector = BackendSelector(runtime)
        self.stage_runner = StageRunner()
        self.stage_executor = PipelineStageExecutor(settings)

    def execute(self, command: DebugPipelineCommand, stop_after: str | None = None) -> DebugPipelineOutcome:
        context = PipelineContext(command=command.render)
        raw_bytes, source_rgba = self._decode_command(command)
        context.raw_bytes = raw_bytes
        context.source_rgba = source_rgba

        decode_def = get_stage_definition("decode")
        stages = [
            StageExecution(
                stage_key="decode",
                title=decode_def.title,
                description=decode_def.description,
                status="completed",
                requested_mode="internal",
                actual_mode="internal",
                requested_backend_kind="internal",
                actual_backend_kind="internal",
                elapsed_ms=0,
                previews=self.runtime.previews.build("decode", source_rgba, context),
            )
        ]
        if stop_after == "decode":
            return DebugPipelineOutcome(request_id=command.render.request_id, stages=stages, warnings=context.warnings)

        if stop_after == "geometry_estimator" and not self.settings.geometry_enabled:
            stages.append(self._skipped_stage("geometry_estimator"))
            return DebugPipelineOutcome(request_id=command.render.request_id, stages=stages, warnings=context.warnings)

        for stage_key in self._stage_order():
            execution = self._run_stage(stage_key, command, context)
            stages.append(execution)
            if execution.status == "failed" or stop_after == stage_key:
                return DebugPipelineOutcome(request_id=command.render.request_id, stages=stages, warnings=context.warnings)

        return DebugPipelineOutcome(request_id=command.render.request_id, stages=stages, warnings=context.warnings)

    def _stage_order(self) -> tuple[str, ...]:
        return self.stage_executor.stage_order()

    def _skipped_stage(self, stage_key: str) -> StageExecution:
        definition = get_stage_definition(stage_key)
        return StageExecution(
            stage_key=stage_key,
            title=definition.title,
            description=definition.description,
            status="skipped",
            requested_mode="internal",
            actual_mode="internal",
            requested_backend_kind="internal",
            actual_backend_kind="internal",
            model_variant="disabled",
            elapsed_ms=0,
            error="stage disabled by SHADOWGEN_GEOMETRY_ENABLED=false",
            details={"enabled": False},
        )

    def _run_stage(self, stage_key: str, command: DebugPipelineCommand, context: PipelineContext) -> StageExecution:
        requested_backend_kind, requested_variant = self._requested_backend(stage_key, command)
        selection = self.selector.select_for_debug(stage_key, requested_backend_kind, requested_variant)
        registered = None if selection.backend_id is None else self.runtime.registry.get(selection.backend_id)
        _, execution = self.stage_runner.execute(
            stage_key=stage_key,
            selection=selection,
            context=context,
            backend=None if registered is None else registered.handler,
            invocation=lambda backend: self.stage_executor.invoke(stage_key, backend, context),
            details_factory=lambda value, stage_selection: self.stage_executor.stage_details(stage_key, value, stage_selection),
            previews_factory=lambda value: self.runtime.previews.build(stage_key, self.stage_executor.assign_stage_value(stage_key, context, value), context),
            capture_errors=True,
        )
        if execution.status == "completed":
            execution.details = dict(execution.details or {})
            execution.details.setdefault("device", execution.device or "cpu")
        return execution

    def _requested_backend(self, stage_key: str, command: DebugPipelineCommand) -> tuple[str, str]:
        legacy_mode = command.stage_modes.get(stage_key)
        if stage_key in command.stage_backend_kinds:
            requested_backend_kind = command.stage_backend_kinds[stage_key]
        elif legacy_mode == "mock":
            requested_backend_kind = "mock"
        else:
            requested_backend_kind = "local"

        if requested_backend_kind == "mock":
            requested_variant = self._mock_variant(stage_key)
        elif stage_key in command.stage_variants:
            requested_variant = command.stage_variants[stage_key]
        elif stage_key == "shadow_generator":
            requested_variant = legacy_mode if legacy_mode in {"mock", "v1-gan", "v2-diff"} else "v1-gan"
        elif stage_key == "normal_estimator":
            requested_variant = self.settings.normals_model_variant
        elif stage_key == "detector":
            requested_variant = "grounding-dino"
        elif stage_key == "segmenter":
            requested_variant = "birefnet"
        elif stage_key == "depth_estimator":
            requested_variant = "depth-anything-v2-small"
        elif stage_key == "geometry_estimator":
            requested_variant = "geocalib"
        elif stage_key == "foreground_refiner":
            requested_variant = "fast-foreground-estimation"
        elif stage_key == "composer":
            requested_variant = "python-composer"
        else:
            requested_variant = "default"
        return requested_backend_kind, requested_variant

    def _mock_variant(self, stage_key: str) -> str:
        mapping = {
            "shadow_generator": "mock",
            "foreground_refiner": "passthrough-v1",
        }
        return mapping.get(stage_key, "mock-v1")

    def _decode_command(self, command: DebugPipelineCommand):
        render = command.render
        if render.pipeline_version != self.settings.default_pipeline_version:
            raise ValidationServiceError(
                f"pipeline_version must be {self.settings.default_pipeline_version}",
                request_id=render.request_id,
            )
        try:
            return decode_image(render.source.image_base64, render.source.mime_type, self.settings.max_image_bytes)
        except ValueError as exc:
            message = str(exc)
            if "mime_type" in message:
                raise UnsupportedInputServiceError(message, request_id=render.request_id) from exc
            raise ValidationServiceError(message, request_id=render.request_id) from exc
