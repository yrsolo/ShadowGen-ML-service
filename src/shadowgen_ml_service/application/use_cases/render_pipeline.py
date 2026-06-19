from __future__ import annotations

from time import perf_counter

from shadowgen_ml_service.application.dependencies import PipelineRuntime
from shadowgen_ml_service.application.models import PipelineContext
from shadowgen_ml_service.application.services.backend_selector import BackendSelector
from shadowgen_ml_service.application.services.pipeline_stage_executor import PipelineStageExecutor
from shadowgen_ml_service.application.services.stage_runner import StageRunner
from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.core.commands import RenderCommand
from shadowgen_ml_service.core.errors import TimeoutServiceError, UnsupportedInputServiceError, ValidationServiceError
from shadowgen_ml_service.core.models import GeometryResult, PreprocessSnapshot, RenderOutcome
from shadowgen_ml_service.utils.images import decode_image, prepare_working_crop


class RenderPipelineUseCase:
    def __init__(self, settings: Settings, runtime: PipelineRuntime) -> None:
        self.settings = settings
        self.runtime = runtime
        self.selector = BackendSelector(runtime)
        self.stage_runner = StageRunner()
        self.stage_executor = PipelineStageExecutor(settings)

    def execute(self, command: RenderCommand) -> RenderOutcome:
        context = PipelineContext(command=command)
        self._append_runtime_warnings(context)

        decode_started = perf_counter()
        raw_bytes, source_rgba = self._decode_command(command)
        context.raw_bytes = raw_bytes
        context.source_rgba = source_rgba
        context.metrics.measure_from("decode_ms", decode_started)

        cache_started = perf_counter()
        cache_key = self.runtime.cache.make_key(
            raw_bytes=raw_bytes,
            runtime_signature=self.runtime.signature,
            padding_px=command.padding_px,
            working_size=self.settings.working_size,
            working_content_scale=self.settings.working_content_scale,
        )
        context.cache_key = cache_key
        snapshot = self.runtime.cache.load(cache_key)
        context.metrics.measure_from("cache_ms", cache_started)

        if snapshot is None:
            context.detection = self._execute_public_stage("detector", context)
            context.geometry = self._execute_public_stage("geometry_estimator", context) if self.settings.geometry_enabled else self._disabled_geometry_result()
            if not self.settings.geometry_enabled:
                context.metrics.set("geometry_ms", 0)
            context.working_crop = prepare_working_crop(
                source_rgba,
                context.detection.bbox,
                self.settings.working_size,
                content_scale=self.settings.working_content_scale,
            )
            context.segmentation = self._execute_public_stage("segmenter", context)
            context.pre_refinement_cutout = context.segmentation.cutout_rgba
            context.foreground_refinement = self._execute_public_stage("foreground_refiner", context)
            context.segmentation = self.stage_executor.merge_refined_cutout(context.segmentation, context.foreground_refinement.cutout_rgba)
            context.depth = self._execute_public_stage("depth_estimator", context)
            context.normals = self._execute_public_stage("normal_estimator", context)

            snapshot = PreprocessSnapshot(
                detection=context.detection,
                geometry=context.geometry,
                segmentation=context.segmentation,
                depth=context.depth,
                normals=context.normals,
                foreground_refinement=context.foreground_refinement,
            )
            self.runtime.cache.save(cache_key, snapshot)
        else:
            context.warnings.append("preprocess_cache_hit")
            context.metrics.values.setdefault("detection_ms", 0)
            context.metrics.values.setdefault("geometry_ms", 0)
            context.metrics.values.setdefault("segmentation_ms", 0)
            context.metrics.values.setdefault("foreground_refinement_ms", 0)
            context.metrics.values.setdefault("depth_ms", 0)
            context.metrics.values.setdefault("normals_ms", 0)
            context.preprocess_snapshot = snapshot
            context.detection = snapshot.detection
            context.geometry = snapshot.geometry
            context.segmentation = snapshot.segmentation
            context.depth = snapshot.depth
            context.normals = snapshot.normals
            context.foreground_refinement = snapshot.foreground_refinement

        context.shadow = self._execute_public_stage("shadow_generator", context)
        context.composition = self._execute_public_stage("composer", context)

        encode_started = perf_counter()
        artifacts = self.runtime.encoder.encode(
            final_image=context.composition.final_image,
            output_format=command.output.format,
            debug_images={
                "cutout": context.segmentation.cutout_rgba,
                "mask": context.segmentation.mask,
                "crop": context.segmentation.crop_rgba,
                "depth": context.depth.depth_map,
                "normals": context.normals.normal_map,
                "shadow": context.shadow.shadow_image,
            },
            return_debug=command.output.return_debug,
        )
        context.metrics.measure_from("encode_ms", encode_started)
        total_ms = context.metrics.total()
        if context.detection.confidence < 0.7:
            context.warnings.append("main_object_low_confidence")
        if self.settings.geometry_enabled and context.geometry.confidence < 0.7:
            context.warnings.append("geometry_estimation_low_confidence")
        if total_ms > self.settings.request_timeout_ms:
            raise TimeoutServiceError("render request exceeded configured timeout", request_id=command.request_id)
        return RenderOutcome(
            request_id=command.request_id,
            artifacts=artifacts,
            metrics=context.metrics.values,
            warnings=context.warnings,
            service_version=self.settings.service_version,
            model_version=self.runtime.descriptor.model_version,
        )

    def _execute_public_stage(self, stage_key: str, context: PipelineContext):
        selection = self._select_public_stage(stage_key, context)
        registered = None if selection.backend_id is None else self.runtime.registry.get(selection.backend_id)
        value, execution = self.stage_runner.execute(
            stage_key=stage_key,
            selection=selection,
            context=context,
            backend=None if registered is None else registered.handler,
            invocation=lambda backend: self.stage_executor.invoke(stage_key, backend, context),
            capture_errors=False,
        )
        metric_key = {
            "geometry_estimator": "geometry_ms",
            "detector": "detection_ms",
            "segmenter": "segmentation_ms",
            "foreground_refiner": "foreground_refinement_ms",
            "depth_estimator": "depth_ms",
            "normal_estimator": "normals_ms",
            "shadow_generator": "shadow_ms",
            "composer": "composition_ms",
        }[stage_key]
        context.metrics.set(metric_key, execution.elapsed_ms or 0)
        return value

    def _select_public_stage(self, stage_key: str, context: PipelineContext):
        if stage_key == "shadow_generator" and context.command.shadow.model:
            backend_kind, variant = self._shadow_model_selection(context.command.shadow.model)
            return self.selector.select_for_debug(stage_key, backend_kind, variant)
        return self.selector.select_for_public(stage_key)

    def _shadow_model_selection(self, model: str) -> tuple[str, str]:
        if model == "v2-diff":
            return "local", "v2-diff"
        return "local", "v1-gan"

    def _decode_command(self, command: RenderCommand):
        if command.pipeline_version != self.settings.default_pipeline_version:
            raise ValidationServiceError(
                f"pipeline_version must be {self.settings.default_pipeline_version}",
                request_id=command.request_id,
            )
        try:
            return decode_image(command.source.image_base64, command.source.mime_type, self.settings.max_image_bytes)
        except ValueError as exc:
            message = str(exc)
            if "mime_type" in message:
                raise UnsupportedInputServiceError(message, request_id=command.request_id) from exc
            raise ValidationServiceError(message, request_id=command.request_id) from exc

    def _append_runtime_warnings(self, context: PipelineContext) -> None:
        for component in self.runtime.descriptor.components:
            if component.using_mock:
                context.warnings.append(f"mock_backend_{component.name}")
            if component.fallback_reason:
                context.warnings.append(f"{component.name}_fallback_active")
            if component.name == "normal_estimator" and component.model_name == "normal-map-from-depth":
                context.warnings.append("normals_neural_backend_unavailable")

    def _disabled_geometry_result(self) -> GeometryResult:
        return GeometryResult(camera_fov=0.0, camera_pitch=0.0, camera_roll=0.0, confidence=1.0)
