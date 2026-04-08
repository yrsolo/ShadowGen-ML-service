from __future__ import annotations

from time import perf_counter

from shadowgen_ml_service.application.dependencies import PipelineRuntime
from shadowgen_ml_service.application.models import PipelineContext, RenderOutcome, StageBackendSelection
from shadowgen_ml_service.application.services.backend_selector import BackendSelector
from shadowgen_ml_service.application.services.stage_runner import StageRunner
from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.core.commands import RenderCommand
from shadowgen_ml_service.core.errors import TimeoutServiceError, UnsupportedInputServiceError, ValidationServiceError
from shadowgen_ml_service.core.models import PreprocessSnapshot, SegmentationResult
from shadowgen_ml_service.utils.images import decode_image, prepare_working_crop


class RenderPipelineUseCase:
    def __init__(self, settings: Settings, runtime: PipelineRuntime) -> None:
        self.settings = settings
        self.runtime = runtime
        self.selector = BackendSelector(runtime)
        self.stage_runner = StageRunner()

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
        )
        context.cache_key = cache_key
        snapshot = self.runtime.cache.load(cache_key)
        context.metrics.measure_from("cache_ms", cache_started)

        if snapshot is None:
            detection = self._execute_public_stage("detector", context, lambda: self.runtime.detector.detect(source_rgba, command.padding_px))
            context.detection = detection

            geometry = self._execute_public_stage("geometry_estimator", context, lambda: self.runtime.geometry.estimate(source_rgba))
            context.geometry = geometry

            context.working_crop = prepare_working_crop(
                source_rgba,
                detection.bbox,
                self.settings.working_size,
                content_scale=self.settings.working_content_scale,
            )

            segmentation = self._execute_public_stage("segmenter", context, lambda: self.runtime.segmenter.segment(context.working_crop))
            context.segmentation = segmentation
            context.pre_refinement_cutout = segmentation.cutout_rgba

            foreground_refinement = self._execute_public_stage(
                "foreground_refiner",
                context,
                lambda: self.runtime.foreground_refiner.refine(
                    context.segmentation.crop_rgba,
                    self._segmentation_alpha(context.segmentation),
                ),
            )
            context.foreground_refinement = foreground_refinement
            context.segmentation = self._merge_refined_cutout(context.segmentation, foreground_refinement.cutout_rgba)

            depth = self._execute_public_stage(
                "depth_estimator",
                context,
                lambda: self.runtime.depth.estimate(context.segmentation.cutout_rgba, context.segmentation.mask),
            )
            context.depth = depth

            normals = self._execute_public_stage(
                "normal_estimator",
                context,
                lambda: self.runtime.normals.estimate(context.segmentation.cutout_rgba, depth.depth_map),
            )
            context.normals = normals

            snapshot = PreprocessSnapshot(
                detection=detection,
                geometry=geometry,
                segmentation=context.segmentation,
                depth=depth,
                normals=normals,
                foreground_refinement=foreground_refinement,
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

        shadow = self._execute_public_stage(
            "shadow_generator",
            context,
            lambda: self.runtime.shadow.generate(
                mask=context.segmentation.mask,
                depth_map=context.depth.depth_map,
                normal_map=context.normals.normal_map,
                geometry=context.geometry,
                shadow=command.shadow,
            ),
        )
        context.shadow = shadow

        composition = self._execute_public_stage(
            "composer",
            context,
            lambda: self.runtime.composer.compose(
                cutout_rgba=context.segmentation.cutout_rgba,
                shadow_rgba=shadow.shadow_rgba,
                background=command.background,
                output=command.output,
            ),
        )
        context.composition = composition

        encode_started = perf_counter()
        artifacts = self.runtime.encoder.encode(
            final_image=composition.final_image,
            output_format=command.output.format,
            debug_images={
                "cutout": context.segmentation.cutout_rgba,
                "mask": context.segmentation.mask,
                "crop": context.segmentation.crop_rgba,
                "depth": context.depth.depth_map,
                "normals": context.normals.normal_map,
                "shadow": shadow.shadow_rgba,
            },
            return_debug=command.output.return_debug,
        )
        context.metrics.measure_from("encode_ms", encode_started)
        total_ms = context.metrics.total()
        if context.detection.confidence < 0.7:
            context.warnings.append("main_object_low_confidence")
        if context.geometry.confidence < 0.7:
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

    def _execute_public_stage(self, stage_key: str, context: PipelineContext, action):
        selection = StageBackendSelection(requested_mode="real", actual_mode=self.selector.actual_mode_for_public(stage_key))
        value, execution = self.stage_runner.execute(stage_key=stage_key, selection=selection, context=context, action=action)
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
        context.warnings.extend(
            f"mock_backend_{component.name}"
            for component in self.runtime.descriptor.components
            if component.using_mock
        )
        if any(component.name == "detector" and component.implementation == "mock-fallback" for component in self.runtime.descriptor.components):
            context.warnings.append("detector_real_fallback_active")
        if any(component.name == "geometry_estimator" and component.implementation == "mock-fallback" for component in self.runtime.descriptor.components):
            context.warnings.append("geometry_real_fallback_active")
        if any(component.name == "segmenter" and component.implementation == "mock-fallback" for component in self.runtime.descriptor.components):
            context.warnings.append("segmenter_real_fallback_active")
        if any(component.name == "foreground_refiner" and component.implementation == "mock-fallback" for component in self.runtime.descriptor.components):
            context.warnings.append("foreground_refiner_real_fallback_active")
        if any(component.name == "depth_estimator" and component.implementation == "mock-fallback" for component in self.runtime.descriptor.components):
            context.warnings.append("depth_real_fallback_active")
        if any(component.name == "normal_estimator" and component.model_name == "normal-map-from-depth" for component in self.runtime.descriptor.components):
            context.warnings.append("normals_neural_backend_unavailable")

    def _segmentation_alpha(self, segmentation: SegmentationResult):
        return segmentation.cutout_rgba.getchannel("A")

    def _merge_refined_cutout(self, segmentation: SegmentationResult, refined_cutout_rgba):
        return SegmentationResult(
            bbox=segmentation.bbox,
            mask=segmentation.mask,
            cutout_rgba=refined_cutout_rgba,
            crop_rgba=segmentation.crop_rgba,
        )
