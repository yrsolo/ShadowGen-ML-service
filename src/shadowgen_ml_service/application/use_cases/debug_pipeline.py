from __future__ import annotations

from shadowgen_ml_service.application.dependencies import PipelineRuntime
from shadowgen_ml_service.application.models import DebugPipelineOutcome, PipelineContext, StageExecution
from shadowgen_ml_service.application.services.backend_selector import BackendSelector
from shadowgen_ml_service.application.services.stage_catalog import get_stage_definition
from shadowgen_ml_service.application.services.stage_runner import StageRunner
from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.core.commands import DebugPipelineCommand
from shadowgen_ml_service.core.errors import UnsupportedInputServiceError, ValidationServiceError
from shadowgen_ml_service.core.stage_io import DepthInput, DetectionInput, NormalsInput, SegmentationInput, ShadowInput
from shadowgen_ml_service.utils.images import decode_image, prepare_working_crop


class DebugPipelineUseCase:
    def __init__(self, settings: Settings, runtime: PipelineRuntime) -> None:
        self.settings = settings
        self.runtime = runtime
        self.selector = BackendSelector(runtime)
        self.stage_runner = StageRunner()

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

        for stage_key in (
            "geometry_estimator",
            "detector",
            "segmenter",
            "foreground_refiner",
            "depth_estimator",
            "normal_estimator",
            "shadow_generator",
            "composer",
        ):
            execution = self._run_stage(stage_key, command, context)
            stages.append(execution)
            if execution.status == "failed" or stop_after == stage_key:
                return DebugPipelineOutcome(request_id=command.render.request_id, stages=stages, warnings=context.warnings)

        return DebugPipelineOutcome(request_id=command.render.request_id, stages=stages, warnings=context.warnings)

    def _run_stage(self, stage_key: str, command: DebugPipelineCommand, context: PipelineContext) -> StageExecution:
        requested_backend_kind, requested_variant = self._requested_backend(stage_key, command)
        selection = self.selector.select_for_debug(stage_key, requested_backend_kind, requested_variant)
        registered = None if selection.backend_id is None else self.runtime.registry.get(selection.backend_id)
        _, execution = self.stage_runner.execute(
            stage_key=stage_key,
            selection=selection,
            context=context,
            backend=None if registered is None else registered.handler,
            invocation=lambda backend: self._invoke_stage(stage_key, backend, context),
            details_factory=lambda value, stage_selection: self._stage_details(stage_key, value, stage_selection),
            previews_factory=lambda value: self.runtime.previews.build(stage_key, self._assign_stage_value(stage_key, context, value), context),
        )
        if execution.status == "completed":
            execution.details = dict(execution.details or {})
            execution.details.setdefault("device", execution.device or "cpu")
        return execution

    def _invoke_stage(self, stage_key: str, backend, context: PipelineContext):
        if stage_key == "geometry_estimator":
            return backend.estimate(context.source_rgba)
        if stage_key == "detector":
            return backend.detect(self._build_detection_input(context))
        if stage_key == "segmenter":
            return backend.segment(self._build_segmentation_input(context))
        if stage_key == "foreground_refiner":
            return backend.refine(context.segmentation.crop_rgba, context.segmentation.cutout_rgba.getchannel("A"))
        if stage_key == "depth_estimator":
            return backend.estimate(self._build_depth_input(context))
        if stage_key == "normal_estimator":
            return backend.estimate(self._build_normals_input(context))
        if stage_key == "shadow_generator":
            return backend.generate(self._build_shadow_input(context))
        if stage_key == "composer":
            return backend.compose(
                cutout_rgba=context.segmentation.cutout_rgba,
                shadow_rgba=context.shadow.shadow_rgba,
                background=context.command.background,
                output=context.command.output,
            )
        raise ValueError(f"unsupported stage key {stage_key}")

    def _stage_details(self, stage_key: str, value, selection) -> dict[str, str | int | float | bool] | None:
        actual_backend = selection.actual_backend_kind
        if stage_key == "geometry_estimator":
            return {
                "camera_fov": round(value.camera_fov, 3),
                "camera_pitch": round(value.camera_pitch, 3),
                "camera_roll": round(value.camera_roll, 3),
                "confidence": round(value.confidence, 4),
                "backend": actual_backend,
                "variant": selection.actual_variant,
            }
        if stage_key == "detector":
            return {
                "bbox_left": value.bbox[0],
                "bbox_top": value.bbox[1],
                "bbox_right": value.bbox[2],
                "bbox_bottom": value.bbox[3],
                "confidence": round(value.confidence, 4),
                "backend": actual_backend,
                "variant": selection.actual_variant,
                "prompt": "mock" if actual_backend == "mock" else self.settings.grounding_dino_prompt,
            }
        if stage_key == "segmenter":
            return {
                "bbox_left": value.bbox[0],
                "bbox_top": value.bbox[1],
                "bbox_right": value.bbox[2],
                "bbox_bottom": value.bbox[3],
                "backend": actual_backend,
                "variant": selection.actual_variant,
                "mask_width": value.mask.width,
                "mask_height": value.mask.height,
            }
        if stage_key == "foreground_refiner":
            return {
                "backend": actual_backend,
                "variant": selection.actual_variant,
                "cutout_width": value.cutout_rgba.width,
                "cutout_height": value.cutout_rgba.height,
            }
        if stage_key == "depth_estimator":
            return {
                "backend": actual_backend,
                "variant": selection.actual_variant,
                "depth_width": value.depth_map.width,
                "depth_height": value.depth_map.height,
            }
        if stage_key == "normal_estimator":
            return {
                "backend": actual_backend,
                "variant": selection.actual_variant,
                "normals_width": value.normal_map.width,
                "normals_height": value.normal_map.height,
            }
        if stage_key == "shadow_generator":
            return {
                "backend": actual_backend,
                "variant": selection.actual_variant,
            }
        return None

    def _assign_stage_value(self, stage_key: str, context: PipelineContext, value):
        if stage_key == "geometry_estimator":
            context.geometry = value
        elif stage_key == "detector":
            context.detection = value
            context.working_crop = prepare_working_crop(
                context.source_rgba,
                value.bbox,
                self.settings.working_size,
                content_scale=self.settings.working_content_scale,
            )
        elif stage_key == "segmenter":
            context.segmentation = value
        elif stage_key == "foreground_refiner":
            context.pre_refinement_cutout = context.segmentation.cutout_rgba
            context.foreground_refinement = value
            context.segmentation = context.segmentation.__class__(
                bbox=context.segmentation.bbox,
                mask=context.segmentation.mask,
                cutout_rgba=value.cutout_rgba,
                crop_rgba=context.segmentation.crop_rgba,
            )
        elif stage_key == "depth_estimator":
            context.depth = value
        elif stage_key == "normal_estimator":
            context.normals = value
        elif stage_key == "shadow_generator":
            context.shadow = value
        elif stage_key == "composer":
            context.composition = value
        return value

    def _prepare_working_crop(self, context: PipelineContext):
        if context.working_crop is None:
            context.working_crop = prepare_working_crop(
                context.source_rgba,
                context.detection.bbox,
                self.settings.working_size,
                content_scale=self.settings.working_content_scale,
            )
        return context.working_crop

    def _build_detection_input(self, context: PipelineContext) -> DetectionInput:
        return DetectionInput(image=context.source_rgba, padding_px=context.command.padding_px)

    def _build_segmentation_input(self, context: PipelineContext) -> SegmentationInput:
        return SegmentationInput(image=self._prepare_working_crop(context))

    def _build_depth_input(self, context: PipelineContext) -> DepthInput:
        return DepthInput(image=context.segmentation.cutout_rgba, mask=context.segmentation.mask)

    def _build_normals_input(self, context: PipelineContext) -> NormalsInput:
        return NormalsInput(image=context.segmentation.cutout_rgba, depth_map=context.depth.depth_map)

    def _build_shadow_input(self, context: PipelineContext) -> ShadowInput:
        return ShadowInput(
            img=context.segmentation.cutout_rgba,
            mask=context.segmentation.mask,
            depth=context.depth.depth_map,
            normal=context.normals.normal_map,
            angle=context.command.shadow.angle_deg,
            elevation=context.command.shadow.elevation_deg,
            softness=context.command.shadow.softness,
            reflection=context.command.shadow.reflection,
            opacity=context.command.shadow.opacity,
        )

    def _requested_backend(self, stage_key: str, command: DebugPipelineCommand) -> tuple[str, str]:
        legacy_mode = command.stage_modes.get(stage_key)
        if stage_key in command.stage_backend_kinds:
            requested_backend_kind = command.stage_backend_kinds[stage_key]
        elif legacy_mode == "mock":
            requested_backend_kind = "mock"
        elif stage_key == "shadow_generator" and legacy_mode == "v2-diff":
            requested_backend_kind = "triton"
        else:
            requested_backend_kind = "local"

        if requested_backend_kind == "mock":
            requested_variant = self._mock_variant(stage_key)
        elif stage_key in command.stage_variants:
            requested_variant = command.stage_variants[stage_key]
        elif stage_key == "shadow_generator":
            requested_variant = legacy_mode if legacy_mode in {"mock", "v1-gan", "v2-diff"} else "v1-gan"
        elif stage_key == "normal_estimator":
            requested_variant = "stable-normal"
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
