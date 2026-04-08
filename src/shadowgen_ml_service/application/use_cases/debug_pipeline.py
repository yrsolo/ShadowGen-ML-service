from __future__ import annotations

from shadowgen_ml_service.application.dependencies import PipelineRuntime
from shadowgen_ml_service.application.models import DebugPipelineOutcome, PipelineContext, StageExecution
from shadowgen_ml_service.application.services.backend_selector import BackendSelector
from shadowgen_ml_service.application.services.stage_catalog import get_stage_definition
from shadowgen_ml_service.application.services.stage_runner import StageRunner
from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.core.commands import DebugPipelineCommand
from shadowgen_ml_service.core.errors import UnsupportedInputServiceError, ValidationServiceError
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
                requested_mode="real",
                actual_mode="internal",
                elapsed_ms=0,
                previews=self.runtime.previews.build("decode", source_rgba, context),
            )
        ]
        if stop_after == "decode":
            return DebugPipelineOutcome(request_id=command.render.request_id, stages=stages, warnings=context.warnings)

        geometry = self._run_stage(
            "geometry_estimator",
            command.stage_modes.get("geometry_estimator", "mock"),
            context,
            lambda: self._geometry_backend(command.stage_modes.get("geometry_estimator", "mock")).estimate(source_rgba),
            lambda value, actual_mode: {
                "camera_fov": round(value.camera_fov, 3),
                "camera_pitch": round(value.camera_pitch, 3),
                "camera_roll": round(value.camera_roll, 3),
                "confidence": round(value.confidence, 4),
                "backend": actual_mode,
                "camera_model": self.settings.geocalib_camera_model if actual_mode == "real" else ("mock-v1" if actual_mode == "mock-fallback" else "mock"),
                "weights": self.settings.geocalib_weights if actual_mode == "real" else "mock-v1",
                "shared_intrinsics": self.settings.geocalib_shared_intrinsics if actual_mode == "real" else False,
            },
        )
        stages.append(geometry)
        if geometry.status == "failed" or stop_after == "geometry_estimator":
            return DebugPipelineOutcome(request_id=command.render.request_id, stages=stages, warnings=context.warnings)
        context.geometry = getattr(context, "geometry")

        detection = self._run_stage(
            "detector",
            command.stage_modes.get("detector", "mock"),
            context,
            lambda: self._detector_backend(command.stage_modes.get("detector", "mock")).detect(source_rgba, command.render.padding_px),
            lambda value, actual_mode: {
                "bbox_left": value.bbox[0],
                "bbox_top": value.bbox[1],
                "bbox_right": value.bbox[2],
                "bbox_bottom": value.bbox[3],
                "confidence": round(value.confidence, 4),
                "backend": actual_mode,
                "prompt": self.settings.grounding_dino_prompt if actual_mode == "real" else "mock",
            },
        )
        stages.append(detection)
        if detection.status == "failed" or stop_after == "detector":
            return DebugPipelineOutcome(request_id=command.render.request_id, stages=stages, warnings=context.warnings)

        segmentation = self._run_stage(
            "segmenter",
            command.stage_modes.get("segmenter", "mock"),
            context,
            lambda: self._segmenter_backend(command.stage_modes.get("segmenter", "mock")).segment(self._prepare_working_crop(context)),
            lambda value, actual_mode: {
                "bbox_left": value.bbox[0],
                "bbox_top": value.bbox[1],
                "bbox_right": value.bbox[2],
                "bbox_bottom": value.bbox[3],
                "backend": actual_mode,
                "mask_width": value.mask.width,
                "mask_height": value.mask.height,
            },
        )
        stages.append(segmentation)
        if segmentation.status == "failed" or stop_after == "segmenter":
            return DebugPipelineOutcome(request_id=command.render.request_id, stages=stages, warnings=context.warnings)

        foreground_refiner = self._run_stage(
            "foreground_refiner",
            command.stage_modes.get("foreground_refiner", "real"),
            context,
            lambda: self._foreground_refiner_backend(command.stage_modes.get("foreground_refiner", "real")).refine(
                context.segmentation.crop_rgba,
                context.segmentation.cutout_rgba.getchannel("A"),
            ),
            lambda value, actual_mode: {
                "backend": actual_mode,
                "cutout_width": value.cutout_rgba.width,
                "cutout_height": value.cutout_rgba.height,
            },
        )
        stages.append(foreground_refiner)
        if foreground_refiner.status == "failed" or stop_after == "foreground_refiner":
            return DebugPipelineOutcome(request_id=command.render.request_id, stages=stages, warnings=context.warnings)

        depth = self._run_stage(
            "depth_estimator",
            command.stage_modes.get("depth_estimator", "mock"),
            context,
            lambda: self._depth_backend(command.stage_modes.get("depth_estimator", "mock")).estimate(
                context.segmentation.cutout_rgba,
                context.segmentation.mask,
            ),
            lambda value, actual_mode: {
                "backend": actual_mode,
                "depth_width": value.depth_map.width,
                "depth_height": value.depth_map.height,
            },
        )
        stages.append(depth)
        if depth.status == "failed" or stop_after == "depth_estimator":
            return DebugPipelineOutcome(request_id=command.render.request_id, stages=stages, warnings=context.warnings)

        normals_mode = command.stage_modes.get("normal_estimator", "real")
        normals_backend = self._normals_backend(normals_mode)
        normals = self._run_stage(
            "normal_estimator",
            normals_mode,
            context,
            lambda: normals_backend.estimate(context.segmentation.cutout_rgba, context.depth.depth_map),
            lambda value, actual_mode: {
                "backend": str(getattr(normals_backend, "backend_name", actual_mode)),
                "variant": str(getattr(normals_backend, "model_variant", "from-depth")),
                "normals_width": value.normal_map.width,
                "normals_height": value.normal_map.height,
            },
        )
        stages.append(normals)
        if normals.status == "failed" or stop_after == "normal_estimator":
            return DebugPipelineOutcome(request_id=command.render.request_id, stages=stages, warnings=context.warnings)

        shadow = self._run_stage(
            "shadow_generator",
            command.stage_modes.get("shadow_generator", "real"),
            context,
            lambda: self.runtime.shadow.generate(
                mask=context.segmentation.mask,
                depth_map=context.depth.depth_map,
                normal_map=context.normals.normal_map,
                geometry=context.geometry,
                shadow=command.render.shadow,
            ),
        )
        stages.append(shadow)
        if shadow.status == "failed" or stop_after == "shadow_generator":
            return DebugPipelineOutcome(request_id=command.render.request_id, stages=stages, warnings=context.warnings)

        composer = self._run_stage(
            "composer",
            command.stage_modes.get("composer", "real"),
            context,
            lambda: self.runtime.composer.compose(
                cutout_rgba=context.segmentation.cutout_rgba,
                shadow_rgba=context.shadow.shadow_rgba,
                background=command.render.background,
                output=command.render.output,
            ),
        )
        stages.append(composer)
        return DebugPipelineOutcome(request_id=command.render.request_id, stages=stages, warnings=context.warnings)

    def _run_stage(self, stage_key: str, requested_mode: str, context: PipelineContext, action, details_factory=None) -> StageExecution:
        selection = self.selector.select_for_debug(stage_key, requested_mode)
        value, execution = self.stage_runner.execute(
            stage_key=stage_key,
            selection=selection,
            context=context,
            action=action,
            details_factory=details_factory,
            previews_factory=lambda stage_value: self.runtime.previews.build(stage_key, self._assign_stage_value(stage_key, context, stage_value), context),
        )
        if execution.status == "completed":
            execution.details = dict(execution.details or {})
            execution.details.setdefault("device", self._device_label_for_stage(stage_key, requested_mode, execution.actual_mode))
            self._assign_stage_value(stage_key, context, value)
        return execution

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

    def _detector_backend(self, requested_mode: str):
        return self.runtime.mock_detector if requested_mode == "mock" else (self.runtime.real_detector or self.runtime.mock_detector)

    def _geometry_backend(self, requested_mode: str):
        return self.runtime.mock_geometry if requested_mode == "mock" else (self.runtime.real_geometry or self.runtime.mock_geometry)

    def _segmenter_backend(self, requested_mode: str):
        return self.runtime.mock_segmenter if requested_mode == "mock" else (self.runtime.real_segmenter or self.runtime.mock_segmenter)

    def _foreground_refiner_backend(self, requested_mode: str):
        return (
            self.runtime.mock_foreground_refiner
            if requested_mode == "mock"
            else (self.runtime.real_foreground_refiner or self.runtime.mock_foreground_refiner)
        )

    def _depth_backend(self, requested_mode: str):
        return self.runtime.mock_depth if requested_mode == "mock" else (self.runtime.real_depth or self.runtime.mock_depth)

    def _normals_backend(self, requested_mode: str):
        return self.runtime.mock_normals if requested_mode == "mock" else (self.runtime.real_normals or self.runtime.mock_normals)

    def _device_label_for_stage(self, stage_key: str, requested_mode: str, actual_mode: str) -> str:
        if actual_mode in {"mock", "mock-fallback", "internal"}:
            return "cpu"
        if stage_key == "detector":
            backend = self._detector_backend(requested_mode)
        elif stage_key == "geometry_estimator":
            backend = self._geometry_backend(requested_mode)
        elif stage_key == "segmenter":
            backend = self._segmenter_backend(requested_mode)
        elif stage_key == "foreground_refiner":
            backend = self._foreground_refiner_backend(requested_mode)
        elif stage_key == "depth_estimator":
            backend = self._depth_backend(requested_mode)
        elif stage_key == "normal_estimator":
            backend = self._normals_backend(requested_mode)
        else:
            return "cpu"
        return str(getattr(backend, "device_label", "cpu"))

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
