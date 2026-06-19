from __future__ import annotations

from shadowgen_ml_service.application.models import PipelineContext
from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.core.models import SegmentationResult
from shadowgen_ml_service.core.stage_io import DepthInput, DetectionInput, NormalsInput, SegmentationInput, ShadowInput
from shadowgen_ml_service.utils.images import alpha_asset, ensure_asset, prepare_working_crop


class PipelineStageExecutor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def stage_order(self) -> tuple[str, ...]:
        stages = [
            "detector",
            "segmenter",
            "foreground_refiner",
            "depth_estimator",
            "normal_estimator",
            "shadow_generator",
            "composer",
        ]
        if self.settings.geometry_enabled:
            stages.insert(0, "geometry_estimator")
        return tuple(stages)

    def invoke(self, stage_key: str, backend, context: PipelineContext):
        if stage_key == "detector":
            return backend.detect(self._build_detection_input(context))
        if stage_key == "geometry_estimator":
            return backend.estimate(ensure_asset(context.source_rgba))
        if stage_key == "segmenter":
            return backend.segment(self._build_segmentation_input(context))
        if stage_key == "foreground_refiner":
            return backend.refine(context.segmentation.crop_rgba, alpha_asset(context.segmentation.cutout_rgba))
        if stage_key == "depth_estimator":
            return backend.estimate(self._build_depth_input(context))
        if stage_key == "normal_estimator":
            return backend.estimate(self._build_normals_input(context))
        if stage_key == "shadow_generator":
            return backend.generate(self._build_shadow_input(context))
        if stage_key == "composer":
            return backend.compose(
                cutout_rgba=context.segmentation.cutout_rgba,
                shadow_image=context.shadow.shadow_image,
                background=context.command.background,
                output=context.command.output,
            )
        raise ValueError(f"unsupported stage key {stage_key}")

    def stage_details(self, stage_key: str, value, selection) -> dict[str, str | int | float | bool] | None:
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

    def assign_stage_value(self, stage_key: str, context: PipelineContext, value):
        if stage_key == "geometry_estimator":
            context.geometry = value
        elif stage_key == "detector":
            context.detection = value
            self._prepare_working_crop(context)
        elif stage_key == "segmenter":
            context.segmentation = value
        elif stage_key == "foreground_refiner":
            context.pre_refinement_cutout = context.segmentation.cutout_rgba
            context.foreground_refinement = value
            context.segmentation = self.merge_refined_cutout(context.segmentation, value.cutout_rgba)
        elif stage_key == "depth_estimator":
            context.depth = value
        elif stage_key == "normal_estimator":
            context.normals = value
        elif stage_key == "shadow_generator":
            context.shadow = value
        elif stage_key == "composer":
            context.composition = value
        return value

    def merge_refined_cutout(self, segmentation: SegmentationResult, refined_cutout_rgba) -> SegmentationResult:
        return SegmentationResult(
            bbox=segmentation.bbox,
            mask=segmentation.mask,
            cutout_rgba=refined_cutout_rgba,
            crop_rgba=segmentation.crop_rgba,
        )

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
        return DetectionInput(image=ensure_asset(context.source_rgba), padding_px=context.command.padding_px)

    def _build_segmentation_input(self, context: PipelineContext) -> SegmentationInput:
        return SegmentationInput(image=ensure_asset(self._prepare_working_crop(context)))

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
