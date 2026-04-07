from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from PIL import Image

from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.pipeline.runtime import PipelineRuntime
from shadowgen_ml_service.pipeline.types import CachedPreprocess, TimedValue
from shadowgen_ml_service.schemas import (
    CapabilitiesResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PipelineDebugResponse,
    PipelineDebugRequest,
    RenderRequest,
    RenderResponse,
    StageExecutionResponse,
    StagePreviewResponse,
)
from shadowgen_ml_service.utils.images import decode_image, draw_geometry_overlay, encode_image, prepare_working_crop


class ServiceError(Exception):
    code = "internal_error"
    http_status = 500

    def __init__(self, message: str, request_id: str | None = None, details: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.request_id = request_id
        self.details = details

    def to_response(self) -> ErrorResponse:
        return ErrorResponse(
            error={
                "code": self.code,
                "message": self.message,
                "details": self.details,
                "request_id": self.request_id,
            }
        )


class ValidationServiceError(ServiceError):
    code = "validation_error"
    http_status = 400


class UnsupportedInputServiceError(ServiceError):
    code = "unsupported_input"
    http_status = 415


class ProcessingFailedServiceError(ServiceError):
    code = "processing_failed"
    http_status = 500


class TimeoutServiceError(ServiceError):
    code = "timeout"
    http_status = 504


@dataclass
class RenderService:
    settings: Settings
    runtime: PipelineRuntime

    STAGE_DEFINITIONS = [
        ("decode", "Decode", "Decode input image and validate the payload."),
        ("geometry_estimator", "Geometry", "Estimate camera geometry from the original image."),
        ("detector", "Detection", "Locate the main foreground object and compute the crop area."),
        ("segmenter", "Segmentation", "Build the foreground mask and cut out the object."),
        ("depth_estimator", "Depth", "Estimate the relative depth map for the foreground crop."),
        ("normal_estimator", "Normals", "Compute surface normals from the depth map."),
        ("shadow_generator", "Shadow", "Generate the shadow layer from geometry and user lighting controls."),
        ("composer", "Composition", "Composite the object and shadow on the target background."),
    ]

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            service_version=self.settings.service_version,
            active_backend_mode=self.runtime.descriptor.mode,
        )

    def capabilities(self) -> CapabilitiesResponse:
        return CapabilitiesResponse(
            service_version=self.settings.service_version,
            model_version=self.runtime.descriptor.model_version,
            supported_input_mime_types=("image/jpeg", "image/png", "image/webp"),
            supported_output_formats=("png", "webp"),
            supports_debug_artifacts=True,
            max_image_bytes=self.settings.max_image_bytes,
            active_backend_mode=self.runtime.descriptor.mode,
            degraded=self.runtime.descriptor.degraded,
            components=[component.__dict__ for component in self.runtime.descriptor.components],
        )

    def render(self, request: RenderRequest) -> RenderResponse:
        started = perf_counter()
        warnings = [f"mock_backend_{component.name}" for component in self.runtime.descriptor.components if component.using_mock]
        if any(component.name == "geometry_estimator" and component.implementation == "mock-fallback" for component in self.runtime.descriptor.components):
            warnings.append("geometry_real_fallback_active")
        metrics: dict[str, int] = {}

        raw_bytes, source_rgba = self._decode_request(request)
        metrics["decode_ms"] = self._elapsed_from(started)

        cache_started = perf_counter()
        cache_key = self.runtime.cache.make_key(
            raw_bytes=raw_bytes,
            runtime_signature=self.runtime.signature,
            padding_px=request.preprocess.padding_px,
            working_size=self.settings.working_size,
        )
        cached = self.runtime.cache.load(cache_key)
        metrics["cache_ms"] = self._elapsed_from(cache_started)

        if cached is None:
            detection_timed = self._timed(lambda: self.runtime.detector.detect(source_rgba, request.preprocess.padding_px))
            metrics["detection_ms"] = detection_timed.elapsed_ms

            geometry_timed = self._timed(lambda: self.runtime.geometry.estimate(source_rgba))
            metrics["geometry_ms"] = geometry_timed.elapsed_ms

            working_crop = prepare_working_crop(
                source_rgba,
                detection_timed.value.bbox,
                self.settings.working_size,
            )
            segmentation_timed = self._timed(lambda: self.runtime.segmenter.segment(working_crop))
            metrics["segmentation_ms"] = segmentation_timed.elapsed_ms
            working_segmentation = segmentation_timed.value.__class__(
                bbox=segmentation_timed.value.bbox,
                mask=segmentation_timed.value.mask,
                cutout_rgba=segmentation_timed.value.cutout_rgba,
                crop_rgba=segmentation_timed.value.crop_rgba,
            )
            working_cutout = working_segmentation.cutout_rgba
            working_mask = working_segmentation.mask

            depth_timed = self._timed(lambda: self.runtime.depth.estimate(working_cutout, working_mask))
            metrics["depth_ms"] = depth_timed.elapsed_ms

            normals_timed = self._timed(lambda: self.runtime.normals.estimate(depth_timed.value.depth_map))
            metrics["normals_ms"] = normals_timed.elapsed_ms

            cached = CachedPreprocess(
                detection=detection_timed.value,
                geometry=geometry_timed.value,
                segmentation=working_segmentation,
                depth=depth_timed.value,
                normals=normals_timed.value,
            )
            self.runtime.cache.save(cache_key, cached)
        else:
            warnings.append("preprocess_cache_hit")
            metrics.setdefault("detection_ms", 0)
            metrics.setdefault("geometry_ms", 0)
            metrics.setdefault("segmentation_ms", 0)
            metrics.setdefault("depth_ms", 0)
            metrics.setdefault("normals_ms", 0)

        shadow_timed = self._timed(
            lambda: self.runtime.shadow.generate(
                mask=cached.segmentation.mask,
                depth_map=cached.depth.depth_map,
                normal_map=cached.normals.normal_map,
                geometry=cached.geometry,
                shadow=request.shadow,
            )
        )
        metrics["shadow_ms"] = shadow_timed.elapsed_ms

        composition_timed = self._timed(
            lambda: self.runtime.composer.compose(
                cutout_rgba=cached.segmentation.cutout_rgba,
                shadow_rgba=shadow_timed.value.shadow_rgba,
                background=request.background,
                output=request.output,
            )
        )
        metrics["composition_ms"] = composition_timed.elapsed_ms

        debug_images = {
            "cutout": cached.segmentation.cutout_rgba,
            "mask": cached.segmentation.mask,
            "crop": cached.segmentation.crop_rgba,
            "depth": cached.depth.depth_map,
            "normals": cached.normals.normal_map,
            "shadow": shadow_timed.value.shadow_rgba,
        }
        encode_timed = self._timed(
            lambda: self.runtime.encoder.encode(
                final_image=composition_timed.value.final_image,
                output_format=request.output.format,
                debug_images=debug_images,
                return_debug=request.output.return_debug,
            )
        )
        metrics["encode_ms"] = encode_timed.elapsed_ms
        metrics["total_ms"] = self._elapsed_from(started)

        if cached.detection.confidence < 0.7:
            warnings.append("main_object_low_confidence")
        if cached.geometry.confidence < 0.7:
            warnings.append("geometry_estimation_low_confidence")
        if metrics["total_ms"] > self.settings.request_timeout_ms:
            raise TimeoutServiceError("render request exceeded configured timeout", request_id=request.request_id)

        return RenderResponse(
            request_id=request.request_id,
            artifacts=encode_timed.value,
            metrics=MetricsResponse(**metrics),
            warnings=warnings,
            model_info=ModelInfoResponse(
                service_version=self.settings.service_version,
                model_version=self.runtime.descriptor.model_version,
            ),
        )

    def run_debug_pipeline(
        self,
        payload: PipelineDebugRequest,
        stop_after: str | None = None,
    ) -> PipelineDebugResponse:
        request = payload.render_request
        warnings: list[str] = []
        stages: list[StageExecutionResponse] = []
        raw_bytes, source_rgba = self._decode_request(request)

        decode_ms = 0
        stages.append(
            self._stage_response(
                stage_key="decode",
                status="completed",
                requested_mode="real",
                actual_mode="internal",
                elapsed_ms=decode_ms,
                previews={"source": source_rgba},
            )
        )
        if stop_after == "decode":
            return PipelineDebugResponse(request_id=request.request_id, stages=stages, warnings=warnings)

        geometry = self._run_stage_with_mode(
            stage_key="geometry_estimator",
            requested_mode=payload.stage_modes.geometry_estimator,
            action=lambda: self.runtime.geometry.estimate(source_rgba),
            previews_factory=lambda value: {
                "geometry_input": source_rgba,
                "geometry_overlay": draw_geometry_overlay(
                    source_rgba,
                    value.camera_fov,
                    value.camera_pitch,
                    value.camera_roll,
                    value.confidence,
                ),
            },
            details_factory=lambda value, actual_mode: {
                "camera_fov": round(value.camera_fov, 3),
                "camera_pitch": round(value.camera_pitch, 3),
                "camera_roll": round(value.camera_roll, 3),
                "confidence": round(value.confidence, 4),
                "backend": actual_mode,
            },
            warnings=warnings,
        )
        stages.append(geometry["response"])
        if geometry["response"].status == "failed" or stop_after == "geometry_estimator":
            return PipelineDebugResponse(request_id=request.request_id, stages=stages, warnings=warnings)

        detection = self._run_stage_with_mode(
            stage_key="detector",
            requested_mode=payload.stage_modes.detector,
            action=lambda: self.runtime.detector.detect(source_rgba, request.preprocess.padding_px),
            previews_factory=lambda value: {
                "crop_for_resize": prepare_working_crop(source_rgba, value.bbox, self.settings.working_size)
            },
            warnings=warnings,
        )
        stages.append(detection["response"])
        if detection["response"].status == "failed" or stop_after == "detector":
            return PipelineDebugResponse(request_id=request.request_id, stages=stages, warnings=warnings)

        working_crop = prepare_working_crop(
            source_rgba,
            detection["value"].bbox,
            self.settings.working_size,
        )
        segmentation = self._run_stage_with_mode(
            stage_key="segmenter",
            requested_mode=payload.stage_modes.segmenter,
            action=lambda: self.runtime.segmenter.segment(working_crop),
            previews_factory=lambda value: {
                "working_crop": value.crop_rgba,
                "cutout": value.cutout_rgba,
                "mask": value.mask,
            },
            warnings=warnings,
        )
        stages.append(segmentation["response"])
        if segmentation["response"].status == "failed" or stop_after == "segmenter":
            return PipelineDebugResponse(request_id=request.request_id, stages=stages, warnings=warnings)

        working_cutout = segmentation["value"].cutout_rgba
        working_mask = segmentation["value"].mask

        depth = self._run_stage_with_mode(
            stage_key="depth_estimator",
            requested_mode=payload.stage_modes.depth_estimator,
            action=lambda: self.runtime.depth.estimate(working_cutout, working_mask),
            previews_factory=lambda value: {"depth": value.depth_map, "working_cutout": working_cutout},
            warnings=warnings,
        )
        stages.append(depth["response"])
        if depth["response"].status == "failed" or stop_after == "depth_estimator":
            return PipelineDebugResponse(request_id=request.request_id, stages=stages, warnings=warnings)

        normals = self._run_stage_with_mode(
            stage_key="normal_estimator",
            requested_mode=payload.stage_modes.normal_estimator,
            action=lambda: self.runtime.normals.estimate(depth["value"].depth_map),
            previews_factory=lambda value: {"normals": value.normal_map},
            warnings=warnings,
        )
        stages.append(normals["response"])
        if normals["response"].status == "failed" or stop_after == "normal_estimator":
            return PipelineDebugResponse(request_id=request.request_id, stages=stages, warnings=warnings)

        shadow = self._run_stage_with_mode(
            stage_key="shadow_generator",
            requested_mode=payload.stage_modes.shadow_generator,
            action=lambda: self.runtime.shadow.generate(
                mask=working_mask,
                depth_map=depth["value"].depth_map,
                normal_map=normals["value"].normal_map,
                geometry=geometry["value"],
                shadow=request.shadow,
            ),
            previews_factory=lambda value: {"shadow": value.shadow_rgba},
            warnings=warnings,
        )
        stages.append(shadow["response"])
        if shadow["response"].status == "failed" or stop_after == "shadow_generator":
            return PipelineDebugResponse(request_id=request.request_id, stages=stages, warnings=warnings)

        composition = self._run_stage_with_mode(
            stage_key="composer",
            requested_mode=payload.stage_modes.composer,
            action=lambda: self.runtime.composer.compose(
                cutout_rgba=working_cutout,
                shadow_rgba=shadow["value"].shadow_rgba,
                background=request.background,
                output=request.output,
            ),
            previews_factory=lambda value: {"final": value.final_image},
            warnings=warnings,
        )
        stages.append(composition["response"])
        return PipelineDebugResponse(request_id=request.request_id, stages=stages, warnings=warnings)

    def _decode_request(self, request: RenderRequest) -> tuple[bytes, object]:
        if request.pipeline_version != self.settings.default_pipeline_version:
            raise ValidationServiceError(
                f"pipeline_version must be {self.settings.default_pipeline_version}",
                request_id=request.request_id,
            )
        try:
            return decode_image(
                request.source.image_base64,
                request.source.mime_type,
                self.settings.max_image_bytes,
            )
        except ValueError as exc:
            message = str(exc)
            if "mime_type" in message:
                raise UnsupportedInputServiceError(message, request_id=request.request_id) from exc
            raise ValidationServiceError(message, request_id=request.request_id) from exc

    def _run_stage_with_mode(self, stage_key: str, requested_mode: str, action, previews_factory, warnings: list[str], details_factory=None) -> dict:
        actual_mode = self._resolve_stage_mode(stage_key, requested_mode)
        if requested_mode == "real" and actual_mode == "unavailable":
            error_message = self._stage_unavailable_message(stage_key)
            warnings.append(f"{stage_key}_real_unavailable")
            return {
                "value": None,
                "response": self._stage_response(
                    stage_key=stage_key,
                    status="failed",
                    requested_mode=requested_mode,
                    actual_mode=actual_mode,
                    error=error_message,
                ),
            }
        if requested_mode == "real" and actual_mode == "mock-fallback":
            warnings.append(f"{stage_key}_mock_fallback")

        timed = self._timed(action)
        previews = previews_factory(timed.value)
        details = details_factory(timed.value, actual_mode) if details_factory is not None else None
        return {
            "value": timed.value,
            "response": self._stage_response(
                stage_key=stage_key,
                status="completed",
                requested_mode=requested_mode,
                actual_mode=actual_mode,
                elapsed_ms=timed.elapsed_ms,
                details=details,
                previews=previews,
            ),
        }

    def _resolve_stage_mode(self, stage_key: str, requested_mode: str) -> str:
        component = next((item for item in self.runtime.descriptor.components if item.name == stage_key), None)
        if component is None:
            if requested_mode == "real":
                return "internal"
            return "mock"
        if requested_mode == "real":
            if stage_key in {"normal_estimator", "shadow_generator", "composer"}:
                return "real"
            if component.implementation == "real" and component.available:
                return "real"
            if stage_key == "geometry_estimator" and component.available and component.using_mock:
                return "mock-fallback"
            return "unavailable"
        return "mock"

    def _stage_unavailable_message(self, stage_key: str) -> str:
        messages = {
            "detector": "Real detector is not wired yet. Configure the model stack and replace the scaffold.",
            "geometry_estimator": "Real geometry estimator is not wired yet. Configure GeoCalib integration first.",
            "segmenter": "Real segmenter is not wired yet. Configure the BiRefNet adapter first.",
            "depth_estimator": "Real depth estimator is not wired yet. Configure the Depth Anything adapter first.",
        }
        return messages.get(stage_key, "Requested real mode is unavailable for this stage.")

    def _stage_response(
        self,
        stage_key: str,
        status: str,
        requested_mode: str,
        actual_mode: str,
        elapsed_ms: int | None = None,
        error: str | None = None,
        details: dict[str, str | int | float | bool] | None = None,
        previews: dict[str, Image.Image] | None = None,
    ) -> StageExecutionResponse:
        title, description = self._stage_meta(stage_key)
        preview_items = []
        if previews:
            for name, image in previews.items():
                mime_type, image_base64 = encode_image(image, "png")
                preview_items.append(StagePreviewResponse(name=name, mime_type=mime_type, image_base64=image_base64))
        return StageExecutionResponse(
            stage_key=stage_key,
            title=title,
            description=description,
            status=status,
            requested_mode=requested_mode,
            actual_mode=actual_mode,
            elapsed_ms=elapsed_ms,
            error=error,
            details=details,
            previews=preview_items,
        )

    def _stage_meta(self, stage_key: str) -> tuple[str, str]:
        for key, title, description in self.STAGE_DEFINITIONS:
            if key == stage_key:
                return title, description
        return stage_key, stage_key

    @staticmethod
    def _timed(callable_obj) -> TimedValue:
        started = perf_counter()
        value = callable_obj()
        return TimedValue(value=value, elapsed_ms=int((perf_counter() - started) * 1000))

    @staticmethod
    def _elapsed_from(started: float) -> int:
        return int((perf_counter() - started) * 1000)
