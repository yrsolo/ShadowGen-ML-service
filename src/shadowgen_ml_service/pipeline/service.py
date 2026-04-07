from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.pipeline.runtime import PipelineRuntime
from shadowgen_ml_service.pipeline.types import CachedPreprocess, TimedValue
from shadowgen_ml_service.schemas import (
    CapabilitiesResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    RenderRequest,
    RenderResponse,
)
from shadowgen_ml_service.utils.images import decode_image, fit_to_square


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

            segmentation_timed = self._timed(lambda: self.runtime.segmenter.segment(source_rgba, detection_timed.value.bbox))
            metrics["segmentation_ms"] = segmentation_timed.elapsed_ms

            working_cutout, working_mask = fit_to_square(
                segmentation_timed.value.cutout_rgba,
                segmentation_timed.value.mask,
                self.settings.working_size,
            )
            working_segmentation = segmentation_timed.value.__class__(
                bbox=segmentation_timed.value.bbox,
                mask=working_mask,
                cutout_rgba=working_cutout,
                crop_rgba=segmentation_timed.value.crop_rgba,
            )

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

    @staticmethod
    def _timed(callable_obj) -> TimedValue:
        started = perf_counter()
        value = callable_obj()
        return TimedValue(value=value, elapsed_ms=int((perf_counter() - started) * 1000))

    @staticmethod
    def _elapsed_from(started: float) -> int:
        return int((perf_counter() - started) * 1000)
