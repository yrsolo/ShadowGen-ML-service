from __future__ import annotations

from shadowgen_ml_service.application.models import CapabilitiesOutcome, DebugPipelineOutcome, HealthOutcome, RenderOutcome
from shadowgen_ml_service.core.commands import BackgroundSpec, DebugPipelineCommand, OutputSpec, RenderCommand, ShadowSpec, SourceImage
from shadowgen_ml_service.core.errors import ServiceError
from shadowgen_ml_service.interfaces.http.dev_schemas import PipelineDebugRequest, PipelineDebugResponse, StageExecutionResponse, StageModesPayload, StagePreviewResponse
from shadowgen_ml_service.interfaces.http.public_schemas import (
    ArtifactResponse,
    CapabilitiesResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    RenderRequest,
    RenderResponse,
)
from shadowgen_ml_service.utils.images import encode_image


def render_request_to_command(payload: RenderRequest) -> RenderCommand:
    return RenderCommand(
        request_id=payload.request_id,
        pipeline_version=payload.pipeline_version,
        source=SourceImage(mime_type=payload.source.mime_type, image_base64=payload.source.image_base64),
        padding_px=payload.preprocess.padding_px,
        shadow=ShadowSpec(**payload.shadow.model_dump()),
        background=BackgroundSpec(**payload.background.model_dump()),
        output=OutputSpec(**payload.output.model_dump()),
    )


def debug_request_to_command(payload: PipelineDebugRequest) -> DebugPipelineCommand:
    return DebugPipelineCommand(
        render=render_request_to_command(payload.render_request),
        stage_modes=payload.stage_modes.model_dump(),
    )


def render_outcome_to_response(outcome: RenderOutcome) -> RenderResponse:
    return RenderResponse(
        request_id=outcome.request_id,
        artifacts=[ArtifactResponse(**artifact.__dict__) for artifact in outcome.artifacts],
        metrics=MetricsResponse(**outcome.metrics),
        warnings=outcome.warnings,
        model_info=ModelInfoResponse(service_version=outcome.service_version, model_version=outcome.model_version),
    )


def health_outcome_to_response(outcome: HealthOutcome) -> HealthResponse:
    return HealthResponse(**outcome.payload.__dict__)


def capabilities_outcome_to_response(outcome: CapabilitiesOutcome) -> CapabilitiesResponse:
    return CapabilitiesResponse(
        service_version=outcome.payload.service_version,
        model_version=outcome.payload.model_version,
        supported_input_mime_types=outcome.payload.supported_input_mime_types,
        supported_output_formats=outcome.payload.supported_output_formats,
        supports_debug_artifacts=outcome.payload.supports_debug_artifacts,
        max_image_bytes=outcome.payload.max_image_bytes,
        active_backend_mode=outcome.payload.active_backend_mode,
        degraded=outcome.payload.degraded,
        components=[component.__dict__ for component in outcome.payload.components],
    )


def debug_outcome_to_response(outcome: DebugPipelineOutcome) -> PipelineDebugResponse:
    stages: list[StageExecutionResponse] = []
    for stage in outcome.stages:
        previews = []
        for name, image in stage.previews.items():
            mime_type, image_base64 = encode_image(image, "png")
            previews.append(StagePreviewResponse(name=name, mime_type=mime_type, image_base64=image_base64))
        stages.append(
            StageExecutionResponse(
                stage_key=stage.stage_key,
                title=stage.title,
                description=stage.description,
                status=stage.status,
                requested_mode=stage.requested_mode,
                actual_mode=stage.actual_mode,
                elapsed_ms=stage.elapsed_ms,
                error=stage.error,
                details=stage.details,
                previews=previews,
            )
        )
    return PipelineDebugResponse(request_id=outcome.request_id, stages=stages, warnings=outcome.warnings)


def service_error_to_response(exc: ServiceError) -> ErrorResponse:
    return ErrorResponse(
        error={
            "code": exc.code,
            "message": exc.message,
            "details": exc.details,
            "request_id": exc.request_id,
        }
    )
