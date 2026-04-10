from __future__ import annotations

from shadowgen_ml_service.application.models import AsyncRenderJobRecord, CapabilitiesOutcome, DebugPipelineOutcome, HealthOutcome, RenderOutcome
from shadowgen_ml_service.core.commands import BackgroundSpec, DebugPipelineCommand, OutputSpec, RenderCommand, ShadowSpec, SourceImage
from shadowgen_ml_service.core.errors import ServiceError
from shadowgen_ml_service.interfaces.http.dev_schemas import PipelineDebugRequest, PipelineDebugResponse, StageExecutionResponse, StageModesPayload, StagePreviewResponse
from shadowgen_ml_service.interfaces.http.public_schemas import (
    ArtifactResponse,
    BackendCapabilities,
    CapabilitiesResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    RenderRequest,
    RenderJobResponse,
    RenderJobSubmitResponse,
    RenderResponse,
)
from shadowgen_ml_service.utils.images import asset_to_base64


def backend_descriptor_to_response(backend) -> BackendCapabilities:
    return BackendCapabilities(
        backend_kind=backend.backend_kind,
        model_variant=backend.model_variant,
        model_name=backend.model_name,
        model_version=backend.model_version,
        available=backend.available,
        detail=backend.detail,
        device=backend.device,
        endpoint=backend.endpoint,
        supports_batching=backend.supports_batching,
        supports_async=backend.supports_async,
    )


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
        stage_backend_kinds=payload.stage_backend_kinds.model_dump(exclude_unset=True),
        stage_variants=payload.stage_variants.model_dump(exclude_unset=True),
        stage_modes=payload.stage_modes.model_dump(exclude_unset=True),
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
        execution_default_backend=outcome.payload.execution_default_backend,
        async_enabled=outcome.payload.async_enabled,
        components=[
            {
                **component.__dict__,
                "backends": [backend_descriptor_to_response(backend).model_dump() for backend in component.backends],
            }
            for component in outcome.payload.components
        ],
    )


def debug_outcome_to_response(outcome: DebugPipelineOutcome) -> PipelineDebugResponse:
    stages: list[StageExecutionResponse] = []
    for stage in outcome.stages:
        previews = []
        for name, image in stage.previews.items():
            previews.append(
                StagePreviewResponse(
                    name=name,
                    mime_type=image.mime_type,
                    image_base64=asset_to_base64(image),
                )
            )
        stages.append(
            StageExecutionResponse(
                stage_key=stage.stage_key,
                title=stage.title,
                description=stage.description,
                status=stage.status,
                requested_mode=stage.requested_mode,
                actual_mode=stage.actual_mode,
                requested_backend_kind=stage.requested_backend_kind,
                actual_backend_kind=stage.actual_backend_kind,
                model_variant=stage.model_variant,
                model_name=stage.model_name,
                model_version=stage.model_version,
                device=stage.device,
                endpoint=stage.endpoint,
                cache_status=stage.cache_status,
                fallback_reason=stage.fallback_reason,
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


def async_job_to_submit_response(record: AsyncRenderJobRecord) -> RenderJobSubmitResponse:
    return RenderJobSubmitResponse(
        job_id=record.job_id,
        request_id=record.request_id,
        status=record.status,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def async_job_to_response(record: AsyncRenderJobRecord) -> RenderJobResponse:
    return RenderJobResponse(
        job_id=record.job_id,
        request_id=record.request_id,
        status=record.status,
        created_at=record.created_at,
        updated_at=record.updated_at,
        error=record.error,
        result=None if record.render_outcome is None else render_outcome_to_response(record.render_outcome),
    )
