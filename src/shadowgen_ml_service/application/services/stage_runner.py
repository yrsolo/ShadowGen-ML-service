from __future__ import annotations

from time import perf_counter
from typing import Callable, Generic, TypeVar

from shadowgen_ml_service.application.models import ExecutionSelection, PipelineContext, StageExecution
from shadowgen_ml_service.application.services.stage_catalog import get_stage_definition
from shadowgen_ml_service.core.errors import BackendFault, ProcessingFailedServiceError, ServiceError, StageFaultKind, TimeoutServiceError


T = TypeVar("T")


class StageRunner(Generic[T]):
    def execute(
        self,
        *,
        stage_key: str,
        selection: ExecutionSelection,
        context: PipelineContext,
        backend: object | None,
        invocation: Callable[[object | None], T],
        details_factory: Callable[[T, ExecutionSelection], dict[str, str | int | float | bool] | None] | None = None,
        previews_factory: Callable[[T], dict[str, object]] | None = None,
        cache_status: str | None = None,
        capture_errors: bool = False,
    ) -> tuple[T | None, StageExecution]:
        definition = get_stage_definition(stage_key)
        if selection.actual_mode == "unavailable":
            execution = StageExecution(
                stage_key=stage_key,
                title=definition.title,
                description=definition.description,
                status="failed",
                requested_mode=selection.requested_mode,
                actual_mode=selection.actual_mode,
                requested_backend_kind=selection.requested_backend_kind,
                actual_backend_kind=selection.actual_backend_kind,
                model_variant=selection.requested_variant,
                fallback_reason=selection.fallback_reason,
                error=selection.unavailable_message,
            )
            return None, execution

        started = perf_counter()
        try:
            value = invocation(backend)
        except Exception as exc:
            elapsed_ms = int((perf_counter() - started) * 1000)
            normalized = self._normalize_error(stage_key=stage_key, context=context, exc=exc)
            if selection.fallback_reason:
                context.warnings.append(f"{stage_key}_fallback_active")
            execution = self._build_execution(
                definition=definition,
                selection=selection,
                status="failed",
                cache_status=cache_status,
                elapsed_ms=elapsed_ms,
                error=normalized.message,
            )
            if capture_errors:
                return None, execution
            raise normalized from exc
        elapsed_ms = int((perf_counter() - started) * 1000)
        if selection.fallback_reason:
            context.warnings.append(f"{stage_key}_fallback_active")
        execution = self._build_execution(
            definition=definition,
            selection=selection,
            status="completed",
            cache_status=cache_status,
            elapsed_ms=elapsed_ms,
            details=details_factory(value, selection) if details_factory is not None else None,
            previews=previews_factory(value) if previews_factory is not None else {},
        )
        return value, execution

    def _build_execution(
        self,
        *,
        definition,
        selection: ExecutionSelection,
        status: str,
        cache_status: str | None,
        elapsed_ms: int | None,
        error: str | None = None,
        details: dict[str, str | int | float | bool] | None = None,
        previews: dict[str, object] | None = None,
    ) -> StageExecution:
        descriptor = selection.descriptor
        return StageExecution(
            stage_key=selection.stage_key,
            title=definition.title,
            description=definition.description,
            status=status,
            requested_mode=selection.requested_mode,
            actual_mode=selection.actual_mode,
            requested_backend_kind=selection.requested_backend_kind,
            actual_backend_kind=selection.actual_backend_kind,
            model_variant=selection.actual_variant,
            model_name=None if descriptor is None else descriptor.model_name,
            model_version=None if descriptor is None else descriptor.model_version,
            device=None if descriptor is None else descriptor.device,
            endpoint=None if descriptor is None else descriptor.endpoint,
            cache_status=cache_status,
            fallback_reason=selection.fallback_reason,
            elapsed_ms=elapsed_ms,
            error=error,
            details=details,
            previews={} if previews is None else previews,
        )

    def _normalize_error(self, *, stage_key: str, context: PipelineContext, exc: Exception) -> ServiceError:
        if isinstance(exc, ServiceError):
            return exc
        details = {"stage_key": stage_key}
        if isinstance(exc, BackendFault):
            details.update(exc.details)
            return self._normalize_backend_fault(stage_key=stage_key, context=context, fault=exc, details=details)
        return ProcessingFailedServiceError(
            f"{stage_key} execution failed: {exc}",
            request_id=context.command.request_id,
            details={**details, "kind": StageFaultKind.BACKEND_RUNTIME_ERROR.value},
        )

    def _normalize_backend_fault(
        self,
        *,
        stage_key: str,
        context: PipelineContext,
        fault: BackendFault,
        details: dict[str, str],
    ) -> ServiceError:
        if fault.kind == StageFaultKind.TRITON_TIMEOUT:
            return TimeoutServiceError(
                f"{stage_key} timed out while waiting for Triton inference",
                request_id=context.command.request_id,
                details=details,
            )
        if fault.kind == StageFaultKind.TRITON_ENDPOINT_UNAVAILABLE:
            return ProcessingFailedServiceError(
                f"{stage_key} could not reach the Triton endpoint",
                request_id=context.command.request_id,
                details={**details, "kind": fault.kind.value},
            )
        if fault.kind == StageFaultKind.TRITON_MODEL_UNAVAILABLE:
            return ProcessingFailedServiceError(
                f"{stage_key} Triton model is unavailable",
                request_id=context.command.request_id,
                details={**details, "kind": fault.kind.value},
            )
        if fault.kind == StageFaultKind.TRITON_SCHEMA_MISMATCH:
            return ProcessingFailedServiceError(
                f"{stage_key} received an incompatible Triton tensor schema",
                request_id=context.command.request_id,
                details={**details, "kind": fault.kind.value},
            )
        if fault.kind == StageFaultKind.TRITON_INVALID_RESPONSE:
            return ProcessingFailedServiceError(
                f"{stage_key} received an invalid Triton response",
                request_id=context.command.request_id,
                details={**details, "kind": fault.kind.value},
            )
        if fault.kind == StageFaultKind.TRITON_BACKEND_ERROR:
            return ProcessingFailedServiceError(
                f"{stage_key} Triton backend error: {fault}",
                request_id=context.command.request_id,
                details={**details, "kind": fault.kind.value},
            )
        return ProcessingFailedServiceError(
            f"{stage_key} execution failed: {fault}",
            request_id=context.command.request_id,
            details={**details, "kind": fault.kind.value},
        )
