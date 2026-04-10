from __future__ import annotations

from time import perf_counter
from typing import Callable, Generic, TypeVar

from shadowgen_ml_service.application.models import ExecutionSelection, PipelineContext, StageExecution
from shadowgen_ml_service.application.services.stage_catalog import get_stage_definition


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
        value = invocation(backend)
        elapsed_ms = int((perf_counter() - started) * 1000)
        if selection.fallback_reason:
            context.warnings.append(f"{stage_key}_fallback_active")
        descriptor = selection.descriptor
        execution = StageExecution(
            stage_key=stage_key,
            title=definition.title,
            description=definition.description,
            status="completed",
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
            details=details_factory(value, selection) if details_factory is not None else None,
            previews=previews_factory(value) if previews_factory is not None else {},
        )
        return value, execution
