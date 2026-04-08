from __future__ import annotations

from time import perf_counter
from typing import Callable, Generic, TypeVar

from shadowgen_ml_service.application.models import PipelineContext, StageBackendSelection, StageExecution
from shadowgen_ml_service.application.services.stage_catalog import get_stage_definition


T = TypeVar("T")


class StageRunner(Generic[T]):
    def execute(
        self,
        *,
        stage_key: str,
        selection: StageBackendSelection,
        context: PipelineContext,
        action: Callable[[], T],
        details_factory: Callable[[T, str], dict[str, str | int | float | bool] | None] | None = None,
        previews_factory: Callable[[T], dict[str, object]] | None = None,
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
                error=selection.unavailable_message,
            )
            return None, execution

        started = perf_counter()
        value = action()
        elapsed_ms = int((perf_counter() - started) * 1000)
        if selection.requested_mode == "real" and selection.actual_mode == "mock-fallback":
            context.warnings.append(f"{stage_key}_mock_fallback")
        execution = StageExecution(
            stage_key=stage_key,
            title=definition.title,
            description=definition.description,
            status="completed",
            requested_mode=selection.requested_mode,
            actual_mode=selection.actual_mode,
            elapsed_ms=elapsed_ms,
            details=details_factory(value, selection.actual_mode) if details_factory is not None else None,
            previews=previews_factory(value) if previews_factory is not None else {},
        )
        return value, execution
