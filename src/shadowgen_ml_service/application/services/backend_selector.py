from __future__ import annotations

from shadowgen_ml_service.application.dependencies import PipelineRuntime
from shadowgen_ml_service.application.models import ExecutionSelection
from shadowgen_ml_service.core.models import StageBackendId


UNAVAILABLE_MESSAGES = {
    "detector": "Requested detector backend is unavailable.",
    "geometry_estimator": "Requested geometry backend is unavailable.",
    "segmenter": "Requested segmenter backend is unavailable.",
    "foreground_refiner": "Requested foreground refinement backend is unavailable.",
    "depth_estimator": "Requested depth backend is unavailable.",
    "normal_estimator": "Requested normals backend is unavailable.",
    "shadow_generator": "Requested shadow backend is unavailable.",
    "composer": "Requested composer backend is unavailable.",
}


class BackendSelector:
    def __init__(self, runtime: PipelineRuntime) -> None:
        self.runtime = runtime

    def select_for_public(self, stage_key: str) -> ExecutionSelection:
        component = next((item for item in self.runtime.descriptor.components if item.name == stage_key), None)
        active = self.runtime.default_backend(stage_key)
        if component is None or active is None:
            return ExecutionSelection(
                stage_key=stage_key,
                backend_id=None,
                requested_backend_kind="internal",
                actual_backend_kind="internal",
                actual_mode="internal",
            )
        return ExecutionSelection(
            stage_key=stage_key,
            backend_id=active.descriptor.backend_id,
            requested_backend_kind=component.backend_kind,
            actual_backend_kind=active.descriptor.backend_kind,
            requested_variant=component.model_variant,
            actual_variant=active.descriptor.model_variant,
            actual_mode=component.implementation,
            fallback_reason=component.fallback_reason,
            descriptor=active.descriptor,
        )

    def select_for_debug(self, stage_key: str, requested_backend_kind: str, requested_variant: str = "default") -> ExecutionSelection:
        if stage_key == "decode":
            return ExecutionSelection(
                stage_key=stage_key,
                backend_id=None,
                requested_backend_kind="internal",
                actual_backend_kind="internal",
                actual_mode="internal",
            )

        fallback_reason = None
        requested_unavailable_reason = None
        for backend_kind, variant in self._fallback_order(stage_key, requested_backend_kind, requested_variant):
            backend = self.runtime.backend(stage_key, backend_kind, variant)
            if backend is None or not backend.descriptor.available or backend.handler is None:
                if backend_kind == requested_backend_kind and variant == requested_variant:
                    requested_unavailable_reason = self._unavailable_reason(
                        stage_key,
                        requested_backend_kind,
                        requested_variant,
                        backend.descriptor.detail if backend is not None else None,
                    )
                continue
            actual_mode = backend_kind
            if backend_kind == "mock" and requested_backend_kind != "mock":
                actual_mode = "mock-fallback"
                fallback_reason = requested_unavailable_reason or self._unavailable_reason(
                    stage_key, requested_backend_kind, requested_variant, None
                )
            elif backend_kind != requested_backend_kind or variant != requested_variant:
                actual_mode = f"{backend_kind}-fallback"
                fallback_reason = requested_unavailable_reason or self._unavailable_reason(
                    stage_key, requested_backend_kind, requested_variant, None
                )
            return ExecutionSelection(
                stage_key=stage_key,
                backend_id=backend.descriptor.backend_id,
                requested_backend_kind=requested_backend_kind,
                actual_backend_kind=backend_kind,
                requested_variant=requested_variant,
                actual_variant=variant,
                actual_mode=actual_mode,
                fallback_reason=fallback_reason,
                descriptor=backend.descriptor,
            )

        return ExecutionSelection(
            stage_key=stage_key,
            backend_id=None,
            requested_backend_kind=requested_backend_kind,
            actual_backend_kind="unavailable",
            requested_variant=requested_variant,
            actual_variant=requested_variant,
            actual_mode="unavailable",
            unavailable_message=UNAVAILABLE_MESSAGES.get(stage_key, "Requested backend is unavailable."),
        )

    def _fallback_order(self, stage_key: str, requested_backend_kind: str, requested_variant: str) -> list[tuple[str, str]]:
        if requested_backend_kind == "mock":
            return [("mock", self._mock_variant(stage_key))]

        fallback: list[tuple[str, str]] = [(requested_backend_kind, requested_variant)]
        if stage_key == "normal_estimator":
            if requested_backend_kind == "triton":
                fallback.extend([("local", "stable-normal"), ("local", "from-depth-v2"), ("mock", "mock-v1")])
            else:
                fallback.extend([("local", "from-depth-v2"), ("mock", "mock-v1")])
            return fallback

        if stage_key == "shadow_generator":
            if requested_backend_kind == "triton":
                fallback.extend([("local", "v1-gan"), ("mock", "mock")])
            elif requested_backend_kind == "local":
                fallback.append(("mock", "mock"))
            return fallback

        if requested_backend_kind == "triton":
            fallback.extend([("local", self._default_variant(stage_key)), ("mock", "mock-v1")])
        elif requested_backend_kind == "local":
            fallback.append(("mock", "mock-v1"))
        return fallback

    def _default_variant(self, stage_key: str) -> str:
        mapping = {
            "detector": "grounding-dino",
            "segmenter": "birefnet",
            "depth_estimator": "depth-anything-v2-small",
            "normal_estimator": "stable-normal",
            "geometry_estimator": "geocalib",
            "foreground_refiner": "fast-foreground-estimation",
            "composer": "python-composer",
        }
        return mapping.get(stage_key, "default")

    def _mock_variant(self, stage_key: str) -> str:
        mapping = {
            "shadow_generator": "mock",
            "foreground_refiner": "passthrough-v1",
        }
        return mapping.get(stage_key, "mock-v1")

    def _unavailable_reason(
        self,
        stage_key: str,
        requested_backend_kind: str,
        requested_variant: str,
        detail: str | None,
    ) -> str:
        base = f"{requested_backend_kind}:{requested_variant} unavailable for {stage_key}"
        return f"{base}: {detail}" if detail else base
