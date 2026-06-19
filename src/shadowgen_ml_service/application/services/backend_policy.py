from __future__ import annotations

from shadowgen_ml_service.application.dependencies import PipelineBackendRegistry, RegisteredStageBackend
from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.core.models import StageBackendId


STAGE_ORDER = (
    "detector",
    "geometry_estimator",
    "segmenter",
    "foreground_refiner",
    "depth_estimator",
    "normal_estimator",
    "shadow_generator",
    "composer",
    "artifact_encoder",
)

DEFAULT_VARIANTS = {
    "detector": "grounding-dino",
    "segmenter": "birefnet",
    "depth_estimator": "depth-anything-v2-small",
    "geometry_estimator": "geocalib",
    "foreground_refiner": "fast-foreground-estimation",
    "composer": "python-composer",
    "artifact_encoder": "default-encoder",
}

MOCK_VARIANTS = {
    "shadow_generator": "mock",
    "foreground_refiner": "passthrough-v1",
}


def effective_default_backend(settings: Settings) -> str:
    if settings.execution_default_backend:
        return settings.execution_default_backend
    return "mock" if settings.runtime_mode.lower() == "mock" else "local"


def stage_backend_preference(stage_override: str, global_default: str) -> str:
    return stage_override or global_default


def default_stage_backends(settings: Settings) -> dict[str, str]:
    execution_default_backend = effective_default_backend(settings)
    return {
        "detector": "mock" if execution_default_backend == "mock" else stage_backend_preference(settings.detector_backend_kind, execution_default_backend),
        "geometry_estimator": "mock" if execution_default_backend == "mock" else ("internal" if not settings.geometry_enabled else settings.geometry_backend_kind),
        "segmenter": "mock" if execution_default_backend == "mock" else stage_backend_preference(settings.segmenter_backend_kind, execution_default_backend),
        "foreground_refiner": "mock" if execution_default_backend == "mock" else settings.foreground_refiner_backend_kind,
        "depth_estimator": "mock" if execution_default_backend == "mock" else stage_backend_preference(settings.depth_backend_kind, execution_default_backend),
        "normal_estimator": "mock" if execution_default_backend == "mock" else stage_backend_preference(settings.normals_backend_kind, execution_default_backend),
        "shadow_generator": "mock" if execution_default_backend == "mock" else stage_backend_preference(settings.shadow_backend_kind, execution_default_backend),
        "composer": "local",
        "artifact_encoder": "internal",
    }


def preferred_variant(stage_key: str, settings: Settings) -> str:
    if stage_key == "shadow_generator":
        return settings.shadow_model_variant.lower()
    if stage_key == "normal_estimator":
        return settings.normals_model_variant
    if stage_key == "segmenter":
        return settings.segmenter_model_variant
    if stage_key == "detector":
        return settings.detector_model_variant
    if stage_key == "geometry_estimator":
        return "geocalib" if settings.geometry_enabled else "disabled"
    return DEFAULT_VARIANTS.get(stage_key, "default")


def fallback_candidate_ids(stage_key: str, preferred_backend_kind: str, preferred_variant: str) -> list[StageBackendId]:
    if stage_key == "normal_estimator":
        return _normal_candidates(stage_key, preferred_backend_kind, preferred_variant)
    if stage_key == "shadow_generator":
        return _shadow_candidates(stage_key, preferred_backend_kind, preferred_variant)
    if stage_key == "segmenter":
        return _variant_stage_candidates(stage_key, preferred_backend_kind, preferred_variant, "birefnet", "mock-v1")
    if stage_key == "detector":
        return _variant_stage_candidates(stage_key, preferred_backend_kind, preferred_variant, "grounding-dino", "mock-v1")

    return _dedupe(
        [
            StageBackendId(
                stage_key=stage_key,
                backend_kind=backend_kind,
                model_variant=default_variant_for_backend(stage_key, backend_kind, preferred_variant),
            )
            for backend_kind in fallback_backend_order(stage_key, preferred_backend_kind)
        ]
    )


def resolve_active_backend(
    *,
    registry: PipelineBackendRegistry,
    stage_key: str,
    preferred_backend_kind: str,
    preferred_variant: str,
) -> tuple[RegisteredStageBackend | None, str | None]:
    candidates = fallback_candidate_ids(stage_key, preferred_backend_kind, preferred_variant)
    for index, backend_id in enumerate(candidates):
        registered = registry.get(backend_id)
        if _registered_is_usable(registered, stage_key):
            fallback_reason = f"{preferred_backend_kind}:{preferred_variant} unavailable for {stage_key}" if index > 0 else None
            return registered, fallback_reason

    if stage_key == "normal_estimator":
        for variant in ("stable-normal", "from-depth-v2"):
            registered = registry.get(StageBackendId(stage_key=stage_key, backend_kind="local", model_variant=variant))
            if _registered_is_usable(registered, stage_key):
                return registered, f"{preferred_backend_kind}:{preferred_variant} unavailable for {stage_key}"

    for registered in registry.list_stage(stage_key):
        if _registered_is_usable(registered, stage_key):
            return registered, f"{preferred_backend_kind}:{preferred_variant} unavailable for {stage_key}"
    return None, None


def fallback_backend_order(stage_key: str, preferred_backend_kind: str) -> list[str]:
    if preferred_backend_kind == "mock":
        return ["mock"]
    if preferred_backend_kind == "internal":
        return ["internal"]
    if stage_key == "normal_estimator":
        if preferred_backend_kind == "triton":
            return ["triton", "local", "local", "mock"]
        return ["local", "local", "mock"]
    if stage_key in {"geometry_estimator", "foreground_refiner", "composer"}:
        return [preferred_backend_kind, "local", "mock"]
    if preferred_backend_kind == "triton":
        return ["triton", "local", "mock"]
    if preferred_backend_kind == "local":
        return [preferred_backend_kind, "mock"]
    return ["local", "mock"]


def default_variant_for_backend(stage_key: str, backend_kind: str, preferred_variant: str) -> str:
    if backend_kind == "internal" and stage_key == "geometry_estimator":
        return preferred_variant
    if backend_kind == "mock":
        return MOCK_VARIANTS.get(stage_key, "mock-v1")
    if stage_key == "normal_estimator" and backend_kind == "local":
        return preferred_variant if preferred_variant in {"stable-normal", "from-depth-v2"} else "stable-normal"
    if stage_key in {"shadow_generator", "detector", "segmenter"}:
        return preferred_variant
    return DEFAULT_VARIANTS.get(stage_key, preferred_variant)


def _normal_candidates(stage_key: str, preferred_backend_kind: str, preferred_variant: str) -> list[StageBackendId]:
    if preferred_backend_kind == "mock":
        return [StageBackendId(stage_key=stage_key, backend_kind="mock", model_variant="mock-v1")]
    if preferred_variant == "from-depth-v2":
        return [
            StageBackendId(stage_key=stage_key, backend_kind="local", model_variant="from-depth-v2"),
            StageBackendId(stage_key=stage_key, backend_kind="mock", model_variant="mock-v1"),
        ]
    candidates = [StageBackendId(stage_key=stage_key, backend_kind=preferred_backend_kind, model_variant="stable-normal")]
    if preferred_backend_kind == "triton":
        candidates.append(StageBackendId(stage_key=stage_key, backend_kind="local", model_variant="stable-normal"))
    candidates.extend(
        [
            StageBackendId(stage_key=stage_key, backend_kind="local", model_variant="from-depth-v2"),
            StageBackendId(stage_key=stage_key, backend_kind="mock", model_variant="mock-v1"),
        ]
    )
    return _dedupe(candidates)


def _shadow_candidates(stage_key: str, preferred_backend_kind: str, preferred_variant: str) -> list[StageBackendId]:
    if preferred_backend_kind == "mock":
        return [StageBackendId(stage_key=stage_key, backend_kind="mock", model_variant="mock")]
    if preferred_backend_kind == "triton":
        return _dedupe(
            [
                StageBackendId(stage_key=stage_key, backend_kind="triton", model_variant=preferred_variant),
                StageBackendId(stage_key=stage_key, backend_kind="local", model_variant=preferred_variant),
                StageBackendId(stage_key=stage_key, backend_kind="local", model_variant="v1-gan"),
                StageBackendId(stage_key=stage_key, backend_kind="mock", model_variant="mock"),
            ]
        )
    candidates = [StageBackendId(stage_key=stage_key, backend_kind=preferred_backend_kind, model_variant=preferred_variant)]
    if preferred_variant == "v2-diff":
        candidates.append(StageBackendId(stage_key=stage_key, backend_kind="local", model_variant="v1-gan"))
    candidates.append(StageBackendId(stage_key=stage_key, backend_kind="mock", model_variant="mock"))
    return _dedupe(candidates)


def _variant_stage_candidates(
    stage_key: str,
    preferred_backend_kind: str,
    preferred_variant: str,
    local_variant: str,
    mock_variant: str,
) -> list[StageBackendId]:
    if preferred_backend_kind == "mock":
        return [StageBackendId(stage_key=stage_key, backend_kind="mock", model_variant=mock_variant)]
    if preferred_backend_kind == "triton":
        return _dedupe(
            [
                StageBackendId(stage_key=stage_key, backend_kind="triton", model_variant=preferred_variant),
                StageBackendId(stage_key=stage_key, backend_kind="triton", model_variant=local_variant),
                StageBackendId(stage_key=stage_key, backend_kind="local", model_variant=local_variant),
                StageBackendId(stage_key=stage_key, backend_kind="mock", model_variant=mock_variant),
            ]
        )
    return _dedupe(
        [
            StageBackendId(stage_key=stage_key, backend_kind=preferred_backend_kind, model_variant=preferred_variant),
            StageBackendId(stage_key=stage_key, backend_kind="local", model_variant=local_variant),
            StageBackendId(stage_key=stage_key, backend_kind="mock", model_variant=mock_variant),
        ]
    )


def _registered_is_usable(registered: RegisteredStageBackend | None, stage_key: str) -> bool:
    if registered is None or not registered.descriptor.available:
        return False
    if registered.handler is not None:
        return True
    return stage_key in {"artifact_encoder", "geometry_estimator"} and registered.descriptor.backend_kind == "internal"


def _dedupe(items: list[StageBackendId]) -> list[StageBackendId]:
    seen: set[StageBackendId] = set()
    result: list[StageBackendId] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
