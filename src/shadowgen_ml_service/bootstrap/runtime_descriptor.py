from __future__ import annotations

from shadowgen_ml_service.core.models import ComponentStatus, RuntimeDescriptor, StageBackendDescriptor, StageBackendId


def backend_descriptor(
    *,
    stage_key: str,
    backend_kind: str,
    model_variant: str,
    model_name: str,
    model_version: str,
    available: bool,
    detail: str | None = None,
    device: str = "cpu",
    endpoint: str | None = None,
    supports_batching: bool = False,
    supports_async: bool = False,
    is_default: bool = False,
) -> StageBackendDescriptor:
    return StageBackendDescriptor(
        backend_id=StageBackendId(stage_key=stage_key, backend_kind=backend_kind, model_variant=model_variant),
        model_name=model_name,
        model_version=model_version,
        available=available,
        detail=detail,
        device=device,
        endpoint=endpoint,
        supports_batching=supports_batching,
        supports_async=supports_async,
        is_default=is_default,
    )


def component_status(
    *,
    name: str,
    implementation: str,
    model_name: str,
    model_version: str,
    available: bool,
    using_mock: bool,
    detail: str | None = None,
    backend_kind: str = "mock",
    model_variant: str = "default",
    device: str = "cpu",
    endpoint: str | None = None,
    supports_batching: bool = False,
    supports_async: bool = False,
    fallback_reason: str | None = None,
    backends: list[StageBackendDescriptor] | None = None,
) -> ComponentStatus:
    return ComponentStatus(
        name=name,
        implementation=implementation,
        model_name=model_name,
        model_version=model_version,
        available=available,
        using_mock=using_mock,
        detail=detail,
        backend_kind=backend_kind,
        model_variant=model_variant,
        device=device,
        endpoint=endpoint,
        supports_batching=supports_batching,
        supports_async=supports_async,
        fallback_reason=fallback_reason,
        backends=list(backends or []),
    )


def build_runtime_descriptor(
    *,
    execution_default_backend: str,
    async_enabled: bool,
    components: list[ComponentStatus],
) -> RuntimeDescriptor:
    fallback_needed = any(component.fallback_reason for component in components)
    mock_only = all(component.backend_kind == "mock" for component in components if component.name != "composer")
    if mock_only:
        active_mode = "mock"
    elif fallback_needed:
        active_mode = f"{execution_default_backend}-fallback"
    else:
        active_mode = execution_default_backend

    version_parts = [f"{component.name}:{component.backend_kind}:{component.model_variant}" for component in components]
    return RuntimeDescriptor(
        mode=active_mode,
        degraded=fallback_needed,
        components=components,
        model_version="|".join(version_parts),
        execution_default_backend=execution_default_backend,
        async_enabled=async_enabled,
    )
