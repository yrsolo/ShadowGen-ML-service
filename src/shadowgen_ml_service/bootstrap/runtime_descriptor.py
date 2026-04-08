from __future__ import annotations

from shadowgen_ml_service.core.models import ComponentStatus, RuntimeDescriptor


def component_status(
    name: str,
    implementation: str,
    model_name: str,
    model_version: str,
    available: bool,
    using_mock: bool,
    detail: str | None = None,
) -> ComponentStatus:
    return ComponentStatus(
        name=name,
        implementation=implementation,
        model_name=model_name,
        model_version=model_version,
        available=available,
        using_mock=using_mock,
        detail=detail,
    )


def build_runtime_descriptor(mode: str, components: list[ComponentStatus]) -> RuntimeDescriptor:
    fallback_needed = any(
        component.using_mock
        for component in components
        if component.name in {"detector", "geometry_estimator", "segmenter", "depth_estimator"}
    )
    if mode == "mock":
        active_mode = "mock"
        degraded = False
    elif mode == "real":
        active_mode = "real-fallback" if fallback_needed else "real"
        degraded = fallback_needed
    else:
        active_mode = "auto-fallback" if fallback_needed else "auto-real"
        degraded = fallback_needed
    return RuntimeDescriptor(
        mode=active_mode,
        degraded=degraded,
        components=components,
        model_version="mock-stack-v1" if degraded or active_mode == "mock" else "real-stack-v1",
    )
