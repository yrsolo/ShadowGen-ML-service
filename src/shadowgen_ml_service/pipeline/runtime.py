from __future__ import annotations

from dataclasses import dataclass

from shadowgen_ml_service.adapters.mock import (
    DefaultArtifactEncoder,
    MockComposer,
    MockDepthEstimator,
    MockDetector,
    MockGeometryEstimator,
    MockNormalEstimator,
    MockSegmenter,
    MockShadowGenerator,
)
from shadowgen_ml_service.adapters.real import probe_birefnet, probe_depth_anything, probe_geocalib, probe_grounding_dino
from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.pipeline.cache import PreprocessCache
from shadowgen_ml_service.pipeline.contracts import ArtifactEncoder, Composer, DepthEstimator, Detector, GeometryEstimator, NormalEstimator, Segmenter, ShadowGenerator
from shadowgen_ml_service.pipeline.types import ComponentStatus, RuntimeDescriptor


@dataclass
class PipelineRuntime:
    detector: Detector
    geometry: GeometryEstimator
    segmenter: Segmenter
    depth: DepthEstimator
    normals: NormalEstimator
    shadow: ShadowGenerator
    composer: Composer
    encoder: ArtifactEncoder
    cache: PreprocessCache
    descriptor: RuntimeDescriptor

    @property
    def signature(self) -> str:
        parts = [
            component.name + ":" + component.implementation + ":" + component.model_version
            for component in self.descriptor.components
        ]
        return "|".join(parts)


def _component_status(
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


def build_runtime(settings: Settings) -> PipelineRuntime:
    mode = settings.runtime_mode.lower()
    grounding = probe_grounding_dino()
    geocalib = probe_geocalib()
    birefnet = probe_birefnet()
    depth_anything = probe_depth_anything()

    detector = MockDetector()
    geometry = MockGeometryEstimator()
    segmenter = MockSegmenter()
    depth = MockDepthEstimator()
    normals = MockNormalEstimator()
    shadow = MockShadowGenerator()
    composer = MockComposer()
    encoder = DefaultArtifactEncoder()

    components = [
        _component_status(
            "detector",
            "mock",
            grounding.model_name,
            "mock-v1",
            True,
            True,
            "deterministic fallback detector" if not grounding.available else "real wrapper scaffold exists",
        ),
        _component_status(
            "geometry_estimator",
            "mock",
            geocalib.model_name,
            "mock-v1",
            True,
            True,
            "deterministic fallback geometry estimator" if not geocalib.available else "real wrapper scaffold exists",
        ),
        _component_status(
            "segmenter",
            "mock",
            birefnet.model_name,
            "mock-v1",
            True,
            True,
            "deterministic fallback matting stage" if not birefnet.available else "real wrapper scaffold exists",
        ),
        _component_status(
            "depth_estimator",
            "mock",
            depth_anything.model_name,
            "mock-v1",
            True,
            True,
            "deterministic fallback depth estimator" if not depth_anything.available else "real wrapper scaffold exists",
        ),
        _component_status("normal_estimator", "internal", "normal-map-from-depth", "v1", True, False),
        _component_status("shadow_generator", "deterministic-stub", "shadow-stub", "v1", True, False),
        _component_status("composer", "python", "solid-background-composer", "v1", True, False),
        _component_status("artifact_encoder", "python", "artifact-encoder", "v1", True, False),
    ]

    fallback_needed = not all(probe.available for probe in (grounding, geocalib, birefnet, depth_anything))
    if mode == "mock":
        active_mode = "mock"
        degraded = False
    elif mode == "real":
        active_mode = "real-fallback" if fallback_needed else "real"
        degraded = fallback_needed
    else:
        active_mode = "auto-fallback" if fallback_needed else "auto-real"
        degraded = fallback_needed

    descriptor = RuntimeDescriptor(
        mode=active_mode,
        degraded=degraded,
        components=components,
        model_version="mock-stack-v1" if degraded or active_mode == "mock" else "real-stack-v1",
    )
    return PipelineRuntime(
        detector=detector,
        geometry=geometry,
        segmenter=segmenter,
        depth=depth,
        normals=normals,
        shadow=shadow,
        composer=composer,
        encoder=encoder,
        cache=PreprocessCache(settings.preprocess_cache_dir),
        descriptor=descriptor,
    )
