from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
from shadowgen_ml_service.adapters.real import (
    RealDetector,
    RealGeometryEstimator,
    RealSegmenter,
    probe_birefnet,
    probe_depth_anything,
    probe_geocalib,
    probe_grounding_dino,
)
from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.pipeline.cache import PreprocessCache
from shadowgen_ml_service.pipeline.contracts import ArtifactEncoder, Composer, DepthEstimator, Detector, GeometryEstimator, NormalEstimator, Segmenter, ShadowGenerator
from shadowgen_ml_service.pipeline.types import ComponentStatus, RuntimeDescriptor


@dataclass
class PipelineRuntime:
    detector: Detector
    mock_detector: Detector
    real_detector: Optional[Detector]
    geometry: GeometryEstimator
    mock_geometry: GeometryEstimator
    real_geometry: Optional[GeometryEstimator]
    segmenter: Segmenter
    mock_segmenter: Segmenter
    real_segmenter: Optional[Segmenter]
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
    birefnet = probe_birefnet(allow_cpu=settings.birefnet_allow_cpu)
    depth_anything = probe_depth_anything()

    mock_detector = MockDetector()
    detector: Detector = mock_detector
    real_detector: Detector | None = None
    mock_geometry = MockGeometryEstimator()
    geometry: GeometryEstimator = mock_geometry
    real_geometry: GeometryEstimator | None = None
    mock_segmenter = MockSegmenter()
    segmenter = mock_segmenter
    real_segmenter: Segmenter | None = None
    depth = MockDepthEstimator()
    normals = MockNormalEstimator()
    shadow = MockShadowGenerator()
    composer = MockComposer()
    encoder = DefaultArtifactEncoder()

    detector_component = _component_status(
        "detector",
        "mock",
        grounding.model_name,
        "mock-v1",
        True,
        True,
        "deterministic fallback detector" if not grounding.available else "real wrapper scaffold exists",
    )
    if mode != "mock" and grounding.available:
        try:
            detector = RealDetector(
                model_id=settings.grounding_dino_model_id,
                prompt=settings.grounding_dino_prompt,
                box_threshold=settings.grounding_dino_box_threshold,
                text_threshold=settings.grounding_dino_text_threshold,
            )
            real_detector = detector
            detector_component = _component_status(
                "detector",
                "real",
                grounding.model_name,
                grounding.model_version,
                True,
                False,
                (
                    "GroundingDINO backend active "
                    f"(model_id={settings.grounding_dino_model_id}, prompt={settings.grounding_dino_prompt!r}, "
                    f"box_threshold={settings.grounding_dino_box_threshold}, "
                    f"text_threshold={settings.grounding_dino_text_threshold})"
                ),
            )
        except Exception as exc:
            detector_component = _component_status(
                "detector",
                "mock-fallback",
                grounding.model_name,
                "mock-v1",
                True,
                True,
                f"GroundingDINO init failed: {exc}",
            )

    geometry_component = _component_status(
        "geometry_estimator",
        "mock",
        geocalib.model_name,
        "mock-v1",
        True,
        True,
        "deterministic fallback geometry estimator" if not geocalib.available else "real wrapper scaffold exists",
    )
    if mode != "mock" and geocalib.available:
        try:
            geometry = RealGeometryEstimator(
                weights=settings.geocalib_weights,
                camera_model=settings.geocalib_camera_model,
                shared_intrinsics=settings.geocalib_shared_intrinsics,
            )
            real_geometry = geometry
            geometry_component = _component_status(
                "geometry_estimator",
                "real",
                geocalib.model_name,
                geocalib.model_version,
                True,
                False,
                (
                    "GeoCalib backend active "
                    f"(weights={settings.geocalib_weights}, camera_model={settings.geocalib_camera_model}, "
                    f"shared_intrinsics={settings.geocalib_shared_intrinsics})"
                ),
            )
        except Exception as exc:
            geometry_component = _component_status(
                "geometry_estimator",
                "mock-fallback",
                geocalib.model_name,
                "mock-v1",
                True,
                True,
                f"GeoCalib init failed: {exc}",
            )

    components = [
        detector_component,
        geometry_component,
        None,
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

    segmenter_component = _component_status(
        "segmenter",
        "mock",
        birefnet.model_name,
        "mock-v1",
        True,
        True,
        birefnet.detail or ("deterministic fallback matting stage" if not birefnet.available else "real wrapper scaffold exists"),
    )
    if mode != "mock" and birefnet.available:
        try:
            segmenter = RealSegmenter(
                model_id=settings.birefnet_model_id,
                resolution=settings.birefnet_resolution,
                mask_threshold=settings.birefnet_mask_threshold,
            )
            real_segmenter = segmenter
            segmenter_component = _component_status(
                "segmenter",
                "real",
                birefnet.model_name,
                birefnet.model_version,
                True,
                False,
                (
                    "BiRefNet backend active "
                    f"(model_id={settings.birefnet_model_id}, resolution={settings.birefnet_resolution}, "
                    f"mask_threshold={settings.birefnet_mask_threshold}, "
                    f"allow_cpu={settings.birefnet_allow_cpu})"
                ),
            )
        except Exception as exc:
            segmenter_component = _component_status(
                "segmenter",
                "mock-fallback",
                birefnet.model_name,
                "mock-v1",
                True,
                True,
                f"BiRefNet init failed: {exc}",
            )

    components[2] = segmenter_component
    fallback_needed = any(component.using_mock for component in components if component is not None and component.name in {"detector", "geometry_estimator", "segmenter", "depth_estimator"})
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
        mock_detector=mock_detector,
        real_detector=real_detector,
        geometry=geometry,
        mock_geometry=mock_geometry,
        real_geometry=real_geometry,
        segmenter=segmenter,
        mock_segmenter=mock_segmenter,
        real_segmenter=real_segmenter,
        depth=depth,
        normals=normals,
        shadow=shadow,
        composer=composer,
        encoder=encoder,
        cache=PreprocessCache(settings.preprocess_cache_dir),
        descriptor=descriptor,
    )
