from __future__ import annotations

from dataclasses import dataclass

from shadowgen_ml_service.core.contracts import (
    ArtifactEncoder,
    Composer,
    DepthEstimator,
    Detector,
    ForegroundColorEstimator,
    GeometryEstimator,
    NormalEstimator,
    PreprocessCacheRepository,
    PreviewBuilderRegistry,
    Segmenter,
    ShadowGenerator,
)
from shadowgen_ml_service.core.models import RuntimeDescriptor


@dataclass
class PipelineRuntime:
    detector: Detector
    mock_detector: Detector
    real_detector: Detector | None
    geometry: GeometryEstimator
    mock_geometry: GeometryEstimator
    real_geometry: GeometryEstimator | None
    segmenter: Segmenter
    mock_segmenter: Segmenter
    real_segmenter: Segmenter | None
    foreground_refiner: ForegroundColorEstimator
    mock_foreground_refiner: ForegroundColorEstimator
    real_foreground_refiner: ForegroundColorEstimator | None
    depth: DepthEstimator
    mock_depth: DepthEstimator
    real_depth: DepthEstimator | None
    normals: NormalEstimator
    mock_normals: NormalEstimator
    real_normals: NormalEstimator | None
    shadow: ShadowGenerator
    composer: Composer
    encoder: ArtifactEncoder
    cache: PreprocessCacheRepository
    previews: PreviewBuilderRegistry
    descriptor: RuntimeDescriptor

    @property
    def signature(self) -> str:
        return "|".join(
            f"{component.name}:{component.implementation}:{component.model_version}"
            for component in self.descriptor.components
        )
