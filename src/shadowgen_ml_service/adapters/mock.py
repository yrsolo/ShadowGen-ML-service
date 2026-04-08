from shadowgen_ml_service.infrastructure.encoding.default import DefaultArtifactEncoder
from shadowgen_ml_service.infrastructure.stages.composition.python_composer import PythonComposer as MockComposer
from shadowgen_ml_service.infrastructure.stages.depth.mock import MockDepthEstimator
from shadowgen_ml_service.infrastructure.stages.detection.mock import MockDetector
from shadowgen_ml_service.infrastructure.stages.foreground_refinement.mock import PassthroughForegroundColorEstimator
from shadowgen_ml_service.infrastructure.stages.geometry.mock import MockGeometryEstimator
from shadowgen_ml_service.infrastructure.stages.normals.from_depth import NormalFromDepthEstimator as MockNormalEstimator
from shadowgen_ml_service.infrastructure.stages.segmentation.mock import MockSegmenter
from shadowgen_ml_service.infrastructure.stages.shadow.stub import DeterministicShadowGenerator as MockShadowGenerator

__all__ = [
    "DefaultArtifactEncoder",
    "MockComposer",
    "MockDepthEstimator",
    "MockDetector",
    "PassthroughForegroundColorEstimator",
    "MockGeometryEstimator",
    "MockNormalEstimator",
    "MockSegmenter",
    "MockShadowGenerator",
]
