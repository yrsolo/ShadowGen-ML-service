from shadowgen_ml_service.infrastructure.stages.detection.grounding_dino import RealDetector, probe_grounding_dino, select_primary_detection
from shadowgen_ml_service.infrastructure.stages.detection.mock import MockDetector

__all__ = ["MockDetector", "RealDetector", "probe_grounding_dino", "select_primary_detection"]
