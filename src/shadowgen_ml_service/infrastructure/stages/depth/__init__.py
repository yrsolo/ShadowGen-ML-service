from shadowgen_ml_service.infrastructure.stages.depth.depth_anything import RealDepthEstimator, probe_depth_anything
from shadowgen_ml_service.infrastructure.stages.depth.mock import MockDepthEstimator

__all__ = ["MockDepthEstimator", "RealDepthEstimator", "probe_depth_anything"]
