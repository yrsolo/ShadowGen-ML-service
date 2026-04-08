from __future__ import annotations

from PIL import Image

from shadowgen_ml_service.core.contracts import NormalEstimator
from shadowgen_ml_service.core.models import NormalResult
from shadowgen_ml_service.utils.images import normals_from_depth


class NormalFromDepthEstimator(NormalEstimator):
    def estimate(self, depth_map: Image.Image) -> NormalResult:
        return NormalResult(normal_map=normals_from_depth(depth_map))
