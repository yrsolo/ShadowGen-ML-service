from __future__ import annotations

from PIL import Image

from shadowgen_ml_service.core.contracts import NormalEstimator
from shadowgen_ml_service.core.models import NormalResult
from shadowgen_ml_service.core.stage_io import NormalsInput
from shadowgen_ml_service.utils.images import normals_from_depth


class NormalFromDepthEstimator(NormalEstimator):
    def __init__(self) -> None:
        self.device_label = "cpu"
        self.backend_name = "from-depth"
        self.model_variant = "from-depth-v2"

    def estimate(self, stage_input: NormalsInput) -> NormalResult:
        if stage_input.depth_map is None:
            raise ValueError("depth_map is required for the depth-derived normals backend")
        return NormalResult(normal_map=normals_from_depth(stage_input.depth_map))
