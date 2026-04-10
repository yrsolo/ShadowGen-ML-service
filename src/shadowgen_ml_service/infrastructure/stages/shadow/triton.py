from __future__ import annotations

from shadowgen_ml_service.core.contracts import ShadowGenerator
from shadowgen_ml_service.core.models import ShadowResult
from shadowgen_ml_service.infrastructure.backends.triton.client import TritonInferenceClient
from shadowgen_ml_service.infrastructure.backends.triton.model_registry import TritonModelBinding
from shadowgen_ml_service.infrastructure.backends.triton.serializers import base64_png_to_image, image_to_base64_png


class TritonShadowGenerator(ShadowGenerator):
    backend_name = "triton-shadow"

    def __init__(self, client: TritonInferenceClient, binding: TritonModelBinding, *, model_variant: str) -> None:
        self.client = client
        self.binding = binding
        self.device_label = "triton"
        self.model_variant = model_variant

    def generate(self, cutout_rgba, mask, depth_map, normal_map, geometry, shadow) -> ShadowResult:
        response = self.client.infer_json(
            self.binding.model_name,
            {
                "img_base64": image_to_base64_png(cutout_rgba.convert("RGBA")),
                "mask_base64": image_to_base64_png(mask.convert("L")),
                "depth_base64": image_to_base64_png(depth_map.convert("L")),
                "normal_base64": image_to_base64_png(normal_map.convert("RGB")),
                "angle": float(shadow.angle_deg),
                "elevation": float(shadow.elevation_deg),
                "softness": float(shadow.softness),
                "reflection": float(shadow.reflection),
                "camera_pitch": float(geometry.camera_pitch),
                "camera_roll": float(geometry.camera_roll),
                "camera_fov": float(geometry.camera_fov),
            },
        )
        return ShadowResult(shadow_rgba=base64_png_to_image(response["shadow_base64"], mode="RGBA"))
