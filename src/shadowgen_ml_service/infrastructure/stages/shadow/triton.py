from __future__ import annotations

from shadowgen_ml_service.core.contracts import ShadowGenerator
from shadowgen_ml_service.core.models import ShadowResult
from shadowgen_ml_service.core.stage_io import ShadowInput
from shadowgen_ml_service.infrastructure.backends.triton.batching import TritonStageBatchCoordinator
from shadowgen_ml_service.infrastructure.backends.triton.client import TritonInferenceClient
from shadowgen_ml_service.infrastructure.backends.triton.model_registry import TritonModelBinding
from shadowgen_ml_service.infrastructure.backends.triton.serializers import (
    batch_input_tensors,
    grayscale_to_nchw_float32_input,
    image_to_nchw_float32_input,
    rgb_to_nchw_float32_input,
    rgba_output_to_image,
    scalar_to_input,
    split_output_tensor,
)
from shadowgen_ml_service.utils.images import ensure_pil, pil_to_asset


class TritonShadowGenerator(ShadowGenerator):
    backend_name = "triton-shadow"

    def __init__(
        self,
        client: TritonInferenceClient,
        binding: TritonModelBinding,
        *,
        model_variant: str,
        batcher: TritonStageBatchCoordinator | None = None,
    ) -> None:
        self.client = client
        self.binding = binding
        self.batcher = batcher
        self.device_label = "triton"
        self.model_variant = model_variant

    def generate(self, stage_input: ShadowInput) -> ShadowResult:
        if self.batcher is None:
            return self._generate_many([stage_input])[0]
        return self.batcher.execute(
            stage_key="shadow_generator",
            model_variant=self.model_variant,
            payload=stage_input,
            run_batch=self._generate_many,
        )

    def _generate_many(self, stage_inputs: list[ShadowInput]) -> list[ShadowResult]:
        request_batches = []
        for stage_input in stage_inputs:
            image = ensure_pil(stage_input.img)
            mask = ensure_pil(stage_input.mask)
            inputs = [
                image_to_nchw_float32_input(self.binding.inputs["img"].tensor_name, image.convert("RGBA")),
                grayscale_to_nchw_float32_input(self.binding.inputs["mask"].tensor_name, mask.convert("L")),
            ]
            if "depth" in self.binding.inputs:
                depth = ensure_pil(stage_input.depth)
                inputs.append(grayscale_to_nchw_float32_input(self.binding.inputs["depth"].tensor_name, depth.convert("L")))
            if "normal" in self.binding.inputs:
                normal = ensure_pil(stage_input.normal)
                inputs.append(rgb_to_nchw_float32_input(self.binding.inputs["normal"].tensor_name, normal.convert("RGB")))
            if "angle" in self.binding.inputs:
                inputs.append(scalar_to_input(self.binding.inputs["angle"].tensor_name, stage_input.angle))
            if "elevation" in self.binding.inputs:
                inputs.append(scalar_to_input(self.binding.inputs["elevation"].tensor_name, stage_input.elevation))
            if "softness" in self.binding.inputs:
                inputs.append(scalar_to_input(self.binding.inputs["softness"].tensor_name, stage_input.softness))
            if "reflection" in self.binding.inputs:
                inputs.append(scalar_to_input(self.binding.inputs["reflection"].tensor_name, stage_input.reflection))
            request_batches.append(inputs)
        batched_inputs = batch_input_tensors(request_batches)
        response = self.client.infer(self.binding, inputs=batched_inputs)
        shadow_tensors = split_output_tensor(response[self.binding.outputs["shadow_image"].tensor_name], len(stage_inputs))
        return [ShadowResult(shadow_image=pil_to_asset(rgba_output_to_image(tensor))) for tensor in shadow_tensors]
