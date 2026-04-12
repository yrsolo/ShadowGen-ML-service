from __future__ import annotations

from shadowgen_ml_service.core.contracts import DepthEstimator
from shadowgen_ml_service.core.models import DepthResult
from shadowgen_ml_service.core.stage_io import DepthInput
from shadowgen_ml_service.infrastructure.backends.triton.batching import TritonStageBatchCoordinator
from shadowgen_ml_service.infrastructure.backends.triton.client import TritonInferenceClient
from shadowgen_ml_service.infrastructure.backends.triton.model_registry import TritonModelBinding
from shadowgen_ml_service.infrastructure.backends.triton.serializers import (
    batch_input_tensors,
    grayscale_output_to_image,
    image_to_nchw_float32_input,
    mask_to_nchw_float32_input,
    split_output_tensor,
)
from shadowgen_ml_service.utils.images import ensure_pil, pil_to_asset


class TritonDepthEstimator(DepthEstimator):
    backend_name = "triton-depth"

    def __init__(self, client: TritonInferenceClient, binding: TritonModelBinding, batcher: TritonStageBatchCoordinator | None = None) -> None:
        self.client = client
        self.binding = binding
        self.batcher = batcher
        self.device_label = "triton"
        self.model_variant = binding.model_variant

    def estimate(self, stage_input: DepthInput) -> DepthResult:
        if self.batcher is None:
            return self._estimate_many([stage_input])[0]
        return self.batcher.execute(
            stage_key="depth_estimator",
            model_variant=self.model_variant,
            payload=stage_input,
            run_batch=self._estimate_many,
        )

    def _estimate_many(self, stage_inputs: list[DepthInput]) -> list[DepthResult]:
        batched_inputs = batch_input_tensors(
            [
                [
                    image_to_nchw_float32_input(self.binding.inputs["image"].tensor_name, ensure_pil(stage_input.image).convert("RGB")),
                    mask_to_nchw_float32_input(self.binding.inputs["mask"].tensor_name, ensure_pil(stage_input.mask).convert("L")),
                ]
                for stage_input in stage_inputs
            ]
        )
        response = self.client.infer(self.binding, inputs=batched_inputs)
        depth_tensors = split_output_tensor(response[self.binding.outputs["depth"].tensor_name], len(stage_inputs))
        return [DepthResult(depth_map=pil_to_asset(grayscale_output_to_image(tensor))) for tensor in depth_tensors]
