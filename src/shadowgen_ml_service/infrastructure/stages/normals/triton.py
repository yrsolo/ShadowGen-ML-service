from __future__ import annotations

from shadowgen_ml_service.core.contracts import NormalEstimator
from shadowgen_ml_service.core.models import NormalResult
from shadowgen_ml_service.core.stage_io import NormalsInput
from shadowgen_ml_service.infrastructure.backends.triton.batching import TritonStageBatchCoordinator
from shadowgen_ml_service.infrastructure.backends.triton.client import TritonInferenceClient
from shadowgen_ml_service.infrastructure.backends.triton.model_registry import TritonModelBinding
from shadowgen_ml_service.infrastructure.backends.triton.serializers import (
    batch_input_tensors,
    grayscale_to_nchw_float32_input,
    image_to_nchw_float32_input,
    rgb_output_to_image,
    split_output_tensor,
)
from shadowgen_ml_service.utils.images import ensure_pil, pil_to_asset


class TritonNormalEstimator(NormalEstimator):
    backend_name = "triton-normals"

    def __init__(self, client: TritonInferenceClient, binding: TritonModelBinding, batcher: TritonStageBatchCoordinator | None = None) -> None:
        self.client = client
        self.binding = binding
        self.batcher = batcher
        self.device_label = "triton"
        self.model_variant = binding.model_variant

    def estimate(self, stage_input: NormalsInput) -> NormalResult:
        if self.batcher is None:
            return self._estimate_many([stage_input])[0]
        return self.batcher.execute(
            stage_key="normal_estimator",
            model_variant=self.model_variant,
            payload=stage_input,
            run_batch=self._estimate_many,
        )

    def _estimate_many(self, stage_inputs: list[NormalsInput]) -> list[NormalResult]:
        request_batches = []
        for stage_input in stage_inputs:
            source = ensure_pil(stage_input.image)
            inputs = [image_to_nchw_float32_input(self.binding.inputs["image"].tensor_name, source.convert("RGB"))]
            if stage_input.depth_map is not None and "depth" in self.binding.inputs:
                inputs.append(
                    grayscale_to_nchw_float32_input(
                        self.binding.inputs["depth"].tensor_name,
                        ensure_pil(stage_input.depth_map).convert("L"),
                    )
                )
            request_batches.append(inputs)
        batched_inputs = batch_input_tensors(request_batches)
        response = self.client.infer(self.binding, inputs=batched_inputs)
        normal_tensors = split_output_tensor(response[self.binding.outputs["normal"].tensor_name], len(stage_inputs))
        return [NormalResult(normal_map=pil_to_asset(rgb_output_to_image(tensor))) for tensor in normal_tensors]
