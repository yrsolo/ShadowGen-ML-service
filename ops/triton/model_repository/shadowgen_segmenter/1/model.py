from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import triton_python_backend_utils as pb_utils
from transformers import AutoModelForImageSegmentation


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class _BackendConfig:
    model_id: str
    resolution: int
    mask_threshold: float
    device: str
    compile_enabled: bool
    compile_mode: str
    compile_backend: str | None
    matmul_precision: str
    local_files_only: bool


class TritonPythonModel:
    def initialize(self, args: dict[str, str]) -> None:
        model_config = json.loads(args["model_config"])
        self.model_config = model_config
        self.output_mask_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(model_config, "mask")["data_type"]
        )
        self.config = self._load_backend_config(model_config)
        self.device = self._resolve_device(self.config.device)
        self._configure_torch_runtime()
        self.model = AutoModelForImageSegmentation.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
            local_files_only=self.config.local_files_only,
        )
        self.model.eval()
        self.model.to(self.device)
        self.model = self._maybe_compile(self.model)
        self.mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(_IMAGENET_STD, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def execute(self, requests):
        responses = [None] * len(requests)
        grouped_indices: dict[tuple[int, int], list[int]] = {}
        arrays: list[np.ndarray] = []

        for index, request in enumerate(requests):
            tensor = pb_utils.get_input_tensor_by_name(request, "image")
            if tensor is None:
                raise pb_utils.TritonModelException("missing required input tensor: image")
            array = tensor.as_numpy().astype(np.float32)
            if array.ndim == 3:
                array = array[None, ...]
            if array.ndim != 4 or array.shape[1] != 3:
                raise pb_utils.TritonModelException(
                    f"segmenter expects NCHW FP32 image input, observed shape {tuple(array.shape)}"
                )
            arrays.append(array)
            grouped_indices.setdefault((int(array.shape[2]), int(array.shape[3])), []).append(index)

        for shape_key, request_indices in grouped_indices.items():
            masks = self._infer_group([arrays[index] for index in request_indices], shape_key)
            for request_index, mask in zip(request_indices, masks, strict=True):
                responses[request_index] = pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("mask", mask.astype(self.output_mask_dtype, copy=False))]
                )

        return responses

    def finalize(self) -> None:
        return

    def _infer_group(self, request_arrays: list[np.ndarray], shape_key: tuple[int, int]) -> list[np.ndarray]:
        merged = np.concatenate(request_arrays, axis=0)
        original_height, original_width = shape_key
        tensor = torch.from_numpy(merged).to(self.device, dtype=torch.float32)
        resized = F.interpolate(
            tensor,
            size=(self.config.resolution, self.config.resolution),
            mode="bilinear",
            align_corners=False,
        )
        normalized = (resized - self.mean) / self.std
        with torch.no_grad():
            prediction = self.model(normalized)[-1].sigmoid()
        if prediction.ndim == 3:
            prediction = prediction.unsqueeze(1)
        if prediction.shape[-2:] != (original_height, original_width):
            prediction = F.interpolate(
                prediction,
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )
        prediction = prediction.clamp(0.0, 1.0)
        if self.config.mask_threshold > 0.0:
            threshold = torch.tensor(self.config.mask_threshold, dtype=prediction.dtype, device=prediction.device)
            prediction = torch.where(prediction >= threshold, prediction, torch.zeros_like(prediction))
        batch_sizes = [int(array.shape[0]) for array in request_arrays]
        split_tensors = torch.split(prediction, batch_sizes, dim=0)
        return [tensor.detach().cpu().numpy() for tensor in split_tensors]

    def _load_backend_config(self, model_config: dict) -> _BackendConfig:
        parameters = model_config.get("parameters", {})

        def _string(name: str, default: str) -> str:
            raw = parameters.get(name)
            if not isinstance(raw, dict):
                return default
            value = raw.get("string_value")
            if value is None:
                return default
            return str(value)

        return _BackendConfig(
            model_id=_string("model_id", "ZhengPeng7/BiRefNet_lite-matting"),
            resolution=int(_string("resolution", "1024")),
            mask_threshold=float(_string("mask_threshold", "0.5")),
            device=_string("device", "cuda"),
            compile_enabled=_as_bool(_string("compile_enabled", "false"), False),
            compile_mode=_string("compile_mode", "reduce-overhead"),
            compile_backend=_string("compile_backend", "") or None,
            matmul_precision=_string("matmul_precision", "high"),
            local_files_only=_as_bool(_string("local_files_only", "false"), False),
        )

    def _resolve_device(self, requested: str) -> torch.device:
        if requested.startswith("cuda") and torch.cuda.is_available():
            return torch.device(requested)
        return torch.device("cpu")

    def _configure_torch_runtime(self) -> None:
        setter = getattr(torch, "set_float32_matmul_precision", None)
        if setter is not None and self.config.matmul_precision:
            try:
                setter(self.config.matmul_precision)
            except Exception:
                pass

    def _maybe_compile(self, model):
        if not self.config.compile_enabled:
            return model
        if self.device.type != "cuda":
            return model
        compiler = getattr(torch, "compile", None)
        if compiler is None:
            return model
        kwargs = {}
        if self.config.compile_mode:
            kwargs["mode"] = self.config.compile_mode
        if self.config.compile_backend:
            kwargs["backend"] = self.config.compile_backend
        try:
            return compiler(model, **kwargs)
        except Exception:
            return model
