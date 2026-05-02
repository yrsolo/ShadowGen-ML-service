from __future__ import annotations

import json
import os
import inspect
from dataclasses import dataclass

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from PIL import Image
from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_override(name: str) -> str | None:
    value = os.getenv(f"SHADOWGEN_TRITON_DETECTOR_{name.upper()}")
    if value is None or value.strip() == "":
        return None
    return value


@dataclass(frozen=True)
class _BackendConfig:
    model_id: str
    prompt: str
    box_threshold: float
    text_threshold: float
    device: str
    matmul_precision: str
    local_files_only: bool


class TritonPythonModel:
    def initialize(self, args: dict[str, str]) -> None:
        model_config = json.loads(args["model_config"])
        self.model_config = model_config
        self.output_bbox_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(model_config, "bbox")["data_type"]
        )
        self.output_confidence_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(model_config, "confidence")["data_type"]
        )
        self.config = self._load_backend_config(model_config)
        self.device = self._resolve_device(self.config.device)
        self._configure_torch_runtime()

        self.processor = GroundingDinoProcessor.from_pretrained(
            self.config.model_id,
            local_files_only=self.config.local_files_only,
        )
        self.model = GroundingDinoForObjectDetection.from_pretrained(
            self.config.model_id,
            local_files_only=self.config.local_files_only,
        )
        self.model.eval()
        self.model.to(self.device)

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                image_tensor = pb_utils.get_input_tensor_by_name(request, "image")
                if image_tensor is None:
                    raise pb_utils.TritonModelException("missing required input tensor: image")
                array = image_tensor.as_numpy().astype(np.float32)
                if array.ndim == 3:
                    array = array[None, ...]
                if array.ndim != 4 or array.shape[1] != 3:
                    raise pb_utils.TritonModelException(
                        f"detector expects NCHW FP32 image input, observed shape {tuple(array.shape)}"
                    )

                boxes, confidences = self._infer_batch(array)
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor("bbox", boxes.astype(self.output_bbox_dtype, copy=False)),
                            pb_utils.Tensor(
                                "confidence",
                                confidences.astype(self.output_confidence_dtype, copy=False),
                            ),
                        ]
                    )
                )
            except pb_utils.TritonModelException:
                raise
            except Exception as exc:
                raise pb_utils.TritonModelException(
                    f"shadowgen_detector inference failed: {type(exc).__name__}: {exc}"
                ) from exc
        return responses

    def finalize(self) -> None:
        return

    def _infer_batch(self, array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        images = [_nchw_to_pil(sample) for sample in array]
        inputs = self.processor(images=images, text=self.config.prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        target_sizes = [image.size[::-1] for image in images]
        with torch.inference_mode():
            outputs = self.model(**inputs)

        post_process_kwargs = _post_process_kwargs(
            self.processor,
            box_threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
        )
        try:
            processed = self.processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs.get("input_ids"),
                target_sizes=target_sizes,
                **post_process_kwargs,
            )
        except TypeError:
            processed = self.processor.post_process_grounded_object_detection(
                outputs,
                target_sizes=target_sizes,
                **post_process_kwargs,
            )

        boxes = []
        confidences = []
        for image, result in zip(images, processed, strict=True):
            selected_box, confidence = _select_primary_detection(result, image.size)
            boxes.append(selected_box)
            confidences.append([confidence])
        return np.asarray(boxes, dtype=np.float32), np.asarray(confidences, dtype=np.float32)

    def _load_backend_config(self, model_config: dict) -> _BackendConfig:
        parameters = model_config.get("parameters", {})

        def _string(name: str, default: str) -> str:
            env_value = _env_override(name)
            if env_value is not None:
                return env_value
            raw = parameters.get(name)
            if not isinstance(raw, dict):
                return default
            value = raw.get("string_value")
            if value is None:
                return default
            return str(value)

        return _BackendConfig(
            model_id=_string("model_id", "IDEA-Research/grounding-dino-base"),
            prompt=_string("prompt", "object."),
            box_threshold=float(_string("box_threshold", "0.25")),
            text_threshold=float(_string("text_threshold", "0.25")),
            device=_string("device", "cuda"),
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


def _nchw_to_pil(sample: np.ndarray) -> Image.Image:
    if sample.ndim != 3 or sample.shape[0] != 3:
        raise pb_utils.TritonModelException(f"expected CHW RGB sample, observed shape {tuple(sample.shape)}")
    data = np.transpose(sample, (1, 2, 0))
    data = np.clip(data, 0.0, 1.0)
    return Image.fromarray((data * 255.0).astype(np.uint8), mode="RGB")


def _select_primary_detection(result: dict, image_size: tuple[int, int]) -> tuple[tuple[float, float, float, float], float]:
    raw_boxes = result.get("boxes")
    raw_scores = result.get("scores")
    if raw_boxes is None or raw_scores is None or len(raw_boxes) == 0:
        raise pb_utils.TritonModelException("GroundingDINO returned no candidate boxes")

    candidates = []
    for box, score in zip(raw_boxes, raw_scores, strict=True):
        bbox = _clamp_bbox(tuple(float(value) for value in box.detach().cpu().tolist()), image_size)
        if _bbox_area(bbox) <= 0:
            continue
        candidates.append((bbox, float(score.detach().cpu().item())))
    if not candidates:
        raise pb_utils.TritonModelException("GroundingDINO returned no valid candidate boxes")

    top_score = max(score for _, score in candidates)
    near_top = [(bbox, score) for bbox, score in candidates if top_score - score <= 0.03]
    bbox, score = max(near_top, key=lambda item: _bbox_area(item[0]))
    return bbox, score


def _post_process_kwargs(processor, *, box_threshold: float, text_threshold: float) -> dict[str, float]:
    signature = inspect.signature(processor.post_process_grounded_object_detection)
    if "box_threshold" in signature.parameters:
        return {"box_threshold": box_threshold, "text_threshold": text_threshold}
    return {"threshold": box_threshold, "text_threshold": text_threshold}


def _clamp_bbox(box: tuple[float, float, float, float], image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    width, height = image_size
    left, top, right, bottom = box
    left = max(0.0, min(float(width), left))
    right = max(0.0, min(float(width), right))
    top = max(0.0, min(float(height), top))
    bottom = max(0.0, min(float(height), bottom))
    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top
    return left, top, right, bottom


def _bbox_area(box: tuple[float, float, float, float]) -> float:
    left, top, right, bottom = box
    return max(0.0, right - left) * max(0.0, bottom - top)
