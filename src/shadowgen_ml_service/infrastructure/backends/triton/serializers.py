from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from shadowgen_ml_service.infrastructure.backends.triton.errors import TritonSchemaMismatchError
from shadowgen_ml_service.infrastructure.backends.triton.model_registry import TritonTensorBinding


TRITON_DTYPE_TO_NUMPY = {
    "BOOL": np.bool_,
    "FP16": np.float16,
    "FP32": np.float32,
    "FP64": np.float64,
    "INT8": np.int8,
    "INT16": np.int16,
    "INT32": np.int32,
    "INT64": np.int64,
    "UINT16": np.uint16,
    "UINT32": np.uint32,
    "UINT8": np.uint8,
}


def _tensor_input(name: str, array: np.ndarray, *, datatype: str) -> dict[str, Any]:
    tensor = np.asarray(array, dtype=TRITON_DTYPE_TO_NUMPY[datatype])
    return {
        "name": name,
        "shape": [int(value) for value in tensor.shape],
        "datatype": datatype,
        "data": tensor.reshape(-1).tolist(),
    }


def tensor_input(name: str, array: np.ndarray, *, datatype: str) -> dict[str, Any]:
    return _tensor_input(name, array, datatype=datatype)


def _image_to_nchw(image: Image.Image, *, mode: str) -> np.ndarray:
    array = np.asarray(image.convert(mode), dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[:, :, None]
    return np.transpose(array, (2, 0, 1))[None, ...]


def image_to_nchw_float32_input(name: str, image: Image.Image) -> dict[str, Any]:
    return _tensor_input(name, _image_to_nchw(image, mode=image.mode if image.mode in {"RGB", "RGBA"} else "RGB"), datatype="FP32")


def rgb_to_nchw_float32_input(name: str, image: Image.Image) -> dict[str, Any]:
    return _tensor_input(name, _image_to_nchw(image, mode="RGB"), datatype="FP32")


def grayscale_to_nchw_float32_input(name: str, image: Image.Image) -> dict[str, Any]:
    return _tensor_input(name, _image_to_nchw(image, mode="L"), datatype="FP32")


def mask_to_nchw_float32_input(name: str, image: Image.Image) -> dict[str, Any]:
    return grayscale_to_nchw_float32_input(name, image)


def scalar_to_input(name: str, value: int | float, *, datatype: str = "FP32") -> dict[str, Any]:
    return _tensor_input(name, np.asarray([value]), datatype=datatype)


def output_request(name: str) -> dict[str, str]:
    return {"name": name}


def tensor_from_output(output: dict[str, Any]) -> np.ndarray:
    datatype = output.get("datatype")
    if datatype not in TRITON_DTYPE_TO_NUMPY:
        raise ValueError(f"unsupported Triton datatype: {datatype}")
    shape = tuple(int(value) for value in output.get("shape", []))
    data = output.get("data")
    if data is None:
        raise ValueError(f"output tensor {output.get('name')} does not contain inline data")
    return np.asarray(data, dtype=TRITON_DTYPE_TO_NUMPY[datatype]).reshape(shape)


def tensor_from_input(input_payload: dict[str, Any]) -> np.ndarray:
    datatype = input_payload.get("datatype")
    if datatype not in TRITON_DTYPE_TO_NUMPY:
        raise ValueError(f"unsupported Triton datatype: {datatype}")
    shape = tuple(int(value) for value in input_payload.get("shape", []))
    data = input_payload.get("data")
    if data is None:
        raise ValueError(f"input tensor {input_payload.get('name')} does not contain inline data")
    return np.asarray(data, dtype=TRITON_DTYPE_TO_NUMPY[datatype]).reshape(shape)


def batch_input_tensors(request_batches: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if not request_batches:
        return []
    tensor_names = [tensor["name"] for tensor in request_batches[0]]
    result: list[dict[str, Any]] = []
    for tensor_name in tensor_names:
        matching = []
        datatype = None
        for batch in request_batches:
            found = next((item for item in batch if item["name"] == tensor_name), None)
            if found is None:
                raise TritonSchemaMismatchError(f"missing input tensor {tensor_name} in batch item")
            if datatype is None:
                datatype = found["datatype"]
            elif datatype != found["datatype"]:
                raise TritonSchemaMismatchError(f"input tensor {tensor_name} has inconsistent datatypes across batch")
            matching.append(tensor_from_input(found))
        prototype = matching[0]
        if prototype.ndim == 0:
            merged = np.stack(matching, axis=0)
        elif prototype.shape[0] == 1:
            for item in matching[1:]:
                if item.shape[1:] != prototype.shape[1:]:
                    raise TritonSchemaMismatchError(f"input tensor {tensor_name} has incompatible shapes for batching")
            merged = np.concatenate(matching, axis=0)
        else:
            for item in matching[1:]:
                if item.shape != prototype.shape:
                    raise TritonSchemaMismatchError(f"input tensor {tensor_name} has incompatible shapes for batching")
            merged = np.stack(matching, axis=0)
        result.append(_tensor_input(tensor_name, merged, datatype=datatype))
    return result


def split_output_tensor(tensor: np.ndarray, batch_size: int) -> list[np.ndarray]:
    if batch_size <= 1:
        return [np.asarray(tensor)]
    tensor = np.asarray(tensor)
    if tensor.ndim == 0 or tensor.shape[0] != batch_size:
        raise TritonSchemaMismatchError(
            f"output tensor cannot be split into batch size {batch_size}: observed shape {tuple(tensor.shape)}"
        )
    if tensor.ndim == 1:
        return [tensor[index : index + 1] for index in range(batch_size)]
    return [tensor[index : index + 1] for index in range(batch_size)]


def tensor_map_from_response(response_payload: dict[str, Any]) -> dict[str, np.ndarray]:
    outputs = response_payload.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("Triton response does not contain an outputs list")
    result: dict[str, np.ndarray] = {}
    for output in outputs:
        name = output.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Triton response output is missing a valid name")
        result[name] = tensor_from_output(output)
    return result


def validate_tensor_against_binding(name: str, tensor: np.ndarray, binding: TritonTensorBinding) -> None:
    if tensor.dtype != np.dtype(TRITON_DTYPE_TO_NUMPY[binding.datatype]):
        raise TritonSchemaMismatchError(
            f"tensor {name} has dtype {tensor.dtype}, expected {binding.datatype}"
        )
    if binding.expected_ranks and tensor.ndim not in binding.expected_ranks:
        raise TritonSchemaMismatchError(
            f"tensor {name} has rank {tensor.ndim}, expected one of {binding.expected_ranks}"
        )
    if binding.shape_policy == "scalar" and tensor.size != 1:
        raise TritonSchemaMismatchError(f"tensor {name} must contain exactly one scalar value")
    if binding.shape_policy == "bbox4" and tensor.size < 4:
        raise TritonSchemaMismatchError(f"tensor {name} must contain at least 4 bbox values")
    if binding.shape_policy == "channel-first":
        if tensor.ndim not in {3, 4}:
            raise TritonSchemaMismatchError(f"tensor {name} must be rank 3 or 4 channel-first image data")
        channel_axis = 1 if tensor.ndim == 4 else 0
        if binding.channels is not None and int(tensor.shape[channel_axis]) != binding.channels:
            raise TritonSchemaMismatchError(
                f"tensor {name} has {int(tensor.shape[channel_axis])} channels, expected {binding.channels}"
            )


def tensor_to_scalar(tensor: np.ndarray) -> float:
    return float(np.asarray(tensor).reshape(-1)[0])


def tensor_to_bbox(tensor: np.ndarray) -> tuple[int, int, int, int]:
    values = np.asarray(tensor).reshape(-1)
    return tuple(int(round(float(value))) for value in values[:4])


def _to_image(array: np.ndarray, *, mode: str) -> Image.Image:
    data = np.asarray(array, dtype=np.float32)
    if data.ndim == 4:
        data = data[0]
    if data.ndim == 3 and data.shape[0] in {1, 3, 4}:
        data = np.transpose(data, (1, 2, 0))
    data = np.clip(data, 0.0, 1.0)
    if mode == "L":
        if data.ndim == 3:
            data = data[:, :, 0]
        return Image.fromarray((data * 255.0).astype(np.uint8), mode="L")
    return Image.fromarray((data * 255.0).astype(np.uint8), mode=mode)


def mask_output_to_image(tensor: np.ndarray) -> Image.Image:
    return _to_image(tensor, mode="L")


def grayscale_output_to_image(tensor: np.ndarray) -> Image.Image:
    return _to_image(tensor, mode="L")


def rgb_output_to_image(tensor: np.ndarray) -> Image.Image:
    return _to_image(tensor, mode="RGB")


def rgba_output_to_image(tensor: np.ndarray) -> Image.Image:
    return _to_image(tensor, mode="RGBA")
