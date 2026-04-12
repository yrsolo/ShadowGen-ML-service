from __future__ import annotations

from dataclasses import dataclass


SEGMENTER_ONNX_INPUT_NAME = "image"
SEGMENTER_ONNX_OUTPUT_NAME = "mask"
SEGMENTER_TRITON_MODEL_NAME = "shadowgen_segmenter"
SEGMENTER_ONNX_OPSET = 17
SEGMENTER_MAX_BATCH_SIZE = 4
SEGMENTER_DYNAMIC_BATCH_DELAY_US = 15_000


@dataclass(frozen=True)
class SegmenterOnnxContract:
    model_name: str = SEGMENTER_TRITON_MODEL_NAME
    input_name: str = SEGMENTER_ONNX_INPUT_NAME
    output_name: str = SEGMENTER_ONNX_OUTPUT_NAME
    opset: int = SEGMENTER_ONNX_OPSET
    max_batch_size: int = SEGMENTER_MAX_BATCH_SIZE
    dynamic_batch_delay_us: int = SEGMENTER_DYNAMIC_BATCH_DELAY_US


def default_segmenter_onnx_contract() -> SegmenterOnnxContract:
    return SegmenterOnnxContract()


def validate_segmenter_export_names(*, input_names: list[str], output_names: list[str]) -> None:
    contract = default_segmenter_onnx_contract()
    if input_names != [contract.input_name]:
        raise ValueError(f"segmenter ONNX export must expose exactly one input named {contract.input_name!r}")
    if output_names != [contract.output_name]:
        raise ValueError(f"segmenter ONNX export must expose exactly one output named {contract.output_name!r}")


def render_segmenter_config_pbtxt(*, contract: SegmenterOnnxContract | None = None) -> str:
    active = contract or default_segmenter_onnx_contract()
    return "\n".join(
        [
            f'name: "{active.model_name}"',
            'platform: "onnxruntime_onnx"',
            f"max_batch_size: {active.max_batch_size}",
            "input [",
            "  {",
            f'    name: "{active.input_name}"',
            '    data_type: TYPE_FP32',
            "    dims: [ 3, -1, -1 ]",
            "  }",
            "]",
            "output [",
            "  {",
            f'    name: "{active.output_name}"',
            '    data_type: TYPE_FP32',
            "    dims: [ 1, -1, -1 ]",
            "  }",
            "]",
            "dynamic_batching {",
            "  preferred_batch_size: [ 2, 4 ]",
            f"  max_queue_delay_microseconds: {active.dynamic_batch_delay_us}",
            "}",
            "instance_group [",
            "  {",
            "    kind: KIND_GPU",
            "    count: 1",
            "  }",
            "]",
            "",
        ]
    )
