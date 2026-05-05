from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
for stream_name in ("stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    if stream is not None and hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from shadowgen_ml_service.config import Settings  # noqa: E402
from shadowgen_ml_service.infrastructure.stages.detection.grounding_dino import load_grounding_dino_classes  # noqa: E402


ONNX_TO_TRITON_DTYPE = {
    1: "TYPE_FP32",
    7: "TYPE_INT64",
}


def parse_args() -> argparse.Namespace:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Experimental model-only GroundingDINO ONNX export.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "ops" / "triton" / "model_repository" / "shadowgen_detector_onnx" / "1" / "model.onnx",
    )
    parser.add_argument("--model-id", default=settings.grounding_dino_model_id)
    parser.add_argument("--prompt", default=settings.grounding_dino_prompt)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--exporter", choices=("auto", "dynamo", "legacy"), default="auto")
    return parser.parse_args()


def export_detector_onnx(args: argparse.Namespace) -> Path:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError("PyTorch is required for ONNX export") from exc
    try:
        import onnx
    except Exception as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError("onnx is required for export validation") from exc

    processor_cls, model_cls = load_grounding_dino_classes()
    processor = processor_cls.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    model = model_cls.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    model.eval()

    encoded = processor(images=[_dummy_pil(args.width, args.height)], text=args.prompt, return_tensors="pt")
    tensor_inputs = {
        key: value
        for key, value in encoded.items()
        if hasattr(value, "shape")
    }
    ordered_keys = ["pixel_values", *[key for key in tensor_inputs if key != "pixel_values"]]

    class GroundingDinoExportWrapper(torch.nn.Module):
        def __init__(self, wrapped_model, input_keys: list[str]) -> None:
            super().__init__()
            self.wrapped_model = wrapped_model
            self.input_keys = input_keys

        def forward(self, *values):
            kwargs = {key: value for key, value in zip(self.input_keys, values, strict=True)}
            outputs = self.wrapped_model(**kwargs)
            return outputs.logits, outputs.pred_boxes

    wrapper = GroundingDinoExportWrapper(model, ordered_keys)
    wrapper.eval()
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failure = _export_with_fallback(
        torch_module=torch,
        wrapper=wrapper,
        inputs=tuple(tensor_inputs[key] for key in ordered_keys),
        input_names=ordered_keys,
        output_path=output_path,
        opset=args.opset,
        exporter=args.exporter,
    )
    if failure is not None:
        raise RuntimeError(failure)
    exported = onnx.load(output_path)
    onnx.checker.check_model(exported)
    _write_triton_config(exported, output_path.parents[1])
    return output_path


def _dummy_pil(width: int, height: int):
    from PIL import Image

    return Image.new("RGB", (width, height), "white")


DETECTOR_OUTPUT_DIMS = {
    "pred_boxes": [900, 4],
}


def _shape_without_batch(value) -> list[int]:
    if value.name in DETECTOR_OUTPUT_DIMS:
        return DETECTOR_OUTPUT_DIMS[value.name]
    dims = []
    for dim in value.type.tensor_type.shape.dim[1:]:
        if dim.dim_value:
            dims.append(int(dim.dim_value))
        else:
            dims.append(-1)
    return dims


def _triton_dtype(value) -> str:
    dtype = ONNX_TO_TRITON_DTYPE.get(value.type.tensor_type.elem_type)
    if dtype is None:
        raise RuntimeError(f"Unsupported ONNX dtype {value.type.tensor_type.elem_type} for tensor {value.name}")
    return dtype


def _write_triton_config(model, model_dir: Path) -> None:
    inputs = list(model.graph.input)
    outputs = list(model.graph.output)

    def _tensor_block(kind: str, values) -> str:
        blocks = []
        for value in values:
            dims = ", ".join(str(item) for item in _shape_without_batch(value))
            blocks.append(
                "\n".join(
                    [
                        "  {",
                        f'    name: "{value.name}"',
                        f"    data_type: {_triton_dtype(value)}",
                        f"    dims: [ {dims} ]",
                        "  }",
                    ]
                )
            )
        return f"{kind} [\n" + ",\n".join(blocks) + "\n]"

    config = "\n".join(
        [
            'name: "shadowgen_detector_onnx"',
            'platform: "onnxruntime_onnx"',
            "max_batch_size: 4",
            _tensor_block("input", inputs),
            _tensor_block("output", outputs),
            "dynamic_batching {",
            "  preferred_batch_size: [ 2, 4 ]",
            "  max_queue_delay_microseconds: 15000",
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
    (model_dir / "config.pbtxt").write_text(config, encoding="utf-8")


def _export_with_fallback(*, torch_module, wrapper, inputs, input_names: list[str], output_path: Path, opset: int, exporter: str) -> str | None:
    exporters = [exporter] if exporter != "auto" else ["dynamo", "legacy"]
    failures = []
    dynamic_axes = {name: {0: "batch"} for name in input_names}
    dynamic_axes["pixel_values"] = {0: "batch", 2: "height", 3: "width"}
    dynamic_axes["pixel_mask"] = {0: "batch", 1: "mask_height", 2: "mask_width"}
    dynamic_axes["logits"] = {0: "batch"}
    dynamic_axes["pred_boxes"] = {0: "batch"}
    for candidate in exporters:
        try:
            torch_module.onnx.export(
                wrapper,
                inputs,
                output_path,
                input_names=input_names,
                output_names=["logits", "pred_boxes"],
                dynamic_axes=dynamic_axes,
                opset_version=opset,
                dynamo=candidate == "dynamo",
            )
            return None
        except Exception as exc:  # pragma: no cover - local export workflow
            failures.append(_normalize_export_failure(candidate, exc))
            if output_path.exists():
                output_path.unlink()
    return (
        "GroundingDINO ONNX export did not complete.\n\n"
        + "\n\n".join(failures)
        + "\n\nThis tool exports only model logits/boxes. Postprocess still needs a separate Triton adapter or ensemble step."
    )


def _normalize_export_failure(exporter: str, exc: Exception) -> str:
    details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    if "MultiScaleDeformableAttention" in details or "custom" in details.lower():
        return f"[{exporter}] blocked by custom attention/export operator: {details}"
    return f"[{exporter}] {details}"


def main() -> int:
    output = export_detector_onnx(parse_args())
    print(f"Exported detector ONNX to {output}")
    print(f"Generated Triton config at {output.parents[1] / 'config.pbtxt'}")
    print("Next step: start-triton.cmd")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
