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

    dummy_image = torch.rand(1, 3, args.height, args.width, dtype=torch.float32)
    encoded = processor(images=[_dummy_pil(args.width, args.height)], text=args.prompt, return_tensors="pt")
    tensor_inputs = {
        key: value
        for key, value in encoded.items()
        if hasattr(value, "shape") and key != "pixel_values"
    }
    tensor_inputs["pixel_values"] = dummy_image
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
    return output_path


def _dummy_pil(width: int, height: int):
    from PIL import Image

    return Image.new("RGB", (width, height), "white")


def _export_with_fallback(*, torch_module, wrapper, inputs, input_names: list[str], output_path: Path, opset: int, exporter: str) -> str | None:
    exporters = [exporter] if exporter != "auto" else ["dynamo", "legacy"]
    failures = []
    dynamic_axes = {name: {0: "batch"} for name in input_names}
    dynamic_axes["pixel_values"] = {0: "batch", 2: "height", 3: "width"}
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
