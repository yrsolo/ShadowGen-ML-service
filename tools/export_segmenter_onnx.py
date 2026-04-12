from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shadowgen_ml_service.config import Settings  # noqa: E402
from shadowgen_ml_service.infrastructure.stages.segmentation.birefnet import load_transformers_auto_classes  # noqa: E402
from shadowgen_ml_service.infrastructure.stages.segmentation.onnx_export import (  # noqa: E402
    default_segmenter_onnx_contract,
    validate_segmenter_export_names,
)


def parse_args() -> argparse.Namespace:
    settings = Settings()
    contract = default_segmenter_onnx_contract()
    parser = argparse.ArgumentParser(description="Export BiRefNet segmenter to ONNX for Triton.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "ops" / "triton" / "model_repository" / contract.model_name / "1" / "model.onnx",
        help="Output ONNX path inside the Triton model repository.",
    )
    parser.add_argument("--model-id", default=settings.birefnet_model_id)
    parser.add_argument("--resolution", type=int, default=settings.birefnet_resolution)
    parser.add_argument("--opset", type=int, default=contract.opset)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=settings.working_size)
    parser.add_argument("--width", type=int, default=settings.working_size)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def export_segmenter_onnx(args: argparse.Namespace) -> Path:
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError("PyTorch is required for ONNX export") from exc

    try:
        import onnx
    except Exception as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError("The 'onnx' package is required for ONNX export validation") from exc

    _, model_cls = load_transformers_auto_classes()
    model = model_cls.from_pretrained(args.model_id, trust_remote_code=True, local_files_only=args.local_files_only)
    model.eval()

    class BiRefNetMaskWrapper(torch.nn.Module):
        def __init__(self, wrapped_model, resolution: int) -> None:
            super().__init__()
            self.model = wrapped_model
            self.resolution = resolution
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))

        def forward(self, image):
            original_size = image.shape[-2:]
            resized = F.interpolate(image, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
            normalized = (resized - self.mean) / self.std
            matte = self.model(normalized)[-1].sigmoid()
            return F.interpolate(matte, size=original_size, mode="bilinear", align_corners=False).clamp(0.0, 1.0)

    wrapper = BiRefNetMaskWrapper(model, args.resolution)
    dummy = torch.rand(args.batch_size, 3, args.height, args.width, dtype=torch.float32)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contract = default_segmenter_onnx_contract()
    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        input_names=[contract.input_name],
        output_names=[contract.output_name],
        dynamic_axes={
            contract.input_name: {0: "batch", 2: "height", 3: "width"},
            contract.output_name: {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=args.opset,
    )

    exported = onnx.load(output_path)
    validate_segmenter_export_names(
        input_names=[node.name for node in exported.graph.input],
        output_names=[node.name for node in exported.graph.output],
    )
    onnx.checker.check_model(exported)
    return output_path


def main() -> int:
    args = parse_args()
    output_path = export_segmenter_onnx(args)
    print(f"Exported segmenter ONNX to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
