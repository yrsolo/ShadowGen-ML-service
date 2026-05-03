from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shadowgen_ml_service.bootstrap.triton_bindings import build_triton_model_registry  # noqa: E402
from shadowgen_ml_service.config import Settings  # noqa: E402
from shadowgen_ml_service.core.stage_io import DetectionInput, SegmentationInput  # noqa: E402
from shadowgen_ml_service.infrastructure.backends.triton.client import TritonInferenceClient  # noqa: E402
from shadowgen_ml_service.infrastructure.backends.triton.config import TritonBackendSettings  # noqa: E402
from shadowgen_ml_service.infrastructure.stages.detection.grounding_dino import RealDetector  # noqa: E402
from shadowgen_ml_service.infrastructure.stages.detection.triton import TritonDetector  # noqa: E402
from shadowgen_ml_service.infrastructure.stages.segmentation.birefnet import RealSegmenter  # noqa: E402
from shadowgen_ml_service.infrastructure.stages.segmentation.triton import TritonSegmenter  # noqa: E402
from shadowgen_ml_service.utils.images import pil_to_asset  # noqa: E402


def _measure(label: str, repeats: int, warmup: int, fn) -> dict:
    samples = []
    for index in range(warmup + repeats):
        started = time.perf_counter()
        result = fn()
        elapsed_ms = (time.perf_counter() - started) * 1000
        if index >= warmup:
            samples.append(elapsed_ms)
    return {
        "label": label,
        "samples_ms": [round(value, 2) for value in samples],
        "mean_ms": round(statistics.mean(samples), 2),
        "median_ms": round(statistics.median(samples), 2),
        "min_ms": round(min(samples), 2),
        "max_ms": round(max(samples), 2),
        "last_result": result,
    }


def _triton_client(settings: Settings, *, transport: str) -> TritonInferenceClient:
    return TritonInferenceClient(
        TritonBackendSettings(
            url=settings.triton_url,
            protocol=settings.triton_protocol,
            timeout_ms=settings.triton_timeout_ms,
            transport=transport,
        )
    )


def _benchmark_detector(settings: Settings, image: Image.Image, repeats: int, warmup: int, transports: list[str]) -> list[dict]:
    stage_input = DetectionInput(image=pil_to_asset(image), padding_px=100)
    results = []
    if "local" in transports:
        detector = RealDetector(
            model_id=settings.grounding_dino_model_id,
            prompt=settings.grounding_dino_prompt,
            box_threshold=settings.grounding_dino_box_threshold,
            text_threshold=settings.grounding_dino_text_threshold,
            target_device=settings.target_device,
        )
        results.append(
            _measure(
                "detector/local/grounding-dino",
                repeats,
                warmup,
                lambda: _detection_summary(detector.detect(stage_input)),
            )
        )
    registry = build_triton_model_registry(settings)
    binding = registry.get("detector", "grounding-dino")
    if binding is not None:
        for transport in transports:
            if not transport.startswith("triton-"):
                continue
            triton_transport = transport.removeprefix("triton-")
            client = _triton_client(settings, transport=triton_transport)
            detector = TritonDetector(client, binding)
            results.append(
                _measure(
                    f"detector/triton/{triton_transport}",
                    repeats,
                    warmup,
                    lambda detector=detector: _detection_summary(detector.detect(stage_input)),
                )
            )
    return results


def _benchmark_segmenter(settings: Settings, image: Image.Image, repeats: int, warmup: int, transports: list[str]) -> list[dict]:
    stage_input = SegmentationInput(image=pil_to_asset(image))
    results = []
    if "local" in transports:
        segmenter = RealSegmenter(
            model_id=settings.birefnet_model_id,
            resolution=settings.birefnet_resolution,
            mask_threshold=settings.birefnet_mask_threshold,
            target_device=settings.target_device,
            compile_enabled=settings.birefnet_compile_enabled,
            compile_mode=settings.birefnet_compile_mode,
            compile_backend=settings.birefnet_compile_backend,
            matmul_precision=settings.birefnet_matmul_precision,
        )
        results.append(
            _measure(
                "segmenter/local/birefnet",
                repeats,
                warmup,
                lambda: _segmentation_summary(segmenter.segment(stage_input)),
            )
        )
    registry = build_triton_model_registry(settings)
    binding = registry.get("segmenter", "birefnet")
    if binding is not None:
        for transport in transports:
            if not transport.startswith("triton-"):
                continue
            triton_transport = transport.removeprefix("triton-")
            client = _triton_client(settings, transport=triton_transport)
            segmenter = TritonSegmenter(client, binding, batcher=None)
            results.append(
                _measure(
                    f"segmenter/triton/{triton_transport}",
                    repeats,
                    warmup,
                    lambda segmenter=segmenter: _segmentation_summary(segmenter.segment(stage_input)),
                )
            )
    return results


def _detection_summary(result) -> dict:
    return {"bbox": result.bbox, "confidence": round(float(result.confidence), 4)}


def _segmentation_summary(result) -> dict:
    return {"bbox": result.bbox, "mask_size": [result.mask.width, result.mask.height]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark local and Triton stage backends on the same image.")
    parser.add_argument("--stage", choices=["detector", "segmenter", "all"], default="all")
    parser.add_argument("--image", type=Path, default=Path(r"C:\Users\solofarm\Pictures\Screenshots\1.jpg"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--max-size", type=int, default=768)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--transport",
        action="append",
        choices=["local", "triton-json", "triton-native"],
        default=None,
        help="May be passed multiple times. Default: local, triton-json, triton-native.",
    )
    args = parser.parse_args(argv)

    if not args.image.exists():
        raise FileNotFoundError(args.image)
    transports = args.transport or ["local", "triton-json", "triton-native"]
    settings = Settings(triton_url=args.base_url, triton_timeout_ms=300_000)
    image = ImageOps.contain(Image.open(args.image).convert("RGB"), (args.max_size, args.max_size))
    results = {
        "image": str(args.image),
        "input_size": list(image.size),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "results": [],
    }
    if args.stage in {"detector", "all"}:
        results["results"].extend(_benchmark_detector(settings, image, args.repeats, args.warmup, transports))
    if args.stage in {"segmenter", "all"}:
        results["results"].extend(_benchmark_segmenter(settings, image, args.repeats, args.warmup, transports))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
