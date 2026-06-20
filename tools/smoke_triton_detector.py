from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shadowgen_ml_service.app import create_app  # noqa: E402
from shadowgen_ml_service.bootstrap.triton_bindings import build_triton_model_registry  # noqa: E402
from shadowgen_ml_service.config import Settings  # noqa: E402
from shadowgen_ml_service.core.stage_io import DetectionInput  # noqa: E402
from shadowgen_ml_service.infrastructure.backends.triton.client import TritonInferenceClient  # noqa: E402
from shadowgen_ml_service.infrastructure.backends.triton.config import TritonBackendSettings  # noqa: E402
from shadowgen_ml_service.infrastructure.stages.detection.triton import TritonDetector  # noqa: E402
from shadowgen_ml_service.infrastructure.stages.detection.triton_onnx import TritonOnnxGroundingDinoDetector  # noqa: E402
from shadowgen_ml_service.utils.images import pil_to_asset  # noqa: E402


def _image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _draw_bbox(image: Image.Image, bbox: tuple[int, int, int, int], confidence: float) -> Image.Image:
    overlay = image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(bbox, outline=(255, 95, 64), width=4)
    draw.rounded_rectangle((bbox[0], max(0, bbox[1] - 28), bbox[0] + 150, bbox[1]), radius=6, fill=(255, 255, 255))
    draw.text((bbox[0] + 8, max(0, bbox[1] - 22)), f"triton {confidence:.3f}", fill=(24, 33, 52))
    return overlay


def _build_detector(settings: Settings, client: TritonInferenceClient, variant: str):
    binding = build_triton_model_registry(settings).get("detector", variant)
    if binding is None:
        raise RuntimeError(f"detector/{variant} Triton binding is not configured")
    if variant == "grounding-dino-onnx":
        return binding, TritonOnnxGroundingDinoDetector(
            client,
            binding,
            model_id=settings.grounding_dino_model_id,
            prompt=settings.grounding_dino_prompt,
            box_threshold=settings.grounding_dino_box_threshold,
            text_threshold=settings.grounding_dino_text_threshold,
        )
    return binding, TritonDetector(client, binding)


def run_direct_smoke(*, base_url: str, image_path: Path, output_dir: Path, max_size: int, timeout_ms: int, variant: str) -> dict:
    settings = Settings(triton_url=base_url, triton_timeout_ms=timeout_ms)
    client = TritonInferenceClient(
        TritonBackendSettings(
            url=settings.triton_url,
            protocol=settings.triton_protocol,
            timeout_ms=settings.triton_timeout_ms,
        )
    )
    binding, detector = _build_detector(settings, client, variant)

    available, detail = client.probe_binding(binding)
    if not available:
        raise RuntimeError(detail)

    image = ImageOps.contain(Image.open(image_path).convert("RGB"), (max_size, max_size))
    result = detector.detect(DetectionInput(image=pil_to_asset(image), padding_px=100))

    output_dir.mkdir(parents=True, exist_ok=True)
    image.save(output_dir / "direct_input.png")
    _draw_bbox(image, result.bbox, result.confidence).save(output_dir / "direct_overlay.png")

    return {
        "probe": detail,
        "variant": variant,
        "input_size": image.size,
        "bbox": result.bbox,
        "confidence": result.confidence,
        "artifacts": [
            str(output_dir / "direct_input.png"),
            str(output_dir / "direct_overlay.png"),
        ],
    }


def run_render_smoke(*, base_url: str, image_path: Path, output_dir: Path, timeout_ms: int, variant: str) -> dict:
    cache_dir = ROOT / "var" / "cache" / "triton-detector-smoke" / uuid4().hex
    settings = Settings(
        triton_url=base_url,
        triton_timeout_ms=timeout_ms,
        request_timeout_ms=timeout_ms,
        preprocess_cache_dir=cache_dir,
        detector_backend_kind="triton",
        detector_model_variant=variant,
        segmenter_backend_kind="mock",
        depth_backend_kind="mock",
        normals_backend_kind="mock",
        shadow_backend_kind="mock",
        shadow_v2_diff_bundle_path=Path("missing-test-v2-diff-bundle"),
    )
    try:
        client = TestClient(create_app(settings))
        capabilities = client.get("/v1/capabilities")
        capabilities.raise_for_status()
        detector = next(item for item in capabilities.json()["components"] if item["name"] == "detector")
        if detector["backend_kind"] != "triton":
            raise RuntimeError(f"detector active backend is not triton: {detector}")

        response = client.post(
            "/v1/render",
            json={
                "request_id": f"triton-detector-render-smoke-{uuid4().hex}",
                "pipeline_version": "ml-shadowgen-v1",
                "source": {"mime_type": "image/jpeg", "image_base64": _image_to_base64(image_path)},
                "shadow": {
                    "angle_deg": 45,
                    "elevation_deg": 35,
                    "softness": 0.5,
                    "opacity": 0.65,
                    "reflection": 0.1,
                    "model": "v1-gan",
                },
                "background": {"mode": "solid", "color_hex": "#FFFFFF"},
                "output": {"format": "png", "width": None, "height": None, "return_debug": True},
            },
        )
        response.raise_for_status()
        payload = response.json()
        metrics = payload.get("metrics") or {}
        if int(metrics.get("detection_ms") or 0) <= 0:
            raise RuntimeError(f"render did not execute detection stage: {metrics}")
        return {
            "detector_capability": {
                key: detector.get(key)
                for key in ("backend_kind", "model_variant", "device", "endpoint", "fallback_reason")
            },
            "metrics": metrics,
            "warnings": payload.get("warnings", []),
        }
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run live smoke checks against the Triton detector.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--image", type=Path, default=Path(r"C:\Users\solofarm\Pictures\Screenshots\1.jpg"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "triton-detector-smoke")
    parser.add_argument("--direct-max-size", type=int, default=768)
    parser.add_argument("--timeout-ms", type=int, default=180_000)
    parser.add_argument("--variant", choices=["grounding-dino", "grounding-dino-onnx"], default="grounding-dino")
    parser.add_argument("--direct-only", action="store_true")
    args = parser.parse_args(argv)

    if not args.image.exists():
        raise FileNotFoundError(args.image)

    direct = run_direct_smoke(
        base_url=args.base_url,
        image_path=args.image,
        output_dir=args.output_dir / "direct",
        max_size=args.direct_max_size,
        timeout_ms=args.timeout_ms,
        variant=args.variant,
    )
    result = {"direct": direct}
    if not args.direct_only:
        result["render"] = run_render_smoke(
            base_url=args.base_url,
            image_path=args.image,
            output_dir=args.output_dir / "render",
            timeout_ms=args.timeout_ms,
            variant=args.variant,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
