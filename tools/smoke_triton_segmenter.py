from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shadowgen_ml_service.app import create_app  # noqa: E402
from shadowgen_ml_service.bootstrap.triton_bindings import build_triton_model_registry  # noqa: E402
from shadowgen_ml_service.config import Settings  # noqa: E402
from shadowgen_ml_service.core.stage_io import SegmentationInput  # noqa: E402
from shadowgen_ml_service.infrastructure.backends.triton.client import TritonInferenceClient  # noqa: E402
from shadowgen_ml_service.infrastructure.backends.triton.config import TritonBackendSettings  # noqa: E402
from shadowgen_ml_service.infrastructure.stages.segmentation.triton import TritonSegmenter  # noqa: E402
from shadowgen_ml_service.utils.images import asset_to_pil, pil_to_asset  # noqa: E402


def _image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _save_artifacts(payload: dict, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for item in payload.get("artifacts", []):
        name = item.get("name")
        image_base64 = item.get("image_base64")
        if not name or not image_base64:
            continue
        target = output_dir / f"{name}.png"
        target.write_bytes(base64.b64decode(image_base64))
        saved.append(str(target))
    return saved


def run_direct_smoke(*, base_url: str, image_path: Path, output_dir: Path, max_size: int, timeout_ms: int, variant: str) -> dict:
    settings = Settings(triton_url=base_url, triton_timeout_ms=timeout_ms)
    client = TritonInferenceClient(
        TritonBackendSettings(
            url=settings.triton_url,
            protocol=settings.triton_protocol,
            timeout_ms=settings.triton_timeout_ms,
        )
    )
    binding = build_triton_model_registry(settings).get("segmenter", variant)
    if binding is None:
        raise RuntimeError(f"segmenter/{variant} Triton binding is not configured")

    available, detail = client.probe_binding(binding)
    if not available:
        raise RuntimeError(detail)

    image = ImageOps.contain(Image.open(image_path).convert("RGB"), (max_size, max_size))
    segmenter = TritonSegmenter(client, binding, batcher=None)
    result = segmenter.segment(SegmentationInput(image=pil_to_asset(image)))
    mask = asset_to_pil(result.mask)

    output_dir.mkdir(parents=True, exist_ok=True)
    image.save(output_dir / "direct_input.png")
    mask.save(output_dir / "direct_mask.png")
    asset_to_pil(result.cutout_rgba).save(output_dir / "direct_cutout.png")

    return {
        "probe": detail,
        "variant": variant,
        "input_size": image.size,
        "bbox": result.bbox,
        "mask_extrema": mask.getextrema(),
        "artifacts": [
            str(output_dir / "direct_input.png"),
            str(output_dir / "direct_mask.png"),
            str(output_dir / "direct_cutout.png"),
        ],
    }


def run_render_smoke(*, base_url: str, image_path: Path, output_dir: Path, timeout_ms: int, variant: str) -> dict:
    cache_dir = ROOT / "var" / "cache" / "triton-full-smoke" / uuid4().hex
    settings = Settings(
        triton_url=base_url,
        triton_timeout_ms=timeout_ms,
        request_timeout_ms=timeout_ms,
        preprocess_cache_dir=cache_dir,
        detector_backend_kind="mock",
        segmenter_backend_kind="triton",
        segmenter_model_variant=variant,
        depth_backend_kind="mock",
        normals_backend_kind="mock",
        shadow_backend_kind="mock",
        shadow_v2_diff_bundle_path=Path("missing-test-v2-diff-bundle"),
    )
    try:
        client = TestClient(create_app(settings))
        capabilities = client.get("/v1/capabilities")
        capabilities.raise_for_status()
        segmenter = next(item for item in capabilities.json()["components"] if item["name"] == "segmenter")
        if segmenter["backend_kind"] != "triton":
            raise RuntimeError(f"segmenter active backend is not triton: {segmenter}")

        response = client.post(
            "/v1/render",
            json={
                "request_id": f"triton-render-smoke-{uuid4().hex}",
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
        if int(metrics.get("segmentation_ms") or 0) <= 0:
            raise RuntimeError(f"render did not execute segmentation stage: {metrics}")
        return {
            "segmenter_capability": {
                key: segmenter.get(key)
                for key in ("backend_kind", "model_variant", "device", "endpoint", "fallback_reason")
            },
            "metrics": metrics,
            "warnings": payload.get("warnings", []),
            "artifacts": _save_artifacts(payload, output_dir),
        }
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run live smoke checks against the Triton segmenter.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--image", type=Path, default=Path(r"C:\Users\solofarm\Pictures\Screenshots\1.jpg"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "triton-smoke")
    parser.add_argument("--direct-max-size", type=int, default=512)
    parser.add_argument("--timeout-ms", type=int, default=180_000)
    parser.add_argument("--variant", choices=["birefnet", "rmbg-2.0"], default="birefnet")
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
