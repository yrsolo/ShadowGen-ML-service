from __future__ import annotations

import argparse
import base64
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from shadowgen_ml_service.application.use_cases.debug_pipeline import DebugPipelineUseCase
from shadowgen_ml_service.bootstrap.container import build_runtime
from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.core.assets import RasterAsset
from shadowgen_ml_service.core.commands import BackgroundSpec, DebugPipelineCommand, OutputSpec, RenderCommand, ShadowSpec, SourceImage
from shadowgen_ml_service.utils.images import ensure_pil


COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "ShadowGen-ML-service sample-pack-preparer/0.1"

DEFAULT_COMMONS_QUERIES = [
    "mug on table photo",
    "camera on table photo",
    "shoe on floor photo",
    "bottle on table photo",
    "book on desk photo",
    "plant pot window photo",
    "backpack on chair photo",
    "toy on floor photo",
    "phone on desk photo",
    "headphones on table photo",
]

DEFAULT_COMMONS_TITLES = [
    "File:Coffee Mug on table with creamer.jpg",
    "File:Nikon camera on a table beside a tripod.jpg",
    "File:Brown leather boots on a wood floor (Unsplash).jpg",
    "File:Personalized backpack book bag on a yellow chair (21312803791).jpg",
    "File:Beer bottle on a picnic table at the beach.jpg",
    "File:Laptop on desk book stacks (Unsplash).jpg",
    "File:Potted succulents in window (Unsplash).jpg",
    "File:Toddler Toy Floor Riga (Unsplash).jpg",
    "File:Computer mouse laptop and phone on a desk.jpg",
    "File:Headphones and smartphone (Unsplash).jpg",
]

CONTROL_PRESETS = [
    {"angle": 35.0, "elevation": 35.0, "softness": 0.25, "reflection": 0.0},
    {"angle": 70.0, "elevation": 45.0, "softness": 0.45, "reflection": 0.0},
    {"angle": 125.0, "elevation": 30.0, "softness": 0.35, "reflection": 0.1},
    {"angle": 180.0, "elevation": 55.0, "softness": 0.2, "reflection": 0.0},
    {"angle": 220.0, "elevation": 25.0, "softness": 0.55, "reflection": 0.15},
    {"angle": 275.0, "elevation": 40.0, "softness": 0.3, "reflection": 0.0},
    {"angle": 315.0, "elevation": 60.0, "softness": 0.65, "reflection": 0.0},
    {"angle": 15.0, "elevation": 20.0, "softness": 0.75, "reflection": 0.2},
    {"angle": 150.0, "elevation": 50.0, "softness": 0.4, "reflection": 0.0},
    {"angle": 250.0, "elevation": 32.0, "softness": 0.5, "reflection": 0.1},
]

HEAVY_STAGE_BACKENDS = {
    "detector": "local",
    "segmenter": "local",
    "foreground_refiner": "local",
    "depth_estimator": "local",
    "normal_estimator": "local",
}

HEAVY_STAGE_VARIANTS = {
    "detector": "grounding-dino",
    "segmenter": "birefnet",
    "foreground_refiner": "fast-foreground-estimation",
    "depth_estimator": "depth-anything-v2-small",
    "normal_estimator": "from-depth-v2",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Shadow V2-DIFF model-input sample pack.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/shadow-v2-sample-pack"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--backend-kind", choices=["local", "mock"], default="local")
    parser.add_argument("--normal-variant", choices=["from-depth-v2", "stable-normal", "mock-v1"], default="from-depth-v2")
    parser.add_argument("--skip-download", action="store_true", help="Reuse already downloaded source images from output-dir/sources.")
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    sources_dir = output_dir / "sources"
    samples_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    sources_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    source_records = load_or_download_sources(
        sources_dir=sources_dir,
        count=args.count,
        skip_download=args.skip_download,
    )
    settings = Settings()
    settings.ensure_local_dirs()
    runtime = build_runtime(settings)
    use_case = DebugPipelineUseCase(settings, runtime)

    manifest: dict[str, Any] = {
        "contract": "shadow_generator/v2-diff",
        "working_size": settings.working_size,
        "backend_kind_requested": args.backend_kind,
        "normal_variant_requested": args.normal_variant,
        "samples": [],
        "sources": source_records,
    }

    for index, source in enumerate(source_records[: args.count], start=1):
        sample_id = f"sample_{index:02d}"
        print(f"[{sample_id}] running pipeline for {source['title']}")
        sample_dir = samples_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        controls = CONTROL_PRESETS[(index - 1) % len(CONTROL_PRESETS)]
        outcome = run_pipeline(
            use_case=use_case,
            settings=settings,
            source_path=Path(source["local_path"]),
            sample_id=sample_id,
            backend_kind=args.backend_kind,
            normal_variant=args.normal_variant,
            controls=controls,
        )
        export_sample(sample_dir=sample_dir, source_path=Path(source["local_path"]), outcome=outcome, controls=controls, source=source)
        manifest["samples"].append(
            {
                "sample_id": sample_id,
                "path": str(sample_dir.as_posix()),
                "source_title": source["title"],
                "controls": controls,
                "stages": [
                    {
                        "stage_key": stage.stage_key,
                        "status": stage.status,
                        "requested_backend_kind": stage.requested_backend_kind,
                        "actual_backend_kind": stage.actual_backend_kind,
                        "model_variant": stage.model_variant,
                        "model_name": stage.model_name,
                        "device": stage.device,
                        "fallback_reason": stage.fallback_reason,
                        "error": stage.error,
                    }
                    for stage in outcome.stages
                ],
            }
        )

    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output_dir)
    print(f"sample pack written to {output_dir}")
    return 0


def load_or_download_sources(*, sources_dir: Path, count: int, skip_download: bool) -> list[dict[str, Any]]:
    if skip_download:
        records = []
        for path in sorted(sources_dir.glob("*.png"))[:count]:
            records.append({"title": path.stem, "local_path": str(path), "source_url": None, "license": "local"})
        if len(records) < count:
            raise RuntimeError(f"--skip-download requested but only found {len(records)} source PNG files")
        return records

    records: list[dict[str, Any]] = []
    used_titles: set[str] = set()
    for source in fetch_commons_images_by_titles(DEFAULT_COMMONS_TITLES):
        if len(records) >= count:
            break
        used_titles.add(source["title"])
        local_path = sources_dir / f"{len(records) + 1:02d}_{slugify(source['title'])}.png"
        if not local_path.exists():
            download_to_png(source["download_url"], local_path)
            time.sleep(1.0)
        source["local_path"] = str(local_path)
        records.append(source)

    for query in DEFAULT_COMMONS_QUERIES:
        if len(records) >= count:
            break
        source = find_commons_image(query=query, used_titles=used_titles)
        if source is None:
            continue
        used_titles.add(source["title"])
        local_path = sources_dir / f"{len(records) + 1:02d}_{slugify(source['title'])}.png"
        if not local_path.exists():
            download_to_png(source["download_url"], local_path)
            time.sleep(1.0)
        source["local_path"] = str(local_path)
        records.append(source)
    if len(records) < count:
        raise RuntimeError(f"only prepared {len(records)} source images, expected {count}")
    return records


def fetch_commons_images_by_titles(titles: list[str]) -> list[dict[str, Any]]:
    params = {
        "action": "query",
        "titles": "|".join(titles),
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "iiurlwidth": "1600",
        "format": "json",
        "formatversion": "2",
    }
    payload = fetch_json(f"{COMMONS_API_URL}?{urllib.parse.urlencode(params)}")
    pages_by_title = {str(page.get("title")): page for page in payload.get("query", {}).get("pages", [])}
    sources: list[dict[str, Any]] = []
    for title in titles:
        page = pages_by_title.get(title)
        if page is None:
            continue
        source = source_from_commons_page(page, query="curated-title")
        if source is not None:
            sources.append(source)
    return sources


def find_commons_image(*, query: str, used_titles: set[str]) -> dict[str, Any] | None:
    params = {
        "action": "query",
        "generator": "search",
        "gsrnamespace": "6",
        "gsrlimit": "10",
        "gsrsearch": query,
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "iiurlwidth": "1600",
        "format": "json",
        "formatversion": "2",
    }
    payload = fetch_json(f"{COMMONS_API_URL}?{urllib.parse.urlencode(params)}")
    for page in payload.get("query", {}).get("pages", []):
        source = source_from_commons_page(page, query=query)
        if source is None:
            continue
        title = source["title"]
        if title in used_titles:
            continue
        return source
    return None


def source_from_commons_page(page: dict[str, Any], *, query: str) -> dict[str, Any] | None:
    title = str(page.get("title") or "")
    info = (page.get("imageinfo") or [{}])[0]
    mime = str(info.get("mime") or "")
    width = int(info.get("width") or 0)
    height = int(info.get("height") or 0)
    download_url = info.get("thumburl") or info.get("url")
    if not title or not download_url or not mime.startswith("image/") or "svg" in mime:
        return None
    if width < 256 or height < 256:
        return None
    metadata = info.get("extmetadata") or {}
    return {
        "title": title,
        "query": query,
        "source_url": info.get("descriptionurl"),
        "download_url": download_url,
        "mime": mime,
        "width": width,
        "height": height,
        "license": clean_metadata(metadata.get("LicenseShortName", {}).get("value")),
        "artist": clean_metadata(metadata.get("Artist", {}).get("value")),
        "credit": clean_metadata(metadata.get("Credit", {}).get("value")),
    }


def fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def download_to_png(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    raw = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 3:
                raise
            time.sleep(5.0 * (attempt + 1))
    if raw is None:
        raise RuntimeError(f"failed to download {url}")
    image = Image.open(BytesIO(raw))
    image.load()
    image.convert("RGBA").save(path)


def run_pipeline(
    *,
    use_case: DebugPipelineUseCase,
    settings: Settings,
    source_path: Path,
    sample_id: str,
    backend_kind: str,
    normal_variant: str,
    controls: dict[str, float],
):
    image_base64 = base64.b64encode(source_path.read_bytes()).decode("ascii")
    variants = dict(HEAVY_STAGE_VARIANTS)
    variants["normal_estimator"] = normal_variant
    command = DebugPipelineCommand(
        render=RenderCommand(
            request_id=f"shadow-v2-pack-{sample_id}",
            pipeline_version=settings.default_pipeline_version,
            source=SourceImage(mime_type="image/png", image_base64=image_base64),
            padding_px=48,
            shadow=ShadowSpec(
                angle_deg=controls["angle"],
                elevation_deg=controls["elevation"],
                softness=controls["softness"],
                opacity=0.8,
                reflection=controls["reflection"],
            ),
            background=BackgroundSpec(mode="solid", color_hex="#ffffff"),
            output=OutputSpec(format="png", width=None, height=None, return_debug=True),
        ),
        stage_backend_kinds={key: backend_kind for key in HEAVY_STAGE_BACKENDS},
        stage_variants=variants,
    )
    if backend_kind == "mock":
        command.stage_backend_kinds.update({key: "mock" for key in HEAVY_STAGE_BACKENDS})
        command.stage_variants.update({key: "mock-v1" for key in HEAVY_STAGE_BACKENDS})
        command.stage_variants["foreground_refiner"] = "passthrough-v1"
    outcome = use_case.execute(command, stop_after="normal_estimator")
    failed = [stage for stage in outcome.stages if stage.status == "failed"]
    if failed:
        first = failed[0]
        raise RuntimeError(f"{sample_id} failed at {first.stage_key}: {first.error}")
    return outcome


def export_sample(*, sample_dir: Path, source_path: Path, outcome, controls: dict[str, float], source: dict[str, Any]) -> None:
    previews = collect_previews(outcome)
    required = {
        "img": previews["foreground_cutout"],
        "mask": previews["mask"],
        "depth": previews["depth"],
        "normal": previews["normals"],
    }
    Image.open(source_path).save(sample_dir / "source.png")
    for name, asset in required.items():
        ensure_pil(asset).save(sample_dir / f"{name}.png")
    if "working_crop" in previews:
        ensure_pil(previews["working_crop"]).save(sample_dir / "working_crop.png")
    if "cutout" in previews:
        ensure_pil(previews["cutout"]).save(sample_dir / "segmenter_cutout.png")

    arrays = {
        "img": image_to_nchw_float32(ensure_pil(required["img"]).convert("RGBA")),
        "mask": image_to_nchw_float32(ensure_pil(required["mask"]).convert("L")),
        "depth": image_to_nchw_float32(ensure_pil(required["depth"]).convert("L")),
        "normal": image_to_nchw_float32(ensure_pil(required["normal"]).convert("RGB")),
        "angle": np.asarray([controls["angle"]], dtype=np.float32),
        "elevation": np.asarray([controls["elevation"]], dtype=np.float32),
        "softness": np.asarray([controls["softness"]], dtype=np.float32),
        "reflection": np.asarray([controls["reflection"]], dtype=np.float32),
    }
    np.savez_compressed(sample_dir / "shadow_input.npz", **arrays)
    (sample_dir / "controls.json").write_text(json.dumps(controls, ensure_ascii=False, indent=2), encoding="utf-8")
    (sample_dir / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8")
    (sample_dir / "stages.json").write_text(
        json.dumps([stage_to_dict(stage) for stage in outcome.stages], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def collect_previews(outcome) -> dict[str, RasterAsset]:
    previews: dict[str, RasterAsset] = {}
    for stage in outcome.stages:
        previews.update(stage.previews)
    return previews


def image_to_nchw_float32(image: Image.Image) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[:, :, None]
    return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)


def stage_to_dict(stage) -> dict[str, Any]:
    data = asdict(stage)
    data.pop("previews", None)
    return data


def write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text(
        """# Shadow V2-DIFF Sample Pack

This folder is generated by `tools/prepare_shadow_v2_sample_pack.py`.

Each `samples/sample_XX/` folder contains:

- `source.png`: downloaded source image
- `img.png`: RGBA refined foreground cutout, maps to model input `img`
- `mask.png`: grayscale foreground mask, maps to model input `mask`
- `depth.png`: grayscale depth normalized inside `mask`, maps to model input `depth`
- `normal.png`: RGB normal map, maps to model input `normal`
- `controls.json`: scalar conditioning values
- `shadow_input.npz`: ready-to-load FP32 tensors in NCHW batch layout
- `source.json`: source URL/license metadata
- `stages.json`: pipeline backend metadata for reproducibility

The sample pack is for model integration/testing and is not committed to git.
""",
        encoding="utf-8",
    )


def clean_metadata(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"<[^>]+>", "", str(value))
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def slugify(value: str) -> str:
    value = value.removeprefix("File:")
    value = re.sub(r"\.[^.]+$", "", value)
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return value[:80] or "image"


if __name__ == "__main__":
    raise SystemExit(main())
