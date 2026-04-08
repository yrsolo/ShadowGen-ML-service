from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from PIL import Image

from shadowgen_ml_service.core.contracts import PreprocessCacheRepository
from shadowgen_ml_service.core.models import DepthResult, DetectionResult, GeometryResult, NormalResult, PreprocessSnapshot, SegmentationResult


class FilesystemPreprocessCacheRepository(PreprocessCacheRepository):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def make_key(self, raw_bytes: bytes, runtime_signature: str, padding_px: int, working_size: int) -> str:
        digest = hashlib.sha256()
        digest.update(raw_bytes)
        digest.update(runtime_signature.encode("utf-8"))
        digest.update(str(padding_px).encode("utf-8"))
        digest.update(str(working_size).encode("utf-8"))
        return digest.hexdigest()

    def _cache_dir(self, key: str) -> Path:
        return self.root / key

    def load(self, key: str) -> PreprocessSnapshot | None:
        cache_dir = self._cache_dir(key)
        metadata_path = cache_dir / "metadata.json"
        if not metadata_path.exists():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        segmentation = SegmentationResult(
            bbox=tuple(metadata["segmentation"]["bbox"]),
            mask=Image.open(cache_dir / "mask.png").convert("L"),
            cutout_rgba=Image.open(cache_dir / "cutout.png").convert("RGBA"),
            crop_rgba=Image.open(cache_dir / "crop.png").convert("RGBA"),
        )
        return PreprocessSnapshot(
            detection=DetectionResult(**metadata["detection"]),
            geometry=GeometryResult(**metadata["geometry"]),
            segmentation=segmentation,
            depth=DepthResult(depth_map=Image.open(cache_dir / "depth.png").convert("L")),
            normals=NormalResult(normal_map=Image.open(cache_dir / "normals.png").convert("RGB")),
            cache_path=cache_dir,
        )

    def save(self, key: str, snapshot: PreprocessSnapshot) -> None:
        cache_dir = self._cache_dir(key)
        cache_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "detection": asdict(snapshot.detection),
            "geometry": asdict(snapshot.geometry),
            "segmentation": {"bbox": list(snapshot.segmentation.bbox)},
        }
        (cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        snapshot.segmentation.mask.save(cache_dir / "mask.png", format="PNG")
        snapshot.segmentation.cutout_rgba.save(cache_dir / "cutout.png", format="PNG")
        snapshot.segmentation.crop_rgba.save(cache_dir / "crop.png", format="PNG")
        snapshot.depth.depth_map.save(cache_dir / "depth.png", format="PNG")
        snapshot.normals.normal_map.save(cache_dir / "normals.png", format="PNG")
