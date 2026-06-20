from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from shadowgen_ml_service.core.contracts import PreprocessCacheRepository
from shadowgen_ml_service.core.models import DepthResult, DetectionResult, ForegroundRefinementResult, GeometryResult, NormalResult, PreprocessSnapshot, SegmentationResult
from shadowgen_ml_service.utils.images import asset_from_file


class FilesystemPreprocessCacheRepository(PreprocessCacheRepository):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def make_key(
        self,
        raw_bytes: bytes,
        runtime_signature: str,
        padding_px: int,
        working_size: int,
        working_content_scale: float,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(raw_bytes)
        digest.update(runtime_signature.encode("utf-8"))
        digest.update(str(padding_px).encode("utf-8"))
        digest.update(str(working_size).encode("utf-8"))
        digest.update(f"{working_content_scale:.4f}".encode("utf-8"))
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
            mask=asset_from_file(cache_dir / "mask.png"),
            cutout_rgba=asset_from_file(cache_dir / "cutout.png"),
            crop_rgba=asset_from_file(cache_dir / "crop.png"),
        )
        return PreprocessSnapshot(
            detection=DetectionResult(**metadata["detection"]),
            geometry=GeometryResult(**metadata["geometry"]),
            segmentation=segmentation,
            depth=DepthResult(depth_map=asset_from_file(cache_dir / "depth.png")),
            normals=NormalResult(normal_map=asset_from_file(cache_dir / "normals.png")),
            foreground_refinement=(
                ForegroundRefinementResult(cutout_rgba=asset_from_file(cache_dir / "foreground-cutout.png"))
                if (cache_dir / "foreground-cutout.png").exists()
                else None
            ),
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
        (cache_dir / "mask.png").write_bytes(snapshot.segmentation.mask.data)
        (cache_dir / "cutout.png").write_bytes(snapshot.segmentation.cutout_rgba.data)
        (cache_dir / "crop.png").write_bytes(snapshot.segmentation.crop_rgba.data)
        if snapshot.foreground_refinement is not None:
            (cache_dir / "foreground-cutout.png").write_bytes(snapshot.foreground_refinement.cutout_rgba.data)
        (cache_dir / "depth.png").write_bytes(snapshot.depth.depth_map.data)
        (cache_dir / "normals.png").write_bytes(snapshot.normals.normal_map.data)
