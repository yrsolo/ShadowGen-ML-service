from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _as_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


@dataclass(frozen=True)
class Settings:
    service_name: str = os.getenv("SHADOWGEN_SERVICE_NAME", "shadowgen-ml-service")
    service_version: str = os.getenv("SHADOWGEN_SERVICE_VERSION", "0.1.0")
    default_pipeline_version: str = os.getenv("SHADOWGEN_PIPELINE_VERSION", "ml-shadowgen-v1")
    runtime_mode: str = os.getenv("SHADOWGEN_RUNTIME_MODE", "auto")
    target_device: str = os.getenv("SHADOWGEN_TARGET_DEVICE", "cuda")
    working_size: int = _as_int("SHADOWGEN_WORKING_SIZE", 512)
    max_image_bytes: int = _as_int("SHADOWGEN_MAX_IMAGE_BYTES", 10 * 1024 * 1024)
    request_timeout_ms: int = _as_int("SHADOWGEN_REQUEST_TIMEOUT_MS", 30_000)
    model_cache_dir: Path = Path(os.getenv("SHADOWGEN_MODEL_CACHE_DIR", ".models"))
    preprocess_cache_dir: Path = Path(os.getenv("SHADOWGEN_PREPROCESS_CACHE_DIR", "var/cache/preprocess"))
    artifact_dir: Path = Path(os.getenv("SHADOWGEN_ARTIFACT_DIR", "artifacts"))

    def ensure_local_dirs(self) -> None:
        for path in (self.model_cache_dir, self.preprocess_cache_dir, self.artifact_dir):
            path.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_local_dirs()
    return settings
