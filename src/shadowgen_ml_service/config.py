from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _as_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def _as_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    return int(raw)


def _as_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw)


def _as_optional_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    return float(raw)


def _as_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _as_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default))


def _validate_choice(name: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {options}")


@dataclass(frozen=True)
class Settings:
    service_name: str = field(default_factory=lambda: _env("SHADOWGEN_SERVICE_NAME", "shadowgen-ml-service"))
    service_version: str = field(default_factory=lambda: _env("SHADOWGEN_SERVICE_VERSION", "0.1.0"))
    default_pipeline_version: str = field(default_factory=lambda: _env("SHADOWGEN_PIPELINE_VERSION", "ml-shadowgen-v1"))
    runtime_mode: str = field(default_factory=lambda: _env("SHADOWGEN_RUNTIME_MODE", "auto"))
    execution_default_backend: str = field(default_factory=lambda: _env("SHADOWGEN_EXECUTION_DEFAULT_BACKEND", ""))
    target_device: str = field(default_factory=lambda: _env("SHADOWGEN_TARGET_DEVICE", "cuda"))
    dev_api_enabled: bool = field(default_factory=lambda: _as_bool("SHADOWGEN_DEV_API_ENABLED", False))
    dev_shutdown_enabled: bool = field(default_factory=lambda: _as_bool("SHADOWGEN_DEV_SHUTDOWN_ENABLED", False))
    async_enabled: bool = field(default_factory=lambda: _as_bool("SHADOWGEN_ASYNC_ENABLED", True))
    async_backend: str = field(default_factory=lambda: _env("SHADOWGEN_ASYNC_BACKEND", "in-memory"))
    job_max_running: int = field(default_factory=lambda: _as_int("SHADOWGEN_JOB_MAX_RUNNING", 2))
    job_max_pending: int = field(default_factory=lambda: _as_int("SHADOWGEN_JOB_MAX_PENDING", 32))
    job_accepting_enabled: bool = field(default_factory=lambda: _as_bool("SHADOWGEN_JOB_ACCEPTING_ENABLED", True))
    job_cancel_mode: str = field(default_factory=lambda: _env("SHADOWGEN_JOB_CANCEL_MODE", "pending_only"))
    batching_enabled: bool = field(default_factory=lambda: _as_bool("SHADOWGEN_BATCHING_ENABLED", True))
    batch_window_ms: int = field(default_factory=lambda: _as_int("SHADOWGEN_BATCH_WINDOW_MS", 15))
    batch_max_size: int = field(default_factory=lambda: _as_int("SHADOWGEN_BATCH_MAX_SIZE", 4))
    batch_segmenter_enabled: bool = field(default_factory=lambda: _as_bool("SHADOWGEN_BATCH_SEGMENTER_ENABLED", True))
    batch_depth_enabled: bool = field(default_factory=lambda: _as_bool("SHADOWGEN_BATCH_DEPTH_ENABLED", True))
    batch_normals_enabled: bool = field(default_factory=lambda: _as_bool("SHADOWGEN_BATCH_NORMALS_ENABLED", True))
    batch_shadow_enabled: bool = field(default_factory=lambda: _as_bool("SHADOWGEN_BATCH_SHADOW_ENABLED", True))
    triton_url: str | None = field(default_factory=lambda: os.getenv("SHADOWGEN_TRITON_URL") or None)
    triton_protocol: str = field(default_factory=lambda: _env("SHADOWGEN_TRITON_PROTOCOL", "http"))
    triton_transport: str = field(default_factory=lambda: _env("SHADOWGEN_TRITON_TRANSPORT", "native"))
    triton_timeout_ms: int = field(default_factory=lambda: _as_int("SHADOWGEN_TRITON_TIMEOUT_MS", 30_000))
    triton_model_repository: str | None = field(default_factory=lambda: os.getenv("SHADOWGEN_TRITON_MODEL_REPOSITORY") or None)
    working_size: int = field(default_factory=lambda: _as_int("SHADOWGEN_WORKING_SIZE", 512))
    max_image_bytes: int = field(default_factory=lambda: _as_int("SHADOWGEN_MAX_IMAGE_BYTES", 10 * 1024 * 1024))
    request_timeout_ms: int = field(default_factory=lambda: _as_int("SHADOWGEN_REQUEST_TIMEOUT_MS", 30_000))
    model_cache_dir: Path = field(default_factory=lambda: _as_path("SHADOWGEN_MODEL_CACHE_DIR", ".models"))
    preprocess_cache_dir: Path = field(default_factory=lambda: _as_path("SHADOWGEN_PREPROCESS_CACHE_DIR", "var/cache/preprocess"))
    artifact_dir: Path = field(default_factory=lambda: _as_path("SHADOWGEN_ARTIFACT_DIR", "artifacts"))
    working_content_scale: float = field(default_factory=lambda: _as_float("SHADOWGEN_WORKING_CONTENT_SCALE", 0.68))
    geocalib_weights: str = field(default_factory=lambda: _env("SHADOWGEN_GEOCALIB_WEIGHTS", "pinhole"))
    geocalib_camera_model: str = field(default_factory=lambda: _env("SHADOWGEN_GEOCALIB_CAMERA_MODEL", "pinhole"))
    geocalib_shared_intrinsics: bool = field(default_factory=lambda: _as_bool("SHADOWGEN_GEOCALIB_SHARED_INTRINSICS", False))
    geometry_enabled: bool = field(default_factory=lambda: _as_bool("SHADOWGEN_GEOMETRY_ENABLED", False))
    geometry_backend_kind: str = field(default_factory=lambda: _env("SHADOWGEN_GEOMETRY_BACKEND_KIND", "local"))
    grounding_dino_model_id: str = field(default_factory=lambda: _env("SHADOWGEN_GROUNDING_DINO_MODEL_ID", "IDEA-Research/grounding-dino-base"))
    grounding_dino_prompt: str = field(default_factory=lambda: _env("SHADOWGEN_GROUNDING_DINO_PROMPT", "object."))
    grounding_dino_box_threshold: float = field(default_factory=lambda: _as_float("SHADOWGEN_GROUNDING_DINO_BOX_THRESHOLD", 0.25))
    grounding_dino_text_threshold: float = field(default_factory=lambda: _as_float("SHADOWGEN_GROUNDING_DINO_TEXT_THRESHOLD", 0.25))
    detector_backend_kind: str = field(default_factory=lambda: _env("SHADOWGEN_DETECTOR_BACKEND_KIND", ""))
    detector_model_variant: str = field(default_factory=lambda: _env("SHADOWGEN_DETECTOR_MODEL_VARIANT", "grounding-dino"))
    birefnet_model_id: str = field(default_factory=lambda: _env("SHADOWGEN_BIREFNET_MODEL_ID", "ZhengPeng7/BiRefNet"))
    birefnet_resolution: int = field(default_factory=lambda: _as_int("SHADOWGEN_BIREFNET_RESOLUTION", 1024))
    birefnet_mask_threshold: float = field(default_factory=lambda: _as_float("SHADOWGEN_BIREFNET_MASK_THRESHOLD", 0.5))
    birefnet_allow_cpu: bool = field(default_factory=lambda: _as_bool("SHADOWGEN_BIREFNET_ALLOW_CPU", False))
    birefnet_compile_enabled: bool = field(default_factory=lambda: _as_bool("SHADOWGEN_BIREFNET_COMPILE_ENABLED", False))
    birefnet_compile_mode: str = field(default_factory=lambda: _env("SHADOWGEN_BIREFNET_COMPILE_MODE", "reduce-overhead"))
    birefnet_compile_backend: str = field(default_factory=lambda: _env("SHADOWGEN_BIREFNET_COMPILE_BACKEND", ""))
    birefnet_matmul_precision: str = field(default_factory=lambda: _env("SHADOWGEN_BIREFNET_MATMUL_PRECISION", "high"))
    segmenter_backend_kind: str = field(default_factory=lambda: _env("SHADOWGEN_SEGMENTER_BACKEND_KIND", ""))
    segmenter_model_variant: str = field(default_factory=lambda: _env("SHADOWGEN_SEGMENTER_MODEL_VARIANT", "birefnet"))
    foreground_refiner_backend_kind: str = field(default_factory=lambda: _env("SHADOWGEN_FOREGROUND_REFINER_BACKEND_KIND", "local"))
    depth_anything_model_id: str = field(default_factory=lambda: _env("SHADOWGEN_DEPTH_ANYTHING_MODEL_ID", "depth-anything/Depth-Anything-V2-Small-hf"))
    depth_backend_kind: str = field(default_factory=lambda: _env("SHADOWGEN_DEPTH_BACKEND_KIND", ""))
    stable_normal_variant: str = field(default_factory=lambda: _env("SHADOWGEN_STABLE_NORMAL_VARIANT", "StableNormal_turbo"))
    stable_normal_resolution: int = field(default_factory=lambda: _as_int("SHADOWGEN_STABLE_NORMAL_RESOLUTION", 1024))
    stable_normal_allow_cpu: bool = field(default_factory=lambda: _as_bool("SHADOWGEN_STABLE_NORMAL_ALLOW_CPU", False))
    normals_backend_kind: str = field(default_factory=lambda: _env("SHADOWGEN_NORMALS_BACKEND_KIND", ""))
    normals_model_variant: str = field(default_factory=lambda: _env("SHADOWGEN_NORMALS_MODEL_VARIANT", "from-depth-v2"))
    shadow_backend_kind: str = field(default_factory=lambda: _env("SHADOWGEN_SHADOW_BACKEND_KIND", ""))
    shadow_model_variant: str = field(default_factory=lambda: _env("SHADOWGEN_SHADOW_MODEL_VARIANT", "v1-gan"))
    shadow_pix2pix_weights_path: Path = field(default_factory=lambda: _as_path("SHADOWGEN_SHADOW_PIX2PIX_WEIGHTS_PATH", ".models/shadow/AveragedModel.pth"))
    shadow_v2_diff_bundle_path: Path = field(
        default_factory=lambda: _as_path("SHADOWGEN_SHADOW_V2_DIFF_BUNDLE_PATH", ".models/shadow/v2-diff/shadowgen_inpaint_lora_prod_current")
    )
    shadow_v2_diff_background_path: Path = field(default_factory=lambda: _as_path("SHADOWGEN_SHADOW_V2_DIFF_BACKGROUND_PATH", ".models/shadow/v2-diff/mean_background.png"))
    shadow_v2_diff_seed: int = field(default_factory=lambda: _as_int("SHADOWGEN_SHADOW_V2_DIFF_SEED", 1234))
    shadow_v2_diff_fast_lcm: bool = field(default_factory=lambda: _as_bool("SHADOWGEN_SHADOW_V2_DIFF_FAST_LCM", True))
    shadow_v2_diff_steps: int | None = field(default_factory=lambda: _as_optional_int("SHADOWGEN_SHADOW_V2_DIFF_STEPS"))
    shadow_v2_diff_guidance_scale: float | None = field(default_factory=lambda: _as_optional_float("SHADOWGEN_SHADOW_V2_DIFF_GUIDANCE_SCALE"))
    shadow_v2_diff_compile_enabled: bool = field(default_factory=lambda: _as_bool("SHADOWGEN_SHADOW_V2_DIFF_COMPILE_ENABLED", False))
    shadow_v2_diff_compile_mode: str = field(default_factory=lambda: _env("SHADOWGEN_SHADOW_V2_DIFF_COMPILE_MODE", "reduce-overhead"))
    shadow_v2_diff_compile_backend: str = field(default_factory=lambda: _env("SHADOWGEN_SHADOW_V2_DIFF_COMPILE_BACKEND", ""))
    triton_detector_model: str = field(default_factory=lambda: _env("SHADOWGEN_TRITON_DETECTOR_MODEL", "shadowgen_detector"))
    triton_detector_onnx_model: str = field(default_factory=lambda: _env("SHADOWGEN_TRITON_DETECTOR_ONNX_MODEL", "shadowgen_detector_onnx"))
    triton_segmenter_model: str = field(default_factory=lambda: _env("SHADOWGEN_TRITON_SEGMENTER_MODEL", "shadowgen_segmenter"))
    triton_segmenter_rmbg2_model: str = field(default_factory=lambda: _env("SHADOWGEN_TRITON_SEGMENTER_RMBG2_MODEL", "shadowgen_segmenter_rmbg2"))
    triton_segmenter_rmbg2_input: str = field(default_factory=lambda: _env("SHADOWGEN_TRITON_SEGMENTER_RMBG2_INPUT", "pixel_values"))
    triton_segmenter_rmbg2_output: str = field(default_factory=lambda: _env("SHADOWGEN_TRITON_SEGMENTER_RMBG2_OUTPUT", "alphas"))
    triton_segmenter_rmbg2_resolution: int = field(default_factory=lambda: _as_int("SHADOWGEN_TRITON_SEGMENTER_RMBG2_RESOLUTION", 1024))
    triton_depth_model: str = field(default_factory=lambda: _env("SHADOWGEN_TRITON_DEPTH_MODEL", "shadowgen_depth"))
    triton_normals_model: str = field(default_factory=lambda: _env("SHADOWGEN_TRITON_NORMALS_MODEL", "shadowgen_normals"))
    triton_shadow_v2_model: str = field(default_factory=lambda: _env("SHADOWGEN_TRITON_SHADOW_V2_MODEL", "shadowgen_shadow_v2"))

    def __post_init__(self) -> None:
        _validate_choice("runtime_mode", self.runtime_mode.lower(), {"auto", "local", "mock", "real"})
        if self.execution_default_backend:
            _validate_choice("execution_default_backend", self.execution_default_backend, {"local", "mock", "triton"})
        for name in (
            "detector_backend_kind",
            "segmenter_backend_kind",
            "depth_backend_kind",
            "normals_backend_kind",
            "shadow_backend_kind",
        ):
            value = getattr(self, name)
            if value:
                _validate_choice(name, value, {"local", "mock", "triton"})
        _validate_choice("geometry_backend_kind", self.geometry_backend_kind, {"local", "mock"})
        _validate_choice("foreground_refiner_backend_kind", self.foreground_refiner_backend_kind, {"local", "mock"})
        _validate_choice("triton_protocol", self.triton_protocol, {"http", "grpc"})
        _validate_choice("triton_transport", self.triton_transport, {"native", "json"})
        _validate_choice("job_cancel_mode", self.job_cancel_mode, {"pending_only"})
        if self.job_max_running < 1:
            raise ValueError("job_max_running must be >= 1")
        if self.job_max_pending < 1:
            raise ValueError("job_max_pending must be >= 1")
        if self.batch_window_ms < 0:
            raise ValueError("batch_window_ms must be >= 0")
        if self.batch_max_size < 1:
            raise ValueError("batch_max_size must be >= 1")
        if self.working_size < 1:
            raise ValueError("working_size must be >= 1")
        if not 0.1 <= self.working_content_scale <= 1.0:
            raise ValueError("working_content_scale must be between 0.1 and 1.0")

    def ensure_local_dirs(self) -> None:
        for path in (self.model_cache_dir, self.preprocess_cache_dir, self.artifact_dir):
            path.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
