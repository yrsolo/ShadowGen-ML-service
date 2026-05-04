from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _as_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def _as_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    service_name: str = os.getenv("SHADOWGEN_SERVICE_NAME", "shadowgen-ml-service")
    service_version: str = os.getenv("SHADOWGEN_SERVICE_VERSION", "0.1.0")
    default_pipeline_version: str = os.getenv("SHADOWGEN_PIPELINE_VERSION", "ml-shadowgen-v1")
    runtime_mode: str = os.getenv("SHADOWGEN_RUNTIME_MODE", "auto")
    execution_default_backend: str = os.getenv("SHADOWGEN_EXECUTION_DEFAULT_BACKEND", "")
    target_device: str = os.getenv("SHADOWGEN_TARGET_DEVICE", "cuda")
    async_enabled: bool = _as_bool("SHADOWGEN_ASYNC_ENABLED", True)
    async_backend: str = os.getenv("SHADOWGEN_ASYNC_BACKEND", "in-memory")
    job_max_running: int = _as_int("SHADOWGEN_JOB_MAX_RUNNING", 2)
    job_max_pending: int = _as_int("SHADOWGEN_JOB_MAX_PENDING", 32)
    job_accepting_enabled: bool = _as_bool("SHADOWGEN_JOB_ACCEPTING_ENABLED", True)
    job_cancel_mode: str = os.getenv("SHADOWGEN_JOB_CANCEL_MODE", "pending_only")
    batching_enabled: bool = _as_bool("SHADOWGEN_BATCHING_ENABLED", True)
    batch_window_ms: int = _as_int("SHADOWGEN_BATCH_WINDOW_MS", 15)
    batch_max_size: int = _as_int("SHADOWGEN_BATCH_MAX_SIZE", 4)
    batch_segmenter_enabled: bool = _as_bool("SHADOWGEN_BATCH_SEGMENTER_ENABLED", True)
    batch_depth_enabled: bool = _as_bool("SHADOWGEN_BATCH_DEPTH_ENABLED", True)
    batch_normals_enabled: bool = _as_bool("SHADOWGEN_BATCH_NORMALS_ENABLED", True)
    batch_shadow_enabled: bool = _as_bool("SHADOWGEN_BATCH_SHADOW_ENABLED", True)
    triton_url: str | None = os.getenv("SHADOWGEN_TRITON_URL") or None
    triton_protocol: str = os.getenv("SHADOWGEN_TRITON_PROTOCOL", "http")
    triton_transport: str = os.getenv("SHADOWGEN_TRITON_TRANSPORT", "native")
    triton_timeout_ms: int = _as_int("SHADOWGEN_TRITON_TIMEOUT_MS", 30_000)
    triton_model_repository: str | None = os.getenv("SHADOWGEN_TRITON_MODEL_REPOSITORY") or None
    working_size: int = _as_int("SHADOWGEN_WORKING_SIZE", 512)
    max_image_bytes: int = _as_int("SHADOWGEN_MAX_IMAGE_BYTES", 10 * 1024 * 1024)
    request_timeout_ms: int = _as_int("SHADOWGEN_REQUEST_TIMEOUT_MS", 30_000)
    model_cache_dir: Path = Path(os.getenv("SHADOWGEN_MODEL_CACHE_DIR", ".models"))
    preprocess_cache_dir: Path = Path(os.getenv("SHADOWGEN_PREPROCESS_CACHE_DIR", "var/cache/preprocess"))
    artifact_dir: Path = Path(os.getenv("SHADOWGEN_ARTIFACT_DIR", "artifacts"))
    working_content_scale: float = float(os.getenv("SHADOWGEN_WORKING_CONTENT_SCALE", "0.68"))
    geocalib_weights: str = os.getenv("SHADOWGEN_GEOCALIB_WEIGHTS", "pinhole")
    geocalib_camera_model: str = os.getenv("SHADOWGEN_GEOCALIB_CAMERA_MODEL", "pinhole")
    geocalib_shared_intrinsics: bool = _as_bool("SHADOWGEN_GEOCALIB_SHARED_INTRINSICS", False)
    geometry_enabled: bool = _as_bool("SHADOWGEN_GEOMETRY_ENABLED", False)
    geometry_backend_kind: str = os.getenv("SHADOWGEN_GEOMETRY_BACKEND_KIND", "local")
    grounding_dino_model_id: str = os.getenv("SHADOWGEN_GROUNDING_DINO_MODEL_ID", "IDEA-Research/grounding-dino-base")
    grounding_dino_prompt: str = os.getenv("SHADOWGEN_GROUNDING_DINO_PROMPT", "object.")
    grounding_dino_box_threshold: float = float(os.getenv("SHADOWGEN_GROUNDING_DINO_BOX_THRESHOLD", "0.25"))
    grounding_dino_text_threshold: float = float(os.getenv("SHADOWGEN_GROUNDING_DINO_TEXT_THRESHOLD", "0.25"))
    detector_backend_kind: str = os.getenv("SHADOWGEN_DETECTOR_BACKEND_KIND", "")
    detector_model_variant: str = os.getenv("SHADOWGEN_DETECTOR_MODEL_VARIANT", "grounding-dino")
    birefnet_model_id: str = os.getenv("SHADOWGEN_BIREFNET_MODEL_ID", "ZhengPeng7/BiRefNet-matting")
    birefnet_resolution: int = _as_int("SHADOWGEN_BIREFNET_RESOLUTION", 1024)
    birefnet_mask_threshold: float = float(os.getenv("SHADOWGEN_BIREFNET_MASK_THRESHOLD", "0.5"))
    birefnet_allow_cpu: bool = _as_bool("SHADOWGEN_BIREFNET_ALLOW_CPU", False)
    birefnet_compile_enabled: bool = _as_bool("SHADOWGEN_BIREFNET_COMPILE_ENABLED", False)
    birefnet_compile_mode: str = os.getenv("SHADOWGEN_BIREFNET_COMPILE_MODE", "reduce-overhead")
    birefnet_compile_backend: str = os.getenv("SHADOWGEN_BIREFNET_COMPILE_BACKEND", "")
    birefnet_matmul_precision: str = os.getenv("SHADOWGEN_BIREFNET_MATMUL_PRECISION", "high")
    segmenter_backend_kind: str = os.getenv("SHADOWGEN_SEGMENTER_BACKEND_KIND", "")
    segmenter_model_variant: str = os.getenv("SHADOWGEN_SEGMENTER_MODEL_VARIANT", "birefnet")
    foreground_refiner_backend_kind: str = os.getenv("SHADOWGEN_FOREGROUND_REFINER_BACKEND_KIND", "local")
    depth_anything_model_id: str = os.getenv("SHADOWGEN_DEPTH_ANYTHING_MODEL_ID", "depth-anything/Depth-Anything-V2-Small-hf")
    depth_backend_kind: str = os.getenv("SHADOWGEN_DEPTH_BACKEND_KIND", "")
    stable_normal_variant: str = os.getenv("SHADOWGEN_STABLE_NORMAL_VARIANT", "StableNormal_turbo")
    stable_normal_resolution: int = _as_int("SHADOWGEN_STABLE_NORMAL_RESOLUTION", 1024)
    stable_normal_allow_cpu: bool = _as_bool("SHADOWGEN_STABLE_NORMAL_ALLOW_CPU", False)
    normals_backend_kind: str = os.getenv("SHADOWGEN_NORMALS_BACKEND_KIND", "")
    normals_model_variant: str = os.getenv("SHADOWGEN_NORMALS_MODEL_VARIANT", "from-depth-v2")
    shadow_backend_kind: str = os.getenv("SHADOWGEN_SHADOW_BACKEND_KIND", "")
    shadow_model_variant: str = os.getenv("SHADOWGEN_SHADOW_MODEL_VARIANT", "v1-gan")
    shadow_pix2pix_weights_path: Path = Path(os.getenv("SHADOWGEN_SHADOW_PIX2PIX_WEIGHTS_PATH", ".models/shadow/AveragedModel.pth"))
    shadow_v2_diff_bundle_path: Path = Path(
        os.getenv("SHADOWGEN_SHADOW_V2_DIFF_BUNDLE_PATH", ".models/shadow/v2-diff/shadowgen_inpaint_lora_prod_current")
    )
    shadow_v2_diff_background_path: Path = Path(os.getenv("SHADOWGEN_SHADOW_V2_DIFF_BACKGROUND_PATH", ".models/shadow/v2-diff/mean_background.png"))
    shadow_v2_diff_seed: int = _as_int("SHADOWGEN_SHADOW_V2_DIFF_SEED", 1234)
    shadow_v2_diff_fast_lcm: bool = _as_bool("SHADOWGEN_SHADOW_V2_DIFF_FAST_LCM", True)
    shadow_v2_diff_steps: int | None = None if os.getenv("SHADOWGEN_SHADOW_V2_DIFF_STEPS") is None else _as_int("SHADOWGEN_SHADOW_V2_DIFF_STEPS", 5)
    shadow_v2_diff_guidance_scale: float | None = (
        None if os.getenv("SHADOWGEN_SHADOW_V2_DIFF_GUIDANCE_SCALE") is None else float(os.getenv("SHADOWGEN_SHADOW_V2_DIFF_GUIDANCE_SCALE", "1.0"))
    )
    shadow_v2_diff_compile_enabled: bool = _as_bool("SHADOWGEN_SHADOW_V2_DIFF_COMPILE_ENABLED", False)
    shadow_v2_diff_compile_mode: str = os.getenv("SHADOWGEN_SHADOW_V2_DIFF_COMPILE_MODE", "reduce-overhead")
    shadow_v2_diff_compile_backend: str = os.getenv("SHADOWGEN_SHADOW_V2_DIFF_COMPILE_BACKEND", "")
    triton_detector_model: str = os.getenv("SHADOWGEN_TRITON_DETECTOR_MODEL", "shadowgen_detector")
    triton_detector_onnx_model: str = os.getenv("SHADOWGEN_TRITON_DETECTOR_ONNX_MODEL", "shadowgen_detector_onnx")
    triton_segmenter_model: str = os.getenv("SHADOWGEN_TRITON_SEGMENTER_MODEL", "shadowgen_segmenter")
    triton_segmenter_rmbg2_model: str = os.getenv("SHADOWGEN_TRITON_SEGMENTER_RMBG2_MODEL", "shadowgen_segmenter_rmbg2")
    triton_segmenter_rmbg2_input: str = os.getenv("SHADOWGEN_TRITON_SEGMENTER_RMBG2_INPUT", "input")
    triton_segmenter_rmbg2_output: str = os.getenv("SHADOWGEN_TRITON_SEGMENTER_RMBG2_OUTPUT", "output")
    triton_depth_model: str = os.getenv("SHADOWGEN_TRITON_DEPTH_MODEL", "shadowgen_depth")
    triton_normals_model: str = os.getenv("SHADOWGEN_TRITON_NORMALS_MODEL", "shadowgen_normals")
    triton_shadow_v2_model: str = os.getenv("SHADOWGEN_TRITON_SHADOW_V2_MODEL", "shadowgen_shadow_v2")

    def ensure_local_dirs(self) -> None:
        for path in (self.model_cache_dir, self.preprocess_cache_dir, self.artifact_dir):
            path.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_local_dirs()
    return settings
