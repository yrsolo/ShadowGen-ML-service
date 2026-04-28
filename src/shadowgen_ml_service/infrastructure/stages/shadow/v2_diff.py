from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image, ImageOps

from shadowgen_ml_service.core.contracts import ShadowGenerator
from shadowgen_ml_service.core.models import ShadowResult
from shadowgen_ml_service.core.stage_io import ShadowInput
from shadowgen_ml_service.infrastructure.stages.shared.model_support import RealAdapterProbe, import_module, module_available
from shadowgen_ml_service.utils.images import ensure_pil, pil_to_asset


def _load_bundle_config(bundle_path: Path) -> dict[str, Any]:
    return json.loads((bundle_path / "bundle_config.json").read_text(encoding="utf-8"))


def _bucket_camera_view(camera_elevation_deg: float | None, buckets: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not buckets:
        return None
    if camera_elevation_deg is None:
        return buckets[min(1, len(buckets) - 1)]
    elevation = float(camera_elevation_deg)
    for bucket in buckets:
        max_elev = bucket.get("max_camera_elevation_deg")
        if max_elev is None:
            return bucket
        if elevation < float(max_elev):
            return bucket
    return buckets[-1]


def _render_prompt(bundle: dict[str, Any], camera_elevation_deg: float | None) -> str:
    base = str(bundle.get("default_prompt_base", "")).strip()
    bucket = _bucket_camera_view(camera_elevation_deg, list(bundle.get("view_buckets", [])))
    phrase = "" if bucket is None else str(bucket.get("phrase", "")).strip()
    if base and phrase:
        return f"{base}, {phrase}"
    return base or phrase


def _compose_conditioning_image(cutout_rgba: Image.Image, mask: Image.Image, background: Image.Image) -> Image.Image:
    size = cutout_rgba.size
    conditioning = background.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    conditioning.paste(cutout_rgba.convert("RGB"), mask=mask.convert("L"))
    return conditioning


class V2DiffShadowGenerator(ShadowGenerator):
    _PIPELINE_CACHE: ClassVar[dict[tuple[str, str], Any]] = {}

    def __init__(
        self,
        *,
        bundle_path: str | Path,
        background_path: str | Path,
        target_device: str = "cuda",
        seed: int = 1234,
        steps: int | None = None,
        guidance_scale: float | None = None,
        fast_lcm: bool = True,
        compile_enabled: bool = False,
        compile_mode: str = "reduce-overhead",
        compile_backend: str | None = None,
        torch_module: Any | None = None,
        pipeline_cls: Any | None = None,
        scheduler_classes: dict[str, Any] | None = None,
    ) -> None:
        self.bundle_path = Path(bundle_path)
        self.background_path = Path(background_path)
        self.target_device = target_device
        self.seed = int(seed)
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.fast_lcm = bool(fast_lcm)
        self.compile_enabled = bool(compile_enabled)
        self.compile_mode = compile_mode
        self.compile_backend = compile_backend or None
        self.backend_name = "v2-diff"
        self.model_variant = "V2-DIFF"
        self._torch = torch_module or import_module("torch")
        self._pipeline_cls = pipeline_cls
        self._scheduler_classes = scheduler_classes
        self._bundle = _load_bundle_config(self.bundle_path)
        self.device_label = self._resolve_device_label()

    def generate(self, stage_input: ShadowInput) -> ShadowResult:
        cutout = ensure_pil(stage_input.img).convert("RGBA")
        mask = ensure_pil(stage_input.mask).convert("L").resize(cutout.size, Image.Resampling.NEAREST)
        inpaint_mask = ImageOps.invert(mask)
        background = Image.open(self.background_path)
        conditioning = _compose_conditioning_image(cutout, mask, background)
        pipe = self._load_pipeline()
        prompt = _render_prompt(self._bundle, stage_input.elevation)
        negative_prompt = "" if self.fast_lcm else str(self._bundle.get("default_negative_prompt", ""))
        generator = self._torch.Generator(device=self.device_label).manual_seed(self.seed)
        steps = int(
            self.steps
            or (
                self._bundle.get("fast_lcm_default_num_inference_steps", 5)
                if self.fast_lcm
                else self._bundle["default_num_inference_steps"]
            )
        )
        guidance = float(
            self.guidance_scale
            if self.guidance_scale is not None
            else (
                self._bundle.get("fast_lcm_default_guidance_scale", 1.0)
                if self.fast_lcm
                else self._bundle["default_guidance_scale"]
            )
        )
        dtype = self._tensor_dtype()
        autocast_context = (
            self._torch.autocast(device_type="cuda", dtype=dtype)
            if str(self.device_label).startswith("cuda") and dtype in {self._torch.float16, getattr(self._torch, "bfloat16", object())}
            else None
        )
        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": conditioning,
            "mask_image": inpaint_mask,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator,
        }
        if autocast_context is None:
            result = pipe(**kwargs)
        else:
            with autocast_context:
                result = pipe(**kwargs)
        output = result.images[0].convert("RGBA")
        if output.size != cutout.size:
            output = output.resize(cutout.size, Image.Resampling.LANCZOS)
        output.putalpha(Image.new("L", output.size, 255))
        return ShadowResult(shadow_image=pil_to_asset(output))

    def _load_pipeline(self):
        cache_key = (
            str(self.bundle_path.resolve()),
            self.device_label,
            self._scheduler_name(),
            "fast-lcm" if self.fast_lcm else "normal",
            "compile" if self.compile_enabled else "eager",
            self.compile_mode,
            self.compile_backend or "",
        )
        cached = self._PIPELINE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        diffusers = import_module("diffusers")
        pipeline_cls = self._pipeline_cls or diffusers.StableDiffusionInpaintPipeline
        pipe = pipeline_cls.from_pretrained(
            str(self._bundle["base_model"]),
            torch_dtype=self._tensor_dtype(),
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = self._make_scheduler()
        pipe.unet.load_attn_procs(self.bundle_path / "checkpoint")
        if self.fast_lcm:
            self._load_fast_lcm_adapter(pipe)
        pipe = pipe.to(self.device_label)
        self._apply_runtime_optimizations(pipe)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        self._PIPELINE_CACHE[cache_key] = pipe
        return pipe

    def _apply_runtime_optimizations(self, pipe) -> None:
        if not str(self.device_label).startswith("cuda"):
            return
        if hasattr(self._torch, "set_float32_matmul_precision"):
            self._torch.set_float32_matmul_precision("high")
        matmul = getattr(getattr(getattr(self._torch, "backends", None), "cuda", None), "matmul", None)
        if matmul is not None and hasattr(matmul, "allow_tf32"):
            matmul.allow_tf32 = True
        channels_last = getattr(self._torch, "channels_last", None)
        if channels_last is not None and hasattr(pipe.unet, "to"):
            pipe.unet.to(memory_format=channels_last)
        if self.compile_enabled and hasattr(self._torch, "compile"):
            compile_kwargs = {"mode": self.compile_mode}
            if self.compile_backend:
                compile_kwargs["backend"] = self.compile_backend
            pipe.unet = self._torch.compile(pipe.unet, **compile_kwargs)

    def _load_fast_lcm_adapter(self, pipe) -> None:
        if not hasattr(pipe, "fuse_lora") or not hasattr(pipe, "load_lora_weights"):
            raise RuntimeError("V2-DIFF fast LCM mode requires a diffusers pipeline with LoRA fuse/load support")
        pipe.fuse_lora()
        if hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()
        pipe.load_lora_weights(
            str(self._bundle["fast_lcm_lora_repo_id"]),
            weight_name="pytorch_lora_weights.safetensors",
        )
        pipe.fuse_lora()
        if hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()

    def _scheduler_name(self) -> str:
        return "lcm" if self.fast_lcm else str(self._bundle.get("default_scheduler", "dpmpp_2m_karras"))

    def _make_scheduler(self):
        scheduler_name = self._scheduler_name()
        base_model = str(self._bundle["base_model"])
        classes = self._scheduler_classes
        if classes is None:
            diffusers = import_module("diffusers")
            classes = {
                "ddim": diffusers.DDIMScheduler,
                "dpmpp_2m": diffusers.DPMSolverMultistepScheduler,
                "dpmpp_2m_karras": diffusers.DPMSolverMultistepScheduler,
                "lcm": diffusers.LCMScheduler,
                "unipc": diffusers.UniPCMultistepScheduler,
            }
        if scheduler_name == "lcm":
            ddim = classes["ddim"].from_pretrained(base_model, subfolder="scheduler")
            return classes["lcm"].from_config(ddim.config)
        if scheduler_name == "ddim":
            return classes["ddim"].from_pretrained(base_model, subfolder="scheduler")
        if scheduler_name in {"dpmpp_2m", "dpmpp_2m_karras"}:
            return classes[scheduler_name].from_pretrained(
                base_model,
                subfolder="scheduler",
                algorithm_type="dpmsolver++",
                solver_order=2,
                use_karras_sigmas=scheduler_name == "dpmpp_2m_karras",
            )
        if scheduler_name == "unipc":
            return classes["unipc"].from_pretrained(base_model, subfolder="scheduler")
        raise ValueError(f"unsupported V2-DIFF scheduler: {scheduler_name}")

    def _tensor_dtype(self):
        if str(self.device_label).startswith("cuda") and bool(getattr(getattr(self._torch, "cuda", None), "is_bf16_supported", lambda: False)()):
            return self._torch.bfloat16
        if str(self.device_label).startswith("cuda"):
            return self._torch.float16
        return self._torch.float32

    def _resolve_device_label(self) -> str:
        if str(self.target_device).startswith("cuda") and bool(getattr(getattr(self._torch, "cuda", None), "is_available", lambda: False)()):
            return str(self.target_device)
        return "cpu"


def probe_shadow_v2_diff(bundle_path: str | Path, background_path: str | Path) -> RealAdapterProbe:
    bundle = Path(bundle_path)
    background = Path(background_path)
    missing_modules = [name for name in ("torch", "diffusers") if not module_available(name)]
    if missing_modules:
        return RealAdapterProbe("shadowgen-inpaint-lora", "unavailable", False, f"requires {', '.join(missing_modules)}")
    if not bundle.exists():
        return RealAdapterProbe("shadowgen-inpaint-lora", "unavailable", False, f"bundle not found: {bundle}")
    if not (bundle / "bundle_config.json").exists():
        return RealAdapterProbe("shadowgen-inpaint-lora", "unavailable", False, f"bundle_config.json not found in {bundle}")
    if not (bundle / "checkpoint").exists():
        return RealAdapterProbe("shadowgen-inpaint-lora", "unavailable", False, f"checkpoint directory not found in {bundle}")
    if not background.exists():
        return RealAdapterProbe("shadowgen-inpaint-lora", "unavailable", False, f"neutral background not found: {background}")
    try:
        config = _load_bundle_config(bundle)
    except Exception as exc:
        return RealAdapterProbe("shadowgen-inpaint-lora", "unavailable", False, f"failed to read bundle config: {exc}")
    return RealAdapterProbe(
        "shadowgen-inpaint-lora",
        str(config.get("experiment_name") or config.get("bundle_format") or "bundle-v1"),
        True,
        f"V2-DIFF local bundle available ({bundle})",
    )
