from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from PIL import Image, ImageFilter

from shadowgen_ml_service.core.commands import ShadowSpec
from shadowgen_ml_service.core.contracts import ShadowGenerator
from shadowgen_ml_service.core.models import GeometryResult, ShadowResult
from shadowgen_ml_service.infrastructure.stages.shared.model_support import RealAdapterProbe, import_module, module_available


def _to_numpy_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _to_numpy_mask(mask: Image.Image) -> np.ndarray:
    mask_array = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    return np.repeat(mask_array[:, :, None], 3, axis=2)


def _compose_input(cutout_rgba: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.asarray(cutout_rgba.getchannel("A"), dtype=np.float32) / 255.0
    alpha_rgb = np.repeat(alpha[:, :, None], 3, axis=2)
    rgb = _to_numpy_rgb(cutout_rgba)
    white = np.ones_like(rgb, dtype=np.float32)
    rgb_on_white = white * (1.0 - alpha_rgb) + rgb * alpha_rgb
    return rgb_on_white, alpha_rgb


def _extract_shadow_rgba(predicted_rgb: np.ndarray, foreground_mask_rgb: np.ndarray, shadow: ShadowSpec) -> Image.Image:
    background_mask = np.clip(1.0 - foreground_mask_rgb[:, :, 0], 0.0, 1.0)
    grayscale = predicted_rgb.mean(axis=2)
    shadow_strength = np.clip(1.0 - grayscale, 0.0, 1.0) * background_mask
    shadow_alpha = np.clip(shadow_strength * float(shadow.opacity), 0.0, 1.0)
    alpha_image = Image.fromarray((shadow_alpha * 255.0).astype(np.uint8), mode="L")
    if shadow.softness > 0:
        alpha_image = alpha_image.filter(ImageFilter.GaussianBlur(radius=max(0.5, float(shadow.softness) * 10.0)))
    shadow_rgba = Image.new("RGBA", alpha_image.size, (0, 0, 0, 0))
    shadow_rgba.putalpha(alpha_image)
    return shadow_rgba


def build_generator(torch_module: Any) -> Any:
    nn_module = import_module("torch.nn")

    class Block(nn_module.Module):
        def __init__(self, in_channels: int, out_channels: int, dropout: bool = False, norm: bool = True, relu: bool = True):
            super().__init__()
            layers = [nn_module.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)]
            if norm:
                layers.append(nn_module.BatchNorm2d(out_channels))
            if relu:
                layers.append(nn_module.ReLU(inplace=True))
            if dropout:
                layers.append(nn_module.Dropout(0.5))
            self.block = nn_module.Sequential(*layers)

        def forward(self, x):
            return self.block(x)

    class Generator(nn_module.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = Block(6, 64, norm=False)
            self.enc2 = Block(64, 128)
            self.enc3 = Block(128, 256)
            self.enc4 = Block(256, 256)
            self.enc5 = Block(256, 256)
            self.enc6 = Block(256, 256, dropout=True)
            self.downsample = nn_module.MaxPool2d(2)
            self.dec6 = Block(256, 256)
            self.dec5 = Block(256 * 2, 256)
            self.dec4 = Block(256 * 2, 256)
            self.dec3 = Block(256 * 2, 128)
            self.dec2 = Block(128 * 2, 64)
            self.dec1 = Block(64 * 2, 3, norm=False, relu=False)
            self.upsample = nn_module.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        def forward(self, x):
            enc1 = self.enc1(x)
            enc2 = self.enc2(self.downsample(enc1))
            enc3 = self.enc3(self.downsample(enc2))
            enc4 = self.enc4(self.downsample(enc3))
            enc5 = self.enc5(self.downsample(enc4))
            enc6 = self.enc6(self.downsample(enc5))
            dec6 = self.dec6(enc6)
            dec5 = self.dec5(torch_module.cat([enc5, self.upsample(dec6)], dim=1))
            dec4 = self.dec4(torch_module.cat([enc4, self.upsample(dec5)], dim=1))
            dec3 = self.dec3(torch_module.cat([enc3, self.upsample(dec4)], dim=1))
            dec2 = self.dec2(torch_module.cat([enc2, self.upsample(dec3)], dim=1))
            dec1 = self.dec1(torch_module.cat([enc1, self.upsample(dec2)], dim=1))
            return torch_module.tanh(dec1)

    return Generator()


class Pix2PixShadowGenerator(ShadowGenerator):
    _RESOURCE_CACHE: ClassVar[dict[tuple[str, str], Any]] = {}

    def __init__(
        self,
        torch_module: Any | None = None,
        *,
        weights_path: str | Path,
        target_device: str = "cuda",
    ) -> None:
        self._torch = torch_module or import_module("torch")
        self.weights_path = Path(weights_path)
        self.target_device = target_device
        self.backend_name = "pix2pix-shadow"
        self.device_label = self._resolve_device_label()
        cache_key = (str(self.weights_path.resolve()), self.device_label)
        if torch_module is not None:
            self._generator = self._load_generator()
        else:
            cached = self._RESOURCE_CACHE.get(cache_key)
            if cached is None:
                cached = self._load_generator()
                self._RESOURCE_CACHE[cache_key] = cached
            self._generator = cached
        self.device_label = self._infer_device_label()

    def generate(
        self,
        cutout_rgba: Image.Image,
        mask: Image.Image,
        depth_map: Image.Image,
        normal_map: Image.Image,
        geometry: GeometryResult,
        shadow: ShadowSpec,
    ) -> ShadowResult:
        colors_on_white, foreground_mask = _compose_input(cutout_rgba)
        input_tensor = self._build_input_tensor(colors_on_white, foreground_mask, shadow.angle_deg)
        with self._torch.inference_mode():
            predicted = self._generator(input_tensor)[0]
        predicted_rgb = predicted.detach().float().cpu().permute(1, 2, 0).numpy()
        predicted_rgb = np.clip(predicted_rgb, 0.0, 1.0)
        shadow_rgba = _extract_shadow_rgba(predicted_rgb, foreground_mask, shadow)
        return ShadowResult(shadow_rgba=shadow_rgba)

    def _build_input_tensor(self, colors_on_white: np.ndarray, foreground_mask: np.ndarray, angle_deg: float):
        background_mask = 1.0 - foreground_mask
        rotate_angle = np.deg2rad(float(angle_deg))
        rot_plane = np.ones((colors_on_white.shape[0], colors_on_white.shape[1], 2), dtype=np.float32)
        rot_plane[:, :, 0] *= np.sin(rotate_angle)
        rot_plane[:, :, 1] *= np.cos(rotate_angle)
        stacked = np.concatenate([colors_on_white, background_mask[:, :, :1], rot_plane], axis=2)
        tensor = self._torch.tensor(stacked, dtype=self._tensor_dtype()).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device_label)

    def _load_generator(self):
        if not self.weights_path.exists():
            raise FileNotFoundError(f"shadow generator weights not found: {self.weights_path}")
        generator = build_generator(self._torch)
        averaged = self._torch.optim.swa_utils.AveragedModel(
            generator,
            multi_avg_fn=self._torch.optim.swa_utils.get_ema_multi_avg_fn(0.999),
        )
        state_dict = self._torch.load(self.weights_path, map_location=self.device_label, weights_only=True)
        averaged.load_state_dict(state_dict)
        averaged.eval()
        averaged.to(device=self.device_label, dtype=self._tensor_dtype())
        return averaged

    def _tensor_dtype(self):
        return self._torch.float16 if str(self.device_label).startswith("cuda") else self._torch.float32

    def _infer_device_label(self) -> str:
        try:
            return str(next(self._generator.parameters()).device)
        except Exception:
            return "cpu"

    def _resolve_device_label(self) -> str:
        if str(self.target_device).startswith("cuda") and bool(getattr(getattr(self._torch, "cuda", None), "is_available", lambda: False)()):
            return str(self.target_device)
        return "cpu"


def probe_shadow_pix2pix(weights_path: str | Path, *, target_device: str = "cuda") -> RealAdapterProbe:
    if not module_available("torch"):
        return RealAdapterProbe("legacy-shadow-pix2pix", "bootstrap-probe", False, "requires torch")
    weights = Path(weights_path)
    if not weights.exists():
        return RealAdapterProbe("legacy-shadow-pix2pix", "bootstrap-probe", False, f"weights not found: {weights}")
    torch_module = import_module("torch")
    has_cuda = bool(getattr(getattr(torch_module, "cuda", None), "is_available", lambda: False)())
    detail = "CUDA runtime detected" if has_cuda and str(target_device).startswith("cuda") else "running on cpu"
    return RealAdapterProbe("legacy-shadow-pix2pix", "bootstrap-probe", True, detail)
