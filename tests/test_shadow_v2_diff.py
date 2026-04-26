from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from shadowgen_ml_service.core.stage_io import ShadowInput
from shadowgen_ml_service.infrastructure.stages.shadow.v2_diff import V2DiffShadowGenerator, probe_shadow_v2_diff
from shadowgen_ml_service.utils.images import pil_to_asset


class FakeGenerator:
    def __init__(self, device: str) -> None:
        self.device = device
        self.seed = None

    def manual_seed(self, seed: int):
        self.seed = seed
        return self


class FakeTorch:
    float16 = "float16"
    float32 = "float32"
    bfloat16 = "bfloat16"

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def is_bf16_supported() -> bool:
            return False

    Generator = FakeGenerator


class FakeScheduler:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return {"args": args, "kwargs": kwargs}


class FakeUnet:
    def __init__(self) -> None:
        self.loaded_path = None

    def load_attn_procs(self, path: Path) -> None:
        self.loaded_path = path


class FakePipeline:
    last_instance = None

    def __init__(self) -> None:
        self.unet = FakeUnet()
        self.scheduler = None
        self.device = None
        self.calls = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.last_instance = cls()
        cls.last_instance.from_pretrained_args = args
        cls.last_instance.from_pretrained_kwargs = kwargs
        return cls.last_instance

    def to(self, device: str):
        self.device = device
        return self

    def set_progress_bar_config(self, **kwargs) -> None:
        self.progress_bar_config = kwargs

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        output = Image.new("RGB", kwargs["image"].size, (120, 121, 122))
        return type("Result", (), {"images": [output]})()


class V2DiffShadowTests(unittest.TestCase):
    def test_generate_uses_bundle_mask_and_returns_full_shadow_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "bundle"
            checkpoint = bundle / "checkpoint"
            checkpoint.mkdir(parents=True)
            (bundle / "bundle_config.json").write_text(
                json.dumps(
                    {
                        "base_model": "base-model",
                        "default_scheduler": "dpmpp_2m_karras",
                        "default_prompt_base": "soft shadow",
                        "default_negative_prompt": "bad",
                        "default_num_inference_steps": 7,
                        "default_guidance_scale": 1.5,
                        "view_buckets": [{"max_camera_elevation_deg": None, "phrase": "top view"}],
                    }
                ),
                encoding="utf-8",
            )
            background = root / "background.png"
            Image.new("RGB", (16, 16), (196, 196, 196)).save(background)
            cutout = Image.new("RGBA", (8, 8), (255, 0, 0, 0))
            mask = Image.new("L", (8, 8), 0)
            for x in range(2, 6):
                for y in range(2, 6):
                    cutout.putpixel((x, y), (255, 0, 0, 255))
                    mask.putpixel((x, y), 255)
            asset = pil_to_asset(cutout)
            mask_asset = pil_to_asset(mask)
            generator = V2DiffShadowGenerator(
                bundle_path=bundle,
                background_path=background,
                target_device="cpu",
                torch_module=FakeTorch(),
                pipeline_cls=FakePipeline,
                scheduler_classes={"dpmpp_2m_karras": FakeScheduler},
            )

            result = generator.generate(
                ShadowInput(
                    img=asset,
                    mask=mask_asset,
                    depth=mask_asset,
                    normal=asset,
                    angle=0,
                    elevation=45,
                    softness=0,
                    reflection=0,
                    opacity=1,
                )
            )

        self.assertEqual(result.shadow_image.width, 8)
        self.assertEqual(result.shadow_image.height, 8)
        pipe = FakePipeline.last_instance
        self.assertEqual(pipe.device, "cpu")
        self.assertEqual(pipe.calls[0]["num_inference_steps"], 7)
        self.assertEqual(pipe.calls[0]["guidance_scale"], 1.5)
        self.assertEqual(pipe.calls[0]["mask_image"].mode, "L")

    def test_probe_reports_missing_bundle(self) -> None:
        probe = probe_shadow_v2_diff("missing-bundle", "missing-background.png")
        self.assertFalse(probe.available)


if __name__ == "__main__":
    unittest.main()
