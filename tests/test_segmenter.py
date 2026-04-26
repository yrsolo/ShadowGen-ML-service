from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from shadowgen_ml_service.adapters.real import RealSegmenter
from shadowgen_ml_service.core.stage_io import SegmentationInput
from shadowgen_ml_service.infrastructure.stages.segmentation.birefnet import apply_binary_mask_to_alpha


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTorch:
    class cuda:
        @staticmethod
        def is_available():
            return True

    @staticmethod
    def no_grad():
        return FakeNoGrad()


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)
        self.last_device = None

    @property
    def shape(self):
        return self.array.shape

    def unsqueeze(self, axis):
        return FakeTensor(np.expand_dims(self.array, axis))

    def sigmoid(self):
        return FakeTensor(1.0 / (1.0 + np.exp(-self.array)))

    def cpu(self):
        return self

    def detach(self):
        return self

    def numpy(self):
        return self.array

    def squeeze(self):
        return FakeTensor(np.squeeze(self.array))

    def __getitem__(self, item):
        return FakeTensor(self.array[item])

    def to(self, device=None, dtype=None):
        self.last_device = device
        return self


class FakeCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        value = image
        for transform in self.transforms:
            value = transform(value)
        return value


class FakeResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return image.resize((self.size[1], self.size[0]))


class FakeToTensor:
    def __call__(self, image):
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        return FakeTensor(array)


class FakeNormalize:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

    def __call__(self, tensor):
        return FakeTensor((tensor.array - self.mean) / self.std)


class FakeTransformsModule:
    Compose = FakeCompose
    Resize = FakeResize
    ToTensor = FakeToTensor
    Normalize = FakeNormalize


class FakeSegmentationModel:
    last_from_pretrained = None

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.last_from_pretrained = (model_id, kwargs)
        return cls()

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def __call__(self, image_tensor):
        _batch, _channels, height, width = image_tensor.shape
        matte = np.zeros((1, 1, height, width), dtype=np.float32)
        matte[:, :, 8:-8, 10:-10] = 8.0
        return [FakeTensor(matte)]


class FakeTransformersSegmentationModule:
    AutoImageProcessor = None
    AutoModelForImageSegmentation = FakeSegmentationModel


class SegmenterTests(unittest.TestCase):
    def test_binary_mask_removes_alpha_leaks_outside_main_component(self) -> None:
        alpha = np.array(
            [
                [0, 240, 230, 0],
                [0, 220, 210, 0],
                [12, 15, 0, 0],
            ],
            dtype=np.uint8,
        )
        binary_mask = np.array(
            [
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )

        cleaned = apply_binary_mask_to_alpha(alpha, binary_mask)

        self.assertEqual(int(cleaned[2, 0]), 0)
        self.assertEqual(int(cleaned[2, 1]), 0)
        self.assertEqual(int(cleaned[0, 1]), 240)

    def test_real_segmenter_returns_mask_and_cutout(self) -> None:
        segmenter = RealSegmenter(
            transformers_module=FakeTransformersSegmentationModule,
            torch_module=FakeTorch(),
            transforms_module=FakeTransformsModule,
            model_id="ZhengPeng7/BiRefNet_lite-matting",
            resolution=64,
            mask_threshold=0.5,
            target_device="cuda:0",
        )
        image = Image.new("RGBA", (96, 80), (255, 255, 255, 255))
        result = segmenter.segment(SegmentationInput(image=image))
        self.assertEqual(result.mask.size, image.size)
        self.assertEqual(result.cutout_rgba.size, image.size)
        self.assertEqual(result.crop_rgba.size, image.size)
        self.assertEqual(result.bbox, (0, 0, image.width, image.height))
        self.assertEqual(FakeSegmentationModel.last_from_pretrained[0], "ZhengPeng7/BiRefNet_lite-matting")
        self.assertEqual(segmenter.device_label, "cuda:0")
        self.assertEqual(segmenter._model.device, "cuda:0")


if __name__ == "__main__":
    unittest.main()
