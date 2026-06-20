from __future__ import annotations

import numpy as np
from PIL import Image


def estimate_foreground_mask(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.int16)
    corners = np.asarray([arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]], dtype=np.float32)
    background = corners.mean(axis=0)
    delta = np.linalg.norm(arr - background, axis=2)
    mask = delta > 30
    if int(mask.sum()) < max(64, arr.shape[0] * arr.shape[1] // 100):
        yy, xx = np.indices(mask.shape)
        cy = arr.shape[0] / 2
        cx = arr.shape[1] / 2
        rx = max(arr.shape[1] * 0.28, 1)
        ry = max(arr.shape[0] * 0.28, 1)
        mask = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1
    return Image.fromarray(mask.astype(np.uint8) * 255)


def bbox_from_mask(mask: Image.Image, padding_px: int) -> tuple[int, int, int, int]:
    arr = np.asarray(mask, dtype=np.uint8)
    coords = np.argwhere(arr > 0)
    if coords.size == 0:
        width, height = mask.size
        return (0, 0, width, height)
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)
    width, height = mask.size
    return (
        max(0, int(min_x) - padding_px),
        max(0, int(min_y) - padding_px),
        min(width, int(max_x) + padding_px + 1),
        min(height, int(max_y) + padding_px + 1),
    )


def crop_to_bbox(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    return image.crop(bbox)


def prepare_working_crop(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    size: int,
    *,
    content_scale: float = 1.0,
) -> Image.Image:
    crop = crop_to_bbox(image, bbox)
    canvas, _ = fit_to_square(crop, Image.new("L", crop.size, 255), size, content_scale=content_scale)
    return canvas


def fit_to_square(image: Image.Image, mask: Image.Image, size: int, content_scale: float = 1.0) -> tuple[Image.Image, Image.Image]:
    content_scale = max(0.1, min(content_scale, 1.0))
    target_size = max(1, int(round(size * content_scale)))
    ratio = min(target_size / image.width, target_size / image.height)
    new_width = max(1, int(image.width * ratio))
    new_height = max(1, int(image.height * ratio))
    resized_rgba = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    resized_mask = mask.resize((new_width, new_height), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    mask_canvas = Image.new("L", (size, size), 0)
    offset = ((size - new_width) // 2, (size - new_height) // 2)
    canvas.alpha_composite(resized_rgba, offset)
    mask_canvas.paste(resized_mask, offset)
    return canvas, mask_canvas


def create_cutout(image: Image.Image, mask: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA").copy()
    rgba.putalpha(mask)
    return rgba


def depth_from_mask(mask: Image.Image) -> Image.Image:
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    height = arr.shape[0]
    gradient = np.linspace(0.2, 1.0, height, dtype=np.float32)[:, None]
    depth = np.clip(arr * gradient, 0.0, 1.0)
    return Image.fromarray((depth * 255).astype(np.uint8))
