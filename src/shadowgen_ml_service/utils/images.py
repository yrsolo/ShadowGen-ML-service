from __future__ import annotations

import base64
import binascii
from io import BytesIO
from math import cos, radians, sin

import numpy as np
from PIL import Image, ImageChops, ImageColor, ImageFilter, ImageOps


MIME_TO_PIL = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}


def decode_image(image_base64: str, mime_type: str, max_image_bytes: int) -> tuple[bytes, Image.Image]:
    try:
        raw = base64.b64decode(image_base64, validate=True)
    except binascii.Error as exc:
        raise ValueError("source.image_base64 must be valid base64") from exc
    if len(raw) > max_image_bytes:
        raise ValueError(f"source payload exceeds {max_image_bytes} bytes")

    try:
        image = Image.open(BytesIO(raw))
        image.load()
    except Exception as exc:  # pragma: no cover
        raise ValueError("source image payload could not be decoded") from exc

    expected = MIME_TO_PIL[mime_type]
    actual = (image.format or "").upper()
    if actual and actual != expected:
        raise ValueError(f"source mime_type {mime_type} does not match decoded format {actual}")
    return raw, image.convert("RGBA")


def encode_image(image: Image.Image, output_format: str) -> tuple[str, str]:
    pil_format = output_format.upper()
    mime_type = f"image/{output_format.lower()}"
    buffer = BytesIO()
    save_image = image.convert("RGB") if pil_format == "JPEG" else image
    save_image.save(buffer, format=pil_format)
    return mime_type, base64.b64encode(buffer.getvalue()).decode("ascii")


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
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


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


def prepare_working_crop(image: Image.Image, bbox: tuple[int, int, int, int], size: int) -> Image.Image:
    crop = crop_to_bbox(image, bbox)
    canvas, _ = fit_to_square(crop, Image.new("L", crop.size, 255), size)
    return canvas


def fit_to_square(image: Image.Image, mask: Image.Image, size: int) -> tuple[Image.Image, Image.Image]:
    ratio = min(size / image.width, size / image.height)
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
    return Image.fromarray((depth * 255).astype(np.uint8), mode="L")


def normals_from_depth(depth_map: Image.Image) -> Image.Image:
    arr = np.asarray(depth_map, dtype=np.float32) / 255.0
    dy, dx = np.gradient(arr)
    nx = -dx
    ny = -dy
    nz = np.ones_like(arr)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    normal = np.stack(
        [
            (nx / norm + 1.0) * 0.5,
            (ny / norm + 1.0) * 0.5,
            (nz / norm + 1.0) * 0.5,
        ],
        axis=-1,
    )
    return Image.fromarray((normal * 255).astype(np.uint8), mode="RGB")


def generate_shadow_layer(
    mask: Image.Image,
    angle_deg: float,
    elevation_deg: float,
    softness: float,
    opacity: float,
    reflection: float,
    camera_pitch: float,
) -> Image.Image:
    size = max(mask.size)
    length = max(12.0, (1.0 - (elevation_deg / 90.0)) * size * 0.35)
    angle = radians(angle_deg)
    offset_x = int(cos(angle) * length)
    offset_y = int(sin(angle) * length)
    scale_y = max(0.35, 1.0 - (elevation_deg / 180.0) - abs(camera_pitch) / 180.0)

    stretched = mask.resize((mask.width, max(1, int(mask.height * scale_y))), Image.Resampling.BILINEAR)
    shadow_mask = Image.new("L", mask.size, 0)
    paste_y = min(max(mask.height - stretched.height, 0), mask.height)
    shadow_mask.paste(stretched, (0, paste_y))
    shadow_mask = ImageChops.offset(shadow_mask, offset_x, offset_y)
    shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(radius=1.5 + softness * 22.0))
    shadow_alpha = shadow_mask.point(lambda value: int(value * opacity))

    shadow_rgba = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    shadow_rgba.putalpha(shadow_alpha)

    if reflection > 0:
        reflection_mask = ImageOps.flip(mask).filter(ImageFilter.GaussianBlur(radius=6 + softness * 8))
        reflection_mask = reflection_mask.point(lambda value: int(value * reflection * 0.35))
        reflection_rgba = Image.new("RGBA", mask.size, (90, 110, 140, 0))
        reflection_rgba.putalpha(reflection_mask)
        shadow_rgba = Image.alpha_composite(shadow_rgba, reflection_rgba)
    return shadow_rgba


def compose_on_background(
    cutout_rgba: Image.Image,
    shadow_rgba: Image.Image,
    color_hex: str,
    width: int | None,
    height: int | None,
) -> Image.Image:
    background = Image.new("RGBA", cutout_rgba.size, ImageColor.getrgb(color_hex) + (255,))
    background.alpha_composite(shadow_rgba)
    background.alpha_composite(cutout_rgba)
    final_image = background.convert("RGBA")
    if width is None and height is None:
        return final_image
    if width is None:
        ratio = height / final_image.height
        width = max(1, int(final_image.width * ratio))
    elif height is None:
        ratio = width / final_image.width
        height = max(1, int(final_image.height * ratio))
    return final_image.resize((width, height), Image.Resampling.LANCZOS)
