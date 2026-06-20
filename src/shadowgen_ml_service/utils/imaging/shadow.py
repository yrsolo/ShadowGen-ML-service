from __future__ import annotations

from math import cos, radians, sin

from PIL import Image, ImageChops, ImageColor, ImageFilter, ImageOps


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

    shadow_image = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    shadow_image.putalpha(shadow_alpha)

    if reflection > 0:
        reflection_mask = ImageOps.flip(mask).filter(ImageFilter.GaussianBlur(radius=6 + softness * 8))
        reflection_mask = reflection_mask.point(lambda value: int(value * reflection * 0.35))
        reflection_image = Image.new("RGBA", mask.size, (90, 110, 140, 0))
        reflection_image.putalpha(reflection_mask)
        shadow_image = Image.alpha_composite(shadow_image, reflection_image)
    return shadow_image


def compose_on_background(
    cutout_rgba: Image.Image,
    shadow_image: Image.Image,
    color_hex: str,
    width: int | None,
    height: int | None,
) -> Image.Image:
    background = Image.new("RGBA", cutout_rgba.size, ImageColor.getrgb(color_hex) + (255,))
    background.alpha_composite(shadow_image.convert("RGBA"))
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
