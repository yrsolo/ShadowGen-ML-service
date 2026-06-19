from __future__ import annotations

from math import cos, radians, sin, tan

from PIL import Image, ImageDraw


def draw_geometry_overlay(
    image: Image.Image,
    camera_fov: float,
    camera_pitch: float,
    camera_roll: float,
    confidence: float,
) -> Image.Image:
    canvas = image.convert("RGBA").copy()
    width, height = canvas.size
    center_x = width / 2
    center_y = height / 2

    roll_rad = radians(camera_roll)
    pitch_shift = tan(radians(camera_pitch)) * height * 0.2
    horizon_y = center_y + pitch_shift
    length = max(width, height) * 0.8
    dx = cos(roll_rad) * length
    dy = sin(roll_rad) * length

    grid_overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    _draw_floor_grid(
        grid_overlay,
        horizon_y=horizon_y,
        camera_fov=camera_fov,
        line_color=(42, 147, 213, 108),
        fill_color=(42, 147, 213, 30),
    )
    if abs(camera_roll) > 0.05:
        grid_overlay = grid_overlay.rotate(
            camera_roll,
            resample=Image.Resampling.BICUBIC,
            center=(center_x, center_y),
        )
    canvas = Image.alpha_composite(canvas, grid_overlay)

    draw = ImageDraw.Draw(canvas)
    horizon_color = (255, 122, 89, 255)
    axis_color = (29, 36, 51, 220)
    draw.line((center_x - dx, horizon_y - dy, center_x + dx, horizon_y + dy), fill=horizon_color, width=4)
    draw.line((center_x, 0, center_x, height), fill=axis_color, width=2)
    draw.ellipse((center_x - 6, center_y - 6, center_x + 6, center_y + 6), fill=horizon_color)
    box_top = max(14, height - 88)
    draw.rounded_rectangle((14, box_top, 336, height - 16), radius=14, fill=(255, 255, 255, 220))
    draw.text((24, box_top + 10), f"fov {camera_fov:.1f} deg", fill=axis_color)
    draw.text((24, box_top + 26), f"pitch {camera_pitch:.1f} deg", fill=axis_color)
    draw.text((24, box_top + 42), f"roll {camera_roll:.1f} deg  conf {confidence:.2f}", fill=axis_color)
    draw.text((24, box_top + 58), "synthetic floor grid", fill=(72, 98, 130, 255))
    return canvas


def draw_detection_overlay(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    confidence: float,
    *,
    label: str = "main object",
) -> Image.Image:
    canvas = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = bbox
    box_color = (255, 122, 89, 255)
    fill_color = (255, 122, 89, 42)
    info_bg = (255, 255, 255, 220)
    text_color = (29, 36, 51, 255)

    draw.rounded_rectangle((left, top, right, bottom), radius=18, outline=box_color, width=4, fill=fill_color)
    card_left = max(12, left)
    card_top = max(12, top - 58)
    card_right = min(canvas.width - 12, card_left + 220)
    card_bottom = min(canvas.height - 12, card_top + 46)
    draw.rounded_rectangle((card_left, card_top, card_right, card_bottom), radius=14, fill=info_bg)
    draw.text((card_left + 12, card_top + 10), label, fill=text_color)
    draw.text((card_left + 12, card_top + 25), f"conf {confidence:.3f}", fill=text_color)
    return canvas


def _draw_floor_grid(
    overlay: Image.Image,
    *,
    horizon_y: float,
    camera_fov: float,
    line_color: tuple[int, int, int, int],
    fill_color: tuple[int, int, int, int],
) -> None:
    width, height = overlay.size
    if horizon_y >= height - 12:
        return

    draw = ImageDraw.Draw(overlay)
    vanishing_x = width / 2
    visible_horizon_y = min(max(horizon_y, 12), height - 36)
    draw.polygon(
        [
            (0, height),
            (width, height),
            (width, visible_horizon_y),
            (0, visible_horizon_y),
        ],
        fill=fill_color,
    )

    span_scale = max(0.65, min(1.35, 70.0 / max(camera_fov, 1.0)))
    lane_count = 6
    half_bottom_span = width * 0.44 * span_scale
    edge_margin = width * 0.08

    for index in range(-lane_count, lane_count + 1):
        ratio = index / max(lane_count, 1)
        x_bottom = vanishing_x + ratio * half_bottom_span
        x_bottom = min(max(x_bottom, edge_margin), width - edge_margin)
        draw.line((x_bottom, height, vanishing_x, visible_horizon_y), fill=line_color, width=2)

    band_steps = 8
    for step in range(1, band_steps + 1):
        progress = step / (band_steps + 1)
        perspective = progress * progress
        y = height - (height - visible_horizon_y) * perspective
        left_x = edge_margin + (vanishing_x - edge_margin) * perspective
        right_x = (width - edge_margin) - ((width - edge_margin) - vanishing_x) * perspective
        draw.line((left_x, y, right_x, y), fill=line_color, width=2)
