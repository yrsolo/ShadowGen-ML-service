from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def normalize_depth_map(predicted_depth: Any, output_size: tuple[int, int], mask: Image.Image | None = None) -> Image.Image:
    depth = predicted_depth
    if hasattr(depth, "detach"):
        depth = depth.detach()
    if hasattr(depth, "cpu"):
        depth = depth.cpu()
    if hasattr(depth, "numpy"):
        depth_array = depth.numpy()
    else:
        depth_array = np.asarray(depth)

    depth_array = np.asarray(depth_array, dtype=np.float32)
    while depth_array.ndim > 2 and depth_array.shape[0] == 1:
        depth_array = depth_array[0]
    if depth_array.ndim == 3 and depth_array.shape[-1] == 1:
        depth_array = depth_array[:, :, 0]
    if depth_array.ndim != 2:
        raise ValueError(f"depth tensor must resolve to a 2D map, got shape {tuple(depth_array.shape)}")

    depth_array = np.asarray(
        Image.fromarray(depth_array, mode="F").resize(output_size, Image.Resampling.BILINEAR),
        dtype=np.float32,
    )

    valid_mask = None
    if mask is not None:
        mask_array = np.asarray(mask.convert("L").resize(output_size, Image.Resampling.BILINEAR), dtype=np.uint8)
        valid_mask = mask_array > 0
        values = depth_array[valid_mask]
    else:
        values = depth_array.reshape(-1)
    if values.size == 0:
        return Image.new("L", output_size, 0)

    min_value = float(values.min())
    max_value = float(values.max())
    if max_value - min_value < 1e-6:
        normalized = np.zeros_like(depth_array, dtype=np.float32)
    else:
        normalized = (depth_array - min_value) / (max_value - min_value)
    normalized = np.clip(normalized, 0.0, 1.0)
    if valid_mask is not None:
        normalized = np.where(valid_mask, normalized, 0.0)
    return Image.fromarray((normalized * 255.0).clip(0, 255).astype(np.uint8), mode="L")
