from __future__ import annotations

import numpy as np
from PIL import Image


def normals_from_depth(depth_map: Image.Image) -> Image.Image:
    import cv2

    depth_u8 = np.asarray(depth_map.convert("L"), dtype=np.uint8)
    valid_mask = (depth_u8 > 0).astype(np.uint8)
    if valid_mask.sum() == 0:
        return Image.new("RGB", depth_map.size, (127, 127, 255))

    depth = depth_u8.astype(np.float32) / 255.0
    valid = valid_mask.astype(np.float32)
    blurred_depth = cv2.GaussianBlur(depth * valid, (0, 0), sigmaX=2.2, sigmaY=2.2)
    blurred_weight = cv2.GaussianBlur(valid, (0, 0), sigmaX=2.2, sigmaY=2.2)
    smoothed_depth = blurred_depth / np.maximum(blurred_weight, 1e-4)
    smoothed_depth = np.where(valid_mask > 0, smoothed_depth, 0.0).astype(np.float32)

    grad_x = cv2.Sobel(smoothed_depth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smoothed_depth, cv2.CV_32F, 0, 1, ksize=3)
    interior_mask = cv2.erode(valid_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
    interior_weight = cv2.GaussianBlur(interior_mask.astype(np.float32), (0, 0), sigmaX=1.2, sigmaY=1.2)
    interior_weight = np.clip(interior_weight, 0.0, 1.0)

    gradient_strength = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    active_gradients = gradient_strength[interior_mask > 0]
    if active_gradients.size > 0:
        scale = max(float(np.percentile(active_gradients, 90)), 1e-3)
    else:
        scale = 1e-3
    slope_gain = 2.4 / scale
    nx = -grad_x * slope_gain * interior_weight
    ny = -grad_y * slope_gain * interior_weight
    nz = np.ones_like(smoothed_depth)

    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    normal = np.stack(
        [
            (nx / norm + 1.0) * 0.5,
            (ny / norm + 1.0) * 0.5,
            (nz / norm + 1.0) * 0.5,
        ],
        axis=-1,
    )

    neutral = np.zeros_like(normal)
    neutral[..., 0] = 0.5
    neutral[..., 1] = 0.5
    neutral[..., 2] = 1.0
    blend_weight = cv2.GaussianBlur(valid_mask.astype(np.float32), (0, 0), sigmaX=0.8, sigmaY=0.8)
    blend_weight = np.clip(blend_weight, 0.0, 1.0)[..., None]
    normal = neutral * (1.0 - blend_weight) + normal * blend_weight
    return Image.fromarray((normal * 255).clip(0, 255).astype(np.uint8), mode="RGB")
