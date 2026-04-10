from __future__ import annotations

from io import BytesIO
import base64

import numpy as np
from PIL import Image


def image_to_base64_png(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def image_to_nhwc_uint8(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.uint8)


def base64_png_to_image(value: str, mode: str | None = None) -> Image.Image:
    raw = base64.b64decode(value.encode("ascii"))
    image = Image.open(BytesIO(raw))
    return image if mode is None else image.convert(mode)
