from __future__ import annotations

import base64
import binascii
from io import BytesIO
from pathlib import Path

from PIL import Image

from shadowgen_ml_service.core.assets import RasterAsset


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


def pil_to_asset(image: Image.Image, *, mime_type: str = "image/png") -> RasterAsset:
    pil_format = MIME_TO_PIL.get(mime_type, "PNG")
    buffer = BytesIO()
    save_image = image.convert("RGB") if pil_format == "JPEG" else image
    save_image.save(buffer, format=pil_format)
    return RasterAsset(
        mime_type=mime_type,
        mode=image.mode,
        width=image.width,
        height=image.height,
        data=buffer.getvalue(),
    )


def asset_to_pil(asset: RasterAsset) -> Image.Image:
    image = Image.open(BytesIO(asset.data))
    image.load()
    if image.mode != asset.mode:
        image = image.convert(asset.mode)
    return image


def ensure_asset(image_or_asset: Image.Image | RasterAsset, *, mime_type: str = "image/png") -> RasterAsset:
    if isinstance(image_or_asset, RasterAsset):
        return image_or_asset
    return pil_to_asset(image_or_asset, mime_type=mime_type)


def ensure_pil(image_or_asset: Image.Image | RasterAsset) -> Image.Image:
    if isinstance(image_or_asset, RasterAsset):
        return asset_to_pil(image_or_asset)
    return image_or_asset


def asset_from_file(path: str | BytesIO | Path, *, mime_type: str = "image/png") -> RasterAsset:
    if hasattr(path, "read"):
        data = path.read()
    else:
        data = Path(path).read_bytes()
    image = Image.open(BytesIO(data))
    image.load()
    return RasterAsset(
        mime_type=mime_type,
        mode=image.mode,
        width=image.width,
        height=image.height,
        data=data,
    )


def asset_to_base64(asset: RasterAsset) -> str:
    return base64.b64encode(asset.data).decode("ascii")


def alpha_asset(asset: RasterAsset) -> RasterAsset:
    return pil_to_asset(ensure_pil(asset).getchannel("A"), mime_type="image/png")
