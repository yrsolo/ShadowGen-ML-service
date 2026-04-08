from __future__ import annotations

from PIL import Image

from shadowgen_ml_service.core.contracts import ArtifactEncoder
from shadowgen_ml_service.core.models import EncodedArtifact
from shadowgen_ml_service.utils.images import encode_image


class DefaultArtifactEncoder(ArtifactEncoder):
    def encode(
        self,
        final_image: Image.Image,
        output_format: str,
        debug_images: dict[str, Image.Image],
        return_debug: bool,
    ) -> list[EncodedArtifact]:
        artifacts: list[EncodedArtifact] = []
        mime_type, image_base64 = encode_image(final_image, output_format)
        artifacts.append(EncodedArtifact(name="final", kind="final", mime_type=mime_type, image_base64=image_base64))
        if return_debug:
            for name, image in debug_images.items():
                debug_mime_type, debug_base64 = encode_image(image, "png")
                artifacts.append(EncodedArtifact(name=name, kind="debug", mime_type=debug_mime_type, image_base64=debug_base64))
        return artifacts
