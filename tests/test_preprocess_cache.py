from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from shadowgen_ml_service.infrastructure.cache.preprocess_cache_repository import FilesystemPreprocessCacheRepository


class PreprocessCacheTests(unittest.TestCase):
    def test_cache_key_changes_with_working_content_scale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = FilesystemPreprocessCacheRepository(Path(tmp))

            tight = cache.make_key(
                raw_bytes=b"image",
                runtime_signature="runtime",
                padding_px=100,
                working_size=512,
                working_content_scale=0.82,
            )
            airy = cache.make_key(
                raw_bytes=b"image",
                runtime_signature="runtime",
                padding_px=100,
                working_size=512,
                working_content_scale=0.68,
            )

        self.assertNotEqual(tight, airy)


if __name__ == "__main__":
    unittest.main()
