from __future__ import annotations

import os
from unittest.mock import patch
import unittest

from shadowgen_ml_service.config import Settings, get_settings


class SettingsTests(unittest.TestCase):
    def test_settings_read_environment_at_instantiation_time(self) -> None:
        with patch.dict(os.environ, {"SHADOWGEN_WORKING_SIZE": "640"}, clear=False):
            first = Settings()
        with patch.dict(os.environ, {"SHADOWGEN_WORKING_SIZE": "768"}, clear=False):
            second = Settings()

        self.assertEqual(first.working_size, 640)
        self.assertEqual(second.working_size, 768)

    def test_get_settings_does_not_cache_environment_values(self) -> None:
        with patch.dict(os.environ, {"SHADOWGEN_SERVICE_VERSION": "test-a"}, clear=False):
            first = get_settings()
        with patch.dict(os.environ, {"SHADOWGEN_SERVICE_VERSION": "test-b"}, clear=False):
            second = get_settings()

        self.assertEqual(first.service_version, "test-a")
        self.assertEqual(second.service_version, "test-b")

    def test_dev_api_flags_default_to_disabled(self) -> None:
        settings = Settings()

        self.assertFalse(settings.dev_api_enabled)
        self.assertFalse(settings.dev_shutdown_enabled)


if __name__ == "__main__":
    unittest.main()
