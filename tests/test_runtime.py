from __future__ import annotations

import unittest

from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.pipeline.runtime import build_runtime


class RuntimeTests(unittest.TestCase):
    def test_auto_runtime_uses_fallback_mode(self) -> None:
        runtime = build_runtime(Settings(runtime_mode="auto"))
        self.assertTrue(runtime.descriptor.degraded)
        self.assertIn("fallback", runtime.descriptor.mode)

    def test_mock_runtime_is_not_degraded(self) -> None:
        runtime = build_runtime(Settings(runtime_mode="mock"))
        self.assertFalse(runtime.descriptor.degraded)
        self.assertEqual(runtime.descriptor.mode, "mock")


if __name__ == "__main__":
    unittest.main()
