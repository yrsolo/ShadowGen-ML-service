from __future__ import annotations

import unittest

from shadowgen_ml_service.infrastructure.stages.segmentation.torch_runtime import (
    TorchCompileConfig,
    configure_torch_runtime,
    maybe_compile_model,
)


class _FakeTorch:
    def __init__(self, *, with_compile: bool = True, compile_raises: bool = False) -> None:
        self.matmul_precision_calls: list[str] = []
        self.compile_calls: list[dict[str, object]] = []
        if with_compile:
            self.compile = self._compile  # type: ignore[attr-defined]
        self._compile_raises = compile_raises

    def set_float32_matmul_precision(self, value: str) -> None:
        self.matmul_precision_calls.append(value)

    def _compile(self, model, **kwargs):
        self.compile_calls.append(kwargs)
        if self._compile_raises:
            raise RuntimeError("boom")
        return {"compiled_model": model, "kwargs": kwargs}


class TorchRuntimeTests(unittest.TestCase):
    def test_configure_torch_runtime_sets_requested_precision_when_supported(self) -> None:
        fake_torch = _FakeTorch()

        configure_torch_runtime(fake_torch, matmul_precision="high")

        self.assertEqual(fake_torch.matmul_precision_calls, ["high"])

    def test_maybe_compile_model_skips_when_compile_disabled(self) -> None:
        fake_torch = _FakeTorch()

        compiled_model, status = maybe_compile_model(
            "model",
            fake_torch,
            device_label="cuda:0",
            config=TorchCompileConfig(enabled=False),
        )

        self.assertEqual(compiled_model, "model")
        self.assertEqual(status, "disabled")
        self.assertEqual(fake_torch.compile_calls, [])

    def test_maybe_compile_model_uses_torch_compile_on_cuda(self) -> None:
        fake_torch = _FakeTorch()

        compiled_model, status = maybe_compile_model(
            "model",
            fake_torch,
            device_label="cuda:0",
            config=TorchCompileConfig(enabled=True, mode="reduce-overhead", backend="inductor"),
        )

        self.assertEqual(status, "compiled")
        self.assertEqual(fake_torch.compile_calls, [{"mode": "reduce-overhead", "backend": "inductor"}])
        self.assertEqual(compiled_model["compiled_model"], "model")

    def test_maybe_compile_model_returns_original_model_when_compile_fails(self) -> None:
        fake_torch = _FakeTorch(compile_raises=True)

        compiled_model, status = maybe_compile_model(
            "model",
            fake_torch,
            device_label="cuda:0",
            config=TorchCompileConfig(enabled=True),
        )

        self.assertEqual(compiled_model, "model")
        self.assertTrue(status.startswith("compile-failed:"))


if __name__ == "__main__":
    unittest.main()
