from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TorchCompileConfig:
    enabled: bool = False
    mode: str = "reduce-overhead"
    backend: str | None = None
    matmul_precision: str = "high"


def configure_torch_runtime(torch_module: Any, *, matmul_precision: str = "high") -> None:
    if not matmul_precision:
        return
    setter = getattr(torch_module, "set_float32_matmul_precision", None)
    if setter is None:
        return
    try:
        setter(matmul_precision)
    except Exception:
        return


def maybe_compile_model(
    model: Any,
    torch_module: Any,
    *,
    device_label: str,
    config: TorchCompileConfig,
) -> tuple[Any, str]:
    if not config.enabled:
        return model, "disabled"
    if not str(device_label).startswith("cuda"):
        return model, "skipped-non-cuda"
    compiler = getattr(torch_module, "compile", None)
    if compiler is None:
        return model, "unsupported"
    kwargs: dict[str, Any] = {}
    if config.mode:
        kwargs["mode"] = config.mode
    if config.backend:
        kwargs["backend"] = config.backend
    try:
        return compiler(model, **kwargs), "compiled"
    except Exception as exc:
        return model, f"compile-failed:{exc}"
