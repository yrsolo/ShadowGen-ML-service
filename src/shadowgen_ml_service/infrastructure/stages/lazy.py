from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any


class LazyStageAdapter:
    def __init__(self, factory: Callable[[], Any], *, backend_name: str, device_label: str = "lazy") -> None:
        self._factory = factory
        self._lock = Lock()
        self._delegate: Any | None = None
        self.backend_name = backend_name
        self.device_label = device_label

    def _get_delegate(self) -> Any:
        if self._delegate is None:
            with self._lock:
                if self._delegate is None:
                    self._delegate = self._factory()
                    self.device_label = getattr(self._delegate, "device_label", self.device_label)
        return self._delegate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_delegate(), name)
