from __future__ import annotations

import json
from urllib import request

from shadowgen_ml_service.infrastructure.backends.triton.config import TritonBackendSettings
from shadowgen_ml_service.infrastructure.backends.triton.errors import TritonBackendError, TritonModelUnavailableError


class TritonInferenceClient:
    def __init__(self, settings: TritonBackendSettings) -> None:
        self.settings = settings

    @property
    def endpoint(self) -> str | None:
        return self.settings.url

    def is_enabled(self) -> bool:
        return self.settings.enabled

    def ping(self) -> bool:
        if not self.settings.enabled:
            return False
        try:
            req = request.Request(f"{self.settings.url}/v2/health/ready", method="GET")
            with request.urlopen(req, timeout=self.settings.timeout_ms / 1000) as response:
                return 200 <= response.status < 300
        except Exception:
            return False

    def infer_json(self, model_name: str, payload: dict) -> dict:
        if not self.settings.enabled:
            raise TritonModelUnavailableError("Triton endpoint is not configured")
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.settings.url}/v2/models/{model_name}/infer",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.settings.timeout_ms / 1000) as response:
                if response.status == 404:
                    raise TritonModelUnavailableError(f"Triton model {model_name} is unavailable")
                return json.loads(response.read().decode("utf-8"))
        except TritonBackendError:
            raise
        except Exception as exc:
            raise TritonBackendError(str(exc)) from exc
