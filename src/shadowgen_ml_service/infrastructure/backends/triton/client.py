from __future__ import annotations

import json
import socket
from urllib import error, request

from shadowgen_ml_service.infrastructure.backends.triton.config import TritonBackendSettings
from shadowgen_ml_service.infrastructure.backends.triton.errors import (
    TritonBackendError,
    TritonEndpointUnavailableError,
    TritonInvalidResponseError,
    TritonModelUnavailableError,
    TritonSchemaMismatchError,
    TritonTimeoutError,
)
from shadowgen_ml_service.infrastructure.backends.triton.serializers import tensor_map_from_response


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

    def infer(self, model_name: str, *, inputs: list[dict], outputs: list[str]) -> dict:
        if not self.settings.enabled:
            raise TritonModelUnavailableError("Triton endpoint is not configured")
        body = json.dumps(
            {
                "inputs": inputs,
                "outputs": [{"name": name} for name in outputs],
            }
        ).encode("utf-8")
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
                payload = json.loads(response.read().decode("utf-8"))
                tensors = tensor_map_from_response(payload)
                missing = [name for name in outputs if name not in tensors]
                if missing:
                    raise TritonSchemaMismatchError(
                        f"Triton model {model_name} did not return expected outputs: {', '.join(missing)}"
                    )
                return tensors
        except error.HTTPError as exc:
            if exc.code == 404:
                raise TritonModelUnavailableError(f"Triton model {model_name} is unavailable") from exc
            raise TritonBackendError(f"Triton HTTP error {exc.code}: {exc.reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise TritonTimeoutError(f"Triton request timed out for model {model_name}") from exc
        except error.URLError as exc:
            raise TritonEndpointUnavailableError(f"Triton endpoint is unavailable: {exc.reason}") from exc
        except ValueError as exc:
            raise TritonInvalidResponseError(f"Malformed Triton response for model {model_name}: {exc}") from exc
        except TritonBackendError:
            raise
        except Exception as exc:
            raise TritonBackendError(str(exc)) from exc
