from __future__ import annotations

import json
import socket
from urllib import error, request
from urllib.parse import urlparse

import numpy as np

from shadowgen_ml_service.infrastructure.backends.triton.config import TritonBackendSettings
from shadowgen_ml_service.infrastructure.backends.triton.errors import (
    TritonBackendError,
    TritonEndpointUnavailableError,
    TritonInvalidResponseError,
    TritonModelUnavailableError,
    TritonSchemaMismatchError,
    TritonTimeoutError,
)
from shadowgen_ml_service.infrastructure.backends.triton.model_registry import TritonModelBinding
from shadowgen_ml_service.infrastructure.backends.triton.serializers import tensor_from_input, tensor_map_from_response, validate_tensor_against_binding


class TritonInferenceClient:
    def __init__(self, settings: TritonBackendSettings) -> None:
        self.settings = settings
        self._native_http_client = None

    @property
    def endpoint(self) -> str | None:
        return self.settings.url

    def is_enabled(self) -> bool:
        return self.settings.enabled

    def ping(self) -> bool:
        if not self.settings.enabled:
            return False
        try:
            with self._request_json("/v2/health/ready", method="GET", expect_json=False) as response:
                return 200 <= response.status < 300
        except Exception:
            return False

    def model_ready(self, model_name: str) -> bool:
        if not self.settings.enabled:
            return False
        try:
            with self._request_json(f"/v2/models/{model_name}/ready", method="GET", expect_json=False) as response:
                return 200 <= response.status < 300
        except Exception:
            return False

    def model_metadata(self, model_name: str) -> dict:
        if not self.settings.enabled:
            raise TritonModelUnavailableError("Triton endpoint is not configured")
        return self._request_json(f"/v2/models/{model_name}", method="GET")

    def probe_binding(self, binding: TritonModelBinding) -> tuple[bool, str]:
        if not self.settings.enabled:
            return False, "Triton endpoint is not configured"
        if not self.model_ready(binding.model_name):
            return False, f"Triton model {binding.model_name} is not ready"
        try:
            metadata = self.model_metadata(binding.model_name)
            self._validate_model_metadata(binding, metadata)
        except TritonBackendError as exc:
            return False, str(exc)
        return True, f"Triton model {binding.model_name} is ready"

    def infer(self, binding: TritonModelBinding, *, inputs: list[dict]) -> dict:
        if not self.settings.enabled:
            raise TritonModelUnavailableError("Triton endpoint is not configured")
        if self._should_use_native_transport():
            try:
                return self._infer_native_http(binding, inputs=inputs)
            except ImportError:
                # Keep development usable without the optional official Triton client.
                pass
        output_names = [item.tensor_name for item in binding.outputs.values()]
        body = json.dumps(
            {
                "inputs": inputs,
                "outputs": [{"name": name} for name in output_names],
            }
        ).encode("utf-8")
        req = request.Request(
            f"{self.settings.url}/v2/models/{binding.model_name}/infer",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.settings.timeout_ms / 1000) as response:
                if response.status == 404:
                    raise TritonModelUnavailableError(f"Triton model {binding.model_name} is unavailable")
                payload = json.loads(response.read().decode("utf-8"))
                tensors = tensor_map_from_response(payload)
                missing = [name for name in output_names if name not in tensors]
                if missing:
                    raise TritonSchemaMismatchError(
                        f"Triton model {binding.model_name} did not return expected outputs: {', '.join(missing)}"
                    )
                for output_binding in binding.outputs.values():
                    validate_tensor_against_binding(
                        output_binding.tensor_name,
                        tensors[output_binding.tensor_name],
                        output_binding,
                    )
                return tensors
        except error.HTTPError as exc:
            if exc.code == 404:
                raise TritonModelUnavailableError(f"Triton model {binding.model_name} is unavailable") from exc
            raise TritonBackendError(self._http_error_message(exc)) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise TritonTimeoutError(f"Triton request timed out for model {binding.model_name}") from exc
        except error.URLError as exc:
            raise TritonEndpointUnavailableError(f"Triton endpoint is unavailable: {exc.reason}") from exc
        except ValueError as exc:
            raise TritonInvalidResponseError(f"Malformed Triton response for model {binding.model_name}: {exc}") from exc
        except TritonBackendError:
            raise
        except Exception as exc:
            raise TritonBackendError(str(exc)) from exc

    def _should_use_native_transport(self) -> bool:
        transport = (self.settings.transport or "native").strip().lower()
        protocol = (self.settings.protocol or "http").strip().lower()
        return transport in {"native", "binary", "http-binary"} and protocol == "http"

    def _infer_native_http(self, binding: TritonModelBinding, *, inputs: list[dict]) -> dict:
        try:
            import tritonclient.http as httpclient
        except ImportError:
            raise

        client = self._get_native_http_client(httpclient)
        infer_inputs = []
        for item in inputs:
            array = np.ascontiguousarray(tensor_from_input(item))
            infer_input = httpclient.InferInput(item["name"], list(array.shape), item["datatype"])
            infer_input.set_data_from_numpy(array, binary_data=True)
            infer_inputs.append(infer_input)
        output_names = [item.tensor_name for item in binding.outputs.values()]
        requested_outputs = [httpclient.InferRequestedOutput(name, binary_data=True) for name in output_names]
        try:
            result = client.infer(
                binding.model_name,
                inputs=infer_inputs,
                outputs=requested_outputs,
            )
        except Exception as exc:
            message = str(exc)
            lowered = message.lower()
            if "timeout" in lowered:
                raise TritonTimeoutError(f"Triton request timed out for model {binding.model_name}: {message}") from exc
            if "unavailable" in lowered or "failed to establish" in lowered or "connection" in lowered:
                raise TritonEndpointUnavailableError(f"Triton endpoint is unavailable: {message}") from exc
            if "not found" in lowered or "unknown model" in lowered:
                raise TritonModelUnavailableError(f"Triton model {binding.model_name} is unavailable: {message}") from exc
            raise TritonBackendError(f"Triton native HTTP error for model {binding.model_name}: {message}") from exc

        tensors: dict[str, np.ndarray] = {}
        for output_binding in binding.outputs.values():
            tensor = result.as_numpy(output_binding.tensor_name)
            if tensor is None:
                raise TritonSchemaMismatchError(
                    f"Triton model {binding.model_name} did not return expected output: {output_binding.tensor_name}"
                )
            tensors[output_binding.tensor_name] = np.asarray(tensor)
            validate_tensor_against_binding(
                output_binding.tensor_name,
                tensors[output_binding.tensor_name],
                output_binding,
            )
        return tensors

    def _get_native_http_client(self, httpclient):
        if self._native_http_client is not None:
            return self._native_http_client
        if not self.settings.url:
            raise TritonModelUnavailableError("Triton endpoint is not configured")
        parsed = urlparse(self.settings.url)
        endpoint = parsed.netloc or parsed.path
        if not endpoint:
            raise TritonEndpointUnavailableError(f"Invalid Triton URL: {self.settings.url}")
        self._native_http_client = httpclient.InferenceServerClient(
            url=endpoint,
            network_timeout=self.settings.timeout_ms / 1000,
            connection_timeout=self.settings.timeout_ms / 1000,
        )
        return self._native_http_client

    def _request_json(self, path: str, *, method: str, expect_json: bool = True):
        req = request.Request(f"{self.settings.url}{path}", method=method)
        try:
            response = request.urlopen(req, timeout=self.settings.timeout_ms / 1000)
        except error.HTTPError as exc:
            if exc.code == 404:
                raise TritonModelUnavailableError(f"Triton resource {path} is unavailable") from exc
            raise TritonBackendError(self._http_error_message(exc)) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise TritonTimeoutError(f"Triton request timed out for resource {path}") from exc
        except error.URLError as exc:
            raise TritonEndpointUnavailableError(f"Triton endpoint is unavailable: {exc.reason}") from exc

        if not expect_json:
            return response

        try:
            with response:
                return json.loads(response.read().decode("utf-8"))
        except ValueError as exc:
            raise TritonInvalidResponseError(f"Malformed Triton response for resource {path}: {exc}") from exc

    def _validate_model_metadata(self, binding: TritonModelBinding, metadata: dict) -> None:
        inputs = self._metadata_index(metadata, "inputs")
        outputs = self._metadata_index(metadata, "outputs")

        for alias, tensor_binding in binding.inputs.items():
            metadata_entry = inputs.get(tensor_binding.tensor_name)
            if metadata_entry is None:
                raise TritonSchemaMismatchError(
                    f"Triton model {binding.model_name} is missing input tensor {tensor_binding.tensor_name} ({alias})"
                )
            self._validate_metadata_entry(tensor_binding.tensor_name, metadata_entry, tensor_binding)

        for alias, tensor_binding in binding.outputs.items():
            metadata_entry = outputs.get(tensor_binding.tensor_name)
            if metadata_entry is None:
                raise TritonSchemaMismatchError(
                    f"Triton model {binding.model_name} is missing output tensor {tensor_binding.tensor_name} ({alias})"
                )
            self._validate_metadata_entry(tensor_binding.tensor_name, metadata_entry, tensor_binding)

    def _metadata_index(self, metadata: dict, key: str) -> dict[str, dict]:
        entries = metadata.get(key)
        if not isinstance(entries, list):
            raise TritonInvalidResponseError(f"Triton model metadata is missing {key}")
        result: dict[str, dict] = {}
        for entry in entries:
            name = entry.get("name")
            if isinstance(name, str) and name:
                result[name] = entry
        return result

    def _validate_metadata_entry(self, tensor_name: str, entry: dict, binding) -> None:
        datatype = entry.get("datatype")
        if datatype != binding.datatype:
            raise TritonSchemaMismatchError(
                f"Triton tensor {tensor_name} has datatype {datatype}, expected {binding.datatype}"
            )

        shape = entry.get("shape")
        if isinstance(shape, list):
            observed_rank = len(shape)
            if binding.expected_ranks:
                valid_ranks = set(binding.expected_ranks)
                valid_ranks.update(rank - 1 for rank in binding.expected_ranks if rank > 0)
                if observed_rank not in valid_ranks:
                    raise TritonSchemaMismatchError(
                        f"Triton tensor {tensor_name} has rank {observed_rank}, expected one of {sorted(valid_ranks)}"
                    )
            if binding.shape_policy == "channel-first" and binding.channels is not None and shape:
                channel_dim = self._coerce_dim(shape[0])
                if channel_dim not in {None, -1, binding.channels}:
                    raise TritonSchemaMismatchError(
                        f"Triton tensor {tensor_name} advertises {channel_dim} channels, expected {binding.channels}"
                    )

    @staticmethod
    def _coerce_dim(value) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return None
        return None

    @staticmethod
    def _http_error_message(exc: error.HTTPError) -> str:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            detail = ""
        suffix = f": {detail}" if detail else ""
        return f"Triton HTTP error {exc.code}: {exc.reason}{suffix}"
