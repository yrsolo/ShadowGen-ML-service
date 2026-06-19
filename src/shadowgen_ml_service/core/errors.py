from __future__ import annotations

from enum import Enum


class StageFaultKind(str, Enum):
    BACKEND_RUNTIME_ERROR = "backend_runtime_error"
    TRITON_BACKEND_ERROR = "triton_backend_error"
    TRITON_ENDPOINT_UNAVAILABLE = "triton_endpoint_unavailable"
    TRITON_INVALID_RESPONSE = "triton_invalid_response"
    TRITON_MODEL_UNAVAILABLE = "triton_model_unavailable"
    TRITON_SCHEMA_MISMATCH = "triton_schema_mismatch"
    TRITON_TIMEOUT = "triton_timeout"


class BackendFault(RuntimeError):
    kind: StageFaultKind = StageFaultKind.BACKEND_RUNTIME_ERROR

    def __init__(
        self,
        message: str = "",
        *,
        kind: StageFaultKind | None = None,
        details: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind or self.kind
        self.details = details or {}


class ServiceError(Exception):
    code = "internal_error"
    http_status = 500

    def __init__(self, message: str, request_id: str | None = None, details: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.request_id = request_id
        self.details = details


class ValidationServiceError(ServiceError):
    code = "validation_error"
    http_status = 400


class UnsupportedInputServiceError(ServiceError):
    code = "unsupported_input"
    http_status = 415


class ProcessingFailedServiceError(ServiceError):
    code = "processing_failed"
    http_status = 500


class TimeoutServiceError(ServiceError):
    code = "timeout"
    http_status = 504


class AsyncDisabledServiceError(ServiceError):
    code = "async_disabled"
    http_status = 503


class QueueFullServiceError(ServiceError):
    code = "queue_full"
    http_status = 429


class NotAcceptingJobsServiceError(ServiceError):
    code = "not_accepting_jobs"
    http_status = 503
