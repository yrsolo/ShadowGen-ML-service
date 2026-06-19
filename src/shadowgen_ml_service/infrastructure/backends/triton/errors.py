from shadowgen_ml_service.core.errors import BackendFault, StageFaultKind


class TritonBackendError(BackendFault):
    """Base Triton transport or model execution error."""

    kind = StageFaultKind.TRITON_BACKEND_ERROR


class TritonModelUnavailableError(TritonBackendError):
    """Raised when the configured Triton model is not available."""

    kind = StageFaultKind.TRITON_MODEL_UNAVAILABLE


class TritonEndpointUnavailableError(TritonBackendError):
    """Raised when the Triton endpoint cannot be reached."""

    kind = StageFaultKind.TRITON_ENDPOINT_UNAVAILABLE


class TritonSchemaMismatchError(TritonBackendError):
    """Raised when Triton returned an unexpected tensor schema."""

    kind = StageFaultKind.TRITON_SCHEMA_MISMATCH


class TritonTimeoutError(TritonBackendError):
    """Raised when a Triton request timed out."""

    kind = StageFaultKind.TRITON_TIMEOUT


class TritonInvalidResponseError(TritonBackendError):
    """Raised when Triton returned a malformed response."""

    kind = StageFaultKind.TRITON_INVALID_RESPONSE
