class TritonBackendError(RuntimeError):
    """Base Triton transport or model execution error."""


class TritonModelUnavailableError(TritonBackendError):
    """Raised when the configured Triton model is not available."""


class TritonEndpointUnavailableError(TritonBackendError):
    """Raised when the Triton endpoint cannot be reached."""


class TritonSchemaMismatchError(TritonBackendError):
    """Raised when Triton returned an unexpected tensor schema."""


class TritonTimeoutError(TritonBackendError):
    """Raised when a Triton request timed out."""


class TritonInvalidResponseError(TritonBackendError):
    """Raised when Triton returned a malformed response."""
