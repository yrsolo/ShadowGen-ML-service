class TritonBackendError(RuntimeError):
    """Base Triton transport or model execution error."""


class TritonModelUnavailableError(TritonBackendError):
    """Raised when the configured Triton model is not available."""
