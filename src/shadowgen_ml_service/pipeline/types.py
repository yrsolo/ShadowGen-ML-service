from dataclasses import dataclass
from typing import Any

from shadowgen_ml_service.core.models import *  # noqa: F401,F403


@dataclass
class TimedValue:
    value: Any
    elapsed_ms: int


CachedPreprocess = PreprocessSnapshot
