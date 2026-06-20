from __future__ import annotations

from dataclasses import dataclass, field
from threading import Condition, Event, Lock
from time import monotonic
from typing import Callable, Generic, TypeVar


PayloadT = TypeVar("PayloadT")
ResultT = TypeVar("ResultT")


@dataclass
class _BatchItem(Generic[PayloadT, ResultT]):
    payload: PayloadT
    event: Event = field(default_factory=Event)
    result: ResultT | None = None
    error: Exception | None = None


@dataclass
class _BatchGroup(Generic[PayloadT, ResultT]):
    condition: Condition
    items: list[_BatchItem[PayloadT, ResultT]] = field(default_factory=list)
    closed: bool = False


class TritonStageBatchCoordinator:
    def __init__(
        self,
        *,
        enabled: bool,
        window_ms: int,
        max_size: int,
        stage_enabled: dict[str, bool] | None = None,
    ) -> None:
        self.enabled = enabled
        self.window_ms = max(0, window_ms)
        self.max_size = max(1, max_size)
        self.stage_enabled = dict(stage_enabled or {})
        self._lock = Lock()
        self._groups: dict[tuple[str, str], _BatchGroup] = {}

    def execute(
        self,
        *,
        stage_key: str,
        model_variant: str,
        payload: PayloadT,
        run_batch: Callable[[list[PayloadT]], list[ResultT]],
    ) -> ResultT:
        if not self.enabled or not self.stage_enabled.get(stage_key, False) or self.max_size <= 1:
            return run_batch([payload])[0]

        key = (stage_key, model_variant)
        leader = False
        with self._lock:
            group = self._groups.get(key)
            if group is None or group.closed:
                group = _BatchGroup(condition=Condition(self._lock))
                self._groups[key] = group
            item: _BatchItem[PayloadT, ResultT] = _BatchItem(payload=payload)
            group.items.append(item)
            leader = len(group.items) == 1
            group.condition.notify_all()

        if leader:
            deadline = monotonic() + (self.window_ms / 1000.0)
            with self._lock:
                active = self._groups.get(key)
                while active is group and len(group.items) < self.max_size and monotonic() < deadline:
                    timeout = max(0.0, deadline - monotonic())
                    if timeout <= 0:
                        break
                    group.condition.wait(timeout=timeout)
                if self._groups.get(key) is group:
                    group.closed = True
                    del self._groups[key]
                items = list(group.items)
            try:
                results = run_batch([queued.payload for queued in items])
                if len(results) != len(items):
                    raise RuntimeError(
                        f"batch runner returned {len(results)} results for {len(items)} payloads in {stage_key}"
                    )
                for queued, result in zip(items, results, strict=True):
                    queued.result = result
                    queued.event.set()
            except Exception as exc:
                for queued in items:
                    queued.error = exc
                    queued.event.set()
        else:
            item.event.wait()

        if item.error is not None:
            raise item.error
        return item.result
