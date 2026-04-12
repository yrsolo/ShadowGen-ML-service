from __future__ import annotations

from abc import ABC, abstractmethod

from shadowgen_ml_service.application.models import AsyncRenderJobRecord, RenderOutcome
from shadowgen_ml_service.core.commands import RenderCommand
from shadowgen_ml_service.core.models import JobExecutionInfo


class JobRepository(ABC):
    @abstractmethod
    def create(self, command: RenderCommand) -> AsyncRenderJobRecord:
        raise NotImplementedError

    @abstractmethod
    def get(self, job_id: str) -> AsyncRenderJobRecord | None:
        raise NotImplementedError

    @abstractmethod
    def get_by_request_id(self, request_id: str) -> AsyncRenderJobRecord | None:
        raise NotImplementedError

    @abstractmethod
    def update(self, job_id: str, *, status: str, error: str | None = None, render_outcome: RenderOutcome | None = None) -> AsyncRenderJobRecord:
        raise NotImplementedError


class JobQueue(ABC):
    @abstractmethod
    def submit(self, job_id: str, work) -> None:
        raise NotImplementedError

    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        raise NotImplementedError


class JobResultStore(ABC):
    @abstractmethod
    def get_result(self, job_id: str) -> RenderOutcome | None:
        raise NotImplementedError


class JobCapacityProvider(ABC):
    @abstractmethod
    def capacity_snapshot(self) -> JobExecutionInfo:
        raise NotImplementedError
