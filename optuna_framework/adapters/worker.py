from abc import ABC
from typing import Any, Dict


class WorkerAdapter(ABC):
    def __init__(self, meta: Dict[str, Any], project: Dict[str, Any]) -> None:
        self.meta = dict(meta or {})
        self.project = dict(project or {})

    def on_worker_start(self, context: Dict[str, Any]) -> None:
        """Hook called once when the worker process starts."""

    def on_worker_end(self, context: Dict[str, Any]) -> None:
        """Hook called once when the worker process ends."""


__all__ = ["WorkerAdapter"]
