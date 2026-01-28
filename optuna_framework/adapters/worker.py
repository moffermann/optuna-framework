from abc import ABC, abstractmethod
from typing import Any, Dict


class WorkerAdapter(ABC):
    def __init__(self, meta: Dict[str, Any], project: Dict[str, Any]) -> None:
        self.meta = dict(meta or {})
        self.project = dict(project or {})

    def init(self, context: Dict[str, Any]) -> None:
        """Hook called once at worker start."""

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> None:
        """Hook called before each trial execution."""
        raise NotImplementedError

    def finish(self, context: Dict[str, Any]) -> None:
        """Hook called after each trial execution."""
