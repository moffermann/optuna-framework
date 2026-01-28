from abc import ABC, abstractmethod
from typing import Any, Dict


class TrialAdapter(ABC):
    def __init__(self, meta: Dict[str, Any], project: Dict[str, Any]) -> None:
        self.meta = dict(meta or {})
        self.project = dict(project or {})

    def init(self, context: Dict[str, Any]) -> None:
        """Hook called before each trial execution."""

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> None:
        """Hook called during trial execution (before objective)."""
        raise NotImplementedError

    def finish(self, context: Dict[str, Any]) -> None:
        """Hook called after each trial execution."""
