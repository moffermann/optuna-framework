from abc import ABC, abstractmethod
from typing import Any, Dict


class MasterAdapter(ABC):
    def __init__(self, meta: Dict[str, Any], project: Dict[str, Any]) -> None:
        self.meta = dict(meta or {})
        self.project = dict(project or {})

    def init(self, context: Dict[str, Any]) -> None:
        """Hook called once at master start."""

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> None:
        """Hook called for each trial on master."""
        raise NotImplementedError

    def finish(self, context: Dict[str, Any]) -> None:
        """Hook called after each trial on master."""


__all__ = ["MasterAdapter"]
