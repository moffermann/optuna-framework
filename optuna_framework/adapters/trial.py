from abc import ABC
from typing import Any, Dict


class TrialAdapter(ABC):
    def __init__(self, meta: Dict[str, Any], project: Dict[str, Any]) -> None:
        self.meta = dict(meta or {})
        self.project = dict(project or {})

    def init(self, context: Dict[str, Any]) -> None:
        """Hook called before each trial execution."""

    def finish(self, context: Dict[str, Any]) -> None:
        """Hook called after each trial execution."""
