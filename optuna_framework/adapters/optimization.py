from abc import ABC
from typing import Any, Dict


class OptimizationAdapter(ABC):
    def __init__(self, meta: Dict[str, Any], project: Dict[str, Any]) -> None:
        self.meta = dict(meta or {})
        self.project = dict(project or {})

    def on_optimization_start(self, context: Dict[str, Any]) -> None:
        """Hook called once before optimization starts."""

    def on_optimization_end(self, context: Dict[str, Any]) -> None:
        """Hook called once after optimization ends."""


__all__ = ["OptimizationAdapter"]
