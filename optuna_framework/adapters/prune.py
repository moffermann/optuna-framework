from abc import ABC, abstractmethod
from typing import Any, Dict


class PruneAdapter(ABC):
    def __init__(self, meta: Dict[str, Any], project: Dict[str, Any]) -> None:
        self.meta = dict(meta or {})
        self.project = dict(project or {})

    def init(self) -> None:
        """Optional hook executed once per worker before pruning."""

    @abstractmethod
    def prune(self, params: Dict[str, Any], trial: Any) -> None:
        """Raise optuna.TrialPruned to stop the trial early."""
        raise NotImplementedError
