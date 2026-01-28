from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

import optuna

from optuna_framework.search_space import normalize_value, suggest_value


@dataclass
class TrialResult:
    value: float
    user_attrs: Dict[str, Any] = field(default_factory=dict)


class ObjectiveAdapter(ABC):
    def __init__(self, meta: Dict[str, Any], project: Dict[str, Any]) -> None:
        self.meta = dict(meta or {})
        self.project = dict(project or {})

    def worker_init(self) -> None:
        """Optional hook executed once per worker before setup."""

    def setup(self) -> None:
        """Optional hook to load heavy data per worker."""

    def teardown(self) -> None:
        """Optional hook to release resources per worker."""

    def validate_search_space(self, search_space: Dict[str, Any]) -> List[str]:
        return []

    def validate_trial_params(self, params: Dict[str, Any]) -> List[str]:
        return []

    def suggest_params(
        self, trial: optuna.trial.Trial, search_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            name: normalize_value(suggest_value(trial, name, spec))
            for name, spec in search_space.items()
        }

    def on_trial_start(self, trial: optuna.trial.Trial, params: Dict[str, Any]) -> None:
        """Optional hook before execute."""

    def on_trial_end(self, trial: optuna.trial.Trial, value: float, params: Dict[str, Any]) -> None:
        """Optional hook after execute."""

    @abstractmethod
    def execute(self, params: Dict[str, Any], trial: optuna.trial.Trial) -> Any:
        """Return float or TrialResult."""
        raise NotImplementedError
