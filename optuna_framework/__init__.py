"""Generic Optuna framework.
"""

from optuna_framework.adapters.objective import ObjectiveAdapter, TrialResult
from optuna_framework.adapters.trial import TrialAdapter
from optuna_framework.adapters.master import MasterAdapter
from optuna_framework.adapters.prune import PruneAdapter
from optuna_framework.objective import ObjectiveCallable
from optuna_framework.runner import optimize_study
from optuna_framework.search_space import (
    build_params_tree,
    flatten_spec_tree,
    normalize_value,
    parse_spec,
    resolve_param_value,
    suggest_value,
)

__all__ = [
    "ObjectiveAdapter",
    "TrialResult",
    "TrialAdapter",
    "MasterAdapter",
    "PruneAdapter",
    "ObjectiveCallable",
    "optimize_study",
    "build_params_tree",
    "flatten_spec_tree",
    "normalize_value",
    "parse_spec",
    "resolve_param_value",
    "suggest_value",
]
