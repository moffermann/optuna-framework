from __future__ import annotations

import optuna

from optuna_framework.adapters.prune import PruneAdapter


class TracePruneAdapter(PruneAdapter):
    def prune(self, params, trial):
        # Demo: prune a specific trial to show the flow
        if trial.number == 1:
            raise optuna.TrialPruned("demo prune: trial_number == 1")
