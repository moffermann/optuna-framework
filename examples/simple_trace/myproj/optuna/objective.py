from __future__ import annotations

import time

from optuna_framework.adapters.objective import ObjectiveAdapter, TrialResult


class TraceObjectiveAdapter(ObjectiveAdapter):
    def execute(self, params, trial):
        print(f"[OBJECTIVE] execute trial={trial.number} params={params}")
        time.sleep(0.2)
        # Dummy score uses numeric params if present; otherwise constant.
        score = 0.0
        for val in params.values():
            if isinstance(val, (int, float)):
                score += float(val)
        return TrialResult(value=score, user_attrs={"score": score})
