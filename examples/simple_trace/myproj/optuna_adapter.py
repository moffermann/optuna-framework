from __future__ import annotations

from optuna_framework.adapters.objective import ObjectiveAdapter, TrialResult


class TraceObjectiveAdapter(ObjectiveAdapter):
    def worker_init(self) -> None:
        print('[OBJECTIVE] worker_init')

    def setup(self) -> None:
        print('[OBJECTIVE] setup')

    def teardown(self) -> None:
        print('[OBJECTIVE] teardown')

    def validate_search_space(self, search_space):
        print('[OBJECTIVE] validate_search_space', search_space)
        return []

    def validate_trial_params(self, params):
        print('[OBJECTIVE] validate_trial_params', params)
        return []

    def on_trial_start(self, trial, params):
        print(f"[OBJECTIVE] on_trial_start trial={trial.number} params={params}")

    def on_trial_end(self, trial, value, params):
        print(f"[OBJECTIVE] on_trial_end trial={trial.number} value={value:.4f} params={params}")

    def execute(self, params, trial):
        print(f"[OBJECTIVE] execute trial={trial.number} params={params}")
        # Dummy score uses numeric params if present; otherwise constant.
        score = 0.0
        for val in params.values():
            if isinstance(val, (int, float)):
                score += float(val)
        return TrialResult(value=score, user_attrs={'score': score})
