from __future__ import annotations

from optuna_framework.adapters.trial import TrialAdapter


class TraceTrialAdapter(TrialAdapter):
    def on_trial_start(self, context):
        trial_number = context.get("trial_number", "?")
        worker_id = context.get("worker_id", "?")
        print(f"[TRIAL {trial_number}] on_trial_start worker={worker_id}", flush=True)

    def on_trial_end(self, context):
        trial_number = context.get("trial_number", "?")
        worker_id = context.get("worker_id", "?")
        state = context.get("state", "?")
        print(f"[TRIAL {trial_number}] on_trial_end worker={worker_id} state={state}", flush=True)
