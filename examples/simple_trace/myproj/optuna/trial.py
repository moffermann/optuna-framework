from __future__ import annotations

from optuna_framework.adapters.trial import TrialAdapter


class TraceTrialAdapter(TrialAdapter):
    def on_trial_start(self, context):
        print('[TRIAL] start', context)

    def on_trial_end(self, context):
        print('[TRIAL] end', context)
