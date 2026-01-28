from __future__ import annotations

from optuna_framework.adapters.trial import TrialAdapter


class TraceTrialAdapter(TrialAdapter):
    def init(self, context):
        print('[TRIAL] init', context)

    def execute(self, context):
        print('[TRIAL] execute', context)

    def finish(self, context):
        print('[TRIAL] finish', context)
