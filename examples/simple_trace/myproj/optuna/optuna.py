from __future__ import annotations

from optuna_framework.adapters.optuna import OptunaAdapter


class TraceOptunaAdapter(OptunaAdapter):
    def on_optuna_start(self, context):
        print('[OPTUNA] start', context)

    def on_optuna_end(self, context):
        print('[OPTUNA] end', context)
