from __future__ import annotations

from optuna_framework.adapters.optimization import OptimizationAdapter


class TraceOptimizationAdapter(OptimizationAdapter):
    def on_optimization_start(self, context):
        print('[OPTIMIZATION] start', context, flush=True)

    def on_optimization_end(self, context):
        print('[OPTIMIZATION] end', context, flush=True)
