from __future__ import annotations

from optuna_framework.adapters.worker import WorkerAdapter


class TraceWorkerAdapter(WorkerAdapter):
    def init(self, context):
        print('[WORKER] init', context)

    def execute(self, context):
        print('[WORKER] execute', context)

    def finish(self, context):
        print('[WORKER] finish', context)
