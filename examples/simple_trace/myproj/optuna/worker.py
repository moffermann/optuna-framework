from __future__ import annotations

from optuna_framework.adapters.worker import WorkerAdapter


class TraceWorkerAdapter(WorkerAdapter):
    def on_worker_start(self, context):
        print('[WORKER] on_worker_start', context)

    def on_worker_end(self, context):
        print('[WORKER] on_worker_end', context)
