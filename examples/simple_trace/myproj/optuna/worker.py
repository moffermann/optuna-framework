from __future__ import annotations

from optuna_framework.adapters.worker import WorkerAdapter


class TraceWorkerAdapter(WorkerAdapter):
    def on_worker_start(self, context):
        worker_id = context.get("worker_id", "?")
        print(f"[WORKER {worker_id}] on_worker_start", flush=True)

    def on_worker_end(self, context):
        worker_id = context.get("worker_id", "?")
        print(f"[WORKER {worker_id}] on_worker_end", flush=True)
