import os
import time
from typing import Any, Dict, Optional

import optuna

from optuna_framework.adapters.objective import ObjectiveAdapter, TrialResult
from optuna_framework.adapters.prune import PruneAdapter
from optuna_framework.imports import load_object


class ObjectiveCallable:
    def __init__(
        self,
        search_space: Dict[str, Any],
        adapter_path: Optional[str],
        prune_adapter_path: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        project: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.search_space = dict(search_space)
        self.adapter_path = str(adapter_path) if adapter_path else None
        self.prune_adapter_path = str(prune_adapter_path) if prune_adapter_path else None
        self._prune_adapter = None
        self.meta = dict(meta or {})
        self.project = dict(project or {})
        self._initialized = False
        self._adapter: Optional[ObjectiveAdapter] = None

    def _lazy_init(self) -> None:
        if self._initialized:
            return
        if not self.adapter_path:
            print(
                "[WARNING] Objective adapter not configured; set meta.objective_adapter or --objective-adapter.",
                flush=True,
            )
            raise RuntimeError("Objective adapter not configured.")
        adapter_cls = load_object(self.adapter_path)
        adapter = adapter_cls(self.meta, self.project)
        if not isinstance(adapter, ObjectiveAdapter):
            raise TypeError("Objective adapter must inherit from ObjectiveAdapter.")
        adapter.worker_init()
        adapter.setup()
        self._adapter = adapter

        if self.prune_adapter_path:
            prune_cls = load_object(self.prune_adapter_path)
            prune_adapter = prune_cls(self.meta, self.project)
            if not isinstance(prune_adapter, PruneAdapter):
                raise TypeError("Prune adapter must inherit from PruneAdapter.")
            prune_adapter.init()
            self._prune_adapter = prune_adapter
        self._initialized = True

    def __call__(self, trial: optuna.trial.Trial) -> float:
        self._lazy_init()
        if self._adapter is None:
            raise RuntimeError("Objective adapter not initialized.")
        t0 = time.perf_counter()
        pid = os.getpid()
        os.environ["TRIAL_ID"] = str(trial.number)
        print(f"[TRIAL] start number={trial.number} pid={pid}", flush=True)
        params = self._adapter.suggest_params(trial, self.search_space)
        errors = self._adapter.validate_trial_params(params)
        if errors:
            reason = "; ".join(errors)
            trial.set_user_attr("prune_reason", reason)
            raise optuna.exceptions.TrialPruned(reason)

        if self._prune_adapter is not None:
            self._prune_adapter.prune(params, trial)

        try:
            self._adapter.on_trial_start(trial, params)
            result = self._adapter.execute(params, trial)
            if isinstance(result, TrialResult):
                value = float(result.value)
                for key, val in result.user_attrs.items():
                    trial.set_user_attr(key, val)
            else:
                value = float(result)
            self._adapter.on_trial_end(trial, value, params)
            elapsed = time.perf_counter() - t0
            print(
                f"[TRIAL] done number={trial.number} score={value:.6f} sec={elapsed:.1f} pid={pid}",
                flush=True,
            )
            return value
        except optuna.exceptions.TrialPruned as exc:
            elapsed = time.perf_counter() - t0
            print(
                f"[TRIAL] pruned number={trial.number} sec={elapsed:.1f} pid={pid} reason={exc}",
                flush=True,
            )
            raise
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(
                f"[TRIAL] error number={trial.number} sec={elapsed:.1f} pid={pid} err={exc}",
                flush=True,
            )
            raise

    def close(self) -> None:
        if self._adapter is not None:
            self._adapter.teardown()
            self._adapter = None
        if self._prune_adapter is not None:
            self._prune_adapter = None
        self._initialized = False
