import multiprocessing as mp
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import optuna
from optuna.storages import RDBStorage
from optuna.trial import TrialState

from optuna_framework.adapters.master import MasterAdapter
from optuna_framework.adapters.worker import WorkerAdapter
from optuna_framework.imports import load_object
from optuna_framework.io import save_params
from optuna_framework.search_space import build_params_tree, normalize_value, resolve_param_value


def _ensure_sqlite_pragmas(path: Path) -> None:
    try:
        with sqlite3.connect(str(path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=30000;")
    except sqlite3.Error:
        pass


def _ensure_positive_int(value: Union[int, str, float], name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if result < 1:
        raise ValueError(f"{name} must be >= 1, got {result}")
    return result


def create_sampler(opt_cfg: Dict[str, Any], seed: int) -> Tuple[Optional[optuna.samplers.BaseSampler], int]:
    n_trials = _ensure_positive_int(opt_cfg.get("n_trials", 100), "n_trials")
    sampler_name = str(opt_cfg.get("sampler", "tpe")).lower()
    sampler: Optional[optuna.samplers.BaseSampler] = None
    if sampler_name == "grid":
        grid_params = opt_cfg.get("grid_params", None)
        if not isinstance(grid_params, dict) or not grid_params:
            raise ValueError("Grid sampler requires optuna.grid_params dict with parameter lists.")
        sampler = optuna.samplers.GridSampler(grid_params)
        grid_size = 1
        for values in grid_params.values():
            grid_size *= len(values)
        if n_trials > grid_size:
            n_trials = grid_size
    elif sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=int(seed))
    elif sampler_name == "tpe":
        sampler = optuna.samplers.TPESampler(seed=int(seed))
    elif sampler_name == "qmc":
        sampler = optuna.samplers.QMCSampler(seed=int(seed))
    else:
        raise ValueError(
            f"Unsupported sampler '{sampler_name}'. Choose from grid/random/tpe/qmc."
        )
    return sampler, n_trials


def _create_storage(storage_url: str, engine_kwargs: Dict[str, Any]) -> RDBStorage:
    return RDBStorage(url=storage_url, engine_kwargs=engine_kwargs)


def _load_run_adapter(
    adapter_path: Optional[str],
    adapter_cls: Any,
    meta: Dict[str, Any],
    project: Dict[str, Any],
    role: str,
) -> Optional[Any]:
    if not adapter_path:
        return None
    loaded = load_object(adapter_path)
    adapter = loaded(meta, project)
    if not isinstance(adapter, adapter_cls):
        raise TypeError(f"{role} adapter must inherit from {adapter_cls.__name__}.")
    return adapter


def _build_context(
    role: str,
    study_name: str,
    trial: Optional[optuna.trial.Trial] = None,
    value: Optional[float] = None,
    state: Optional[str] = None,
    error: Optional[BaseException] = None,
    phase: Optional[str] = None,
) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {"role": role, "study_name": study_name}
    if phase:
        ctx["phase"] = phase
    if trial is not None:
        ctx["trial_number"] = int(trial.number)
        ctx["params"] = dict(trial.params)
        ctx["user_attrs"] = dict(trial.user_attrs)
    if value is not None:
        ctx["value"] = float(value)
    if state is not None:
        ctx["state"] = state
    if error is not None:
        ctx["error"] = str(error)
    return ctx


def _worker_loop(
    storage_url: str,
    study_name: str,
    objective: Callable[[optuna.trial.Trial], float],
    timeout_sec: Optional[int],
    n_trials: int,
    engine_kwargs: Dict[str, Any],
    worker_adapter_path: Optional[str],
    meta: Dict[str, Any],
    project: Dict[str, Any],
) -> None:
    os.environ["OPTUNA_WORKER_ROLE"] = "worker"
    pid = os.getpid()
    print(
        f"[WORKER] started pid={pid} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES','')}",
        flush=True,
    )
    adapter = _load_run_adapter(worker_adapter_path, WorkerAdapter, meta, project, "Worker")
    if adapter is not None:
        adapter.init(_build_context("worker", study_name, phase="init"))
    t_start = time.time()
    storage = _create_storage(storage_url, engine_kwargs)
    study = optuna.load_study(study_name=study_name, storage=storage)

    consecutive_storage_errors = 0
    max_storage_errors = 10

    while True:
        if timeout_sec is not None and (time.time() - t_start) > float(timeout_sec):
            print(f"[WORKER pid={pid}] timeout reached, exiting", flush=True)
            break
        if n_trials > 0:
            try:
                if len(study.trials) >= int(n_trials):
                    print(f"[WORKER pid={pid}] n_trials={n_trials} reached, exiting", flush=True)
                    break
                consecutive_storage_errors = 0
            except Exception as exc:
                consecutive_storage_errors += 1
                print(
                    f"[WORKER pid={pid}] error checking trial count ({consecutive_storage_errors}/{max_storage_errors}): {exc}",
                    flush=True,
                )
                if consecutive_storage_errors >= max_storage_errors:
                    print(f"[WORKER pid={pid}] too many storage errors, exiting", flush=True)
                    break
                time.sleep(0.5)
                continue
        try:
            trial = study.ask()
            consecutive_storage_errors = 0
        except Exception as exc:
            print(f"[WORKER pid={pid}] error asking for trial: {exc}", flush=True)
            break

        if adapter is not None:
            adapter.execute(_build_context("worker", study_name, trial=trial, phase="start"))

        value: Optional[float] = None
        state_name = None
        error: Optional[BaseException] = None
        try:
            value = objective(trial)
            study.tell(trial, value)
            state_name = TrialState.COMPLETE.name
        except optuna.exceptions.TrialPruned as exc:
            error = exc
            state_name = TrialState.PRUNED.name
            print(f"[WORKER pid={pid}] trial {trial.number} pruned", flush=True)
            study.tell(trial, state=TrialState.PRUNED)
        except Exception as exc:
            error = exc
            state_name = TrialState.FAIL.name
            print(f"[WORKER pid={pid}] trial {trial.number} failed: {exc}", flush=True)
            study.tell(trial, state=TrialState.FAIL)
        finally:
            if adapter is not None:
                adapter.finish(
                    _build_context(
                        "worker",
                        study_name,
                        trial=trial,
                        value=value,
                        state=state_name,
                        error=error,
                        phase="finish",
                    )
                )


def format_study_name(meta_name: str, version: Optional[int]) -> str:
    if version is None:
        return meta_name
    return f"{meta_name}_v{version}"


def get_study_name(
    storage_url: Optional[str],
    meta_name: str,
    study_version: Optional[int],
    continue_study: bool,
) -> Tuple[str, Optional[int]]:
    candidate = format_study_name(meta_name, study_version)
    if continue_study:
        return candidate, study_version
    if study_version is None or not storage_url:
        return candidate, study_version
    try:
        summaries = optuna.study.get_all_study_summaries(storage=storage_url)
    except Exception as exc:
        print(f"[OPTUNA] warning: could not list existing studies: {exc}", flush=True)
        return candidate, study_version
    if not summaries:
        return candidate, study_version
    prefix = f"{meta_name}_v"
    matching = [
        s
        for s in summaries
        if s.study_name == meta_name or s.study_name.startswith(prefix)
    ]
    if not matching:
        return candidate, study_version
    names = {s.study_name for s in matching}
    if candidate not in names:
        return candidate, study_version
    return get_study_name(storage_url, meta_name, study_version + 1, False)


def optimize_study(
    objective: Callable[[optuna.trial.Trial], float],
    payload: Dict[str, Any],
    params_path: str,
    opt_cfg: Dict[str, Any],
    meta: Dict[str, Any],
    search_space: Dict[str, Any],
    search_space_tree: Dict[str, Any],
    continue_study: bool,
    seed: int,
    project: Optional[Dict[str, Any]] = None,
    worker_adapter_path: Optional[str] = None,
    master_adapter_path: Optional[str] = None,
) -> Tuple[optuna.Study, float, Dict[str, Any], Dict[str, Any], Optional[int]]:
    timeout_sec = int(opt_cfg.get("timeout_sec", 0))
    if timeout_sec <= 0:
        timeout_sec = None
    n_jobs = _ensure_positive_int(opt_cfg.get("n_jobs", 1), "n_jobs")
    meta_name = str(meta.get("name", "optuna_study")).strip()
    study_version = int(meta["study_version"]) if "study_version" in meta else None
    storage_url = opt_cfg.get("storage_url", None)
    storage_sqlite = opt_cfg.get("storage_sqlite", None)
    if storage_url is None and storage_sqlite:
        storage_url = f"sqlite:///{Path(str(storage_sqlite)).as_posix()}"

    sampler, n_trials = create_sampler(opt_cfg, seed)

    study_name, resolved_version = get_study_name(
        storage_url, meta_name, study_version, continue_study
    )
    if resolved_version is not None and resolved_version != study_version:
        meta["study_version"] = int(resolved_version)
        payload["meta"] = meta
        save_params(Path(params_path), payload)
        study_version = int(resolved_version)

    storage_engine = None
    engine_kwargs = dict(opt_cfg.get("storage_engine_kwargs", {}))
    connect_args = engine_kwargs.get("connect_args", {})
    if storage_url and str(storage_url).startswith("sqlite:///"):
        connect_args.setdefault("timeout", int(opt_cfg.get("sqlite_timeout", 30)))
        connect_args.setdefault("check_same_thread", False)
    else:
        connect_timeout = int(opt_cfg.get("pg_connect_timeout", 30))
        connect_args.setdefault("connect_timeout", connect_timeout)
    engine_kwargs["connect_args"] = connect_args

    if storage_url:
        if storage_sqlite:
            _ensure_sqlite_pragmas(Path(str(storage_sqlite)))
        storage_engine = RDBStorage(url=storage_url, engine_kwargs=engine_kwargs)
    else:
        raise ValueError("Multiprocess optimization requires a persistent Optuna storage URL.")

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage_engine,
        load_if_exists=True,
        sampler=sampler,
    )

    project = dict(project or {})
    master_adapter = _load_run_adapter(master_adapter_path, MasterAdapter, meta, project, "Master")
    if master_adapter is not None:
        master_adapter.init(_build_context("master", study_name, phase="init"))

    ctx = mp.get_context("spawn")
    procs = []
    for _ in range(n_jobs):
        p = ctx.Process(
            target=_worker_loop,
            args=(
                storage_url,
                study_name,
                objective,
                timeout_sec,
                n_trials,
                engine_kwargs,
                worker_adapter_path,
                meta,
                project,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    failed_workers = [i for i, p in enumerate(procs) if p.exitcode != 0]
    if failed_workers:
        print(f"[OPTUNA] warning: {len(failed_workers)} worker(s) exited with non-zero code", flush=True)

    study = optuna.load_study(study_name=study_name, storage=storage_engine)

    if master_adapter is not None:
        for trial in study.trials:
            ctx = _build_context(
                "master",
                study_name,
                trial=trial,
                value=trial.value,
                state=trial.state.name,
                phase="trial",
            )
            master_adapter.execute(ctx)
            master_adapter.finish(ctx)

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(
            f"No completed trials in study '{study_name}'. "
            f"Total trials: {len(study.trials)}, failed workers: {len(failed_workers)}"
        )

    best = study.best_trial
    best_value = float(best.value)
    best_params_full = {
        name: resolve_param_value(name, spec, best.params)
        for name, spec in search_space.items()
    }
    best_params_full = {k: normalize_value(v) for k, v in best_params_full.items()}
    best_params_tree = build_params_tree(search_space_tree, best_params_full)
    return study, best_value, best_params_full, best_params_tree, study_version
