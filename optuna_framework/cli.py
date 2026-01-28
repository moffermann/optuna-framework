import argparse
from pathlib import Path
from typing import Any, Dict

from optuna_framework.adapters.objective import ObjectiveAdapter
from optuna_framework.imports import load_object
from optuna_framework.io import load_params
from optuna_framework.objective import ObjectiveCallable
from optuna_framework.reporting import build_best_payload, write_best_json
from optuna_framework.runner import optimize_study
from optuna_framework.search_space import flatten_spec_tree


def _ensure_dict(payload: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"params['{key}'] must be a dict.")
    return value


def _resolve_adapter_path(args: argparse.Namespace, meta: Dict[str, Any]) -> str:
    return (
        args.objective_adapter
        or args.adapter
        or meta.get("objective_adapter")
        or meta.get("adapter")
        or meta.get("adapter_path")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna framework runner.")
    parser.add_argument("--params", "-p", required=True, help="Path to Optuna params file (JSON/YAML).")
    parser.add_argument(
        "--objective-adapter",
        default=None,
        help="Objective adapter class path (e.g. myproj.optuna_adapter:MyObjectiveAdapter).",
    )
    parser.add_argument(
        "--trial-adapter",
        default=None,
        help="Trial adapter class path (e.g. myproj.optuna.trial:MyTrialAdapter).",
    )
    parser.add_argument(
        "--worker-adapter",
        default=None,
        help="Worker adapter class path (e.g. myproj.optuna.worker:MyWorkerAdapter).",
    )
    parser.add_argument(
        "--optuna-adapter",
        default=None,
        help="Optuna adapter class path (e.g. myproj.optuna.optuna:MyOptunaAdapter).",
    )
    parser.add_argument(
        "--prune-adapter",
        default=None,
        help="Prune adapter class path (e.g. myproj.optuna_prune:MyPruneAdapter).",
    )
    parser.add_argument(
        "--adapter",
        "-a",
        default=None,
        help="(Legacy) Objective adapter class path.",
    )
    parser.add_argument(
        "--continue-study",
        "-c",
        action="store_true",
        help="Reuse study_version even if it already exists in storage.",
    )
    parser.add_argument(
        "--trials",
        "-t",
        type=int,
        default=None,
        help="Override optuna n_trials for quick runs.",
    )
    args = parser.parse_args()

    params_path = Path(args.params)
    payload = load_params(params_path)
    meta = _ensure_dict(payload, "meta")
    opt_cfg = _ensure_dict(payload, "optuna")
    project = _ensure_dict(payload, "project") if "project" in payload else {}

    search_space_tree = payload.get("search_space", {})
    if not isinstance(search_space_tree, dict):
        raise ValueError("params['search_space'] must be a dict.")
    search_space = flatten_spec_tree(search_space_tree)

    objective_adapter_path = _resolve_adapter_path(args, meta)
    if not objective_adapter_path:
        print(
            "[WARNING] Objective adapter not configured; set meta.objective_adapter or --objective-adapter.",
            flush=True,
        )
        raise ValueError("Objective adapter not configured.")

    if args.trials is not None:
        opt_cfg["n_trials"] = args.trials

    n_trials = int(opt_cfg.get("n_trials", 100))
    n_jobs = int(opt_cfg.get("n_jobs", 1))
    if n_trials < n_jobs:
        raise ValueError(
            f"n_trials ({n_trials}) must be >= n_jobs ({n_jobs})."
        )

    seed = int(meta.get("seed", 42))

    adapter_cls = load_object(str(objective_adapter_path))
    adapter = adapter_cls(meta, project)
    if not isinstance(adapter, ObjectiveAdapter):
        raise TypeError("Objective adapter must inherit from ObjectiveAdapter.")
    errors = adapter.validate_search_space(search_space)
    if errors:
        raise ValueError("Invalid search_space configuration:\n  " + "\n  ".join(errors))
    adapter.teardown()

    objective = ObjectiveCallable(
        search_space,
        str(objective_adapter_path),
        meta=meta,
        project=project,
        prune_adapter_path=args.prune_adapter or meta.get("prune_adapter"),
    )

    study, best_value, best_params_full, best_params_tree, study_version = optimize_study(
        objective,
        payload,
        str(params_path),
        opt_cfg,
        meta,
        search_space,
        search_space_tree,
        args.continue_study,
        seed,
        project=project,
        trial_adapter_path=args.trial_adapter or meta.get("trial_adapter"),
        worker_adapter_path=args.worker_adapter or meta.get("worker_adapter"),
        optuna_adapter_path=args.optuna_adapter or meta.get("optuna_adapter"),
    )

    best = study.best_trial
    out_path = Path(opt_cfg.get("out_path", "optuna_best.json"))
    payload_out = build_best_payload(
        study_name=study.study_name,
        study_version=study_version,
        best_value=best_value,
        best_params=best.params,
        best_params_full=best_params_full,
        best_params_grouped=best_params_tree,
        best_user_attrs=best.user_attrs,
    )
    write_best_json(out_path, payload_out)

    print(f"[DONE] best_value={best_value:.6f} -> {out_path}")


if __name__ == "__main__":
    main()
