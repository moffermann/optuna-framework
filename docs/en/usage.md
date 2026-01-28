# Optuna Framework - Usage

This framework extracts Optuna's multiprocess runner and makes it generic through adapters.

## 1) Parameters file (YAML)

Minimal format:

```yaml
meta:
  name: exp001
  seed: 42
  study_version: 1
  objective_adapter: myproj.optuna.objective:MyObjectiveAdapter
  trial_adapter: myproj.optuna.trial:MyTrialAdapter
  master_adapter: myproj.optuna.master:MyMasterAdapter

optuna:
  n_trials: 100
  n_jobs: 4
  storage_url: sqlite:///optuna.db
  sampler: tpe
  timeout_sec: 0
  out_path: results/optuna_best.json

search_space:
  lr:
    range: [1e-4, 1e-2]
    log: true
  batch_size: [16, 32, 64]
  dropout: [0.0, 0.5]

project:
  train_csv: data/train.csv
  target_col: y
```

Notes:
- `storage_url` is required for multiprocessing (sqlite or postgres).
- `search_space` supports `range`, `choices`, lists, and fixed values.
- `project` is free-form for your project.
- JSON is still supported if you already have it.
- YAML requires PyYAML (`pip install pyyaml`).

## 2) ObjectiveAdapter (objective function)

The main hook is `ObjectiveAdapter.execute(params, trial)`.
Parameter suggestions are generated automatically from `search_space` via `suggest_params`.

Example:

```python
from optuna_framework.adapters.objective import ObjectiveAdapter, TrialResult

class MyObjectiveAdapter(ObjectiveAdapter):
    def execute(self, params, trial):
        # Train/evaluate and return a score (float)
        score = 0.123
        return TrialResult(value=score, user_attrs={"metric": score})
```

Available hooks in ObjectiveAdapter:
- `execute(params, trial)` (required)
- `suggest_params(trial, search_space)`
- `validate_search_space(search_space)` and `validate_trial_params(params)`
- `on_trial_start(trial, params)` / `on_trial_end(trial, value, params)`
- `worker_init()` and `setup()` / `teardown()` (optional)

If ObjectiveAdapter is not configured, the runner emits a warning and exits with error.

## 3) TrialAdapter and MasterAdapter (execution hooks)

TrialAdapter interface:
- `on_trial_start(context)` runs before each trial.
- `on_trial_end(context)` runs after each trial.

The `context` includes `role`, `study_name`, `trial_number`, `params`, `user_attrs`, `value`, `state`.

Example (TrialAdapter):

```python
from optuna_framework.adapters.trial import TrialAdapter

class MyTrialAdapter(TrialAdapter):
    def on_trial_start(self, context):
        # Hook before the trial
        pass

    def on_trial_end(self, context):
        # Hook after the trial
        pass
```

## 4) Run

```bash
python optuna-framework/main.py --params path/to/parameters.yaml
```

If you don't want to store the adapter in YAML:

```bash
python optuna-framework/main.py --params path/to/parameters.yaml --objective-adapter myproj.optuna.objective:MyObjectiveAdapter
```

Useful options:
- `--trials 50` for quick runs.
- `--continue-study` to avoid auto-incrementing `study_version`.
- `--trial-adapter` and `--master-adapter` for extra hooks.

## Result

The runner writes a JSON with the best result to `optuna.out_path` (default `optuna_best.json`).
It includes `best_value`, `best_params`, `best_params_full`, `best_params_grouped`, and `best_user_attrs`.

## 5) Optional pruning adapter

Add `prune_adapter` in `meta` or pass `--prune-adapter` to centralize pruning logic:

```yaml
meta:
  prune_adapter: myproj.optuna.prune:MyPruneAdapter
```

```bash
python optuna-framework/main.py --params path/to/parameters.yaml --prune-adapter myproj.optuna.prune:MyPruneAdapter
```

The framework will call the prune adapter **before** `execute()`.
