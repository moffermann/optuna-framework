# Pruning in this Framework

Pruning lets you stop a trial early when it is clearly not promising. This keeps the search fast and avoids wasting compute.

## Why use a PruneAdapter

In many projects, pruning checks are long and clutter the objective code. A `PruneAdapter` lets you centralize pruning logic in a separate module.

The framework calls the prune adapter **before** `execute()`, so you can reject bad parameter combinations early.
You can still prune inside `execute()` when you need intermediate metrics.

## How it works

1) Optuna suggests parameters from `search_space`.
2) The framework calls `PruneAdapter.prune(params, trial)`.
3) If the prune adapter raises `optuna.TrialPruned`, the trial is marked as PRUNED.
4) Otherwise, the objective `execute()` runs normally.

## Example PruneAdapter

```python
import optuna
from optuna_framework.adapters.prune import PruneAdapter

class MyPruneAdapter(PruneAdapter):
    def prune(self, params, trial):
        # Example: prune invalid combination
        if params["lr"] > 0.01 and params["batch_size"] < 16:
            raise optuna.TrialPruned("invalid lr/batch_size combo")
```

## Example ObjectiveAdapter usage

```python
class MyObjectiveAdapter(ObjectiveAdapter):
    def execute(self, params, trial):
        # Your training logic uses suggested params
        score = train_model(**params)
        return score
```

## Still need in-execute pruning?

If you prune based on intermediate metrics, keep using `trial.report()` and `trial.should_prune()` inside `execute()`.
