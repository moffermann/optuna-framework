# Search Space and Suggest Values

This framework uses Optuna to suggest hyperparameters for each trial. You define the search space once, and the framework generates trial parameters automatically.

## Two files: parameters.yaml and search_space.json

- `parameters.yaml` contains your official run configuration (meta, optuna settings, project config, etc.).
- `search_space.json` contains only the search space definitions.

During optimization, the framework **overrides the model parameters for each trial** with the values suggested from `search_space`.
Think of `parameters.yaml` as your base config, and `search_space` as the source of trial-specific values.

## How suggestions work

The default implementation calls:
- `ObjectiveAdapter.suggest_params(trial, search_space)`
- Which uses `optuna_framework.search_space.suggest_value()` to call Optuna's `suggest_*` methods.

Supported search space formats:

```json
{
  "lr": {"range": [1e-4, 1e-2], "log": true},
  "batch_size": [16, 32, 64],
  "dropout": [0.0, 0.5],
  "fixed_flag": true
}
```

Rules:
- `{"range": [lo, hi], "log": true}` -> float range, log-uniform.
- Lists like `[16, 32, 64]` -> categorical.
- Scalars like `true` or `0.5` -> fixed values.

## Using the suggested params in your model

In your ObjectiveAdapter, you receive the suggested `params` per trial. Use them directly in your training logic.

```python
class MyObjectiveAdapter(ObjectiveAdapter):
    def execute(self, params, trial):
        # Example: use suggested parameters
        lr = params["lr"]
        batch_size = params["batch_size"]
        dropout = params["dropout"]

        score = train_model(lr=lr, batch_size=batch_size, dropout=dropout)
        return score
```

## Advanced: custom suggestions

If you need conditional or dynamic suggestions, override `suggest_params`:

```python
class MyObjectiveAdapter(ObjectiveAdapter):
    def suggest_params(self, trial, search_space):
        params = super().suggest_params(trial, search_space)
        if params["model"] == "cnn":
            params["kernel_size"] = trial.suggest_int("kernel_size", 3, 7)
        return params
```
