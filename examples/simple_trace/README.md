# Simple Trace Example

This example uses adapters that only print hook events so you can see the full execution flow.

## Run

```bash
python main.py --params examples/simple_trace/parameters.yaml
```

## Notes
- Uses SQLite storage under `examples/simple_trace/optuna_trace.db`.
- Writes best trial output to `examples/simple_trace/optuna_best.json`.
- The objective is a dummy score: sum of numeric params.
- JSON params are still supported, but YAML is preferred.
