# Optuna Framework - Uso

Este framework extrae el runner multiproceso de Optuna y lo hace genérico mediante Adapters.

## 1) Archivo de parámetros (JSON)

Formato mínimo:

```json
{
  "meta": {
    "name": "exp001",
    "seed": 42,
    "study_version": 1,
    "objective_adapter": "myproj.optuna_adapter:MyObjectiveAdapter",
    "worker_adapter": "myproj.optuna_worker:MyWorkerAdapter",
    "master_adapter": "myproj.optuna_master:MyMasterAdapter"
  },
  "optuna": {
    "n_trials": 100,
    "n_jobs": 4,
    "storage_url": "sqlite:///optuna.db",
    "sampler": "tpe",
    "timeout_sec": 0,
    "out_path": "results/optuna_best.json"
  },
  "search_space": {
    "lr": {"range": [1e-4, 1e-2], "log": true},
    "batch_size": [16, 32, 64],
    "dropout": [0.0, 0.5]
  },
  "project": {
    "train_csv": "data/train.csv",
    "target_col": "y"
  }
}
```

Notas:
- `storage_url` es obligatorio para multiproceso (sqlite o postgres).
- `search_space` soporta `range`, `choices`, listas y valores fijos.
- `project` es libre para tu proyecto.

## 2) ObjectiveAdapter (función objetivo)

El hook principal de la función objetivo es `ObjectiveAdapter.execute(params, trial)`.
La sugerencia de parámetros se hace automáticamente con `search_space`, vía `suggest_params`.

Ejemplo:

```python
from optuna_framework.adapters.objective import ObjectiveAdapter, TrialResult

class MyObjectiveAdapter(ObjectiveAdapter):
    def setup(self):
        # Cargar data o modelos base por worker
        pass

    def validate_trial_params(self, params):
        errors = []
        if params.get("lr", 0) <= 0:
            errors.append("lr must be > 0")
        return errors

    def execute(self, params, trial):
        # Entrenar/evaluar y devolver un score (float)
        score = 0.123
        return TrialResult(value=score, user_attrs={"metric": score})
```

Hooks disponibles en ObjectiveAdapter:
- `worker_init()` y `setup()` / `teardown()`
- `suggest_params(trial, search_space)`
- `validate_search_space(search_space)` y `validate_trial_params(params)`
- `on_trial_start(trial, params)` / `on_trial_end(trial, value, params)`
- `execute(params, trial)` (obligatorio)

Si no se configura el ObjectiveAdapter, el runner emite un warning y termina con error.

## 3) WorkerAdapter y MasterAdapter (hooks de ejecución)

Ambos comparten la misma interfaz:
- `init(context)` se ejecuta una vez al inicio.
- `execute(context)` se ejecuta antes de cada trial.
- `finish(context)` se ejecuta al final de cada trial.

El `context` incluye `role`, `study_name`, `trial_number`, `params`, `user_attrs`, `value`, `state`.

Ejemplo (WorkerAdapter):

```python
from optuna_framework.adapters.worker import WorkerAdapter

class MyWorkerAdapter(WorkerAdapter):
    def init(self, context):
        pass

    def execute(self, context):
        # Hook antes del trial
        pass

    def finish(self, context):
        # Hook después del trial
        pass
```

## 4) Ejecutar

```bash
python optuna-framework/main.py --params path/to/params.json
```

Si no quieres guardar el adapter en el JSON:

```bash
python optuna-framework/main.py --params path/to/params.json --objective-adapter myproj.optuna_adapter:MyObjectiveAdapter
```

Opciones útiles:
- `--trials 50` para pruebas rápidas.
- `--continue-study` para no auto-incrementar `study_version`.
- `--worker-adapter` y `--master-adapter` para hooks adicionales.

## Resultado

El runner escribe un JSON con el mejor resultado en `optuna.out_path` (por defecto `optuna_best.json`).
Incluye `best_value`, `best_params`, `best_params_full`, `best_params_grouped` y `best_user_attrs`.
