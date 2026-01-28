# Optuna Framework - Uso

Este framework extrae el runner multiproceso de Optuna y lo hace genérico mediante adapters.

## 1) Archivo de parámetros (YAML)

Formato mínimo:

```yaml
meta:
  name: exp001
  seed: 42
  study_version: 1
  objective_adapter: myproj.optuna.adapter:MyObjectiveAdapter
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

Notas:
- `storage_url` es obligatorio para multiproceso (sqlite o postgres).
- `search_space` soporta `range`, `choices`, listas y valores fijos.
- `project` es libre para tu proyecto.
- JSON también es aceptado si ya lo tienes.
- Para YAML necesitas PyYAML (`pip install pyyaml`).

## 2) ObjectiveAdapter (función objetivo)

El hook principal es `ObjectiveAdapter.execute(params, trial)`.
Las sugerencias de parámetros se generan automáticamente desde `search_space` vía `suggest_params`.

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

Si el ObjectiveAdapter no se configura, el runner emite un warning y termina con error.

## 3) TrialAdapter y MasterAdapter (hooks de ejecución)

Ambos comparten la misma interfaz:
- `init(context)` se ejecuta una vez al inicio (por worker o master).
- `execute(context)` se ejecuta antes de cada trial.
- `finish(context)` se ejecuta al final de cada trial.

El `context` incluye `role`, `study_name`, `trial_number`, `params`, `user_attrs`, `value`, `state`.

Ejemplo (TrialAdapter):

```python
from optuna_framework.adapters.trial import TrialAdapter

class MyTrialAdapter(TrialAdapter):
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
python optuna-framework/main.py --params path/to/parameters.yaml
```

Si no quieres guardar el adapter en el YAML:

```bash
python optuna-framework/main.py --params path/to/parameters.yaml --objective-adapter myproj.optuna.adapter:MyObjectiveAdapter
```

Opciones útiles:
- `--trials 50` para pruebas rápidas.
- `--continue-study` para no auto-incrementar `study_version`.
- `--trial-adapter` y `--master-adapter` para hooks adicionales.

## Resultado

El runner escribe un JSON con el mejor resultado en `optuna.out_path` (por defecto `optuna_best.json`).
Incluye `best_value`, `best_params`, `best_params_full`, `best_params_grouped` y `best_user_attrs`.

## 5) Adapter de poda opcional

Agrega `prune_adapter` en `meta` o pasa `--prune-adapter` para centralizar la poda:

```yaml
meta:
  prune_adapter: myproj.optuna.prune:MyPruneAdapter
```

```bash
python optuna-framework/main.py --params path/to/parameters.yaml --prune-adapter myproj.optuna.prune:MyPruneAdapter
```

El framework llama al adapter de poda **antes** de `execute()`.
