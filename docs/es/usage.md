# Optuna Framework - Uso

Este framework extrae el runner multiproceso de Optuna y lo hace genérico mediante adapters.

## 1) Archivo de parámetros (YAML)

Formato mínimo:

```yaml
meta:
  name: exp001
  seed: 42
  study_version: 1
  objective_adapter: myproj.optuna.objective:MyObjectiveAdapter
  worker_adapter: myproj.optuna.worker:MyWorkerAdapter
  trial_adapter: myproj.optuna.trial:MyTrialAdapter
  optuna_adapter: myproj.optuna.optuna:MyOptunaAdapter

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
    def execute(self, params, trial):
        # Entrenar/evaluar y devolver un score (float)
        score = 0.123
        return TrialResult(value=score, user_attrs={"metric": score})
```

Hooks disponibles en ObjectiveAdapter:
- `execute(params, trial)` (obligatorio)
- `suggest_params(trial, search_space)`
- `validate_search_space(search_space)` y `validate_trial_params(params)`
- `on_trial_start(trial, params)` / `on_trial_end(trial, value, params)`
- `worker_init()` y `setup()` / `teardown()` (opcional)

Si el ObjectiveAdapter no se configura, el runner emite un warning y termina con error.

## 3) TrialAdapter, WorkerAdapter y OptunaAdapter (hooks de ejecución)

Interfaz de TrialAdapter:
- `on_trial_start(context)` se ejecuta antes de cada trial.
- `on_trial_end(context)` se ejecuta al final de cada trial.

Interfaz de WorkerAdapter:
- `on_worker_start(context)` se ejecuta una vez al iniciar el worker.
- `on_worker_end(context)` se ejecuta una vez al terminar el worker.

El `context` incluye `role`, `study_name`, `trial_number`, `params`, `user_attrs`, `value`, `state`.

Ejemplo (TrialAdapter):

```python
from optuna_framework.adapters.trial import TrialAdapter

class MyTrialAdapter(TrialAdapter):
    def on_trial_start(self, context):
        # Hook antes del trial
        pass

    def on_trial_end(self, context):
        # Hook después del trial
        pass

Ejemplo (WorkerAdapter):

```python
from optuna_framework.adapters.worker import WorkerAdapter

class MyWorkerAdapter(WorkerAdapter):
    def on_worker_start(self, context):
        pass

    def on_worker_end(self, context):
        pass

Ejemplo (OptunaAdapter):

```python
from optuna_framework.adapters.optuna import OptunaAdapter

class MyOptunaAdapter(OptunaAdapter):
    def on_optuna_start(self, context):
        pass

    def on_optuna_end(self, context):
        pass
```
```
```

## 4) Ejecutar

```bash
python optuna-framework/main.py --params path/to/parameters.yaml
```

Si no quieres guardar el adapter en el YAML:

```bash
python optuna-framework/main.py --params path/to/parameters.yaml --objective-adapter myproj.optuna.objective:MyObjectiveAdapter
```

Opciones útiles:
- `--trials 50` para pruebas rápidas.
- `--continue-study` para no auto-incrementar `study_version`.
- `--trial-adapter`, `--worker-adapter` y `--optuna-adapter` para hooks adicionales.

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
