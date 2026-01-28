# Poda en este Framework

La poda permite detener un trial temprano cuando ya no es prometedor. Esto acelera la búsqueda y evita gastar cómputo.

## Por qué usar un PruneAdapter

En muchos proyectos, las validaciones de poda son largas y ensucian el código del objetivo. Un `PruneAdapter` permite centralizar la lógica en un módulo aparte.

El framework llama al adapter de poda **antes** de `execute()`, así puedes rechazar combinaciones malas temprano.
Igual puedes podar dentro de `execute()` cuando necesitas métricas intermedias.

## Cómo funciona

1) Optuna sugiere parámetros desde `search_space`.
2) El framework llama `PruneAdapter.prune(params, trial)`.
3) Si el adapter lanza `optuna.TrialPruned`, el trial queda PRUNED.
4) Si no, el objetivo `execute()` corre normalmente.

## Ejemplo de PruneAdapter

```python
import optuna
from optuna_framework.adapters.prune import PruneAdapter

class MyPruneAdapter(PruneAdapter):
    def prune(self, params, trial):
        # Ejemplo: podar combinación inválida
        if params["lr"] > 0.01 and params["batch_size"] < 16:
            raise optuna.TrialPruned("combinación inválida lr/batch_size")
```

## Ejemplo de ObjectiveAdapter

```python
class MyObjectiveAdapter(ObjectiveAdapter):
    def execute(self, params, trial):
        score = train_model(**params)
        return score
```

## Poda dentro de execute()

Si podas según métricas intermedias, sigue usando `trial.report()` y `trial.should_prune()` dentro de `execute()`.
