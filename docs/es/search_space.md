# Search Space y Suggest Values

Este framework usa Optuna para sugerir hiperparámetros en cada trial. Tú defines el search space una vez, y el framework genera los parámetros automáticamente.

## Dos archivos: parameters.yaml y search_space.json

- `parameters.yaml` contiene la configuración oficial de la corrida (meta, optuna, project, etc.).
- `search_space.json` contiene solo las definiciones del search space.

Durante la optimización, el framework **sobrescribe los parámetros del modelo en cada trial** con los valores sugeridos desde `search_space`.
Piensa en `parameters.yaml` como tu configuración base y en `search_space` como la fuente de valores por trial.

## Cómo funcionan las sugerencias

La implementación por defecto llama:
- `ObjectiveAdapter.suggest_params(trial, search_space)`
- Que usa `optuna_framework.search_space.suggest_value()` para llamar a los métodos `suggest_*` de Optuna.

Formatos soportados:

```json
{
  "lr": {"range": [1e-4, 1e-2], "log": true},
  "batch_size": [16, 32, 64],
  "dropout": [0.0, 0.5],
  "fixed_flag": true
}
```

Reglas:
- `{"range": [lo, hi], "log": true}` -> rango float, log-uniforme.
- Listas como `[16, 32, 64]` -> categórico.
- Escalares como `true` o `0.5` -> valores fijos.

## Uso de parámetros sugeridos en tu modelo

En tu ObjectiveAdapter recibes los `params` sugeridos por trial. Úsalos directamente en tu lógica de entrenamiento.

```python
class MyObjectiveAdapter(ObjectiveAdapter):
    def execute(self, params, trial):
        lr = params["lr"]
        batch_size = params["batch_size"]
        dropout = params["dropout"]

        score = train_model(lr=lr, batch_size=batch_size, dropout=dropout)
        return score
```

## Avanzado: sugerencias personalizadas

Si necesitas sugerencias condicionales o dinámicas, sobrescribe `suggest_params`:

```python
class MyObjectiveAdapter(ObjectiveAdapter):
    def suggest_params(self, trial, search_space):
        params = super().suggest_params(trial, search_space)
        if params["model"] == "cnn":
            params["kernel_size"] = trial.suggest_int("kernel_size", 3, 7)
        return params
```
