from typing import Any, Dict, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

import optuna


def parse_spec(spec: Any, name: str) -> Dict[str, Any]:
    if isinstance(spec, dict) and "range" in spec:
        lo, hi = spec["range"]
        step = spec.get("step", None)
        log = bool(spec.get("log", False))
        return {"type": "range", "lo": lo, "hi": hi, "step": step, "log": log, "name": name}
    if isinstance(spec, dict) and "choices" in spec:
        return {"type": "cat", "choices": spec["choices"], "name": name}
    if isinstance(spec, list):
        has_bool = any(isinstance(x, bool) for x in spec)
        if len(spec) == 2 and all(isinstance(x, (int, float)) for x in spec) and not has_bool:
            return {"type": "range", "lo": spec[0], "hi": spec[1], "step": None, "log": False, "name": name}
        return {"type": "cat", "choices": spec, "name": name}
    return {"type": "fixed", "value": spec, "name": name}


def is_param_spec_dict(spec: Any) -> bool:
    return isinstance(spec, dict) and ("range" in spec or "choices" in spec)


def flatten_spec_tree(tree: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in tree.items():
        if is_param_spec_dict(value):
            flat[key] = value
        elif isinstance(value, dict):
            flat.update(flatten_spec_tree(value))
        else:
            flat[key] = value
    return flat


def normalize_value(value: Any) -> Any:
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return tuple(normalize_value(v) for v in value)
    return value


def suggest_value(trial: optuna.trial.Trial, name: str, spec: Any) -> Any:
    ps = parse_spec(spec, name)
    if ps["type"] == "fixed":
        return ps["value"]
    if ps["type"] == "cat":
        return trial.suggest_categorical(name, ps["choices"])
    lo = ps["lo"]
    hi = ps["hi"]
    step = ps.get("step", None)
    is_int = isinstance(lo, int) and isinstance(hi, int)
    if step is not None:
        try:
            step_is_int = float(step).is_integer()
        except (TypeError, ValueError):
            step_is_int = False
    else:
        step_is_int = True
    if is_int and step_is_int:
        if ps.get("log", False) and step is not None:
            raise ValueError(f"Param '{name}' cannot use log with a step for int range.")
        if step is None:
            return int(trial.suggest_int(name, int(lo), int(hi), log=bool(ps.get("log", False))))
        return int(trial.suggest_int(name, int(lo), int(hi), step=int(step)))
    return float(
        trial.suggest_float(
            name,
            float(lo),
            float(hi),
            step=step,
            log=bool(ps.get("log", False)),
        )
    )


def resolve_param_value(name: str, spec: Any, best_params: Dict[str, Any]) -> Any:
    if name in best_params:
        return best_params[name]
    ps = parse_spec(spec, name)
    if ps["type"] == "fixed":
        return ps["value"]
    raise ValueError(f"Missing param '{name}' in best_params, and spec is not fixed.")


def build_params_tree(spec_tree: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, spec in spec_tree.items():
        if is_param_spec_dict(spec):
            if key in values:
                out[key] = values[key]
        elif isinstance(spec, dict):
            out[key] = build_params_tree(spec, values)
        else:
            if key in values:
                out[key] = values[key]
    return out


