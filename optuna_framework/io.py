import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be a dict: {path}")
    return data


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def load_params(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML params. Install with 'pip install pyyaml'.")
        if not path.exists():
            raise FileNotFoundError(f"YAML not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"YAML root must be a dict: {path}")
        return data
    if suffix == ".json":
        return load_json(path)
    raise ValueError(f"Unsupported params file extension '{path.suffix}'. Use .json or .yaml/.yml.")


def save_params(path: Path, payload: Dict[str, Any]) -> None:
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        if yaml is None:
            raise ImportError("PyYAML is required to save YAML params. Install with 'pip install pyyaml'.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
        return
    if suffix == ".json":
        save_json(path, payload)
        return
    raise ValueError(f"Unsupported params file extension '{path.suffix}'. Use .json or .yaml/.yml.")
