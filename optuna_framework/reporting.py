from pathlib import Path
from typing import Any, Dict, Optional

from optuna_framework.io import save_json


def build_best_payload(
    study_name: str,
    study_version: Optional[int],
    best_value: float,
    best_params: Dict[str, Any],
    best_params_full: Dict[str, Any],
    best_params_grouped: Dict[str, Any],
    best_user_attrs: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "study_name": study_name,
        "study_version": study_version,
        "best_value": best_value,
        "best_params": best_params,
        "best_params_full": best_params_full,
        "best_params_grouped": best_params_grouped,
        "best_user_attrs": best_user_attrs,
    }


def write_best_json(out_path: Path, payload: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, payload)


