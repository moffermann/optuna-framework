import importlib
from typing import Any


def load_object(path: str) -> Any:
    if ":" in path:
        module_name, attr_name = path.split(":", 1)
    else:
        module_name, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


