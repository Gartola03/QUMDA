import yaml
from pathlib import Path
from .paths import CONFIG_DIR

__all__ = [
    "load_config",
    "save_config",
    "update_config",
    "add_config_param",
    "delete_config_param"
]

CONFIG_PATH = CONFIG_DIR/ "config.yaml"

def load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        with open(path, "w") as f:
            f.write("{}\n")
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config

def save_config(config: dict, path: Path = CONFIG_PATH):
    with open(path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

def update_config(updates: dict, path: Path = CONFIG_PATH):
    config = load_config(path)
    config.update(updates)
    save_config(config, path)

def add_config_param(key: str, value, path: Path = CONFIG_PATH):
    config = load_config(path)
    parts = key.split('.')
    d = config
    for p in parts[:-1]:
        if p not in d or not isinstance(d[p], dict):
            d[p] = {}
        d = d[p]
    d[parts[-1]] = value
    save_config(config, path)

def delete_config_param(key: str, path: Path = CONFIG_PATH):
    config = load_config(path)
    parts = key.split('.')
    d = config
    for p in parts[:-1]:
        if p in d and isinstance(d[p], dict):
            d = d[p]
        else:
            return
    d.pop(parts[-1], None)
    save_config(config, path)