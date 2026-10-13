from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import get_type_hints

import yaml

from .schema import ExperimentConfig
from .validation import validate_config


def merge(base, updates):
    result = deepcopy(base)
    for key, value in updates.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def read_yaml(path, visited=None):
    path = Path(path).resolve()
    visited = set() if visited is None else set(visited)
    if path in visited:
        raise ValueError(f"Cyclic configuration inheritance: {path}")
    visited.add(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a YAML mapping: {path}")
    parent = raw.pop("base", None)
    return merge(read_yaml(path.parent / parent, visited), raw) if parent else raw


def from_dict(cls, raw):
    names = {item.name for item in fields(cls)}
    if set(raw) - names:
        raise KeyError(f"Unknown {cls.__name__} fields: {sorted(set(raw) - names)}")
    hints = get_type_hints(cls)
    values = {}
    for key, value in raw.items():
        target = hints[key]
        if isinstance(target, type) and is_dataclass(target):
            if not isinstance(value, dict):
                raise TypeError(f"{key} must be a mapping")
            value = from_dict(target, value)
        values[key] = value
    return cls(**values)


def load_config(path, overrides=()):
    raw = read_yaml(path)
    for override in overrides:
        if "=" not in override:
            raise ValueError("Overrides must have the form section.key=value")
        key, value = override.split("=", 1)
        node = raw
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    config = from_dict(ExperimentConfig, raw)
    validate_config(config)
    return config


def save_config(config, path):
    Path(path).write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
