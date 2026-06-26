from __future__ import annotations

from pathlib import Path
from typing import Any


def load_simple_yaml(path: Path | str) -> dict[str, Any]:
    """Load the small YAML subset used by this project without requiring PyYAML."""
    try:
        import yaml  # type: ignore

        with Path(path).open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Top-level YAML value must be a mapping")
        return loaded
    except ModuleNotFoundError:
        return _load_indented_mapping(Path(path))


def _load_indented_mapping(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, result)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, sep, value = raw_line.strip().partition(":")
        if not sep:
            raise ValueError(f"Unsupported YAML line: {raw_line}")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip():
            parent[key] = _coerce_scalar(value.strip())
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
    return result


def _coerce_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value

