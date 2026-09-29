"""Strict JSON normalization and atomic writers for R2 experiment outputs."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _child_path(path: str, key: Any) -> str:
    return f"{path}[{key}]" if isinstance(key, int) else f"{path}.{key}"


def to_jsonable(value: Any, path: str = "$") -> Any:
    """Recursively convert explicitly supported scalar/container types to JSON-native values."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return to_jsonable(value.item(), path)
        raise TypeError(f"non-scalar numpy array at {path} is not an allowed JSON output: "
                        f"shape={value.shape}, dtype={value.dtype}")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)

    value_type = type(value)
    type_module, type_name = value_type.__module__, value_type.__name__
    if type_module.startswith("torch") and type_name == "Tensor":
        if value.numel() != 1:
            raise TypeError(f"non-scalar torch tensor at {path} is not an allowed JSON output: "
                            f"shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")
        return to_jsonable(value.detach().cpu().item(), path)
    if type_module.startswith("torch") and type_name in {"device", "dtype"}:
        return str(value)

    if isinstance(value, Mapping):
        output = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object key at {path} must be str, got {type(key).__name__}: {key!r}")
            output[key] = to_jsonable(item, _child_path(path, key))
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item, _child_path(path, index)) for index, item in enumerate(value)]
    raise TypeError(f"unsupported JSON value at {path}: {type_module}.{type_name}")


def json_dumps(value: Any, *, indent: int | None = None, ensure_ascii: bool = False) -> str:
    return json.dumps(to_jsonable(value), indent=indent, ensure_ascii=ensure_ascii, allow_nan=False)


def _atomic_write_text(destination: Path, text: str) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent,
                                         prefix=f".{destination.name}.", suffix=".tmp",
                                         delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_write_json(destination: Path, value: Any, *, indent: int = 2,
                      ensure_ascii: bool = False) -> None:
    _atomic_write_text(Path(destination), json_dumps(value, indent=indent, ensure_ascii=ensure_ascii) + "\n")


def atomic_write_jsonl(destination: Path, rows: Iterable[Any], *, ensure_ascii: bool = False) -> None:
    text = "".join(json_dumps(row, ensure_ascii=ensure_ascii) + "\n" for row in rows)
    _atomic_write_text(Path(destination), text)
