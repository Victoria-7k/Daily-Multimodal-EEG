from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_WINDOW_SECONDS = 10.0
DEFAULT_TEMPORAL_TOKEN_SECONDS = 2.0


@dataclass(frozen=True)
class TemporalSlice:
    token_id: int
    start_seconds: float
    end_seconds: float


def make_temporal_slices(
    *,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    token_seconds: float = DEFAULT_TEMPORAL_TOKEN_SECONDS,
) -> tuple[TemporalSlice, ...]:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if token_seconds <= 0:
        raise ValueError("token_seconds must be positive")
    token_count = window_seconds / token_seconds
    rounded = round(token_count)
    if abs(token_count - rounded) > 1e-9:
        raise ValueError(
            f"window_seconds={window_seconds} must be divisible by token_seconds={token_seconds}"
        )
    return tuple(
        TemporalSlice(
            token_id=index,
            start_seconds=round(index * token_seconds, 9),
            end_seconds=round((index + 1) * token_seconds, 9),
        )
        for index in range(int(rounded))
    )


def build_temporal_token_index(
    rows: Iterable[dict[str, Any]],
    *,
    token_seconds: float = DEFAULT_TEMPORAL_TOKEN_SECONDS,
    window_seconds: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    counts: dict[int, int] = {}
    for row in rows:
        row_window_seconds = float(window_seconds if window_seconds is not None else row.get("window_size_seconds", DEFAULT_WINDOW_SECONDS))
        slices = make_temporal_slices(window_seconds=row_window_seconds, token_seconds=token_seconds)
        token_start = [_clean_number(item.start_seconds) for item in slices]
        token_end = [_clean_number(item.end_seconds) for item in slices]
        next_row = dict(row)
        next_row.update(
            {
                "temporal_token_seconds": _clean_number(token_seconds),
                "temporal_token_count": len(slices),
                "token_start_seconds": token_start,
                "token_end_seconds": token_end,
                "temporal_token_version": "2s_fixed_window_v1",
            }
        )
        output.append(next_row)
        counts[len(slices)] = counts.get(len(slices), 0) + 1
    audit = {
        "row_count": len(output),
        "token_seconds": _clean_number(token_seconds),
        "window_seconds": _clean_number(window_seconds) if window_seconds is not None else "from_index",
        "token_count_distribution": {str(key): value for key, value in sorted(counts.items())},
        "temporal_token_version": "2s_fixed_window_v1",
    }
    return output, audit


def read_jsonl(path: Path | str, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def write_json(data: dict[str, Any], path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _clean_number(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    number = float(value)
    if number.is_integer():
        return int(number)
    return number
