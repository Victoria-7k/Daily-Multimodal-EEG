from __future__ import annotations

from daily_multimodal.schema import CONTEXT_COLUMNS, EMOTION_LABEL_COLUMNS


def label_columns() -> list[str]:
    return [*EMOTION_LABEL_COLUMNS, *CONTEXT_COLUMNS]

