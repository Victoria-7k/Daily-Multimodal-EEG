from __future__ import annotations

from datetime import datetime


_TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%d-%b-%Y %H:%M:%S",
)


def parse_absolute_time(value: str) -> datetime:
    """Parse time strings observed in EEG JSON, behavior TSV, and wear CSV files."""
    text = value.strip()
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported absolute time format: {value!r}")


def subject_to_video_subject(subject_id: str) -> str:
    """Convert BIDS-like subject id, e.g. sub-02, to video folder id, e.g. sub2."""
    prefix, number = subject_id.split("-", 1)
    return f"{prefix}{int(number)}"


def time_to_video_day(value: datetime) -> str:
    """Return the MMDD folder name used by the video tree."""
    return value.strftime("%m%d")

