from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


_WEAR_RE = re.compile(
    r"UID(?P<uid>\d+).*?_ID(?P<device_id>\d+)_"
    r"(?P<start>\d{14})_(?P<end>\d{14})"
    r"(?:_(?P<modality>ACC|GSR|PPG|min))?\.(?P<extension>csv|mat)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WearFile:
    path: Path
    uid: str
    device_id: str
    start_time: datetime
    end_time: datetime
    modality: str
    extension: str


def parse_wear_filename(path: Path | str) -> WearFile:
    """Parse a DailyEmotion wear-data filename into interval and modality metadata."""
    file_path = Path(path)
    match = _WEAR_RE.search(file_path.name)
    if not match:
        raise ValueError(f"Unrecognized wear filename: {file_path.name}")

    extension = match.group("extension").lower()
    raw_modality = match.group("modality")
    if raw_modality:
        modality = raw_modality.upper()
    elif extension == "mat":
        modality = "SUMMARY_MAT"
    else:
        modality = "SUMMARY"

    return WearFile(
        path=file_path,
        uid=match.group("uid"),
        device_id=match.group("device_id"),
        start_time=datetime.strptime(match.group("start"), "%Y%m%d%H%M%S"),
        end_time=datetime.strptime(match.group("end"), "%Y%m%d%H%M%S"),
        modality=modality,
        extension=extension,
    )


def discover_wear_files(root: Path | str) -> list[WearFile]:
    """Return all parseable wear files under the flat out directory."""
    root_path = Path(root)
    files: list[WearFile] = []
    for path in sorted(root_path.iterdir()):
        if not path.is_file():
            continue
        try:
            files.append(parse_wear_filename(path))
        except ValueError:
            continue
    return files

