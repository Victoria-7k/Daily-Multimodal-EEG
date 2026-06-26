from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.config import load_simple_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Check server environment and source roots.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_simple_yaml(args.config)
    roots = config["data_roots"]
    failed = False
    for label, path_text in roots.items():
        path = Path(path_text)
        ok = path.exists()
        print(f"{label} root {'exists' if ok else 'missing'}: {path}")
        failed = failed or not ok
    ffprobe = shutil.which("ffprobe")
    print(f"ffprobe {'available' if ffprobe else 'missing'}: {ffprobe or ''}")
    failed = failed or ffprobe is None
    print(f"python: {sys.version.split()[0]}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

