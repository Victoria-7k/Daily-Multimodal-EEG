#!/usr/bin/env python3
"""Run resumable event-supervised 11-label EEGPT partial fine-tuning."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main() -> int:
    source = Path(__file__).with_name("93_run_eegpt_single_label.py")
    spec = importlib.util.spec_from_file_location("_single_label_eeg_runner", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runner: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return int(module.main(default_mode="multitask"))


if __name__ == "__main__":
    raise SystemExit(main())
