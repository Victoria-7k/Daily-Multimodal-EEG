from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.training.within_subject_runner import (  # noqa: E402
    load_backend_decision,
    run_within_subject_matrix,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the leakage-audited within-subject fusion matrix.")
    parser.add_argument("--config", default="configs/within_subject_fusion.yaml")
    parser.add_argument("--protocol", choices=["event_grouped_5fold", "session_held_out"])
    parser.add_argument("--out-dir", default="outputs/reports/fusion_matrix_within_subject_120s10s")
    parser.add_argument("--model-dir", default="outputs/models/fusion_matrix_within_subject_120s10s")
    parser.add_argument("--backend-decision")
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--screen-subjects")
    parser.add_argument("--screen-experiments")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--video-adapter-epochs", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    runtime = _effective_runtime(args, config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = out_dir / "within_subject_fusion_runtime.json"
    runtime_path.write_text(json.dumps(runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"runtime_path={runtime_path}")
    print(f"device={runtime['device']}")
    print(f"workers={runtime['workers']}")
    if args.dry_run:
        return 0
    selected_subjects = None
    if args.screen_subjects:
        selected_subjects = [value.strip() for value in args.screen_subjects.split(",") if value.strip()]
    selected_experiments = None
    if args.screen_experiments:
        selected_experiments = [value.strip() for value in args.screen_experiments.split(",") if value.strip()]
    summary = run_within_subject_matrix(
        config_path=args.config,
        out_dir=args.out_dir,
        model_dir=args.model_dir,
        protocol=args.protocol or "event_grouped_5fold",
        device=runtime["device"],
        workers=runtime["workers"],
        screen_subjects=selected_subjects,
        screen_experiments=selected_experiments,
        epochs=runtime["epochs"],
        hidden_dim=runtime["hidden_dim"],
        video_adapter_epochs=args.video_adapter_epochs,
        resume=runtime["resume"],
        production=runtime["production"],
    )
    print(f"job_count={summary['job_count']}")
    print(f"summary_path={Path(args.out_dir) / 'within_subject_fusion_summary.json'}")
    return 0


def _effective_runtime(args: argparse.Namespace, config: dict) -> dict:
    decision = None
    decision_hash = None
    if args.backend_decision:
        decision = load_backend_decision(args.backend_decision)
        decision_hash = sha256_file(args.backend_decision)
    explicit_runtime = args.device is not None and args.workers is not None
    if decision is not None and not explicit_runtime:
        device = decision["device"]
        workers = decision["workers"]
    else:
        device = args.device or "cpu"
        workers = int(args.workers or 1)
    production = config.get("production", {})
    return {
        "config": str(args.config),
        "protocol": args.protocol,
        "out_dir": str(args.out_dir),
        "model_dir": str(args.model_dir),
        "device": device,
        "workers": workers,
        "backend_decision": None if args.backend_decision is None else str(args.backend_decision),
        "backend_decision_sha256": decision_hash,
        "production": bool(args.production),
        "resume": bool(args.resume),
        "screen_subjects": args.screen_subjects,
        "epochs": int(args.epochs if args.epochs is not None else production.get("epochs", 200)),
        "hidden_dim": int(args.hidden_dim if args.hidden_dim is not None else production.get("hidden_dim", 128)),
        "split_seed": config.get("split_seed"),
        "model_seed": config.get("model_seed"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
