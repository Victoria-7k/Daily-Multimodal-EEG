from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np  # noqa: E402

from daily_multimodal.training.cross_attention_fusion import build_fusion_dataset  # noqa: E402
from daily_multimodal.training.fusion_matrix import (  # noqa: E402
    branches_for_experiment,
    load_fusion_matrix_config,
    matrix_experiment_specs,
)
from daily_multimodal.training.within_subject_splits import (  # noqa: E402
    build_global_paired_cohort,
    build_split_manifest,
    load_window_metadata,
    write_cohort_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare frozen within-subject fusion cohort and split manifests.")
    parser.add_argument("--config", default="configs/within_subject_fusion.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fusion_config_path = Path(config["fusion_config"])
    window_index_path = Path(config["window_index"])
    cohort_manifest_path = Path(config["cohort_manifest"])
    split_manifest_path = Path(config["split_manifest"])
    split_seed = int(config.get("split_seed", 17))

    matrix_config = load_fusion_matrix_config(fusion_config_path)
    specs = matrix_experiment_specs(matrix_config)
    sample_ids_by_experiment: dict[str, np.ndarray] = {}
    native_counts: dict[str, int] = {}
    native_datasets = {}
    for spec in specs:
        dataset = build_fusion_dataset(
            branches=branches_for_experiment(matrix_config, spec.name),
            experiment=spec,
            metadata_source=matrix_config.metadata_source,
        )
        native_datasets[spec.name] = dataset
        sample_ids_by_experiment[spec.name] = dataset.sample_id
        native_counts[spec.name] = int(len(dataset.sample_id))

    reference = native_datasets[specs[0].name].sample_id
    cohort = build_global_paired_cohort(sample_ids_by_experiment, reference_order=reference)
    strict_datasets = {}
    for spec in specs:
        strict = build_fusion_dataset(
            branches=branches_for_experiment(matrix_config, spec.name),
            experiment=spec,
            base_sample_ids=cohort,
            metadata_source=matrix_config.metadata_source,
        )
        if strict.sample_id.astype(str).tolist() != cohort.astype(str).tolist():
            raise ValueError(f"{spec.name} strict dataset does not preserve the paired cohort order")
        strict_datasets[spec.name] = strict
    _assert_strict_metadata_match(strict_datasets)

    metadata = load_window_metadata(window_index_path, cohort)
    source_hashes = {
        "within_subject_config": _sha256_file(config_path),
        "fusion_config": _sha256_file(fusion_config_path),
        "window_index": _sha256_file(window_index_path),
    }
    cohort_manifest = write_cohort_manifest(
        cohort=cohort,
        native_counts=native_counts,
        source_hashes=source_hashes,
    )
    split_manifest = build_split_manifest(cohort, metadata, split_seed=split_seed)
    split_manifest["cohort_sha256"] = cohort_manifest["sample_id_sha256"]
    split_manifest["window_index_sha256"] = source_hashes["window_index"]

    print(f"native_experiment_count={len(specs)}")
    print(f"paired_cohort_count={len(cohort)}")
    print("protocols=event_grouped_5fold,session_held_out")
    print(f"cohort_manifest={cohort_manifest_path}")
    print(f"split_manifest={split_manifest_path}")
    if args.dry_run:
        return 0

    _write_manifest(cohort_manifest_path, cohort_manifest, force=args.force_rebuild)
    _write_manifest(split_manifest_path, split_manifest, force=args.force_rebuild)
    return 0


def _assert_strict_metadata_match(datasets: dict[str, object]) -> None:
    first_name = next(iter(datasets))
    first = datasets[first_name]
    for name, dataset in datasets.items():
        if dataset.sample_id.astype(str).tolist() != first.sample_id.astype(str).tolist():
            raise ValueError(f"{name} sample_id values differ from {first_name}")
        if dataset.event_id.astype(str).tolist() != first.event_id.astype(str).tolist():
            raise ValueError(f"{name} event_id values differ from {first_name}")
        if dataset.subject_id.astype(str).tolist() != first.subject_id.astype(str).tolist():
            raise ValueError(f"{name} subject_id values differ from {first_name}")
        np.testing.assert_allclose(dataset.target, first.target)


def _write_manifest(path: Path, payload: dict, *, force: bool) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text and not force:
        raise ValueError(f"{path} already exists with different content; pass --force-rebuild")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
