from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPORT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/reports")
PROTOCOLS = ["cross_subject", "cross_day", "within_subject_day"]


def main() -> int:
    summaries = {
        protocol: json.loads((REPORT_ROOT / f"b0_fusion_matrix_{protocol}_summary.json").read_text(encoding="utf-8"))
        for protocol in PROTOCOLS
    }
    rows: list[dict[str, Any]] = []
    for protocol, summary in summaries.items():
        for experiment in summary["experiments"]:
            rows.append(
                {
                    "protocol": protocol,
                    "experiment": experiment["experiment"],
                    "rmse": experiment["test"]["rmse"],
                    "mae": experiment["test"]["mae"],
                    "pooled_r": experiment["pooled_raw_pearson_r"],
                    "centered_r": experiment["within_subject_centered_r"],
                    "per_subject_r_mean": experiment["per_subject_r"]["mean"],
                    "per_subject_r_std": experiment["per_subject_r"]["std"],
                    "split_counts": experiment["split_counts"],
                }
            )
    best_rmse = {
        protocol: min(_protocol_rows(rows, protocol), key=lambda row: float(row["rmse"]))
        for protocol in PROTOCOLS
    }
    best_pooled_r = {
        protocol: max(
            [row for row in _protocol_rows(rows, protocol) if row["pooled_r"] is not None],
            key=lambda row: float(row["pooled_r"]),
        )
        for protocol in PROTOCOLS
    }
    payload = {
        "protocols": PROTOCOLS,
        "best_rmse": best_rmse,
        "best_pooled_r": best_pooled_r,
        "rows": rows,
    }
    json_path = REPORT_ROOT / "b0_fusion_matrix_all_protocols_summary.json"
    md_path = REPORT_ROOT / "b0_fusion_matrix_all_protocols_summary.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")
    print(f"json={json_path}")
    print(f"md={md_path}")
    return 0


def _protocol_rows(rows: list[dict[str, Any]], protocol: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["protocol"] == protocol]


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# B0 Fusion Matrix All Protocols Summary",
        "",
        "| protocol | experiment | RMSE | MAE | pooled r | centered r | per-subject r mean/std |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for protocol in payload["protocols"]:
        for row in _protocol_rows(payload["rows"], protocol):
            lines.append(
                "| {protocol} | {experiment} | {rmse} | {mae} | {pooled_r} | {centered_r} | {ps_mean}/{ps_std} |".format(
                    protocol=protocol,
                    experiment=row["experiment"],
                    rmse=_fmt(row["rmse"]),
                    mae=_fmt(row["mae"]),
                    pooled_r=_fmt(row["pooled_r"]),
                    centered_r=_fmt(row["centered_r"]),
                    ps_mean=_fmt(row["per_subject_r_mean"]),
                    ps_std=_fmt(row["per_subject_r_std"]),
                )
            )
    lines.extend(
        [
            "",
            "## Best By Protocol",
            "",
            "| protocol | best RMSE | best pooled r | split counts |",
            "| --- | --- | --- | --- |",
        ]
    )
    for protocol in payload["protocols"]:
        best_rmse = payload["best_rmse"][protocol]
        best_r = payload["best_pooled_r"][protocol]
        lines.append(
            "| {protocol} | {rmse_exp} ({rmse}) | {r_exp} ({r}) | {splits} |".format(
                protocol=protocol,
                rmse_exp=best_rmse["experiment"],
                rmse=_fmt(best_rmse["rmse"]),
                r_exp=best_r["experiment"],
                r=_fmt(best_r["pooled_r"]),
                splits=best_rmse["split_counts"],
            )
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
