#!/usr/bin/env python3
"""Run the EEGPT-aligned raw/centered multi-task loss experiment matrix."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from daily_multimodal.training.centered_metrics import evaluate_regression_with_centered


LABEL_NAMES = [
    "inspired",
    "alert",
    "determined",
    "attentive",
    "active",
    "hostile",
    "nervous",
    "upset",
    "afraid",
    "ashamed",
    "fatigue",
]
DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_SPLITS_ROOT = Path("/vePFS-0x0d/DailyEEG/splits_new")
DEFAULT_EMBEDDINGS_ROOT = Path("/vePFS-0x0d/DailyEEG_multimodal/embeddings")
DEFAULT_EXPERIMENTS = (
    "cross_day:A1_Wphysio_no_audio",
    "within_subject_day:A1_Wphysio_no_audio",
    "within_subject_day:B0_Wdeep_no_audio",
)
DEFAULT_PROTOCOLS = ("cross_subject", "cross_day", "within_subject_day")


@dataclass(frozen=True)
class Branch:
    name: str
    modality: str
    filename: str
    emb_key: str
    mask_key: str
    modality_index: int


BRANCHES = {
    "eeg": Branch("eeg", "eeg", "eeg/eeg_eegpt_eeg23win_embeddings.npz", "eeg_emb", "eeg_mask", 0),
    "eeg_eegpt_frozen_v1": Branch("eeg_eegpt_frozen_v1", "eeg", "{eeg_token_root}/{protocol}/eegpt_frozen_v1/seed_{eeg_seed}.npz", "eeg_emb", "eeg_mask", 0),
    "eeg_eegpt_partial_ft_v1": Branch("eeg_eegpt_partial_ft_v1", "eeg", "{eeg_token_root}/{protocol}/eegpt_partial_ft_v1/seed_{eeg_seed}.npz", "eeg_emb", "eeg_mask", 0),
    "eeg_cbramod_frozen_v1": Branch("eeg_cbramod_frozen_v1", "eeg", "{eeg_token_root}/{protocol}/cbramod_frozen_v1/seed_{eeg_seed}.npz", "eeg_emb", "eeg_mask", 0),
    "eeg_cbramod_partial_ft_v1": Branch("eeg_cbramod_partial_ft_v1", "eeg", "{eeg_token_root}/{protocol}/cbramod_partial_ft_v1/seed_{eeg_seed}.npz", "eeg_emb", "eeg_mask", 0),
    "eeg_de_5band_1s_avg_v1": Branch("eeg_de_5band_1s_avg_v1", "eeg", "{eeg_token_root}/{protocol}/eeg_de_5band_1s_avg_v1/seed_{eeg_seed}.npz", "eeg_emb", "eeg_mask", 0),
    "wear_physio": Branch("wear_physio", "wear", "wear/wear_physio_preprocessed_eeg23win_embeddings.npz", "wear_emb", "wear_mask", 1),
    "wear_deep": Branch("wear_deep", "wear", "wear/wear_deep_sequence_preprocessed_eeg23win_embeddings.npz", "wear_emb", "wear_mask", 1),
    "wear_moment_frozen_v1": Branch("wear_moment_frozen_v1", "wear", "wear_tokens/{protocol}/wear_moment_frozen_v1/seed_{wear_seed}.npz", "wear_emb", "wear_mask", 1),
    "wear_moment_partial_ft_v1": Branch("wear_moment_partial_ft_v1", "wear", "wear_tokens/{protocol}/wear_moment_partial_ft_v1/seed_{wear_seed}.npz", "wear_emb", "wear_mask", 1),
    "video_B0": Branch("video_B0", "video", "video/video_B0_2xroi_eeg23win_embeddings.npz", "video_emb", "video_mask", 2),
    "video_A1": Branch("video_A1", "video", "video/video_A1_2xroi_eeg23win_embeddings.npz", "video_emb", "video_mask", 2),
    "video_A2": Branch("video_A2", "video", "video/video_A2_2xroi_eeg23win_embeddings.npz", "video_emb", "video_mask", 2),
    "audio": Branch("audio", "audio", "audio/audio_opensmile_eeg23win_embeddings.npz", "audio_emb", "audio_mask", 3),
}

EXPERIMENT_BRANCHES = {
    "B0_Wphysio_full": ("eeg", "wear_physio", "video_B0", "audio"),
    "B0_Wphysio_no_audio": ("eeg", "wear_physio", "video_B0"),
    "B0_Wphysio_no_video": ("eeg", "wear_physio", "audio"),
    "B0_Wphysio_bio_only": ("eeg", "wear_physio"),
    "B0_Wdeep_full": ("eeg", "wear_deep", "video_B0", "audio"),
    "B0_Wdeep_no_audio": ("eeg", "wear_deep", "video_B0"),
    "B0_Wdeep_no_video": ("eeg", "wear_deep", "audio"),
    "B0_Wdeep_bio_only": ("eeg", "wear_deep"),
    "A1_Wphysio_full": ("eeg", "wear_physio", "video_A1", "audio"),
    "A1_Wphysio_no_audio": ("eeg", "wear_physio", "video_A1"),
    "A1_Wdeep_full": ("eeg", "wear_deep", "video_A1", "audio"),
    "A1_Wdeep_no_audio": ("eeg", "wear_deep", "video_A1"),
    "A2_Wphysio_full": ("eeg", "wear_physio", "video_A2", "audio"),
    "A2_Wphysio_no_audio": ("eeg", "wear_physio", "video_A2"),
    "A2_Wdeep_full": ("eeg", "wear_deep", "video_A2", "audio"),
    "A2_Wdeep_no_audio": ("eeg", "wear_deep", "video_A2"),
    "A1_Wmoment_frozen_full": ("eeg", "wear_moment_frozen_v1", "video_A1", "audio"),
    "A1_Wmoment_ft_full": ("eeg", "wear_moment_partial_ft_v1", "video_A1", "audio"),
    "A1_Wmoment_frozen_no_audio": ("eeg", "wear_moment_frozen_v1", "video_A1"),
    "A1_Wmoment_ft_no_audio": ("eeg", "wear_moment_partial_ft_v1", "video_A1"),
    "B0_Wmoment_frozen_full": ("eeg", "wear_moment_frozen_v1", "video_B0", "audio"),
    "B0_Wmoment_frozen_no_audio": ("eeg", "wear_moment_frozen_v1", "video_B0"),
    "B0_Wmoment_ft_full": ("eeg", "wear_moment_partial_ft_v1", "video_B0", "audio"),
    "B0_Wmoment_ft_no_audio": ("eeg", "wear_moment_partial_ft_v1", "video_B0"),
    "A2_Wmoment_frozen_full": ("eeg", "wear_moment_frozen_v1", "video_A2", "audio"),
    "A2_Wmoment_frozen_no_audio": ("eeg", "wear_moment_frozen_v1", "video_A2"),
    "A2_Wmoment_ft_full": ("eeg", "wear_moment_partial_ft_v1", "video_A2", "audio"),
    "A2_Wmoment_ft_no_audio": ("eeg", "wear_moment_partial_ft_v1", "video_A2"),
}
VIDEO_ONLY_EXPERIMENTS = tuple(
    name
    for name in EXPERIMENT_BRANCHES
    if not name.endswith("_no_video") and not name.endswith("_bio_only")
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--embeddings-root", type=Path, default=DEFAULT_EMBEDDINGS_ROOT)
    parser.add_argument("--experiments", default=",".join(DEFAULT_EXPERIMENTS), help="protocol:experiment pairs")
    parser.add_argument(
        "--experiment-set",
        choices=("custom", "video_only"),
        default="custom",
        help="Use video_only to run all full/no_audio video-using routes and drop no_video/bio_only controls.",
    )
    parser.add_argument("--protocols", default=",".join(DEFAULT_PROTOCOLS), help="Protocols used when --experiment-set is not custom.")
    parser.add_argument(
        "--eeg-branches",
        default="eeg",
        help="Comma-separated EEG branches. Additional options include eeg_eegpt_frozen_v1,eeg_eegpt_partial_ft_v1,eeg_cbramod_frozen_v1,eeg_cbramod_partial_ft_v1,eeg_de_5band_1s_avg_v1.",
    )
    parser.add_argument("--eeg-token-seed", type=int, default=240800, help="Seed used in protocol-specific EEG prediction-token filenames.")
    parser.add_argument(
        "--eeg-token-root",
        default="eeg_encoder_tokens",
        help="Directory under --embeddings-root for protocol/profile EEG branch files. Use eeg_encoder_256d_tokens for full hidden embeddings.",
    )
    parser.add_argument(
        "--wear-token-seed",
        type=int,
        default=None,
        help="Seed used in protocol-specific wear-moment token filenames. Defaults to --eeg-token-seed.",
    )
    parser.add_argument("--loss-modes", default="raw_centered_mse,raw_centered_corr")
    parser.add_argument("--lambdas", default="0.1,0.3,0.5,1.0")
    parser.add_argument(
        "--heads",
        default="regression",
        help="Comma-separated heads: regression, ordinal_cumulative, classification_expectation, hybrid_regression_ordinal.",
    )
    parser.add_argument("--head-lambdas", default="0.1,0.3,1.0", help="Lambdas for hybrid_regression_ordinal.")
    parser.add_argument("--no-raw-baseline", action="store_true")
    parser.add_argument("--target-label", default="fatigue")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=240729)
    parser.add_argument(
        "--experiment-seed-fixed",
        action="store_true",
        help="Use the exact --seed value for every run instead of seed + run_number, so paired "
        "experiments (e.g. wear routes) share the identical training seed.",
    )
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--subject-balanced-batches", action="store_true")
    parser.add_argument(
        "--sampler",
        choices=("default", "subject_balanced", "label_balanced", "label_subject_balanced"),
        default="default",
        help="Training batch sampler. --subject-balanced-batches maps default to subject_balanced for backwards compatibility.",
    )
    parser.add_argument(
        "--fusion-variant",
        choices=("attention", "concat", "attention_multihead_pma", "eeg_anchor"),
        default="attention",
        help="Fusion encoder variant: attention (current default), concat (no attention), "
        "attention_multihead_pma (latent-query cross-attention), eeg_anchor (EEG token as query).",
    )
    parser.add_argument("--attn-num-heads", type=int, default=4, help="Attention heads for attention_multihead_pma; must divide --hidden-dim.")
    parser.add_argument("--attn-num-latent", type=int, default=8, help="Number of latent queries for attention_multihead_pma.")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.torch_threads)))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    rows = _load_jsonl(args.root / "index/eeg_aligned_window_index.jsonl")
    sample_id = np.asarray([str(row["sample_id"]) for row in rows], dtype=str)
    subject_id = np.asarray([_norm_subject(row.get("subject_id")) for row in rows], dtype=str)
    event_id = np.asarray([str(row.get("event_id", "")) for row in rows], dtype=str)
    target = np.asarray([_target_value(row, args.target_label) for row in rows], dtype=np.float32)
    requested = _requested_experiments(args)
    eeg_branches = _split_csv(args.eeg_branches)
    for eeg_branch in eeg_branches:
        if eeg_branch not in BRANCHES or BRANCHES[eeg_branch].modality != "eeg":
            raise ValueError(f"unsupported EEG branch: {eeg_branch}")
    branch_cache: dict[tuple[str, str], dict[str, Any]] = {}
    loss_modes = [value.strip() for value in args.loss_modes.split(",") if value.strip()]
    lambdas = [float(value.strip()) for value in args.lambdas.split(",") if value.strip()]
    heads = [value.strip() for value in args.heads.split(",") if value.strip()]
    head_lambdas = [float(value.strip()) for value in args.head_lambdas.split(",") if value.strip()]
    for mode in loss_modes:
        if mode not in {"raw", "raw_centered_mse", "raw_centered_corr", "weighted_mse_label_bins", "huber_extreme_weight", "mse_variance_reg"}:
            raise ValueError(f"unsupported centered loss mode: {mode}")
    for head in heads:
        if head not in {"regression", "ordinal_cumulative", "classification_expectation", "hybrid_regression_ordinal"}:
            raise ValueError(f"unsupported head: {head}")
    results: list[dict[str, Any]] = []
    run_number = 0
    for protocol, experiment in requested:
        split = _load_split(args.splits_root / protocol, len(rows))
        for eeg_branch in eeg_branches:
            branches = _replace_eeg_branch(EXPERIMENT_BRANCHES[experiment], eeg_branch)
            branch_data = _load_all_branches(
                args.embeddings_root,
                sample_id,
                protocol=protocol,
                eeg_seed=args.eeg_token_seed,
                wear_seed=args.wear_token_seed if args.wear_token_seed is not None else args.eeg_token_seed,
                eeg_token_root=args.eeg_token_root,
                required_names=branches,
                cache=branch_cache,
            )
            tokens, token_mask, branch_report = _build_tokens(branch_data, branches)
            loss_specs: list[tuple[str, float]] = []
            if not args.no_raw_baseline and "raw" not in loss_modes:
                loss_specs.append(("raw", 0.0))
            for mode in loss_modes:
                if mode == "raw":
                    loss_specs.append(("raw", 0.0))
                elif _loss_mode_uses_lambda(mode):
                    loss_specs.extend((mode, value) for value in lambdas)
                else:
                    loss_specs.append((mode, 0.0))
            run_specs: list[tuple[str, float, str, float]] = []
            for loss_mode, centered_lambda in loss_specs:
                for head in heads:
                    if head == "hybrid_regression_ordinal":
                        run_specs.extend((loss_mode, centered_lambda, head, value) for value in head_lambdas)
                    else:
                        run_specs.append((loss_mode, centered_lambda, head, 0.0))
            for loss_mode, centered_lambda, head, head_lambda in run_specs:
                run_seed = int(args.seed) if args.experiment_seed_fixed else int(args.seed) + run_number
                run_number += 1
                print(
                    f"starting protocol={protocol} experiment={experiment} eeg_branch={eeg_branch} "
                    f"loss_mode={loss_mode} lambda={centered_lambda} head={head} head_lambda={head_lambda} seed={run_seed}",
                    flush=True,
                )
                model, train_audit = _fit_model(
                    tokens=tokens,
                    token_mask=token_mask,
                    target=target,
                    subjects=subject_id,
                    train_idx=split["train"],
                    val_idx=split["val"],
                    loss_mode=loss_mode,
                    centered_lambda=centered_lambda,
                    head=head,
                    head_lambda=head_lambda,
                    hidden_dim=args.hidden_dim,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    dropout=args.dropout,
                    patience=args.patience,
                    seed=run_seed,
                    device=args.device,
                    subject_balanced_batches=args.subject_balanced_batches,
                    sampler=args.sampler,
                    fusion_variant=args.fusion_variant,
                    attn_num_heads=args.attn_num_heads,
                    attn_num_latent=args.attn_num_latent,
                )
                predictions = {
                    name: _predict(model, tokens, token_mask, indices=indices, device=args.device)
                    for name, indices in split.items()
                    if name in {"train", "val", "test"}
                }
                test_metrics = _metric_aliases(
                    evaluate_regression_with_centered(
                        target[split["test"]], predictions["test"], subject_id[split["test"]]
                    )
                )
                result = {
                    "protocol": protocol,
                    "experiment": experiment,
                    "eeg_branch": eeg_branch,
                    "branches": list(branches),
                    "enabled_modalities": [BRANCHES[name].modality for name in branches],
                    "loss_mode": loss_mode,
                    "centered_lambda": float(centered_lambda),
                    "head": head,
                    "head_lambda": float(head_lambda),
                    "seed": run_seed,
                    "eeg_token_seed": int(args.eeg_token_seed),
                    "fusion_variant": args.fusion_variant,
                    "attn_num_heads": int(args.attn_num_heads),
                    "attn_num_latent": int(args.attn_num_latent),
                    "row_count": len(rows),
                    "target_label": args.target_label,
                    "split_counts": {name: int(len(values)) for name, values in split.items()},
                    "mask_coverage_by_split": _mask_coverage(token_mask, branches, split),
                    "train": _metric_aliases(evaluate_regression_with_centered(target[split["train"]], predictions["train"], subject_id[split["train"]])),
                    "val": _metric_aliases(evaluate_regression_with_centered(target[split["val"]], predictions["val"], subject_id[split["val"]])),
                    "test": test_metrics,
                    "train_raw_loss": train_audit.get("best_train_raw_loss"),
                    "train_centered_loss": train_audit.get("best_train_centered_loss"),
                    "batch_centered_subject_count_mean": train_audit.get("batch_centered_subject_count_mean"),
                    "train_audit": train_audit,
                    "branch_report": branch_report,
                }
                if args.predictions_dir:
                    pred_path = (
                        args.predictions_dir
                        / protocol
                        / eeg_branch
                        / experiment
                        / f"{head}_{loss_mode}_lambda_{centered_lambda:g}_headlambda_{head_lambda:g}.npz"
                    )
                    pred_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        pred_path,
                        train_index=split["train"],
                        val_index=split["val"],
                        test_index=split["test"],
                        train_prediction=predictions["train"],
                        val_prediction=predictions["val"],
                        test_prediction=predictions["test"],
                        target=target,
                        sample_id=sample_id,
                        subject_id=subject_id,
                        event_id=event_id,
                    )
                    result["prediction_path"] = str(pred_path)
                results.append(result)
                print(
                    f"completed protocol={protocol} experiment={experiment} eeg_branch={eeg_branch} "
                    f"loss_mode={loss_mode} lambda={centered_lambda} head={head} head_lambda={head_lambda} "
                    f"rmse={_fmt(test_metrics['rmse'])} "
                    f"raw_r={_fmt(test_metrics['raw_r'])} centered_r={_fmt(test_metrics['within_subject_centered_r'])}",
                    flush=True,
                )

    output = {
        "stage": 2,
        "target_label": args.target_label,
        "root": str(args.root),
        "splits_root": str(args.splits_root),
        "embeddings_root": str(args.embeddings_root),
        "runtime": {
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "patience": args.patience,
            "seed": args.seed,
            "device": args.device,
            "eeg_branches": list(eeg_branches),
            "eeg_token_seed": int(args.eeg_token_seed),
            "eeg_token_root": args.eeg_token_root,
            "heads": heads,
            "head_lambdas": head_lambdas,
            "subject_balanced_batches": args.subject_balanced_batches,
            "sampler": args.sampler,
            "fusion_variant": args.fusion_variant,
            "attn_num_heads": int(args.attn_num_heads),
            "attn_num_latent": int(args.attn_num_latent),
            "train_rule": "pretrain + finetune from splits_new",
            "normalization": "train_only",
        },
        "run_count": len(results),
        "results": results,
    }
    _write_json(output, args.out_json)
    _write_markdown(output, args.out_md)
    print(f"run_count={len(results)}")
    print(f"out_json={args.out_json}")
    print(f"out_md={args.out_md}")
    return 0


def _fit_model(
    *,
    tokens: np.ndarray,
    token_mask: np.ndarray,
    target: np.ndarray,
    subjects: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    loss_mode: str,
    centered_lambda: float,
    head: str,
    head_lambda: float,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    patience: int,
    seed: int,
    device: str,
    subject_balanced_batches: bool,
    sampler: str,
    fusion_variant: str = "attention",
    attn_num_heads: int = 4,
    attn_num_latent: int = 8,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _seed_everything(seed)
    x_mean, x_std = _fit_token_normalization(tokens, token_mask, train_idx)
    y_mean = float(target[train_idx].mean())
    y_std = float(target[train_idx].std()) or 1.0
    dev = torch.device(device)
    label_values = ((np.arange(1, 6, dtype=np.float32) - y_mean) / y_std).astype(np.float32)
    module = AttentionRegressor(
        modality_count=tokens.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
        head=head,
        normalized_label_values=label_values,
        variant=fusion_variant,
        num_heads=attn_num_heads,
        num_latent=attn_num_latent,
    ).to(dev)
    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x_train = torch.as_tensor(_normalize_tokens(tokens[train_idx], x_mean, x_std), dtype=torch.float32, device=dev)
    m_train = torch.as_tensor(token_mask[train_idx], dtype=torch.bool, device=dev)
    y_train = torch.as_tensor((target[train_idx] - y_mean) / y_std, dtype=torch.float32, device=dev)
    raw_y_train = target[train_idx].astype(np.float32)
    raw_y_train_tensor = torch.as_tensor(raw_y_train, dtype=torch.float32, device=dev)
    train_subjects = subjects[train_idx]
    resolved_sampler = _resolve_sampler(sampler, subject_balanced_batches)
    phase2_loss_config = _phase2_loss_config(loss_mode, raw_y_train, y_mean=y_mean, y_std=y_std)
    train_sample_weights = torch.as_tensor(
        _sample_weights_for_loss(raw_y_train, phase2_loss_config),
        dtype=torch.float32,
        device=dev,
    )
    x_val = torch.as_tensor(_normalize_tokens(tokens[val_idx], x_mean, x_std), dtype=torch.float32, device=dev)
    m_val = torch.as_tensor(token_mask[val_idx], dtype=torch.bool, device=dev)
    y_val = torch.as_tensor((target[val_idx] - y_mean) / y_std, dtype=torch.float32, device=dev)
    raw_y_val = torch.as_tensor(target[val_idx].astype(np.float32), dtype=torch.float32, device=dev)
    val_subjects = subjects[val_idx]
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    epoch_audits: list[dict[str, float]] = []
    rng = np.random.default_rng(seed)
    for epoch in range(max(1, epochs)):
        module.train()
        batch_losses: list[float] = []
        batch_raw_losses: list[float] = []
        batch_centered_losses: list[float] = []
        batch_subject_counts: list[int] = []
        for batch in _make_batches(train_subjects, raw_y_train, batch_size, rng, resolved_sampler):
            loss, loss_audit = _head_loss_components(
                module,
                x_train[batch],
                m_train[batch],
                y_train[batch],
                raw_y_train_tensor[batch],
                train_subjects[batch],
                loss_mode=loss_mode,
                centered_lambda=centered_lambda,
                head=head,
                head_lambda=head_lambda,
                sample_weights=train_sample_weights[batch],
                phase2_config=phase2_loss_config,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
            batch_raw_losses.append(float(loss_audit["raw_loss"]))
            batch_centered_losses.append(float(loss_audit["aux_loss"]))
            batch_subject_counts.append(int(loss_audit["eligible_subject_count"]))
        module.eval()
        with torch.no_grad():
            val_loss_tensor, val_loss_audit = _head_loss_components(
                module,
                x_val,
                m_val,
                y_val,
                raw_y_val,
                val_subjects,
                loss_mode=loss_mode,
                centered_lambda=centered_lambda,
                head=head,
                head_lambda=head_lambda,
                sample_weights=None,
                phase2_config=phase2_loss_config,
            )
            val_loss = float(val_loss_tensor.detach().cpu().item())
        audit = {
            "epoch": int(epoch + 1),
            "train_loss": float(np.mean(batch_losses)) if batch_losses else math.nan,
            "train_raw_loss": float(np.mean(batch_raw_losses)) if batch_raw_losses else math.nan,
            "train_centered_loss": float(np.mean(batch_centered_losses)) if batch_centered_losses else math.nan,
            "batch_centered_subject_count_mean": float(np.mean(batch_subject_counts)) if batch_subject_counts else 0.0,
            "val_loss": val_loss,
            "val_raw_loss": float(val_loss_audit["raw_loss"]),
            "val_centered_loss": float(val_loss_audit["aux_loss"]),
            "val_centered_subject_count": int(val_loss_audit["eligible_subject_count"]),
            "val_head_loss": float(val_loss_audit["head_loss"]),
        }
        epoch_audits.append(audit)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        module.load_state_dict(best_state)
    best_audit = epoch_audits[max(0, best_epoch - 1)] if epoch_audits else {}
    module.eval()
    return {
        "module": module,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }, {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "initial_train_loss": epoch_audits[0]["train_loss"] if epoch_audits else math.nan,
        "final_train_loss": epoch_audits[-1]["train_loss"] if epoch_audits else math.nan,
        "best_train_raw_loss": best_audit.get("train_raw_loss"),
        "best_train_centered_loss": best_audit.get("train_centered_loss"),
        "batch_centered_subject_count_mean": best_audit.get("batch_centered_subject_count_mean", 0.0),
        "normalization": "train_only",
        "train_count": int(len(train_idx)),
        "loss_mode": loss_mode,
        "centered_lambda": float(centered_lambda),
        "head": head,
        "head_lambda": float(head_lambda),
        "sampler": resolved_sampler,
        "phase2_loss_config": phase2_loss_config,
        "epoch_count": len(epoch_audits),
        "history": epoch_audits,
    }


class AttentionRegressor(torch.nn.Module):
    def __init__(
        self,
        *,
        modality_count: int,
        hidden_dim: int,
        dropout: float,
        head: str = "regression",
        normalized_label_values: np.ndarray | None = None,
        variant: str = "attention",
        num_heads: int = 4,
        num_latent: int = 8,
    ) -> None:
        super().__init__()
        if variant not in {"attention", "concat", "attention_multihead_pma", "eeg_anchor"}:
            raise ValueError(f"unsupported fusion variant: {variant}")
        if variant == "attention_multihead_pma" and hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        if variant == "eeg_anchor" and modality_count < 1:
            raise ValueError("eeg_anchor requires at least one modality token with EEG at index 0")
        self.variant = variant
        self.head_kind = head
        self.input_projection = torch.nn.Linear(256, hidden_dim)
        self.modality_embedding = torch.nn.Parameter(torch.zeros(1, modality_count, hidden_dim))
        torch.nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)
        self.self_attention = torch.nn.MultiheadAttention(hidden_dim, 1, dropout=dropout, batch_first=True)
        self.query = torch.nn.Parameter(torch.zeros(hidden_dim))
        torch.nn.init.normal_(self.query, mean=0.0, std=0.02)
        self.dropout = torch.nn.Dropout(dropout)
        if variant == "concat":
            self.concat_projection = torch.nn.Linear(256 * modality_count + modality_count, hidden_dim)
        elif variant == "attention_multihead_pma":
            self.latent_queries = torch.nn.Parameter(torch.zeros(1, num_latent, hidden_dim))
            torch.nn.init.normal_(self.latent_queries, mean=0.0, std=0.02)
            self.cross_attention = torch.nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.regression_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.ordinal_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 4),
        )
        self.classification_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 5),
        )
        values = np.arange(1, 6, dtype=np.float32) if normalized_label_values is None else normalized_label_values.astype(np.float32)
        self.register_buffer("normalized_label_values", torch.as_tensor(values, dtype=torch.float32))

    def encode(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.variant == "concat":
            masked = tokens * mask[:, :, None].to(tokens.dtype)
            flat = torch.cat([masked.reshape(tokens.shape[0], -1), mask.to(tokens.dtype)], dim=1)
            return self.concat_projection(flat)
        x = self.input_projection(tokens) + self.modality_embedding
        if self.variant == "attention_multihead_pma":
            x = x.masked_fill(~mask[:, :, None], 0.0)
            queries = self.latent_queries.expand(tokens.shape[0], -1, -1)
            attended, _ = self.cross_attention(queries, x, x, key_padding_mask=~mask, need_weights=False)
            return attended.mean(dim=1)
        if self.variant == "eeg_anchor":
            x = x.masked_fill(~mask[:, :, None], 0.0)
            attended, _ = self.self_attention(x[:, :1], x, x, key_padding_mask=~mask, need_weights=False)
            return attended[:, 0]
        attended, _ = self.self_attention(x, x, x, key_padding_mask=~mask, need_weights=False)
        attended = self.dropout(attended)
        scores = torch.matmul(attended, self.query).masked_fill(~mask, -1.0e9)
        weights = torch.softmax(scores, dim=1) * mask.to(dtype=scores.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.sum(attended * weights.unsqueeze(-1), dim=1)

    def forward_outputs(self, tokens: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = self.encode(tokens, mask)
        outputs: dict[str, torch.Tensor] = {}
        if self.head_kind in {"regression", "hybrid_regression_ordinal"}:
            outputs["regression"] = self.regression_head(pooled).reshape(-1)
        if self.head_kind in {"ordinal_cumulative", "hybrid_regression_ordinal"}:
            outputs["ordinal_logits"] = self.ordinal_head(pooled)
        if self.head_kind == "classification_expectation":
            outputs["class_logits"] = self.classification_head(pooled)
        return outputs

    def prediction_from_outputs(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.head_kind in {"regression", "hybrid_regression_ordinal"}:
            return outputs["regression"].reshape(-1)
        if self.head_kind == "classification_expectation":
            probs = torch.softmax(outputs["class_logits"], dim=1)
            return torch.sum(probs * self.normalized_label_values.reshape(1, -1), dim=1)
        if self.head_kind == "ordinal_cumulative":
            return self._ordinal_expectation(outputs["ordinal_logits"])
        raise ValueError(f"unsupported head: {self.head_kind}")

    def _ordinal_expectation(self, logits: torch.Tensor) -> torch.Tensor:
        cumulative = torch.sigmoid(logits)
        cumulative = torch.cummin(cumulative, dim=1).values
        probs = torch.cat(
            [
                1.0 - cumulative[:, :1],
                cumulative[:, :1] - cumulative[:, 1:2],
                cumulative[:, 1:2] - cumulative[:, 2:3],
                cumulative[:, 2:3] - cumulative[:, 3:4],
                cumulative[:, 3:4],
            ],
            dim=1,
        ).clamp_min(0.0)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.sum(probs * self.normalized_label_values.reshape(1, -1), dim=1)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.prediction_from_outputs(self.forward_outputs(tokens, mask))


def _head_loss_components(
    module: AttentionRegressor,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    raw_target: torch.Tensor,
    subjects: np.ndarray,
    *,
    loss_mode: str,
    centered_lambda: float,
    head: str,
    head_lambda: float,
    sample_weights: torch.Tensor | None,
    phase2_config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float | int]]:
    outputs = module.forward_outputs(tokens, mask)
    prediction = module.prediction_from_outputs(outputs)
    raw_loss, aux_loss, eligible_subject_count = _loss_components(
        prediction,
        target,
        subjects,
        loss_mode,
        sample_weights=sample_weights,
        phase2_config=phase2_config,
    )
    base_loss = raw_loss + float(centered_lambda) * aux_loss
    head_loss = torch.zeros((), dtype=target.dtype, device=target.device)
    if head in {"ordinal_cumulative", "hybrid_regression_ordinal"}:
        head_loss = _ordinal_cumulative_loss(outputs["ordinal_logits"], raw_target)
    elif head == "classification_expectation":
        head_loss = _classification_loss(outputs["class_logits"], raw_target)
    if head == "regression":
        total = base_loss
    elif head == "hybrid_regression_ordinal":
        total = base_loss + float(head_lambda) * head_loss
    else:
        total = head_loss + 0.1 * raw_loss
    return total, {
        "raw_loss": float(raw_loss.detach().cpu().item()),
        "aux_loss": float(aux_loss.detach().cpu().item()),
        "head_loss": float(head_loss.detach().cpu().item()),
        "eligible_subject_count": int(eligible_subject_count),
    }


def _ordinal_cumulative_loss(logits: torch.Tensor, raw_target: torch.Tensor) -> torch.Tensor:
    labels = torch.clamp(torch.round(raw_target), 1, 5)
    thresholds = torch.arange(1, 5, dtype=raw_target.dtype, device=raw_target.device).reshape(1, -1)
    binary = (labels.reshape(-1, 1) > thresholds).to(dtype=logits.dtype)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, binary)


def _classification_loss(logits: torch.Tensor, raw_target: torch.Tensor) -> torch.Tensor:
    labels = torch.clamp(torch.round(raw_target), 1, 5).to(dtype=torch.long) - 1
    return torch.nn.functional.cross_entropy(logits, labels)


def _loss_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    subjects: np.ndarray,
    loss_mode: str,
    *,
    sample_weights: torch.Tensor | None = None,
    phase2_config: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    residual = prediction - target
    phase2_config = phase2_config or {}
    if loss_mode == "weighted_mse_label_bins":
        if sample_weights is None:
            raw_loss = torch.mean(residual**2)
        else:
            raw_loss = torch.sum(sample_weights * residual**2) / sample_weights.sum().clamp_min(1e-8)
        return raw_loss, torch.zeros_like(raw_loss), 0
    if loss_mode == "huber_extreme_weight":
        delta = float(phase2_config.get("huber_delta_standardized", 1.0))
        abs_residual = torch.abs(residual)
        quadratic = torch.minimum(abs_residual, torch.as_tensor(delta, dtype=abs_residual.dtype, device=abs_residual.device))
        linear = abs_residual - quadratic
        loss_values = 0.5 * quadratic**2 + delta * linear
        if sample_weights is not None:
            raw_loss = torch.sum(sample_weights * loss_values) / sample_weights.sum().clamp_min(1e-8)
        else:
            raw_loss = torch.mean(loss_values)
        return raw_loss, torch.zeros_like(raw_loss), 0
    raw_loss = torch.mean(residual**2)
    if loss_mode == "mse_variance_reg":
        variance_loss, eligible = _variance_regularization_loss(prediction, target, subjects)
        return raw_loss, variance_loss, eligible
    if loss_mode == "raw":
        return raw_loss, torch.zeros_like(raw_loss), 0
    subject_values = np.asarray(subjects).reshape(-1)
    centered_prediction: list[torch.Tensor] = []
    centered_target: list[torch.Tensor] = []
    eligible = 0
    for subject in np.unique(subject_values):
        indices = np.flatnonzero(subject_values == subject)
        if len(indices) < 2:
            continue
        index = torch.as_tensor(indices, dtype=torch.long, device=prediction.device)
        centered_prediction.append(prediction[index] - prediction[index].mean())
        centered_target.append(target[index] - target[index].mean())
        eligible += 1
    if not centered_prediction:
        return raw_loss, torch.zeros_like(raw_loss), eligible
    pred = torch.cat(centered_prediction)
    truth = torch.cat(centered_target)
    if loss_mode == "raw_centered_mse":
        centered_loss = torch.mean((pred - truth) ** 2)
    elif loss_mode == "raw_centered_corr":
        denominator = torch.sqrt(torch.sum(pred * pred) * torch.sum(truth * truth)).clamp_min(1e-8)
        centered_loss = 1.0 - torch.sum(pred * truth) / denominator
    else:
        raise ValueError(f"unsupported loss mode: {loss_mode}")
    return raw_loss, centered_loss, eligible


def _variance_regularization_loss(prediction: torch.Tensor, target: torch.Tensor, subjects: np.ndarray) -> tuple[torch.Tensor, int]:
    subject_values = np.asarray(subjects).reshape(-1)
    losses: list[torch.Tensor] = []
    for subject in np.unique(subject_values):
        indices = np.flatnonzero(subject_values == subject)
        if len(indices) < 2:
            continue
        index = torch.as_tensor(indices, dtype=torch.long, device=prediction.device)
        target_std = torch.std(target[index], unbiased=False)
        if float(target_std.detach().cpu().item()) <= 1e-6:
            continue
        pred_std = torch.std(prediction[index], unbiased=False)
        losses.append(torch.relu(target_std - pred_std) ** 2)
    if not losses and len(prediction) >= 2:
        target_std = torch.std(target, unbiased=False)
        if float(target_std.detach().cpu().item()) > 1e-6:
            losses.append(torch.relu(target_std - torch.std(prediction, unbiased=False)) ** 2)
    if not losses:
        return torch.zeros((), dtype=prediction.dtype, device=prediction.device), 0
    return torch.stack(losses).mean(), len(losses)


def _make_batches(subjects: np.ndarray, labels: np.ndarray, batch_size: int, rng: np.random.Generator, sampler: str) -> list[np.ndarray]:
    if sampler == "default":
        order = rng.permutation(len(subjects))
        return [order[start : start + max(1, batch_size)] for start in range(0, len(order), max(1, batch_size))]
    if sampler == "subject_balanced":
        queues = [list(rng.permutation(np.flatnonzero(subjects == subject)).tolist()) for subject in np.unique(subjects)]
    elif sampler == "label_balanced":
        label_bins = _label_bins(labels)
        queues = [list(rng.permutation(np.flatnonzero(label_bins == label_bin)).tolist()) for label_bin in np.unique(label_bins)]
    elif sampler == "label_subject_balanced":
        label_bins = _label_bins(labels)
        keys = sorted({(int(label_bin), str(subject)) for label_bin, subject in zip(label_bins, subjects)})
        queues = [
            list(rng.permutation(np.flatnonzero((label_bins == label_bin) & (subjects == subject))).tolist())
            for label_bin, subject in keys
        ]
    else:
        raise ValueError(f"unsupported sampler: {sampler}")
    order: list[int] = []
    while any(queues):
        subject_order = rng.permutation(len(queues)).tolist()
        for queue_index in subject_order:
            if queues[queue_index]:
                order.append(queues[queue_index].pop())
    return [np.asarray(order[start : start + max(1, batch_size)], dtype=np.int64) for start in range(0, len(order), max(1, batch_size))]


def _resolve_sampler(sampler: str, subject_balanced_batches: bool) -> str:
    if sampler == "default" and subject_balanced_batches:
        return "subject_balanced"
    return sampler


def _loss_mode_uses_lambda(loss_mode: str) -> bool:
    return loss_mode in {"raw_centered_mse", "raw_centered_corr", "mse_variance_reg"}


def _label_bins(labels: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(labels).astype(np.int64), 1, 5)


def _phase2_loss_config(loss_mode: str, raw_y_train: np.ndarray, *, y_mean: float, y_std: float) -> dict[str, Any]:
    label_bins = _label_bins(raw_y_train)
    counts = {str(label): int(np.sum(label_bins == label)) for label in range(1, 6)}
    if loss_mode == "weighted_mse_label_bins":
        nonzero = np.asarray([count for count in counts.values() if count > 0], dtype=np.float64)
        mean_count = float(np.mean(nonzero)) if len(nonzero) else 1.0
        weights = {
            str(label): float(np.clip(mean_count / max(1, counts[str(label)]), 0.5, 3.0))
            for label in range(1, 6)
        }
        return {
            "kind": loss_mode,
            "label_counts": counts,
            "label_weights": weights,
            "weight_clip": [0.5, 3.0],
        }
    if loss_mode == "huber_extreme_weight":
        return {
            "kind": loss_mode,
            "label_counts": counts,
            "low_threshold": 2.0,
            "high_threshold": 4.0,
            "base_weight": 1.0,
            "extreme_weight": 2.0,
            "huber_delta_raw": 1.0,
            "huber_delta_standardized": float(1.0 / (float(y_std) or 1.0)),
            "target_normalization_mean": float(y_mean),
            "target_normalization_std": float(y_std),
        }
    if loss_mode == "mse_variance_reg":
        return {
            "kind": loss_mode,
            "label_counts": counts,
            "variance_scope": "subject_in_batch_with_batch_fallback",
        }
    return {"kind": loss_mode, "label_counts": counts}


def _sample_weights_for_loss(raw_y: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    kind = config.get("kind")
    if kind == "weighted_mse_label_bins":
        label_weights = {int(label): float(weight) for label, weight in config.get("label_weights", {}).items()}
        bins = _label_bins(raw_y)
        return np.asarray([label_weights.get(int(label), 1.0) for label in bins], dtype=np.float32)
    if kind == "huber_extreme_weight":
        weights = np.ones(len(raw_y), dtype=np.float32) * float(config.get("base_weight", 1.0))
        extreme = (raw_y <= float(config.get("low_threshold", 2.0))) | (raw_y >= float(config.get("high_threshold", 4.0)))
        weights[extreme] = float(config.get("extreme_weight", 2.0))
        return weights
    return np.ones(len(raw_y), dtype=np.float32)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _target_value(row: dict[str, Any], target: str) -> float:
    labels = row.get("labels")
    if isinstance(labels, dict):
        return float(labels[target])
    if isinstance(labels, list):
        return float(labels[LABEL_NAMES.index(target)])
    return float(row[target])


def _load_all_branches(
    embedding_root: Path,
    sample_id: np.ndarray,
    *,
    protocol: str,
    eeg_seed: int,
    wear_seed: int,
    eeg_token_root: str,
    required_names: tuple[str, ...],
    cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = {}
    for name in required_names:
        branch = BRANCHES[name]
        filename = branch.filename.format(
            protocol=protocol, eeg_seed=int(eeg_seed), wear_seed=int(wear_seed), eeg_token_root=eeg_token_root
        )
        cache_key = (name, filename)
        if cache_key in cache:
            result[name] = cache[cache_key]
            continue
        path = embedding_root / filename
        with np.load(path, allow_pickle=True) as loaded:
            loaded_ids = loaded["sample_id"].astype(str)
            if not np.array_equal(loaded_ids, sample_id):
                raise ValueError(f"{name} sample_id order does not match canonical index")
            emb_key = branch.emb_key if branch.emb_key in loaded.files else "face_emb"
            embedding = loaded[emb_key].astype(np.float32)
            mask = loaded[branch.mask_key].astype(bool) if branch.mask_key in loaded.files else loaded["modality_mask"][:, branch.modality_index].astype(bool)
            if embedding.shape != (len(sample_id), 256) or mask.shape != (len(sample_id),):
                raise ValueError(f"invalid shape for {name}: {embedding.shape}, {mask.shape}")
        result[name] = {"embedding": embedding, "mask": mask, "path": str(path), "mask_sum": int(mask.sum())}
        cache[cache_key] = result[name]
    return result


def _build_tokens(data: dict[str, dict[str, Any]], names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    tokens = np.stack([data[name]["embedding"] for name in names], axis=1).astype(np.float32)
    masks = np.stack([data[name]["mask"] for name in names], axis=1).astype(bool)
    report = {name: {"path": data[name]["path"], "mask_sum": data[name]["mask_sum"], "modality": BRANCHES[name].modality} for name in names}
    return tokens, masks, report


def _load_split(path: Path, n_rows: int) -> dict[str, np.ndarray]:
    split = {name: _load_indices(path / f"{name}.json", n_rows) for name in ("pretrain", "finetune", "val", "test")}
    split["train"] = np.asarray(split["pretrain"].tolist() + split["finetune"].tolist(), dtype=np.int64)
    return split


def _load_indices(path: Path, n_rows: int) -> np.ndarray:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("indices", value.get("index", value.get("rows")))
    values = np.asarray(value, dtype=np.int64).reshape(-1)
    if values.size and (values.min() < 0 or values.max() >= n_rows):
        raise ValueError(f"{path} contains out-of-range indices")
    return values


def _predict(model: dict[str, Any], tokens: np.ndarray, mask: np.ndarray, *, indices: np.ndarray, device: str) -> np.ndarray:
    module: AttentionRegressor = model["module"]
    device_obj = torch.device(device)
    values = []
    with torch.no_grad():
        for start in range(0, len(indices), 1024):
            chunk = indices[start : start + 1024]
            x = torch.as_tensor(_normalize_tokens(tokens[chunk], model["x_mean"], model["x_std"]), dtype=torch.float32, device=device_obj)
            m = torch.as_tensor(mask[chunk], dtype=torch.bool, device=device_obj)
            values.append((module(x, m).detach().cpu().numpy() * float(model["y_std"]) + float(model["y_mean"])).astype(np.float32))
    return np.concatenate(values) if values else np.zeros((0,), dtype=np.float32)


def _fit_token_normalization(tokens: np.ndarray, mask: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    available = np.where(mask[indices, :, None], tokens[indices], np.nan)
    mean = np.nanmean(available, axis=(0, 1), keepdims=True)
    std = np.nanstd(available, axis=(0, 1), keepdims=True)
    return np.where(np.isfinite(mean), mean, 0.0).astype(np.float32), np.where(np.isfinite(std) & (std >= 1e-6), std, 1.0).astype(np.float32)


def _normalize_tokens(tokens: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((tokens.astype(np.float32) - mean) / std).astype(np.float32)


def _mask_coverage(mask: np.ndarray, branches: tuple[str, ...], split: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            branch: {"valid": int(mask[indices, col].sum()), "total": int(len(indices)), "coverage": float(mask[indices, col].mean()) if len(indices) else 0.0}
            for col, branch in enumerate(branches)
        }
        for name, indices in split.items()
        if name in {"train", "val", "test"}
    }


def _metric_aliases(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["pearson_r"] = result.get("raw_r")
    result["per_subject_r_mean"] = result.get("per_subject_r", {}).get("mean")
    result["per_subject_r_std"] = result.get("per_subject_r", {}).get("std")
    return result


def _parse_experiments(value: str) -> list[tuple[str, str]]:
    items = []
    for raw in value.split(","):
        protocol, experiment = raw.strip().split(":", 1)
        if experiment not in EXPERIMENT_BRANCHES:
            raise ValueError(f"unsupported experiment: {experiment}")
        items.append((protocol, experiment))
    return items


def _requested_experiments(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.experiment_set == "custom":
        return _parse_experiments(args.experiments)
    protocols = _split_csv(args.protocols)
    return [(protocol, experiment) for protocol in protocols for experiment in VIDEO_ONLY_EXPERIMENTS]


def _replace_eeg_branch(branches: tuple[str, ...], eeg_branch: str) -> tuple[str, ...]:
    return tuple(eeg_branch if name == "eeg" else name for name in branches)


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _norm_subject(value: Any) -> str:
    text = str(value)
    if text.startswith("sub-"):
        return text
    try:
        return f"sub-{int(float(text)):02d}"
    except (TypeError, ValueError):
        return text


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(output: dict[str, Any], path: Path) -> None:
    lines = [
        "# EEGPT Raw + Centered Multi-Task Loss",
        "",
        f"target_label: `{output['target_label']}`",
        f"run_count: `{output['run_count']}`",
        "",
        "| protocol | EEG branch | experiment | fusion | loss | lambda | head | head lambda | RMSE | MAE | raw r | centered r | per-subject r mean | best epoch |",
        "| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in output["results"]:
        test = row["test"]
        lines.append(
            f"| {row['protocol']} | {row.get('eeg_branch', 'eeg')} | {row['experiment']} | {row.get('fusion_variant', 'attention')} | "
            f"{row['loss_mode']} | {row['centered_lambda']:.3g} | "
            f"{row.get('head', 'regression')} | {float(row.get('head_lambda', 0.0)):.3g} | "
            f"{_fmt(test['rmse'])} | {_fmt(test['mae'])} | {_fmt(test['raw_r'])} | "
            f"{_fmt(test['within_subject_centered_r'])} | {_fmt(test['per_subject_r_mean'])} | {row['train_audit']['best_epoch']} |"
        )
    lines.extend(["", "`centered_loss` is computed within each training batch for subjects with at least two samples; validation selection uses the same composite objective.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
