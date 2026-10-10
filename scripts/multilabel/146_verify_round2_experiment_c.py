#!/usr/bin/env python3
"""C checkpoint/input/prediction verification, including lambda1 diagnostics."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from daily_multimodal.daily_affect.training import load_bag_dataset, fit_token_normalization
from daily_multimodal.daily_affect.regression import DailyAffectRegressionConfig
from daily_multimodal.training.structure_emotion import MultiEmotionStructureModel, _predict
from daily_multimodal.training.event_loss_adaptation import window_diagnostics

spec=importlib.util.spec_from_file_location("c_selection_paths",Path(__file__).with_name("145_select_report_round2_experiment_c.py"))
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);args=p.parse_args();r=args.root
    torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(True)
    if not (r/"ROUND2_STAGE_D_VAL_SELECTION_FROZEN").exists():raise ValueError("val selection must precede test replay")
    for n,sha in json.loads((r/"inputs/c_execution_source_manifest.json").read_text())["files"].items():
        if v.c.a.sha(r/n)!=sha:raise ValueError("executed source changed: "+n)
    fixed=json.loads((r/"inputs/c_fixed_config.json").read_text())
    datasets={s:load_bag_dataset(Path(path)) for s,path in fixed["bag_paths"].items()}
    for s,path in fixed["bag_paths"].items():
        if v.c.a.sha(path)!=fixed["bag_sha256"][s]:raise ValueError("immutable source bag changed")
    for row in fixed["reuse"]:
        for n,sha in row["sha256"].items():
            if v.c.a.sha(Path(row["directory"])/n)!=sha:raise ValueError("reuse source changed")
    audit={"status":"pass","scope":"C_only","cells":[],"fixed_encoder_tokens":True,
        "selection_sha256":v.h.sha(r/"reports/stage_d/selection_val_only.json"),"test_performance_disclosed":False}
    dev=torch.device("cuda")
    out=r/"reports/experiment_c"
    for route in v.C_ROUTES:
        source="B0" if route.startswith("B0") else "C"
        ds=datasets[source]
        xm,xs=fit_token_normalization(ds.tokens,ds.modality_mask,ds.split_indices()["train"],scope="per_modality")
        for seed in v.SEEDS:
            d=v.route_path(r,route,seed);cp=torch.load(d/"best_checkpoint.pt",map_location="cpu",weights_only=False)
            if not np.array_equal(cp["x_mean"],xm) or not np.array_equal(cp["x_std"],xs):raise ValueError("C train normalization changed")
            model=MultiEmotionStructureModel(DailyAffectRegressionConfig(model_id="window_replicated",hidden_dim=128,dropout=.1,
                adapter_mode="per_modality",temporal_policy="uniform",lambda_d=.25)).to(dev)
            model.load_state_dict(cp["state_dict"],strict=True)
            errors={};diagnostics={"label_names":np.asarray(v.c.a.LABEL_NAMES)}
            with np.load(d/"event_predictions.npz",allow_pickle=False) as saved:
                for leaf in ("val","test"):
                    ix=ds.split_indices()[leaf]
                    actual=_predict(model,ds,ix,cp["x_mean"],cp["x_std"],cp["y_mean"],cp["y_std"],dev)
                    error=float(np.abs(actual-saved[leaf+"_prediction"]).max())
                    if error>1e-6 or not np.isfinite(actual).all():raise ValueError("C independent prediction replay failed")
                    for k,value in (("event_id",ds.event_id),("subject_id",ds.subject_id),("day_id",ds.day_id)):
                        if not np.array_equal(value[ix],saved[leaf+"_"+k]):raise ValueError("C event identity changed")
                    diag=window_diagnostics(model,ds,ix,xm,xs,cp["y_mean"],cp["y_std"],dev)
                    if np.max(np.abs(diag["event_prediction"]-actual))>1e-6:raise ValueError("window diagnostics differ")
                    for k,x in diag.items():diagnostics[leaf+"_"+k]=x
                    errors[leaf]={"event_count":len(ix),"max_abs_error":error}
            if route not in ("F_C","B0_W"):
                with np.load(d/"window_diagnostics.npz",allow_pickle=False) as saved:
                    for k,x in diagnostics.items():
                        if not np.array_equal(x,saved[k]):raise ValueError("saved C window diagnostics differ")
            else:np.savez_compressed(out/f"baseline_diagnostics_{route}_seed_{seed}.npz",**diagnostics)
            audit["cells"].append({"route":route,"seed":seed,"source":source,"checkpoint_sha256":v.c.a.sha(d/"best_checkpoint.pt"),"replay":errors})
            print(json.dumps(audit["cells"][-1]),flush=True)
            del model
    (out/"completion_audit.json").write_text(json.dumps(audit,indent=2))
    (r/"EXPERIMENT_C_REPLAY_VERIFIED").write_text("pass\n")

if __name__=="__main__":main()
