#!/usr/bin/env python3
"""C validation selection; freeze complete Stage-D choices before test reporting."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from daily_multimodal.daily_affect.ema_bags import LABEL_NAMES

def load(name, file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

c=load("round2_c_paths","144_run_round2_experiment_c.py")
h=c.h
SEEDS=h.SEEDS
C_ROUTES=("B0_W","B0_EVENT","B0_EVENT_WEAK","F_C","C_EVENT","C_EVENT_WEAK")
C_COEFFICIENTS={"C_EVENT-F_C":(0,0,0,-1,1,0),"C_EVENT_WEAK-F_C":(0,0,0,-1,0,1),
    "B0_EVENT-B0_W":(-1,1,0,0,0,0),"B0_EVENT_WEAK-B0_W":(-1,0,1,0,0,0),
    "event_loss_interaction":(1,-1,0,-1,1,0),"weak_loss_interaction":(1,0,-1,-1,0,1),
    "F_C-B0_W":(-1,0,0,1,0,0),"C_EVENT-B0_EVENT":(0,-1,0,0,1,0),"C_EVENT_WEAK-B0_EVENT_WEAK":(0,0,-1,0,0,1)}
ALL_ROUTES=("B0_W","F_C","F_P","F_A_C","F_A_P","B_T2_STD","B_T2_LOW","B_T1_LOW","C_EVENT","C_EVENT_WEAK","B0_EVENT","B0_EVENT_WEAK")

def route_path(root,route,seed):
    if route=="B0_W":return c.controls("B0",seed)
    if route in ("F_C","F_P"):return c.a.controls(route[-1],seed)
    return root/f"fusion/{route}/formal/cross_day/seed_{seed}"

def select_loss(prefix,gates,means):
    control="F_C" if prefix=="C" else "B0_W"
    eligible=[v for v in (prefix+"_EVENT",prefix+"_EVENT_WEAK") if gates[v+"-"+control]["passed"]]
    if not eligible:return None
    minimum=min(means[v] for v in eligible)
    return min([v for v in eligible if means[v]<=minimum+1e-6],key=lambda v:0 if v.endswith("WEAK") else 1)

def summarize(tensor,routes):
    return {v:{m:{"mean":float(tensor[i,:,:,j].mean()),"sd":float(tensor[i,:,:,j].mean(axis=1).std(ddof=1))}
        for j,m in enumerate(h.METRICS)} for i,v in enumerate(routes)}

def contrast_rows(deltas,leaf):
    rows=[]
    for name,delta in deltas.items():
        for si,seed in enumerate(SEEDS):
            for li,label in enumerate((*LABEL_NAMES,"macro")):
                value=delta[si,li] if li<11 else delta[si].mean(axis=0)
                rows.append({"leaf":leaf,"contrast":name,"seed":seed,"label":label,**dict(zip(h.METRICS,value))})
    return rows

def bootstrap_rows(arrays,metrics,routes,coefficients,deltas,leaf):
    rows=[]
    for si,seed in enumerate(SEEDS):
        scale=np.asarray(metrics[routes[0],seed]["train_target_std"])
        for route in routes[1:]:
            if not np.array_equal(scale,np.asarray(metrics[route,seed]["train_target_std"])):raise ValueError("label scale differs")
        intervals,blocks=h.bootstraps([arrays[v,seed] for v in routes],scale,coefficients,2000,seed+(121000 if leaf=="val" else 151000))
        for name,data in intervals.items():
            for li,label in enumerate((*LABEL_NAMES,"macro")):
                ci=data["per_label"][:,li] if li<11 else data["macro"]
                point=deltas[name][si,li] if li<11 else deltas[name][si].mean(axis=0)
                for j,metric in enumerate(h.METRICS):
                    rows.append({"leaf":leaf,"contrast":name,"seed":seed,"label":label,"metric":metric,"delta":float(point[j]),
                        "ci_low":float(ci[0,j]),"ci_high":float(ci[1,j]),"subject_day_blocks":blocks,"iterations":2000})
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",required=True,type=Path);args=p.parse_args();r=args.root
    for marker in ("C_STAGE0_COMPLETE","C_CONTROL_EQUIVALENCE_VERIFIED","C_SMOKE_COMPLETE","C_FORMAL_COMPLETE"):
        if not (r/marker).exists():raise ValueError("missing C prerequisite: "+marker)
    out=r/"reports/experiment_c";out.mkdir(parents=True,exist_ok=True)
    contract=json.loads((r/"inputs/c_fixed_config.json").read_text())
    for row in contract["reuse"]:
        for n,sha in row["sha256"].items():
            if h.sha(Path(row["directory"])/n)!=sha:raise ValueError("reused control changed")
    metrics,arrays,values,provenance={},{},{},{}
    for route in ALL_ROUTES:
        for seed in SEEDS:
            d=route_path(r,route,seed)
            metrics[route,seed],arrays[route,seed],values[route,seed]=h.read_val(d)
            provenance[str(d)]={n:h.sha(d/n) for n in ("metrics.json","event_predictions.npz")}
            h.paired(arrays["B0_W",seed],arrays[route,seed])
    tensor=np.stack([np.stack([values[v,s] for s in SEEDS]) for v in C_ROUTES])
    deltas={n:np.tensordot(co,tensor,axes=(0,0)) for n,co in C_COEFFICIENTS.items()}
    gates={n:h.gate(deltas[n].mean(axis=1)) for n in tuple(C_COEFFICIENTS)[:4]}
    means={v:float(tensor[i,:,:,3].mean()) for i,v in enumerate(C_ROUTES)}
    selection={"scope":"experiment_C","leaf":"val_only","gates":gates,"selected_MAE_loss":select_loss("C",gates,means),
        "selected_B0_loss":select_loss("B0",gates,means),"seeds":list(SEEDS),"provenance":provenance,
        "selection_rule":"joint_gate_vs_same_representation_lambda1; mean_sRMSE_min; tolerance_1e-6_prefer_lambda0.1"}
    h.write(out/"selection_val_only.json",selection)
    final=r/"reports/stage_d";final.mkdir(exist_ok=True)
    constituents={}
    for name,path in (("A",r/"reports/selection_val_only.json"),("B",r/"reports/experiment_b/selection_val_only.json"),("C",out/"selection_val_only.json")):
        constituents[name]={"path":str(path),"sha256":h.sha(path),"selection":json.loads(path.read_text())}
    unified={"scope":"all_round2_core_ABC","leaf":"val_only","frozen_at_utc":datetime.now(timezone.utc).isoformat(),
        "constituents":constituents,"all_36_cells_provenance":provenance,"seeds":list(SEEDS),
        "test_metrics_used_for_selection":False,"optional_joint_confirmation":"not_started_separate_budget",
        "within_subject_day":"stopped_protocol_overlap","evidence_status":"exploratory_fixed_plan"}
    record=final/"selection_val_only.json"
    if record.exists():
        saved=json.loads(record.read_text());unified["frozen_at_utc"]=saved["frozen_at_utc"]
        if saved!=unified:raise ValueError("Stage D frozen selection changed")
    else:h.write(record,unified)
    (r/"ROUND2_STAGE_D_VAL_SELECTION_FROZEN").write_text(h.sha(record)+"\n")
    summaries=summarize(tensor,C_ROUTES)
    rows=[]
    for i,route in enumerate(C_ROUTES):
        for si,seed in enumerate(SEEDS):
            for li,label in enumerate((*LABEL_NAMES,"macro")):
                v=tensor[i,si,li] if li<11 else tensor[i,si].mean(axis=0)
                rows.append({"route":route,"seed":seed,"label":label,**dict(zip(h.METRICS,v))})
    h.csv_write(out/"per_label_val_metrics.csv",rows)
    h.csv_write(out/"paired_val_deltas.csv",contrast_rows(deltas,"val"))
    h.csv_write(out/"paired_val_bootstrap.csv",bootstrap_rows(arrays,metrics,C_ROUTES,C_COEFFICIENTS,deltas,"val"))
    cells=[{"protocol":protocol,"route":route,"seed":seed,"status":"completed" if protocol=="cross_day" else "stopped_protocol_overlap",
        "reused":route in ("F_C","B0_W")} for protocol in ("cross_day","within_subject_day") for route in C_ROUTES for seed in SEEDS]
    h.write(out/"cell_status.json",cells)
    h.write(out/"results_val.json",{"summaries":summaries,"gates":gates,"selection":selection,
        "contrasts":{n:{m:float(d.mean(axis=(0,1))[j]) for j,m in enumerate(h.METRICS)} for n,d in deltas.items()}})
    lines=["# C：冻结表征的事件损失对照（val）","","原C与B0各自比较λ=1/0/0.1；三个匹配下游seed。", "",
        "| 配置 | raw r mean±SD | centered r mean±SD | sRMSE mean±SD |","|---|---:|---:|---:|"]
    for v in C_ROUTES:lines.append("| "+v+" | "+" | ".join(f"{summaries[v][k]['mean']:.6f} ± {summaries[v][k]['sd']:.6f}" for k in (h.METRICS[0],h.METRICS[1],h.METRICS[3]))+" |")
    lines += ["",f"MAE选择：`{selection['selected_MAE_loss']}`；B0选择：`{selection['selected_B0_loss']}`。", "",
        "完整A/B/C val选择记录已先行冻结于 `reports/stage_d/selection_val_only.json`，test报告随后独立执行。", "",
        "所有新损失的encoder/token固定。λ=1损失、梯度与3-epoch完整旧trainer训练权重/历史已逐值验收。within_subject_day保持协议stop。"]
    (out/"experiment_c_val_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (r/"EXPERIMENT_C_COMPLETE_WITH_PROTOCOL_STOP").write_text("complete\n")
    print(json.dumps({"summaries":summaries,"gates":gates,"selected_MAE":selection["selected_MAE_loss"],"selected_B0":selection["selected_B0_loss"]}),flush=True)

if __name__=="__main__":main()
