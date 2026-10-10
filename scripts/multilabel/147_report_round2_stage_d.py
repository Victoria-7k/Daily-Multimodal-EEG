#!/usr/bin/env python3
"""Full ABC val/test paired report after immutable validation selection."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location("round2_stage_d_helpers",Path(__file__).with_name("145_select_report_round2_experiment_c.py"))
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
h=v.h
ROUTES=v.ALL_ROUTES
SEEDS=v.SEEDS

def coefficient(terms):return tuple(terms.get(route,0) for route in ROUTES)

COEFFICIENTS={"F_A_C-F_C":coefficient({"F_A_C":1,"F_C":-1}),"F_A_P-F_P":coefficient({"F_A_P":1,"F_P":-1}),
    "input_adaptation_interaction":coefficient({"F_A_P":1,"F_P":-1,"F_A_C":-1,"F_C":1}),
    "F_A_P-F_A_C":coefficient({"F_A_P":1,"F_A_C":-1}),
    "B_T2_LOW-B_T2_STD":coefficient({"B_T2_LOW":1,"B_T2_STD":-1}),
    "B_T1_LOW-B_T2_LOW":coefficient({"B_T1_LOW":1,"B_T2_LOW":-1})}
for route in ("B_T2_STD","B_T2_LOW","B_T1_LOW"):COEFFICIENTS[route+"-F_C"]=coefficient({route:1,"F_C":-1})
for name,co in v.C_COEFFICIENTS.items():COEFFICIENTS[name]=coefficient(dict(zip(v.C_ROUTES,co)))
for route in ("F_P","F_A_C","F_A_P","B_T2_STD","B_T2_LOW","B_T1_LOW"):
    COEFFICIENTS[route+"-B0_W"]=coefficient({route:1,"B0_W":-1})

def read_leaf(directory,leaf,*,selection_frozen=False,expected_count=None):
    if leaf=="test" and not selection_frozen:raise ValueError("test reader requires frozen validation selection")
    if leaf not in ("val","test"):raise ValueError("unknown leaf")
    expected_count=expected_count or (269 if leaf=="val" else 253)
    if leaf=="val":return h.read_val(directory,expected_count=expected_count)
    m=json.loads((directory/"metrics.json").read_text())
    if m["status"]!="ok" or m["protocol"]!="cross_day":raise ValueError("incomplete test cell")
    with np.load(directory/"event_predictions.npz",allow_pickle=False) as z:
        if tuple(z["label_names"].astype(str))!=v.c.a.LABEL_NAMES:raise ValueError("label order changed")
        a={k:z[leaf+"_"+k] for k in ("event_id","subject_id","day_id","target","prediction")}
    if a["target"].shape!=(expected_count,11) or not np.isfinite(a["prediction"]).all():raise ValueError("test array contract failed")
    scale=np.asarray(m["train_target_std"],np.float64)
    actual=h.scores(a["target"],a["prediction"],a["subject_id"],scale)
    stored=np.asarray([[m[leaf]["per_label"][label][metric] for metric in h.METRICS] for label in v.c.a.LABEL_NAMES])
    if not np.isfinite(stored).all() or np.max(np.abs(actual-stored))>2e-6:raise ValueError("test prediction/metric mismatch")
    return m,a,stored

def validate_frozen_selection(root):
    path=root/"reports/stage_d/selection_val_only.json"
    if not (root/"ROUND2_STAGE_D_VAL_SELECTION_FROZEN").exists():raise ValueError("all ABC val decisions required")
    frozen=json.loads(path.read_text())
    if frozen["leaf"]!="val_only" or frozen["test_metrics_used_for_selection"] or len(frozen["all_36_cells_provenance"])!=36:
        raise ValueError("Stage D val scope invalid")
    if (root/"ROUND2_STAGE_D_VAL_SELECTION_FROZEN").read_text().strip()!=h.sha(path):raise ValueError("frozen selection digest differs")
    for record in frozen["constituents"].values():
        if h.sha(Path(record["path"]))!=record["sha256"]:raise ValueError("constituent val choice changed")
    for directory,files in frozen["all_36_cells_provenance"].items():
        for n,sha in files.items():
            if h.sha(Path(directory)/n)!=sha:raise ValueError("frozen cell artifact changed")
    return frozen

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);args=p.parse_args();r=args.root
    for name in ("A","B","C"):
        if not (r/f"EXPERIMENT_{name}_REPLAY_VERIFIED").exists():raise ValueError("ABC independent replay required")
    frozen=validate_frozen_selection(r)
    out=r/"reports/stage_d"
    summaries,contrasts,metric_rows,delta_rows,bootstrap_rows,diagnostic_rows,date_rows={},{},[],[],[],[],[]
    total_recomputed=0
    for leaf in ("val","test"):
        metrics,arrays,values={},{},{}
        for route in ROUTES:
            for seed in SEEDS:
                metrics[route,seed],arrays[route,seed],values[route,seed]=read_leaf(v.route_path(r,route,seed),leaf,selection_frozen=True)
                h.paired(arrays["B0_W",seed],arrays[route,seed])
                total_recomputed+=1
        tensor=np.stack([np.stack([values[route,seed] for seed in SEEDS]) for route in ROUTES])
        summaries[leaf]=v.summarize(tensor,ROUTES)
        ds={n:np.tensordot(co,tensor,axes=(0,0)) for n,co in COEFFICIENTS.items()}
        contrasts[leaf]={n:{"means":dict(zip(h.METRICS,d.mean(axis=(0,1)).tolist())),
            "seed_macro_deltas":d.mean(axis=1).tolist(),"seed_sd":dict(zip(h.METRICS,d.mean(axis=1).std(axis=0,ddof=1).tolist())),
            "raw_r_positive_seeds":int((d.mean(axis=1)[:,0]>0).sum())} for n,d in ds.items()}
        delta_rows.extend(v.contrast_rows(ds,leaf))
        for i,route in enumerate(ROUTES):
            for si,seed in enumerate(SEEDS):
                a=arrays[route,seed]
                for li,label in enumerate((*v.c.a.LABEL_NAMES,"macro")):
                    value=tensor[i,si,li] if li<11 else tensor[i,si].mean(axis=0)
                    sd_y=a["target"].std(axis=0);sd_p=a["prediction"].std(axis=0)
                    metric_rows.append({"leaf":leaf,"route":route,"seed":seed,"label":label,
                        **dict(zip(h.METRICS,value)),"target_std":float(sd_y[li] if li<11 else sd_y.mean()),
                        "prediction_std":float(sd_p[li] if li<11 else sd_p.mean()),"event_count":len(a["event_id"])})
                if route in v.C_ROUTES:
                    path=out.parent/"experiment_c"/f"baseline_diagnostics_{route}_seed_{seed}.npz" if route in ("B0_W","F_C") else v.route_path(r,route,seed)/"window_diagnostics.npz"
                    with np.load(path,allow_pickle=False) as z:
                        variance=z[leaf+"_window_variance"];minimum=z[leaf+"_window_min"];maximum=z[leaf+"_window_max"]
                        for li,label in enumerate(v.c.a.LABEL_NAMES):
                            diagnostic_rows.append({"leaf":leaf,"route":route,"seed":seed,"label":label,
                                "mean_within_event_window_std":float(np.sqrt(variance[:,li]).mean()),
                                "mean_window_range":float((maximum[:,li]-minimum[:,li]).mean()),
                                "window_min":float(minimum[:,li].min()),"window_max":float(maximum[:,li].max()),
                                "event_prediction_std":float(a["prediction"][:,li].std()),"target_std":float(a["target"][:,li].std())})
                    keys=np.char.add(np.char.add(a["subject_id"].astype(str),"/"),a["day_id"].astype(str))
                    for key in np.unique(keys):
                        ix=np.flatnonzero(keys==key)
                        for li,label in enumerate(v.c.a.LABEL_NAMES):
                            date_rows.append({"leaf":leaf,"route":route,"seed":seed,"subject_day":key,"label":label,"events":len(ix),
                                "target_mean":float(a["target"][ix,li].mean()),"prediction_mean":float(a["prediction"][ix,li].mean()),
                                "prediction_std":float(a["prediction"][ix,li].std()),"rmse":float(np.sqrt(((a["target"][ix,li]-a["prediction"][ix,li])**2).mean()))})
        print(f"{leaf}: 36 prediction leaves recomputed; paired bootstrap starting",flush=True)
        bootstrap_rows.extend(v.bootstrap_rows(arrays,metrics,ROUTES,COEFFICIENTS,ds,leaf))
    eeg_results,eeg_rows,eeg_delta_rows,eeg_bootstrap_rows={},[],[],[]
    for source in ("C","P"):
        for leaf in ("val","test"):
            values,arrays,metrics=[],[],[]
            for mode in ("PROBE","ADAPT"):
                directory=r/f"adaptation/A_{source}_{mode}/formal/cross_day/seed_240800"
                m,a,score=read_leaf(directory,leaf,selection_frozen=True)
                values.append(score);arrays.append(a);metrics.append(m)
                eeg_results[f"A_{source}_{mode}_{leaf}"]=dict(zip(h.METRICS,score.mean(axis=0).tolist()))
                for li,label in enumerate((*v.c.a.LABEL_NAMES,"macro")):
                    point=score[li] if li<11 else score.mean(axis=0)
                    eeg_rows.append({"leaf":leaf,"route":f"A_{source}_{mode}","adaptation_seed":240800,"label":label,**dict(zip(h.METRICS,point))})
            co={f"A_{source}_ADAPT-PROBE":(-1,1)}
            intervals,blocks=h.bootstraps(arrays,np.asarray(metrics[0]["train_target_std"]),co,2000,240800+(171000 if leaf=="val" else 181000))
            delta=values[1]-values[0]
            for li,label in enumerate((*v.c.a.LABEL_NAMES,"macro")):
                point=delta[li] if li<11 else delta.mean(axis=0)
                eeg_delta_rows.append({"leaf":leaf,"contrast":next(iter(co)),"adaptation_seed":240800,"label":label,**dict(zip(h.METRICS,point))})
                ci=intervals[next(iter(co))]["per_label"][:,li] if li<11 else intervals[next(iter(co))]["macro"]
                for j,metric in enumerate(h.METRICS):
                    eeg_bootstrap_rows.append({"leaf":leaf,"contrast":next(iter(co)),"adaptation_seed":240800,"label":label,"metric":metric,
                        "delta":float(point[j]),"ci_low":float(ci[0,j]),"ci_high":float(ci[1,j]),"iterations":2000,"subject_day_blocks":blocks})
    for name,rows in (("per_label_val_test_metrics.csv",metric_rows),("paired_val_test_deltas.csv",delta_rows),
        ("paired_val_test_bootstrap.csv",bootstrap_rows),("window_variability.csv",diagnostic_rows),("subject_day_diagnostics.csv",date_rows),
        ("eeg_only_val_test_metrics.csv",eeg_rows),("eeg_only_paired_deltas.csv",eeg_delta_rows),("eeg_only_paired_bootstrap.csv",eeg_bootstrap_rows)):
        h.csv_write(out/name,rows)
    cells=[{"protocol":protocol,"route":route,"seed":seed,"status":"completed" if protocol=="cross_day" else "stopped_protocol_overlap",
        "reused":route in ("B0_W","F_C","F_P"),"path":str(v.route_path(r,route,seed)) if protocol=="cross_day" else None}
        for protocol in ("cross_day","within_subject_day") for route in ROUTES for seed in SEEDS]
    for source in ("C","P"):
        for mode in ("PROBE","ADAPT"):cells.append({"protocol":"cross_day","route":f"A_{source}_{mode}","seed":240800,"status":"completed","scope":"EEG_only_single_adaptation_seed"})
    h.write(out/"cell_status.json",cells)
    result={"status":"complete","summaries":summaries,"contrasts":contrasts,"eeg_only":eeg_results,
        "frozen_val_selection":frozen,"val_selection_sha256":h.sha(out/"selection_val_only.json"),
        "unique_fusion_cells":36,"new_fusion_cells":27,"reused_fusion_cells":9,"eeg_only_cells":4,
        "bootstrap_unit":"subject_day_within_each_seed_common_resampling_all_12_routes", "bootstrap_draws":2000,
        "upstream_ssl_seed_count":1,"adaptation_seed_count":1,"downstream_seed_count":3,
        "test_used_for_posthoc_selection":False,"evidence_status":"exploratory_fixed_plan"}
    h.write(out/"results_val_test.json",result)
    lines=["# 第二轮 A/B/C：Stage D 统一 val/test 报告","","全部36个唯一融合cell与4个EEG-only cell完成；val选择先冻结，test保持同checkpoint，不增加调参。", "",
        "## Test：11情绪等权宏平均（三seed均值±样本SD）","","| 配置 | raw r | centered r | sRMSE |","|---|---:|---:|---:|"]
    for route in ROUTES:
        s=summaries["test"][route]
        lines.append("| "+route+" | "+" | ".join(f"{s[k]['mean']:.6f} ± {s[k]['sd']:.6f}" for k in (h.METRICS[0],h.METRICS[1],h.METRICS[3]))+" |")
    lines += ["","## 匹配差值与val决策","","| 对照 | val Δraw r | test Δraw r | test正向seed | test Δcentered r | test ΔsRMSE |","|---|---:|---:|---:|---:|---:|"]
    for name in COEFFICIENTS:
        val,test=contrasts["val"][name],contrasts["test"][name]
        lines.append(f"| {name} | {val['means']['raw_r']:+.6f} | {test['means']['raw_r']:+.6f} | {test['raw_r_positive_seeds']}/3 | {test['means']['within_subject_centered_r']:+.6f} | {test['means']['standardized_rmse']:+.6f} |")
    lines += ["","A/B/C val门槛与候选保持 `selection_val_only.json` 的冻结内容。Test配对方向与区间用于评估保留效果，全部cell均报告，test不重新选择候选。", "",
        "`per_label_val_test_metrics.csv`包含逐11情绪五项指标、目标/预测SD；`paired_val_test_bootstrap.csv`每seed按subject-day blocks共同重采样2,000次，保留宏/逐情绪95%区间。", "",
        "`window_variability.csv`与`subject_day_diagnostics.csv`展示六条C路线的事件内窗口变化、极值、event预测SD及日期级误差。窗口预测为弱监督输出，EMA标签不提供逐窗真值。", "",
        "EEG-only A在`eeg_only_*`单列一个适配seed；下游三seed不扩展为上游seed稳健性。within_subject_day按原信号/context重叠停止。历史test已被查看，本轮按预定方案的探索性机制验证报告。可选联合确认保持独立预算。"]
    (out/"stage_d_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    h.write(out/"completion_audit.json",{"status":"pass","fusion_prediction_leaves_recomputed":total_recomputed,
        "eeg_only_prediction_leaves_recomputed":8,"all36_membership_matched":True,"val_frozen_before_test_report":True,
        "frozen_val_selection_sha256":h.sha(out/"selection_val_only.json"),"test_report_sha256":h.sha(out/"results_val_test.json"),
        "independent_ABC_replay_markers_verified":True,"within_subject_day":"stopped_protocol_overlap"})
    (r/"ROUND2_STAGE_D_TEST_REPORT_COMPLETE").write_text("complete\n")
    (r/"ROUND2_CORE_ABC_COMPLETE_WITH_PROTOCOL_STOP").write_text("complete\n")
    print(json.dumps({"status":"complete","unique_fusion_cells":36,"test_summaries":summaries["test"],"test_contrasts":contrasts["test"]}),flush=True)

if __name__=="__main__":main()
