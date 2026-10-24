"""Step 5: replay every policy causally on the same outer-test streams.

Outputs (results/replay/):
  per_participant.csv   one row per (fold, participant, policy) with every metric
  pooled.csv            one row per policy, metrics on all test samples pooled
  confusion_<policy>.csv
  traces/fold<k>/<pid>__<policy>.csv   decision traces
"""
import argparse

import numpy as np
import pandas as pd

import config as C
import data as D
import metrics as M
from int8_model import CachedPredictor, Int8Model
from policies import FeatureCache, LvtaoParams, Policy, run_policy


def fold_policies(fold_id):
    """Fixed strides, LVTAO (main + composite ablation) and the budget-matched baseline for one fold."""
    cal = D.load_json(D.fold_dir(fold_id) / "lvtao_params.json")
    policies = [Policy.fixed(s) for s in C.FIXED_STRIDES]
    policies.append(Policy.lvtao(LvtaoParams(**cal["lvtao"]), "lvtao"))
    if "lvtao_composite" in cal:
        policies.append(Policy.lvtao(LvtaoParams(**cal["lvtao_composite"]), "lvtao_composite"))
    policies.append(Policy.fixed(cal["budget_matched_stride"], "budget_matched"))
    return policies


def evaluate(policies, pids, model, traces_dir=None, fold_id=None):
    """Per-participant metric rows and pooled (y_true, y_pred) per policy."""
    rows, pooled = [], {p.name: ([], []) for p in policies}
    for pid in pids:
        x, y, bounds = D.load(pid)
        predict, features = CachedPredictor(model, x), FeatureCache(x)
        for policy in policies:
            trace = run_policy(policy, x, y, bounds, predict, features)
            m, y_pred = M.evaluate_trace(trace, y, bounds)
            rows.append({"fold": fold_id, "participant": pid, "policy": policy.name, **m})
            pooled[policy.name][0].append(y)
            pooled[policy.name][1].append(y_pred)
            if traces_dir is not None:
                pd.DataFrame(trace).to_csv(traces_dir / f"{pid}__{policy.name}.csv", index=False)
    return rows, {k: (np.concatenate(a), np.concatenate(b)) for k, (a, b) in pooled.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, help="replay one fold (default: all)")
    args = ap.parse_args()
    out = C.RESULTS_DIR / "replay"
    out.mkdir(parents=True, exist_ok=True)

    all_rows, pooled_all = [], {}
    for fold in D.load_folds():
        if args.fold is not None and fold["fold"] != args.fold:
            continue
        traces_dir = out / "traces" / f"fold{fold['fold']}"
        traces_dir.mkdir(parents=True, exist_ok=True)
        model = Int8Model(fold["fold"])
        rows, pooled = evaluate(fold_policies(fold["fold"]), fold["test"], model, traces_dir, fold["fold"])
        all_rows.extend(rows)
        for name, (yt, yp) in pooled.items():
            pooled_all.setdefault(name, ([], []))
            pooled_all[name][0].append(yt)
            pooled_all[name][1].append(yp)
        print(f"fold {fold['fold']}: replayed {len(fold['test'])} test participants")

    pd.DataFrame(all_rows).to_csv(out / "per_participant.csv", index=False)

    # Budget-matched strides differ per fold, so the pooled row for it mixes folds' strides by design.
    df = pd.DataFrame(all_rows)
    summary = []
    for name, (yts, yps) in pooled_all.items():
        yt, yp = np.concatenate(yts), np.concatenate(yps)
        m = M.classification(yt, yp)
        sub = df[df.policy == name]
        m.update({
            "policy": name,
            "subject_macro_f1": float(sub["macro_f1"].mean()),
            "transition_macro_f1_mean": float(sub["transition_macro_f1"].mean()),
            "delay_median_s": float(sub["delay_median_s"].median()),
            "delay_p90_s": float(sub["delay_p90_s"].median()),
            "inferences": int(sub["inferences"].sum()),
            "inferences_per_min": float(sub["inferences"].sum() / sub["stream_minutes"].sum()),
        })
        summary.append(m)
        pd.DataFrame(M.confusion(yt, yp), index=C.CLASSES, columns=C.CLASSES).to_csv(out / f"confusion_{name}.csv")
    summary = pd.DataFrame(summary).set_index("policy")
    summary.to_csv(out / "pooled.csv")
    cols = ["macro_f1", "subject_macro_f1", "transition_macro_f1_mean", "delay_median_s", "inferences_per_min"]
    print(summary[cols].round(3).to_string())


if __name__ == "__main__":
    main()
