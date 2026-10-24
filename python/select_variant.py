"""Step 4b: choose the main LVTAO rule on VALIDATION participants only.

Rule (fixed before any test replay of the variants): the main variant is the one with
the largest mean validation macro-F1 gain over a fixed stride that spends the same
number of inferences (its budget-matched stride). Run after calibrate.py; copy the
choice into config.LVTAO_RULE and run calibrate.py again so "lvtao" points to it.

Output: results/lvtao_variant_selection.csv
"""
import pandas as pd

import config as C
import data as D

VARIANTS = {"lvtao_intensity": "max", "lvtao_composite": "composite", "lvtao_change": "change"}


def main():
    rows = []
    for fold in D.load_folds():
        cal = D.load_json(D.fold_dir(fold["fold"]) / "lvtao_params.json")
        for key, rule in VARIANTS.items():
            v, b = cal[f"{key}_validation"], cal[f"{key}_budget_validation"]
            rows.append({"fold": fold["fold"], "variant": key, "rule": rule,
                         "val_macro_f1": v["macro_f1"], "val_inferences_per_min": v["inferences_per_min"],
                         "budget_stride": cal[f"{key}_budget_stride"], "budget_val_macro_f1": b["macro_f1"],
                         "gain_vs_budget": v["macro_f1"] - b["macro_f1"]})
    df = pd.DataFrame(rows)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.RESULTS_DIR / "lvtao_variant_selection.csv", index=False)
    summary = df.groupby(["variant", "rule"])[["gain_vs_budget", "val_macro_f1", "val_inferences_per_min"]].mean()
    print(summary.round(4).to_string())
    best = summary.gain_vs_budget.idxmax()
    print(f"\nCHOICE: LVTAO_RULE = \"{best[1]}\" ({best[0]})")


if __name__ == "__main__":
    main()
