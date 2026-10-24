"""Step 1b: choose training settings on VALIDATION participants only.

Every candidate keeps the Table III architecture; only class weighting and the
training-window stride change. For each fold, a model is trained per candidate and
scored by window-level macro F1 on that fold's validation participants. Test
participants are never loaded.

The winner is the highest mean validation macro F1 across folds. Candidates within
SELECTION_TOLERANCE of it count as tied, and the tie goes to the cheaper setting
(stride 64 before 32, then none < sqrt < balanced weighting).
Copy the printed choice into config.py (TRAIN_STRIDE, CLASS_WEIGHTING).

Output: results/model_selection.csv
"""
import itertools

import pandas as pd

import config as C
import data as D
from train import fit, val_macro_f1

WEIGHTINGS = ("none", "sqrt", "balanced")
STRIDES = (64, 32)
SELECTION_TOLERANCE = 0.005


def main():
    rows = []
    for fold, (stride, weighting) in itertools.product(D.load_folds(), itertools.product(STRIDES, WEIGHTINGS)):
        model, *_, seconds, n_train, val = fit(fold, weighting, stride, verbose=0)
        f1 = val_macro_f1(model, val)
        rows.append({"fold": fold["fold"], "train_stride": stride, "class_weighting": weighting,
                     "val_macro_f1": f1, "train_windows": n_train, "seconds": round(seconds, 1)})
        print(f"fold {fold['fold']} stride {stride:>2} {weighting:<8} val macro F1 {f1:.3f}  ({seconds:.0f} s)", flush=True)

    df = pd.DataFrame(rows)
    out = C.RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "model_selection.csv", index=False)

    summary = (df.groupby(["train_stride", "class_weighting"]).val_macro_f1
               .agg(["mean", "std", "min"]).reset_index())
    best = summary["mean"].max()
    tied = summary[summary["mean"] >= best - SELECTION_TOLERANCE].copy()
    tied["cost"] = tied.train_stride.map({64: 0, 32: 1}) * 10 + tied.class_weighting.map(
        {w: i for i, w in enumerate(WEIGHTINGS)})
    choice = tied.sort_values("cost").iloc[0]
    print("\nmean validation macro F1 across folds:")
    print(summary.round(3).to_string(index=False))
    print(f"\nCHOICE: TRAIN_STRIDE = {int(choice.train_stride)}, CLASS_WEIGHTING = \"{choice.class_weighting}\" "
          f"(mean {choice['mean']:.3f}; best {best:.3f}; tolerance {SELECTION_TOLERANCE})")


if __name__ == "__main__":
    main()
