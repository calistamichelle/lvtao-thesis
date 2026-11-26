"""Step 7: chapter 4 tables and figures.

Paired per-participant bootstrap 95% CIs for LVTAO minus each baseline, the
non-inferiority verdict against the best fixed stride (chosen on validation), and
an accuracy-cadence Pareto plot. Outputs go to results/report/.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import data as D

METRICS = ["macro_f1", "transition_macro_f1", "delay_median_s", "inferences_per_min"]


def bootstrap_ci(diffs, rng):
    diffs = np.asarray(diffs, float)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return np.nan, np.nan, np.nan
    means = rng.choice(diffs, size=(C.BOOTSTRAP_ITERS, len(diffs))).mean(axis=1)
    return diffs.mean(), *np.percentile(means, [2.5, 97.5])


def paired_table(df, energy):
    """LVTAO vs every comparator, paired by participant. 'best_fixed' is each fold's validation pick."""
    best = {f["fold"]: D.load_json(D.fold_dir(f["fold"]) / "lvtao_params.json")["best_fixed_stride"]
            for f in D.load_folds() if (D.fold_dir(f["fold"]) / "lvtao_params.json").exists()}
    df = df.copy()
    if energy is not None:
        df = df.merge(energy[["participant", "policy", "energy_awake_mJ", "energy_light_sleep_mJ"]],
                      on=["participant", "policy"], how="left")
    metrics = METRICS + [m for m in ("energy_awake_mJ", "energy_light_sleep_mJ") if m in df]
    best_rows = df[df.apply(lambda r: r.policy == f"fixed_{best.get(r.fold)}", axis=1)].assign(policy="best_fixed")
    df = pd.concat([df, best_rows])
    wide = df.pivot_table(index="participant", columns="policy", values=metrics)

    rng = np.random.default_rng(C.SEED)
    rows = []
    comparators = ["best_fixed", *[f"fixed_{s}" for s in C.FIXED_STRIDES], "budget_matched"]
    for comp in comparators:
        if ("macro_f1", comp) not in wide:
            continue
        for m in metrics:
            mean, lo, hi = bootstrap_ci(wide[(m, "lvtao")] - wide[(m, comp)], rng)
            rows.append({"comparator": comp, "metric": m, "mean_diff": mean, "ci_low": lo, "ci_high": hi,
                         "participants": int(wide[(m, comp)].notna().sum())})
    return pd.DataFrame(rows)


def ieee_table(df, number, caption):
    """Markdown table with an IEEE-style caption placed above it."""
    return f"TABLE {number}. {caption.upper()}\n\n{df.to_markdown(index=False, floatfmt='.3f')}\n"


def main():
    out = C.RESULTS_DIR / "report"
    out.mkdir(parents=True, exist_ok=True)
    per = pd.read_csv(C.RESULTS_DIR / "replay" / "per_participant.csv")
    pooled = pd.read_csv(C.RESULTS_DIR / "replay" / "pooled.csv")
    energy_path = C.RESULTS_DIR / "energy" / "per_participant.csv"
    energy = pd.read_csv(energy_path) if energy_path.exists() else None

    paired = paired_table(per, energy)
    paired.to_csv(out / "paired_differences.csv", index=False)

    f1 = paired[(paired.comparator == "best_fixed") & (paired.metric == "macro_f1")]
    verdict = "n/a"
    if len(f1):
        lo = f1.ci_low.iloc[0]
        verdict = "NON-INFERIOR" if lo >= -C.NONINFERIORITY_DELTA else "NOT shown non-inferior"
    delay = pooled.set_index("policy").loc["lvtao", "delay_median_s"]

    cols = ["policy", "macro_f1", "subject_macro_f1", "transition_macro_f1_mean", "delay_median_s",
            "delay_p90_s", "inferences_per_min"]
    text = [
        "# Results summary\n",
        f"Time-weighted macro F1, LVTAO vs best fixed stride (validation-chosen): {verdict} "
        f"at delta = {C.NONINFERIORITY_DELTA} (lower 95% CI bound vs -delta).",
        f"LVTAO median detection delay {delay:.2f} s; limit {C.MAX_MEDIAN_DELAY_S:.2f} s.\n",
        ieee_table(pooled[cols], "IV", "Recognition and cadence per scheduling policy (pooled outer-test participants)"),
        ieee_table(paired, "V", "Paired differences, LVTAO minus comparator, with bootstrap 95% CI"),
    ]
    if energy is not None:
        es = pd.read_csv(C.RESULTS_DIR / "energy" / "summary.csv")
        text.append(ieee_table(es, "VI", "Estimated energy per policy (datasheet currents, measured on-device time)"))
    (out / "summary.md").write_text("\n".join(text))

    fig, ax = plt.subplots(figsize=(3.5, 2.6), dpi=200)   # one IEEE column wide
    for _, r in pooled.iterrows():
        ax.scatter(r.inferences_per_min, r.macro_f1, s=20 + 40 * np.nan_to_num(r.delay_median_s),
                   marker="*" if r.policy.startswith("lvtao") else "o")
        ax.annotate(r.policy, (r.inferences_per_min, r.macro_f1), fontsize=5, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xlabel("Inferences per minute")
    ax.set_ylabel("Time-weighted macro F1")
    ax.tick_params(labelsize=6)
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)
    fig.tight_layout()
    fig.savefig(out / "pareto.png")
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
