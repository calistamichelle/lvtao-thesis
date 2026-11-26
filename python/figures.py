"""Step 9: the paper's result figures and the numbers quoted next to them.

Reads only files written by earlier steps (results/replay/, models/fold<k>/lvtao_params.json, the
preprocessed data and the fold-0 Keras model). Nothing here changes a result; it only draws and counts.

Outputs (results/figures/):
  fig2_tradeoff.png     Fig. 2  macro F1 and median detection delay against the inference rate (pooled test)
  fig3_timeline.png     Fig. 3  4-minute excerpt: signal, true activity, change score, inference instants
  fig4_transitions.png  Fig. 4  (a) inference rate after an activity change, (b) share of changes detected
  numbers.json          every other number the text quotes with these figures (see main())
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import data as D
import metrics as M

OUT = C.RESULTS_DIR / "figures"
SERIF = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 7}
THIN = {**SERIF, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# One color per policy in every figure.
POLICY_STYLE = {
    "lvtao": ("LVTAO", "#2a78d6", "-"),
    "budget_matched": ("Budget-matched fixed", "#eb6834", "--"),
    "fixed_128": ("Fixed 128", "#1baf7a", "-."),
    "fixed_32": ("Fixed 32", "#eda100", ":"),
}
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MAX_SINCE_S, BIN_S = 30, 2


def trace(fold, pid, policy):
    return pd.read_csv(C.RESULTS_DIR / "replay" / "traces" / f"fold{fold}" / f"{pid}__{policy}.csv")


def test_participants():
    for f in D.load_folds():
        for pid in f["test"]:
            yield f["fold"], pid


# --- Fig. 2 ---------------------------------------------------------------------------------------------

def fig_tradeoff():
    markers = {
        "lvtao": ("LVTAO", "*", "#2a78d6", 70),
        "budget_matched": ("Budget-matched fixed", "s", "#eb6834", 22),
    }
    pooled = pd.read_csv(C.RESULTS_DIR / "replay" / "pooled.csv").set_index("policy")
    fixed = pooled.loc[[f"fixed_{s}" for s in C.FIXED_STRIDES]]
    with plt.rc_context(SERIF):
        fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=300)
        for ax, col, label in ((axes[0], "macro_f1", "Time-weighted macro F1"),
                               (axes[1], "delay_median_s", "Median detection delay (s)")):
            ax.plot(fixed.inferences_per_min, fixed[col], "-o", color="black", ms=2.5, lw=0.7, label="Fixed stride")
            for s, (x, y) in zip(C.FIXED_STRIDES, zip(fixed.inferences_per_min, fixed[col])):
                ax.annotate(str(s), (x, y), fontsize=5, xytext=(2, 2), textcoords="offset points")
            for key, (name, m, c, size) in markers.items():
                ax.scatter(pooled.loc[key, "inferences_per_min"], pooled.loc[key, col], marker=m, color=c, s=size,
                           label=name, zorder=3, linewidths=0)
            ax.set_xlabel("Inferences per minute")
            ax.set_ylabel(label)
            ax.grid(lw=0.3, alpha=0.5)
            ax.tick_params(labelsize=6)
        axes[0].legend(fontsize=5, loc="lower right", frameon=False)
        fig.tight_layout(pad=0.2, w_pad=0.6)
        fig.savefig(OUT / "fig2_tradeoff.png")
        plt.close(fig)


# --- Fig. 4 ---------------------------------------------------------------------------------------------

def seconds_since_change(y, bounds):
    """Per sample: seconds since the latest change between two scored activities in its segment (NaN before any)."""
    since = np.full(len(y), np.nan)
    n_changes = 0
    for start, _, label, prev in M.runs(y, bounds):
        if prev is not None and prev != C.INVALID and label != C.INVALID:
            n_changes += 1
            seg_end = next(e for s, e in bounds if s <= start < e)
            since[start:seg_end] = np.arange(seg_end - start) / C.FS   # overwritten by the next change
    return since, n_changes


def analyse_transitions():
    edges = np.append(np.arange(0, MAX_SINCE_S + BIN_S, BIN_S), np.inf)   # last bin: steady state (> 30 s)
    counts = {p: np.zeros(len(edges) - 1) for p in POLICY_STYLE}
    exposure = np.zeros(len(edges) - 1)                                    # minutes of stream per bin
    n_changes = 0
    delays = {p: [] for p in POLICY_STYLE}
    missed = {p: 0 for p in POLICY_STYLE}
    for fold, pid in test_participants():
        _, y, bounds = D.load(pid)
        since, n = seconds_since_change(y, bounds)
        n_changes += n
        ok = ~np.isnan(since)
        exposure += np.histogram(since[ok], edges)[0] / C.FS / 60
        for p in POLICY_STYLE:
            t = trace(fold, pid, p)
            at = since[t.end_sample.to_numpy() - 1]
            counts[p] += np.histogram(at[~np.isnan(at)], edges)[0]
            y_pred = M.hold_predictions(t.to_dict("records"), len(y), bounds)
            d, m = M.detection_delays(y, y_pred, bounds)
            delays[p].extend(d)
            missed[p] += m
    rate = {p: counts[p] / exposure for p in POLICY_STYLE}               # inferences per minute
    return edges, rate, n_changes, delays, missed


def fig_transitions(edges, rate, delays, missed):
    with plt.rc_context(THIN):
        fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.65), dpi=300)
        centers = (edges[:-2] + edges[1:-1]) / 2
        ax = axes[0]
        for p in ("lvtao", "budget_matched", "fixed_128"):
            name, color, ls = POLICY_STYLE[p]
            ax.plot(centers, rate[p][:-1], ls=ls, color=color, lw=1.0, label=name)
        steady = rate["lvtao"][-1]
        ax.axhline(steady, color=POLICY_STYLE["lvtao"][1], lw=0.5, ls=(0, (1, 2)))
        ax.text(MAX_SINCE_S, steady, "LVTAO\n> 30 s", fontsize=5, color=MUTED, ha="right", va="top")
        ax.set_xlabel("Time since activity change (s)")
        ax.set_ylabel("Inferences per minute")
        ax.set_xlim(0, MAX_SINCE_S)
        ax.legend(fontsize=5, frameon=False, loc="upper right")
        ax.set_title("(a)", fontsize=7, loc="left")
        ax = axes[1]
        for p, (name, color, ls) in POLICY_STYLE.items():
            d = np.sort(delays[p])
            total = len(d) + missed[p]
            ax.plot(d, np.arange(1, len(d) + 1) / total, ls=ls, color=color, lw=1.0, label=name)
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Detection delay (s)")
        ax.set_ylabel("Fraction of changes detected")
        ax.legend(fontsize=5, frameon=False, loc="lower right")
        ax.set_title("(b)", fontsize=7, loc="left")
        for ax in axes:
            ax.grid(color=GRID, lw=0.4)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=6, colors=MUTED)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        fig.tight_layout(pad=0.2, w_pad=0.8)
        fig.savefig(OUT / "fig4_transitions.png")
        plt.close(fig)


# --- Fig. 3 ---------------------------------------------------------------------------------------------

def pick_excerpt(length_s=240):
    """A test excerpt with several clean activity changes and many distinct activities."""
    best = None
    n = length_s * C.FS
    for fold, pid in test_participants():
        _, y, bounds = D.load(pid)
        for s, e in bounds:
            for start in range(s, e - n, 10 * C.FS):
                seg = y[start:start + n]
                if (seg == C.INVALID).any():
                    continue
                changes = int(np.count_nonzero(np.diff(seg)))
                distinct = len(np.unique(seg))
                if changes > 8:          # avoid flickering labels
                    continue
                key = (distinct, changes)
                if best is None or key > best[0]:
                    best = (key, fold, pid, start)
    return best[1], best[2], best[3], n


def fig_timeline():
    fold, pid, start, n = pick_excerpt()
    x, y, _ = D.load(pid)
    t = (np.arange(start, start + n) - start) / C.FS
    mag = np.linalg.norm(x[start:start + n], axis=1)
    lv = trace(fold, pid, "lvtao")
    bm = trace(fold, pid, "budget_matched")
    cal = D.load_json(D.fold_dir(fold) / "lvtao_params.json")
    t1, t2 = cal["lvtao"]["t1"], cal["lvtao"]["t2"]
    sel = lv[(lv.end_sample >= start) & (lv.end_sample < start + n)]
    sel_bm = bm[(bm.end_sample >= start) & (bm.end_sample < start + n)]
    labels = y[start:start + n]
    cuts = np.concatenate([[0], np.flatnonzero(np.diff(labels)) + 1, [n]])

    with plt.rc_context(THIN):
        fig, axes = plt.subplots(3, 1, figsize=(3.5, 2.55), dpi=300, sharex=True,
                                 gridspec_kw={"height_ratios": [1.5, 1.0, 0.55]})
        ax = axes[0]
        for a, b in zip(cuts[:-1], cuts[1:]):
            cls = labels[a]
            ax.axvspan(t[a], t[b - 1], color=CLASS_COLORS[cls], alpha=0.18, lw=0)
            width = (b - a) / C.FS
            if width > 14:
                ax.text((t[a] + t[b - 1]) / 2, 0.97, C.CLASSES[cls], transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=5.5, color=INK)
            elif width > 4:      # short bout: label it vertically inside the band
                ax.text((t[a] + t[b - 1]) / 2, 0.97, C.CLASSES[cls], transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=5, color=INK, rotation=90)
        ax.plot(t, mag, color=INK, lw=0.35)
        ax.set_ylabel("|a| (g)")
        ax = axes[1]
        st = (sel.end_sample.to_numpy() - start) / C.FS
        ax.step(st, sel.score, where="post", color=POLICY_STYLE["lvtao"][1], lw=0.8)
        for thr, name, va in ((t1, "$T_1$", "top"), (t2, "$T_2$", "bottom")):
            ax.axhline(thr, color=MUTED, lw=0.5, ls="--")
            ax.text(t[-1], thr, f" {name}", va=va, ha="left", fontsize=5.5, color=MUTED)
        ax.set_ylabel("Change\nscore $s$")
        ax = axes[2]
        ax.eventplot([(sel_bm.end_sample.to_numpy() - start) / C.FS, st], lineoffsets=[0, 1], linelengths=0.7,
                     colors=[POLICY_STYLE["budget_matched"][1], POLICY_STYLE["lvtao"][1]], linewidths=0.6)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([f"Fixed {int(cal['budget_matched_stride'])}", "LVTAO"], fontsize=5.5)
        ax.set_ylim(-0.6, 1.6)
        ax.set_xlabel("Time (s)")
        for ax in axes:
            ax.tick_params(labelsize=6, colors=MUTED)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        axes[0].set_xlim(0, t[-1])
        fig.tight_layout(pad=0.2, h_pad=0.3)
        fig.savefig(OUT / "fig3_timeline.png")
        plt.close(fig)
    return {"participant": pid, "fold": fold, "start_s": start / C.FS, "length_s": n / C.FS,
            "lvtao_inferences": int(len(sel)), "budget_inferences": int(len(sel_bm)),
            "activities": [C.CLASSES[c] for c in labels[cuts[:-1]]]}


# --- Numbers quoted in the text ---------------------------------------------------------------------------

def stride_use():
    """LVTAO's stride use, averaged over test participants: share of stream time waiting with stride 128
    (stride-weighted) and share of decisions that chose stride 32."""
    time_128, dec_32 = [], []
    for fold, pid in test_participants():
        s = trace(fold, pid, "lvtao").next_stride.to_numpy()
        time_128.append(s[s == 128].sum() / s.sum())
        dec_32.append(np.mean(s == 32))
    return {"time_share_stride_128": round(float(np.mean(time_128)), 3),
            "decision_share_stride_32": round(float(np.mean(dec_32)), 3)}


def standing_as_walking():
    """Where LVTAO's standing->walking errors fall: in the first window (2.56 s) of a standing bout,
    and in standing bouts shorter than 10 s."""
    walk, stand = C.CLASSES.index("walking"), C.CLASSES.index("standing")
    total = early = short = 0
    for fold, pid in test_participants():
        _, y, bounds = D.load(pid)
        y_pred = M.hold_predictions(trace(fold, pid, "lvtao").to_dict("records"), len(y), bounds)
        for start, end, label, _ in M.runs(y, bounds):
            if label != stand:
                continue
            idx = np.arange(start, end)
            err = idx[y_pred[idx] == walk]
            total += len(err)
            early += int(np.sum(err - start < C.WINDOW))
            short += len(err) if end - start < 10 * C.FS else 0
    return {"samples": int(total), "share_in_first_window": round(early / total, 3),
            "share_in_bouts_under_10s": round(short / total, 3)}


def operation_counts():
    """Arithmetic per LVTAO decision (from window_features in policies.py and controller.cpp) against the
    multiply-accumulates of one CNN inference (from the fold-0 Keras model's layer shapes)."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    n = C.WINDOW
    # per axis: mean (n adds), variance (n subtractions, n multiplications, n adds),
    # temporal difference (n-1 subtractions, n-1 absolute values, n-1 adds)
    controller = len(C.AXES) * (n + 3 * n + 3 * (n - 1))
    model = tf.keras.models.load_model(D.fold_dir(C.FIRMWARE_FOLD) / "model.keras", compile=False)
    macs = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv1D):
            macs += layer.output.shape[1] * layer.kernel_size[0] * layer.input.shape[-1] * layer.filters
        elif isinstance(layer, tf.keras.layers.Dense):
            macs += layer.input.shape[-1] * layer.units
    return {"controller_ops_per_decision": int(controller), "controller_logarithms": 2,
            "cnn_multiply_accumulates": int(macs), "cnn_ops": int(2 * macs),
            "controller_share_of_cnn_ops": round(controller / (2 * macs), 4)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_tradeoff()
    edges, rate, n_changes, delays, missed = analyse_transitions()
    fig_transitions(edges, rate, delays, missed)
    numbers = {"changes": n_changes}
    for p in POLICY_STYLE:
        r = rate[p]
        d = np.array(delays[p])
        numbers[p] = {"rate_0_2s": round(float(r[0]), 1), "rate_0_10s": round(float(r[:5].mean()), 1),
                      "rate_10_30s": round(float(r[5:-1].mean()), 1), "rate_steady_over_30s": round(float(r[-1]), 1),
                      "peak_rate": round(float(r[:-1].max()), 1),
                      "peak_bin_start_s": float(edges[int(r[:-1].argmax())]),
                      "detected": int(len(d)), "missed": int(missed[p]),
                      "within_1s": round(float(np.mean(d <= 1.0) * len(d) / (len(d) + missed[p])), 3),
                      "within_2s": round(float(np.mean(d <= 2.0) * len(d) / (len(d) + missed[p])), 3),
                      "delay_median": round(float(np.median(d)), 2), "delay_mean": round(float(d.mean()), 2)}
    numbers["timeline"] = fig_timeline()
    numbers["lvtao_stride_use"] = stride_use()
    numbers["standing_predicted_as_walking"] = standing_as_walking()
    numbers["operations"] = operation_counts()
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=2))
    print(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()
