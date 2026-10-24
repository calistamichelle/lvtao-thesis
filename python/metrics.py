"""Time-aligned recognition, transition and cadence metrics for one decision trace.

Irregular predictions are mapped to the 50 Hz sample grid with a causal hold rule:
a prediction made when window [end-128, end) completes is shown from sample `end`
until the next prediction (or the end of the segment). Every sample then has equal
weight, so macro F1 on this grid is the time-weighted macro F1.
"""
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import config as C


def hold_predictions(trace, n, bounds):
    pred = np.full(n, C.INVALID, np.int8)
    seg_end_of = {}
    for s, e in bounds:
        seg_end_of[s] = e
    seg_starts = np.array([s for s, _ in bounds])
    for i, row in enumerate(trace):
        start = row["end_sample"]
        seg_end = seg_end_of[seg_starts[np.searchsorted(seg_starts, row["start_sample"], side="right") - 1]]
        stop = seg_end
        if i + 1 < len(trace) and trace[i + 1]["end_sample"] <= seg_end:
            stop = trace[i + 1]["end_sample"]
        pred[start:stop] = row["prediction"]
    return pred


def classification(y_true, y_pred, labels=None):
    mask = (y_true != C.INVALID) & (y_pred != C.INVALID)
    yt, yp = y_true[mask], y_pred[mask]
    if labels is None:
        labels = np.unique(yt)
    if len(yt) == 0:
        return {"scored_samples": 0}
    p, r, f, support = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(yt, yp, labels=labels, average="weighted", zero_division=0)
    out = {
        "scored_samples": int(mask.sum()),
        "accuracy": float(np.mean(yt == yp)),
        "balanced_accuracy": float(np.mean(r[support > 0])),
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "macro_f1": float(np.mean(f)),
        "weighted_precision": float(pw),
        "weighted_recall": float(rw),
        "weighted_f1": float(fw),
    }
    for lab, pi, ri, fi in zip(labels, p, r, f):
        name = C.CLASSES[lab]
        out[f"precision_{name}"], out[f"recall_{name}"], out[f"f1_{name}"] = float(pi), float(ri), float(fi)
    return out


def confusion(y_true, y_pred):
    mask = (y_true != C.INVALID) & (y_pred != C.INVALID)
    return confusion_matrix(y_true[mask], y_pred[mask], labels=range(len(C.CLASSES)))


def runs(y, bounds):
    """(start, end, label) runs of constant label inside each segment."""
    out = []
    for s, e in bounds:
        seg = y[s:e]
        if len(seg) == 0:
            continue
        cuts = np.flatnonzero(np.diff(seg)) + 1
        starts = np.concatenate([[0], cuts]) + s
        ends = np.concatenate([cuts, [len(seg)]]) + s
        prev = None
        for a, b in zip(starts.tolist(), ends.tolist()):
            out.append((a, b, int(y[a]), prev))
            prev = int(y[a])
    return out


def transition_mask(y, bounds):
    """Samples within TRANSITION_BAND of a boundary between two scorable activities."""
    mask = np.zeros(len(y), bool)
    for start, _, label, prev in runs(y, bounds):
        if prev is not None and prev != C.INVALID and label != C.INVALID:
            mask[max(0, start - C.TRANSITION_BAND):start + C.TRANSITION_BAND] = True
    return mask


def detection_delays(y, y_pred, bounds):
    """Seconds from each activity change until the new activity is first predicted."""
    delays, missed = [], 0
    for start, end, label, prev in runs(y, bounds):
        if prev is None or prev == C.INVALID or label == C.INVALID or end - start < C.MIN_RUN_FOR_DELAY:
            continue
        hit = np.flatnonzero(y_pred[start:end] == label)
        if len(hit):
            delays.append(hit[0] / C.FS)
        else:
            missed += 1
    return delays, missed


def cadence(trace, bounds):
    stream_samples = sum(e - s for s, e in bounds)
    minutes = stream_samples / C.FS / 60
    strides = np.array([r["next_stride"] for r in trace]) if trace else np.zeros(0)
    out = {"inferences": len(trace), "stream_minutes": minutes,
           "inferences_per_min": len(trace) / minutes if minutes else 0.0}
    total = strides.sum() if len(strides) else 1
    for s in sorted(set(C.LVTAO_STRIDES) | set(strides.tolist())):
        out[f"share_decisions_stride_{s}"] = float(np.mean(strides == s)) if len(strides) else 0.0
        out[f"share_time_stride_{s}"] = float(strides[strides == s].sum() / total) if len(strides) else 0.0
    return out


def evaluate_trace(trace, y, bounds):
    """All per-participant metrics for one policy. Also returns the held prediction grid."""
    y_pred = hold_predictions(trace, len(y), bounds)
    m = classification(y, y_pred)
    band = transition_mask(y, bounds)
    trans = classification(np.where(band, y, C.INVALID), y_pred)
    m["transition_macro_f1"] = trans.get("macro_f1", float("nan"))
    m["transition_scored_samples"] = trans.get("scored_samples", 0)
    delays, missed = detection_delays(y, y_pred, bounds)
    m["delay_median_s"] = float(np.median(delays)) if delays else float("nan")
    m["delay_p90_s"] = float(np.percentile(delays, 90)) if delays else float("nan")
    m["transitions_detected"] = len(delays)
    m["transitions_missed"] = missed
    m["uncovered_or_unscored_share"] = float(np.mean((y_pred == C.INVALID) | (y == C.INVALID)))
    for i, name in enumerate(C.CLASSES):
        m[f"time_true_min_{name}"] = float(np.sum(y == i)) / C.FS / 60
        m[f"predicted_windows_{name}"] = int(sum(r["prediction"] == i for r in trace))
    m.update(cadence(trace, bounds))
    return m, y_pred
