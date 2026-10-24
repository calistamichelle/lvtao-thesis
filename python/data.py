"""Shared helpers for loading processed participants and cutting windows."""
import json

import numpy as np

import config as C


def participants():
    return sorted(p.stem for p in C.PROCESSED_DIR.glob("*.npz"))


def load(pid):
    """Return (x float32 [T,3] in g, y int8 [T] class index or INVALID, segment bounds [(start, end)])."""
    d = np.load(C.PROCESSED_DIR / f"{pid}.npz")
    return d["x"], d["y"], segment_bounds(d["seg"])


def segment_bounds(seg):
    """Contiguous [start, end) ranges with the same segment id."""
    if len(seg) == 0:
        return []
    cuts = np.flatnonzero(np.diff(seg)) + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [len(seg)]])
    return list(zip(starts.tolist(), ends.tolist()))


def load_folds():
    with open(C.PROCESSED_DIR / "folds.json") as f:
        return json.load(f)


def fold_dir(fold):
    d = C.MODELS_DIR / f"fold{fold}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def window_label(y_win):
    """Majority scorable label, or INVALID if too few samples are scorable."""
    valid = y_win[y_win != C.INVALID]
    if len(valid) < C.MIN_VALID_FRACTION * len(y_win):
        return C.INVALID
    return int(np.bincount(valid, minlength=len(C.CLASSES)).argmax())


def window_ends(bounds, stride):
    """End indices (exclusive) of fixed-stride windows that stay inside one segment."""
    ends = []
    for s, e in bounds:
        ends.extend(range(s + C.WINDOW, e + 1, stride))
    return ends


def labelled_windows(pids, stride):
    """Raw (un-normalised) windows and labels; windows without a valid label are dropped."""
    xs, ys = [], []
    for pid in pids:
        x, y, bounds = load(pid)
        for end in window_ends(bounds, stride):
            label = window_label(y[end - C.WINDOW:end])
            if label != C.INVALID:
                xs.append(x[end - C.WINDOW:end])
                ys.append(label)
    if not xs:
        return np.zeros((0, C.WINDOW, len(C.AXES)), np.float32), np.zeros(0, np.int64)
    return np.stack(xs).astype(np.float32), np.array(ys, np.int64)


def fit_norm(pids):
    """Channel-wise mean/std over all samples of the given (training) participants."""
    x = np.concatenate([load(pid)[0] for pid in pids]).astype(np.float64)
    return x.mean(axis=0).astype(np.float32), x.std(axis=0).astype(np.float32)


def normalize(x, mean, std):
    return ((x.astype(np.float32) - mean) / std).astype(np.float32)


def c_float(v):
    """C++ float literal: '0f' and '2f' are invalid, so always include a decimal point or exponent."""
    s = f"{float(v):.9g}"
    if not any(ch in s for ch in ".eni"):
        s += ".0"
    return s + "f"


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
