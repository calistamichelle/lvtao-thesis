"""Inference scheduling policies: fixed stride and the LVTAO adaptive-stride controller.

The arithmetic here is mirrored line-for-line in firmware/src/controller.cpp so the
ESP32 makes the same decisions as this simulation (checked by host_stream.py).
"""
from dataclasses import asdict, dataclass

import numpy as np

import config as C
import data as D


def _seq_sum(a):
    """Left-to-right float32 sum per column, the same order as the C loop (numpy's sum is pairwise)."""
    return np.cumsum(a, axis=0, dtype=np.float32)[-1]


def window_features(window):
    """(variance, temporal difference) of one raw window, averaged over the 3 axes, in float32."""
    w = window.astype(np.float32)
    n = np.float32(len(w))
    mean = _seq_sum(w) / n
    d = w - mean
    var = _seq_sum(d * d) / n
    td = _seq_sum(np.abs(w[1:] - w[:-1])) / np.float32(len(w) - 1)
    three = np.float32(3)
    return np.float32((var[0] + var[1] + var[2]) / three), np.float32((td[0] + td[1] + td[2]) / three)


@dataclass
class LvtaoParams:
    mu_log_var: float
    sd_log_var: float
    mu_log_td: float
    sd_log_td: float
    t1: float = 0.0
    t2: float = 0.0
    hysteresis: float = 0.0
    dwell: int = 0
    rule: str = "max"   # "max"/"composite": signal intensity; "change": change since the previous inference

    def normalized(self, var, td):
        eps = np.float32(C.FEATURE_EPS)
        nv = (np.log(np.float32(var) + eps) - np.float32(self.mu_log_var)) / np.float32(self.sd_log_var)
        nt = (np.log(np.float32(td) + eps) - np.float32(self.mu_log_td)) / np.float32(self.sd_log_td)
        return np.float32(nv), np.float32(nt)

    def score(self, nv, nt, prev=None):
        """`prev` is (nv, nt) of the previous inference window in the segment, or None for the first one."""
        if self.rule == "change":
            # A transition shows up as a jump in either statistic between consecutive inference windows.
            if prev is None:
                return np.float32(0.0)
            return np.float32(max(abs(np.float32(nv - prev[0])), abs(np.float32(nt - prev[1]))))
        if self.rule == "composite":
            return np.float32(np.float32(C.COMPOSITE_ALPHA) * nv + np.float32(C.COMPOSITE_BETA) * nt)
        # Either signal being high is enough to ask for denser inference.
        return np.float32(max(nv, nt))

    def to_dict(self):
        return asdict(self)


def level_of(score, t1, t2):
    return 0 if score < t1 else (1 if score < t2 else 2)


class LvtaoController:
    """Moves up immediately; moves down only after `dwell` decisions and with a hysteresis margin."""

    def __init__(self, params):
        self.p = params
        self.reset()

    def reset(self):
        self.level = C.LVTAO_INITIAL_LEVEL
        self.dwell = 0
        self.prev = None
        self.score = np.float32(0.0)

    def step(self, nv, nt):
        """Score the window from its normalized features and return the next stride."""
        p = self.p
        score = self.score = p.score(nv, nt, self.prev)
        self.prev = (nv, nt)
        # float32 throughout so the comparisons match the C floats exactly.
        t1, t2, h = np.float32(p.t1), np.float32(p.t2), np.float32(p.hysteresis)
        up = level_of(score, t1, t2)
        if up > self.level:
            self.level, self.dwell = up, 0
        else:
            down = level_of(score, np.float32(t1 - h), np.float32(t2 - h))
            if down < self.level and self.dwell >= p.dwell:
                self.level, self.dwell = down, 0
            else:
                self.dwell += 1
        return C.LVTAO_STRIDES[self.level]


def fit_feature_stats(pids):
    """Log-feature mean/std from training participants' windows."""
    logs_v, logs_t = [], []
    for pid in pids:
        x, _, bounds = D.load(pid)
        for end in D.window_ends(bounds, C.EVAL_STRIDE):
            var, td = window_features(x[end - C.WINDOW:end])
            logs_v.append(np.log(var + np.float32(C.FEATURE_EPS)))
            logs_t.append(np.log(td + np.float32(C.FEATURE_EPS)))
    return LvtaoParams(float(np.mean(logs_v)), float(np.std(logs_v)) or 1.0,
                       float(np.mean(logs_t)), float(np.std(logs_t)) or 1.0)


class Policy:
    """`fixed:<stride>` or `lvtao` (with params)."""

    def __init__(self, name, stride=None, params=None):
        self.name, self.stride, self.params = name, stride, params

    @staticmethod
    def fixed(stride, name=None):
        return Policy(name or f"fixed_{stride}", stride=stride)

    @staticmethod
    def lvtao(params, name="lvtao"):
        return Policy(name, params=params)

    @property
    def adaptive(self):
        return self.params is not None


class FeatureCache:
    def __init__(self, x):
        self.x, self.cache = x, {}

    def __call__(self, end):
        if end not in self.cache:
            self.cache[end] = window_features(self.x[end - C.WINDOW:end])
        return self.cache[end]


def run_policy(policy, x, y, bounds, predict, features=None):
    """Causally replay one participant. Returns one dict per inference (the decision trace)."""
    features = features or FeatureCache(x)
    ctrl = LvtaoController(policy.params) if policy.adaptive else None
    trace = []
    for seg_start, seg_end in bounds:
        if ctrl:
            ctrl.reset()
        end, selected = seg_start + C.WINDOW, 0
        while end <= seg_end:
            row = {"window_id": len(trace), "start_sample": end - C.WINDOW, "end_sample": end,
                   "selected_stride": selected, "prediction": predict(end),
                   "majority_label": D.window_label(y[end - C.WINDOW:end])}
            if ctrl:
                var, td = features(end)
                nv, nt = policy.params.normalized(var, td)
                stride = ctrl.step(nv, nt)
                score = ctrl.score
                row.update(variance=float(var), temporal_difference=float(td),
                           normalized_variance=float(nv), normalized_temporal_difference=float(nt),
                           score=float(score), controller_level=ctrl.level, controller_dwell=ctrl.dwell)
            else:
                stride = policy.stride
            row["next_stride"] = stride
            trace.append(row)
            end, selected = end + stride, stride
    return trace
