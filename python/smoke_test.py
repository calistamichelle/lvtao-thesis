"""End-to-end check of the pipeline on a small synthetic HARTH-shaped dataset.

Runs every step in a temporary folder (your real data/, models/ and results/ are not
touched) with a tiny training budget. Use it after setup or after changing code:

  python python/smoke_test.py

The numbers it prints are meaningless; it only checks the steps run and fit together.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# HARTH label, gravity direction (x, y, z), oscillation frequency Hz, amplitude g
ACTIVITIES = {
    1: ((-1, 0, 0), 1.9, 0.35), 2: ((-1, 0, 0), 2.8, 1.1), 4: ((-0.9, 0.3, 0), 1.6, 0.5),
    6: ((-1, 0, 0), 0, 0), 7: ((0, 0, -1), 0, 0), 8: ((0, 1, 0), 0, 0),
    13: ((-0.3, 0, -0.9), 1.2, 0.2), 3: ((-1, 0, 0), 0.7, 0.1),
}


def synthetic_participant(rng, minutes, path):
    fs, rows, t = 50, [], 0.0
    start = pd.Timestamp("2019-01-01")
    labels = list(ACTIVITIES)
    while t < minutes * 60:
        label = int(rng.choice(labels, p=[.2, .08, .08, .14, .2, .12, .1, .08]))
        dur = rng.uniform(20, 90)
        n = int(dur * fs)
        g, f, amp = ACTIVITIES[label]
        k = np.arange(n) / fs
        osc = amp * np.sin(2 * np.pi * f * k + rng.uniform(0, 6))
        xyz = np.array(g)[None] + osc[:, None] * np.array([0.6, 0.3, 0.5])[None]
        xyz += rng.normal(0, 0.02 + 0.05 * amp, size=xyz.shape)
        ts = start + pd.to_timedelta(t + k, unit="s")
        rows.append(pd.DataFrame({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S.%f"),
                                  "back_x": 0.0, "back_y": 0.0, "back_z": 0.0,
                                  "thigh_x": xyz[:, 0], "thigh_y": xyz[:, 1], "thigh_z": xyz[:, 2],
                                  "label": label}))
        t += dur + (5.0 if rng.random() < 0.05 else 0.0)   # occasional recording gap
    pd.concat(rows).to_csv(path, index=False)


def run(step, env, *args):
    print(f"\n=== {step} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, str(HERE / f"{step}.py"), *args], env=env, check=True)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = tmp / "harth"
        raw.mkdir()
        rng = np.random.default_rng(0)
        for i in range(8):
            synthetic_participant(rng, minutes=12, path=raw / f"S{i:03d}.csv")

        env = {**os.environ, "LVTAO_DATA_DIR": str(raw), "LVTAO_WORK_DIR": str(tmp),
               "LVTAO_FIRMWARE_GEN_DIR": str(tmp / "generated"), "LVTAO_N_OUTER_FOLDS": "4",
               "LVTAO_N_VAL_PARTICIPANTS": "2", "LVTAO_MAX_EPOCHS": "3", "TF_CPP_MIN_LOG_LEVEL": "2"}
        run("preprocess", env)
        run("train", env)
        run("quantize", env)
        run("calibrate", env)
        run("replay", env)

        # No board here: fake the bench file so the energy and report steps can run.
        (tmp / "results" / "device").mkdir(parents=True)
        (tmp / "results" / "device" / "bench.json").write_text(
            '{"stats_us": {"pre_us": {"mean": 150}, "invoke_us": {"mean": 2500}, "ctrl_us": {"mean": 120}}}')
        run("energy_estimate", env)
        run("report", env)

        generated = sorted(p.name for p in (tmp / "generated").iterdir())
        assert generated == ["bench_data.h", "controller_params.h", "model_data.h", "norm.h"], generated
        out = os.environ.get("LVTAO_SMOKE_KEEP_HEADERS")
        if out:  # used to compile the firmware without real data
            for p in (tmp / "generated").iterdir():
                (Path(out) / p.name).write_bytes(p.read_bytes())
        print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
