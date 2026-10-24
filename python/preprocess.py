"""Step 1: HARTH CSVs -> per-participant arrays + participant-disjoint folds.

Outputs (in data/processed/):
  <pid>.npz            x (T,3) float32 thigh acceleration in g, y (T,) int8 class or INVALID, seg (T,) segment id
  participants.csv     samples, minutes, segments, invalid share, per-class minutes
  folds.json           outer train/val/test participants for every fold
"""
import json

import numpy as np
import pandas as pd

import config as C

CLASS_INDEX = {name: i for i, name in enumerate(C.CLASSES)}


def load_participant(path):
    df = pd.read_csv(path, usecols=lambda c: c in ("timestamp", "label", *C.AXES))
    missing = {"timestamp", "label", *C.AXES} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

    t = pd.to_datetime(df["timestamp"]).to_numpy().astype("datetime64[ns]").astype(np.int64) / 1e9
    dt = np.diff(t)
    median_dt = float(np.median(dt)) if len(dt) else 1.0 / C.FS
    if abs(median_dt - 1.0 / C.FS) > 0.05 / C.FS:
        print(f"  WARNING {path.stem}: median sample interval {median_dt * 1000:.2f} ms, expected {1000 / C.FS:.0f} ms")

    # A new segment starts at every gap or backwards jump; nothing is interpolated across it.
    breaks = (dt > C.GAP_TOLERANCE_S) | (dt <= 0)
    seg = np.concatenate([[0], np.cumsum(breaks)]).astype(np.int32)

    y = df["label"].map(lambda v: CLASS_INDEX.get(C.HARTH_LABEL_MAP.get(int(v)), C.INVALID)).to_numpy(np.int8)
    x = df[list(C.AXES)].to_numpy(np.float32)
    return x, y, seg


def make_folds(pids, rng):
    order = [str(p) for p in rng.permutation(pids)]
    folds = []
    for k in range(C.N_OUTER_FOLDS):
        test = sorted(order[k::C.N_OUTER_FOLDS])
        pool = [p for p in order if p not in test]
        val = sorted(rng.choice(pool, size=C.N_VAL_PARTICIPANTS, replace=False).tolist())
        train = sorted(p for p in pool if p not in val)
        assert not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))
        folds.append({"fold": k, "train": train, "val": val, "test": test})
    every_test = sorted(p for f in folds for p in f["test"])
    assert every_test == sorted(pids), "each participant must be in exactly one outer test fold"
    return folds


def main():
    files = sorted(C.DATA_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No HARTH CSVs found in {C.DATA_DIR}. See README for the download link.")
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in files:
        x, y, seg = load_participant(path)
        np.savez_compressed(C.PROCESSED_DIR / f"{path.stem}.npz", x=x, y=y, seg=seg)
        row = {
            "participant": path.stem,
            "samples": len(y),
            "minutes": len(y) / C.FS / 60,
            "segments": int(seg[-1]) + 1 if len(seg) else 0,
            "invalid_share": float(np.mean(y == C.INVALID)),
        }
        for i, name in enumerate(C.CLASSES):
            row[f"min_{name}"] = float(np.sum(y == i)) / C.FS / 60
        rows.append(row)
        print(f"{path.stem}: {row['minutes']:.1f} min, {row['segments']} segment(s), {row['invalid_share']:.1%} unscored")

    pd.DataFrame(rows).to_csv(C.PROCESSED_DIR / "participants.csv", index=False)

    pids = [r["participant"] for r in rows]
    if len(pids) < C.N_OUTER_FOLDS + C.N_VAL_PARTICIPANTS + 1:
        raise SystemExit(f"Only {len(pids)} participants: too few for {C.N_OUTER_FOLDS} folds")
    folds = make_folds(pids, np.random.default_rng(C.SEED))
    with open(C.PROCESSED_DIR / "folds.json", "w") as f:
        json.dump(folds, f, indent=2)
    for fold in folds:
        print(f"fold {fold['fold']}: train {len(fold['train'])}, val {fold['val']}, test {fold['test']}")


if __name__ == "__main__":
    main()
