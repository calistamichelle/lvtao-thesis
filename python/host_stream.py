"""Talk to the ESP32 firmware over USB serial.

  python python/host_stream.py --port /dev/cu.usbserial-XXXX info
  python python/host_stream.py --port /dev/cu.usbserial-XXXX bench
  python python/host_stream.py --port /dev/cu.usbserial-XXXX replay --participant S006 --policy lvtao

`replay` streams a participant's HARTH samples to the board, which runs the same
policy as policies.py and reports timings for every inference. The script then
checks the board made the same decisions as the Python simulation.
Outputs go to results/device/.
"""
import argparse
import json
import struct
import time

import numpy as np
import pandas as pd
import serial

import config as C
import data as D
from int8_model import CachedPredictor, Int8Model
from policies import LvtaoParams, Policy, run_policy

OUT = C.RESULTS_DIR / "device"
TRACE_COLS = ["window_id", "start_sample", "end_sample", "selected_stride", "next_stride", "prediction",
              "variance", "temporal_difference", "normalized_variance", "normalized_temporal_difference",
              "score", "controller_level", "pre_us", "invoke_us", "ctrl_us"]
SUMMARY_COLS = ["samples", "inferences", "push_us_total", "pre_us_total", "invoke_us_total",
                "ctrl_us_total", "max_sample_work_us"]


class Board:
    def __init__(self, port):
        self.ser = serial.Serial(port, C.SERIAL_BAUD, timeout=0.1)
        # Opening the port usually resets a DevKitC; wait for the banner, but carry on if it never comes.
        deadline = time.time() + 6
        while time.time() < deadline:
            if "LVTAO_FW READY" in self.ser.readline().decode(errors="replace"):
                break
        self.ser.reset_input_buffer()

    def send(self, data):
        self.ser.write(data if isinstance(data, bytes) else data.encode())
        self.ser.flush()

    def lines_until(self, *stops, timeout=60):
        """Yield decoded lines until one starts with a stop marker (which is yielded too)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            raw = self.ser.readline()
            if not raw:
                continue
            line = raw.decode(errors="replace").strip()
            if line.startswith("ERR"):
                raise RuntimeError(f"board reported: {line}")
            yield line
            if line.startswith(stops):
                return
            deadline = time.time() + timeout
        raise TimeoutError(f"no {stops} from the board within {timeout} s")


def cmd_info(board):
    board.send("INFO\n")
    info = {}
    for line in board.lines_until("INFO_END"):
        parts = line.split(",")
        if parts[0] == "INFO" and len(parts) == 3:
            try:
                info[parts[1]] = float(parts[2])
            except ValueError:
                info[parts[1]] = parts[2]
    OUT.mkdir(parents=True, exist_ok=True)
    D.save_json(OUT / "info.json", info)
    print(json.dumps(info, indent=2))


def cmd_bench(board):
    board.send("BENCH\n")
    preds, stats = {}, {}
    for line in board.lines_until("BENCH_END", timeout=120):
        parts = line.split(",")
        if parts[0] == "PRED":
            preds[int(parts[1])] = int(parts[2])
        elif parts[0] == "STAT":
            stats[parts[1]] = dict(zip(["mean", "median", "p95", "max"], map(float, parts[2:6])))
    expected = D.load_json(D.fold_dir(C.FIRMWARE_FOLD) / "bench_expected.json")["expected"]
    agree = np.mean([preds.get(i) == e for i, e in enumerate(expected)])
    result = {"stats_us": stats, "parity_agreement": float(agree), "windows": len(expected)}
    OUT.mkdir(parents=True, exist_ok=True)
    D.save_json(OUT / "bench.json", result)
    print(json.dumps(result, indent=2))
    if agree < 1.0:
        print("WARNING: board and Python INT8 predictions differ; check norm.h / model_data.h are from the same fold")


def firmware_policy(name):
    cal = D.load_json(D.fold_dir(C.FIRMWARE_FOLD) / "lvtao_params.json")
    if name == "lvtao":
        return Policy.lvtao(LvtaoParams(**cal["lvtao"]), "lvtao"), "REPLAY LVTAO\n"
    stride = cal["budget_matched_stride"] if name == "budget_matched" else int(name.split("_")[1])
    return Policy.fixed(stride, name), f"REPLAY FIXED {stride}\n"


def cmd_replay(board, pid, policy_name, max_minutes):
    fold = next(f for f in D.load_folds() if f["fold"] == C.FIRMWARE_FOLD)
    if pid not in fold["test"]:
        raise SystemExit(f"{pid} is not a test participant of fold {C.FIRMWARE_FOLD} (the flashed model). "
                         f"Choose one of {fold['test']}")
    x, y, bounds = D.load(pid)
    if max_minutes:
        n = int(max_minutes * 60 * C.FS)
        x, y = x[:n], y[:n]
        bounds = [(s, min(e, n)) for s, e in bounds if s < n]
    policy, command = firmware_policy(policy_name)

    board.send(command)
    for line in board.lines_until("READY", timeout=10):
        pass
    rows, summary = [], None
    t0 = time.time()
    for s, e in bounds:
        board.send(b"R")
        for b in range(s, e, C.REPLAY_BLOCK):
            block = x[b:min(b + C.REPLAY_BLOCK, e)].astype("<f4")
            board.send(b"B" + struct.pack("<H", len(block)) + block.tobytes())
            for line in board.lines_until("K", timeout=10):
                if line.startswith("D,"):
                    rows.append(line[2:].split(","))
        print(f"\r  streamed {e / C.FS / 60:.1f} min, {len(rows)} inferences", end="", flush=True)
    board.send(b"E")
    for line in board.lines_until("DONE", timeout=10):
        if line.startswith("SUMMARY,"):
            summary = dict(zip(SUMMARY_COLS, map(float, line.split(",")[1:])))
    print(f"\n  done in {time.time() - t0:.0f} s")

    device = pd.DataFrame(rows, columns=TRACE_COLS).apply(pd.to_numeric)
    OUT.mkdir(parents=True, exist_ok=True)
    device.to_csv(OUT / f"replay_{pid}__{policy.name}.csv", index=False)

    # Same policy in Python on the same samples: decisions must match.
    sim = pd.DataFrame(run_policy(policy, x, y, bounds, CachedPredictor(Int8Model(C.FIRMWARE_FOLD), x)))
    merged = device.merge(sim, on="end_sample", how="outer", suffixes=("_dev", "_py"), indicator=True)
    both = merged[merged["_merge"] == "both"]
    parity = {
        "device_inferences": len(device), "python_inferences": len(sim),
        "matched_windows": len(both),
        "prediction_agreement": float(np.mean(both.prediction_dev == both.prediction_py)) if len(both) else 0.0,
        "stride_agreement": float(np.mean(both.next_stride_dev == both.next_stride_py)) if len(both) else 0.0,
    }
    result = {"participant": pid, "policy": policy.name, "stream_minutes": sum(e - s for s, e in bounds) / C.FS / 60,
              "summary": summary, "parity": parity,
              "deadline_us": 1e6 / C.FS,
              "deadline_margin_us": 1e6 / C.FS - summary["max_sample_work_us"] if summary else None}
    D.save_json(OUT / f"replay_{pid}__{policy.name}.json", result)
    print(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="e.g. /dev/cu.usbserial-0001 (macOS) or COM5 (Windows)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    sub.add_parser("bench")
    rp = sub.add_parser("replay")
    rp.add_argument("--participant", required=True)
    rp.add_argument("--policy", default="lvtao",
                    help="lvtao, budget_matched, or fixed_<stride> (e.g. fixed_64)")
    rp.add_argument("--max-minutes", type=float, help="only stream the first N minutes")
    args = ap.parse_args()

    board = Board(args.port)
    if args.cmd == "info":
        cmd_info(board)
    elif args.cmd == "bench":
        cmd_bench(board)
    else:
        cmd_replay(board, args.participant, args.policy, args.max_minutes)


if __name__ == "__main__":
    main()
