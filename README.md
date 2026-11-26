# LVTAO: adaptive-stride TinyML HAR on ESP32

Code for the thesis *Resource-Aware Adaptive-Stride Inference for Fixed-Input TinyML Human Activity Recognition on ESP32*
(Bina Nusantara University).

LVTAO keeps one fixed 128×3 input and one INT8 1D-CNN, and decides **how many new samples to wait
for before the next inference** (the stride: 128, 64 or 32) from the variance and temporal
difference of the current window. This repo trains the model, calibrates the controller, compares
it with fixed strides on participant-disjoint HARTH streams, and measures it on a real ESP32 by
streaming HARTH data to the board over USB (no wearable sensor needed).

## Layout

```
python/      training, calibration, replay, metrics, device communication, report, figures
firmware/    PlatformIO (ESP-IDF) project for the ESP32 DevKitC V4
data/harth/  put the HARTH CSVs here (not committed)
models/      per-fold models and calibrated thresholds (generated, not committed)
results/     metrics, traces, device logs, tables, figures (generated, not committed)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python python/smoke_test.py        # ~2 min: runs the whole pipeline on synthetic data
```

Download HARTH (CC BY 4.0) from the UCI repository: <https://archive.ics.uci.edu/dataset/779/harth>.
Unzip it so the per-participant files sit directly in `data/harth/` (`S006.csv`, `S008.csv`, …).

## Pipeline

Run from the repo root, in order. Each step reads the previous step's outputs.

| Step | Command | Produces |
|---|---|---|
| 1 | `python python/preprocess.py` | per-participant arrays, gap-split segments, `folds.json` (participant-disjoint) |
| 1b | `python python/select_training.py` | class weighting × training stride compared on **validation** participants; copy the choice into `config.py` |
| 2 | `python python/train.py` | Table III 1D-CNN per fold, train/val curves, parameter count, training time |
| 3 | `python python/quantize.py` | full-INT8 model, float-vs-INT8 F1, firmware headers for `FIRMWARE_FOLD` |
| 4 | `python python/calibrate.py` | LVTAO thresholds T1, T2 on the change score, chosen on **validation** participants, plus the budget-matched stride (same inference rate) |
| 5 | `python python/replay.py` | every policy on the same outer-test streams: F1, transitions, delay, cadence, traces |
| 6 | on the board, see below | latency, RAM, flash, controller overhead, deadline margin, parity |
| 7 | `python python/energy_estimate.py` | **estimated** energy per policy (needs step 6's `bench.json`) |
| 8 | `python python/report.py` | paired bootstrap CIs, non-inferiority verdict, IEEE tables, Pareto plot |
| 9 | `python python/figures.py` | the paper's result figures (Figs. 2–4) and the other numbers quoted with them, in `results/figures/` |

Classes: walking, running, stairs, standing, sedentary (HARTH sitting + lying, which a single thigh
sensor cannot separate), cycling. Shuffling is kept in the stream but not scored.

All experiment constants (window, strides, folds, δ margin, delay limit, calibration grid,
datasheet currents) are in [`python/config.py`](python/config.py). **Fix them before running step 5
on the test participants**, and do not tune them afterwards.

## On the ESP32

Needs an ESP32 DevKitC V4 (or any ESP32-WROOM board) and a USB cable. No sensor.

```bash
cd firmware
pio run -t upload          # build + flash (first build downloads the ESP-IDF toolchain)
pio run -t size            # exact flash / static RAM usage
cd ..
pio device list            # find the port, e.g. /dev/cu.usbserial-0001
python python/host_stream.py --port /dev/cu.usbserial-0001 info
python python/host_stream.py --port /dev/cu.usbserial-0001 bench
python python/host_stream.py --port /dev/cu.usbserial-0001 replay --participant S006 --policy lvtao
python python/host_stream.py --port /dev/cu.usbserial-0001 replay --participant S006 --policy fixed_64
```

- `info`: model size, tensor arena used, free heap, stack, CPU clock
- `bench`: µs for preprocessing, `Invoke()` and the controller over 512 runs, plus a check that the board's INT8 predictions match Python's
- `replay`: the board runs the policy on a participant's stream and logs every decision with timings. The script checks it made the same decisions as `replay.py` and reports the worst-case work per 20 ms sample period (the deadline margin). Use a test participant of `FIRMWARE_FOLD`; `--max-minutes 10` makes it quicker.

### Without a board: Espressif's QEMU emulator

The same firmware image runs in Espressif's ESP32 emulator. It is exact for memory use and
decisions (parity), but **its timing is not the real chip's**, so don't report latency or energy from it.

```bash
brew install libgcrypt glib pixman sdl2 libslirp
# download qemu-xtensa for your OS from https://github.com/espressif/qemu/releases and unpack it
cd firmware && pio run && cd ..
python ~/.platformio/packages/tool-esptoolpy/esptool.py --chip esp32 merge_bin --fill-flash-size 4MB \
  -o flash.bin 0x1000 firmware/.pio/build/esp32dev/bootloader.bin \
  0x8000 firmware/.pio/build/esp32dev/partitions.bin 0x10000 firmware/.pio/build/esp32dev/firmware.bin
qemu/bin/qemu-system-xtensa -nographic -machine esp32 -drive file=flash.bin,if=mtd,format=raw \
  -serial tcp:127.0.0.1:5555,server=on,wait=off -monitor none &
python python/host_stream.py --port socket://127.0.0.1:5555 --out results/device_qemu bench
```

The Python reference runs the INT8 model with TensorFlow Lite's reference kernels, which use the
same integer arithmetic as TFLM, so device and simulation agree bit for bit.

The firmware needs the headers in `firmware/src/generated/`, which steps 3 and 4 write.
The fold-0 headers are committed, so the firmware builds straight from a clone.

## What the numbers mean

- **Time-weighted macro F1** (primary): predictions are held until the next inference and scored
  on every 20 ms sample, so a policy that infers less often is penalised for stale predictions.
- **Transition F1 / detection delay**: scored within ±1 window of true activity changes; delay is
  the time until the new activity is first predicted.
- **Energy is an estimate**, from measured on-device CPU time × ESP32 datasheet currents, not a
  power measurement. `energy_estimate.py` also prints the inference duty cycle: when inference is a
  tiny share of the time the CPU is awake, scheduling barely changes total energy, so the gain is
  in compute budget rather than battery life unless the device sleeps between inferences.
- The live MPU6050 wearable test is out of scope (Limitations / Future Work).
