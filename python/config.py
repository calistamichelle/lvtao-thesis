"""Frozen experiment constants.

Everything that defines the experiment lives here. Decide these values before
running anything on the outer-test participants and do not change them after.
A few values can be overridden with environment variables so the smoke test can
run a tiny version of the pipeline; real runs should not set them.
"""
import os
from pathlib import Path


def _env_int(name, default):
    return int(os.environ.get(name, default))


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("LVTAO_DATA_DIR", ROOT / "data" / "harth"))
WORK_DIR = Path(os.environ.get("LVTAO_WORK_DIR", ROOT))
PROCESSED_DIR = WORK_DIR / "data" / "processed"
MODELS_DIR = WORK_DIR / "models"
RESULTS_DIR = WORK_DIR / "results"
FIRMWARE_GEN_DIR = Path(os.environ.get("LVTAO_FIRMWARE_GEN_DIR", ROOT / "firmware" / "src" / "generated"))

# --- Signal -----------------------------------------------------------------
FS = 50                      # Hz, HARTH native rate
WINDOW = 128                 # samples = 2.56 s
AXES = ("thigh_x", "thigh_y", "thigh_z")
GAP_TOLERANCE_S = 1.5 / FS   # consecutive timestamps further apart start a new segment

# HARTH label -> thesis class. Unlisted labels (3 = shuffling) stay in the stream
# so the device still sees them, but are marked invalid and never scored.
# Sitting (7) and lying (8) are merged: with one thigh sensor the thigh is horizontal in
# both postures, and the 7-class validation confusion matrix showed them heavily mixed.
CLASSES = ("walking", "running", "stairs", "standing", "sedentary", "cycling")
HARTH_LABEL_MAP = {
    1: "walking", 2: "running", 4: "stairs", 5: "stairs", 6: "standing",
    7: "sedentary", 8: "sedentary", 13: "cycling", 14: "cycling", 130: "cycling", 140: "cycling",
}
INVALID = -1

# --- Participant-disjoint folds --------------------------------------------
SEED = 42
N_OUTER_FOLDS = _env_int("LVTAO_N_OUTER_FOLDS", 5)
N_VAL_PARTICIPANTS = _env_int("LVTAO_N_VAL_PARTICIPANTS", 3)   # drawn from each outer-training pool

# --- Training (identical for every scheduling policy) ----------------------
# TRAIN_STRIDE and CLASS_WEIGHTING were chosen by select_training.py on validation participants only.
TRAIN_STRIDE = _env_int("LVTAO_TRAIN_STRIDE", 32)
MIN_VALID_FRACTION = 0.5     # a window needs at least this share of scorable samples to get a label
BATCH_SIZE = 64
MAX_EPOCHS = _env_int("LVTAO_MAX_EPOCHS", 60)
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 8
LR_PLATEAU_PATIENCE = 3      # halve the learning rate after this many epochs without val-loss improvement
CLASS_WEIGHTING = os.environ.get("LVTAO_CLASS_WEIGHTING", "sqrt")   # "balanced", "sqrt" or "none"
EVAL_STRIDE = 64             # window-level validation/test scoring, independent of TRAIN_STRIDE
N_REPRESENTATIVE = 500       # training windows used to calibrate INT8 quantization

# --- Scheduling policies ----------------------------------------------------
FIXED_STRIDES = (128, 96, 64, 32)
LVTAO_STRIDES = (128, 64, 32)   # level 0 (persistent, low variation) -> 128 ... level 2 -> 32
LVTAO_INITIAL_LEVEL = 1
# Main controller rule ("max" = signal intensity, "change" = change since the previous inference).
# Chosen on validation participants: see results/lvtao_variant_selection.csv.
LVTAO_RULE = os.environ.get("LVTAO_RULE", "change")
FEATURE_EPS = 1e-6
COMPOSITE_ALPHA = 0.5           # ablation rule only: score = alpha*var + beta*td
COMPOSITE_BETA = 0.5

# --- Decision criteria (fix BEFORE looking at test results) ----------------
NONINFERIORITY_DELTA = 0.02     # macro F1 may drop at most 2 percentage points vs the best fixed stride
MAX_MEDIAN_DELAY_S = WINDOW / FS
TRANSITION_BAND = WINDOW        # samples either side of a true activity boundary
MIN_RUN_FOR_DELAY = WINDOW      # activity runs shorter than one window are not used for delay

# --- LVTAO calibration grid (validation participants only) -----------------
THRESHOLD_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
HYSTERESIS_GRID = (0.0, 0.25, 0.5)
DWELL_GRID = (0, 2, 4)

# --- Embedded --------------------------------------------------------------
FIRMWARE_FOLD = 0               # whose model/thresholds are flashed; latency does not depend on the fold
N_BENCH_WINDOWS = 64
SERIAL_BAUD = 115200            # ESP-IDF console default; CH340 boards garble 921600
REPLAY_BLOCK = 32               # samples per serial block before the device acknowledges

# --- Energy ESTIMATE (no power meter) --------------------------------------
# ESP32 datasheet, "Power consumption by power modes". Check these against the
# exact datasheet revision you cite and say in the paper that energy is estimated.
SUPPLY_V = 3.3
I_ACTIVE_MA = 68.0       # Modem-sleep, CPU at 240 MHz, running (upper end of the 30-68 mA range)
I_IDLE_AWAKE_MA = 30.0   # Modem-sleep, CPU at 240 MHz, mostly idle (lower end of the range)
I_LIGHT_SLEEP_MA = 0.8   # Light-sleep

# --- Statistics ------------------------------------------------------------
BOOTSTRAP_ITERS = 5000
