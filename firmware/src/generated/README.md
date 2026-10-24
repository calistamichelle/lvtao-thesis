Headers in this folder are written by the Python pipeline, not by hand:

- `model_data.h`, `norm.h`, `bench_data.h`: `python/quantize.py` (for `FIRMWARE_FOLD`)
- `controller_params.h`: `python/calibrate.py` (for `FIRMWARE_FOLD`)

Run those two steps before building the firmware.
