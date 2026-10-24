"""Step 6: ESTIMATED energy per policy (no power meter).

Uses per-inference CPU time measured on the ESP32 (results/device/bench.json) and the
inference counts from the Python replay (results/replay/per_participant.csv):

  active time   = inferences x (preprocessing + invoke [+ controller for LVTAO]) mean µs
  E_awake       = V x (I_active x active + I_idle_awake x (T - active))   CPU stays awake between inferences
  E_light_sleep = V x (I_active x active + I_light_sleep x (T - active))  idealised: no wake-up or sensor cost

Both are estimates from datasheet currents and must be reported as such. The inference
duty cycle shows how much of total energy inference scheduling can affect at all.
Outputs: results/energy/per_participant.csv, results/energy/summary.csv
"""
import pandas as pd

import config as C
import data as D


def main():
    bench = D.load_json(C.RESULTS_DIR / "device" / "bench.json")["stats_us"]
    pre, invoke, ctrl = bench["pre_us"]["mean"], bench["invoke_us"]["mean"], bench["ctrl_us"]["mean"]
    df = pd.read_csv(C.RESULTS_DIR / "replay" / "per_participant.csv")

    adaptive = df.policy.str.startswith("lvtao")
    per_inf_us = pre + invoke + adaptive * ctrl
    df["active_s"] = df.inferences * per_inf_us / 1e6
    df["duration_s"] = df.stream_minutes * 60
    idle_s = df.duration_s - df.active_s
    v = C.SUPPLY_V
    df["energy_awake_mJ"] = v * (C.I_ACTIVE_MA * df.active_s + C.I_IDLE_AWAKE_MA * idle_s)
    df["energy_light_sleep_mJ"] = v * (C.I_ACTIVE_MA * df.active_s + C.I_LIGHT_SLEEP_MA * idle_s)
    df["inference_duty_cycle"] = df.active_s / df.duration_s

    out = C.RESULTS_DIR / "energy"
    out.mkdir(parents=True, exist_ok=True)
    keep = ["fold", "participant", "policy", "inferences", "duration_s", "active_s", "inference_duty_cycle",
            "energy_awake_mJ", "energy_light_sleep_mJ"]
    df[keep].to_csv(out / "per_participant.csv", index=False)

    g = df.groupby("policy")[["inferences", "duration_s", "active_s", "energy_awake_mJ", "energy_light_sleep_mJ"]].sum()
    g["inference_duty_cycle"] = g.active_s / g.duration_s
    g["avg_power_awake_mW"] = g.energy_awake_mJ / g.duration_s
    g["avg_power_light_sleep_mW"] = g.energy_light_sleep_mJ / g.duration_s
    g.to_csv(out / "summary.csv")
    print("ESTIMATED energy (datasheet currents, measured on-device µs):")
    print(g[["inference_duty_cycle", "avg_power_awake_mW", "avg_power_light_sleep_mW"]].round(4).to_string())
    print(f"\nper inference: pre {pre:.0f} µs, invoke {invoke:.0f} µs, controller {ctrl:.0f} µs")


if __name__ == "__main__":
    main()
