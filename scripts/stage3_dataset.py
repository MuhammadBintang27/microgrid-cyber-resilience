"""Stage 3: labelled dataset of many one-day scenarios.
    .venv\\Scripts\\python -m scripts.stage3_dataset
Each scenario has random benign disturbances (passing clouds, load spikes) and, except for
'normal' scenarios, one attack with random onset, duration, intensity, channel and detection latency.
Rows are sampled every LOG_STEP_MIN minutes, like SCADA logging, and labelled
normal / fdia / dos / load / delay.
Writes results/stage3/dataset.csv (one row per logged step) and results/stage3/scenarios.csv (one row per scenario).
"""
import numpy as np
import pandas as pd
import config as C
from src import grid, profiles, twin
from src.attacks import Attack


def random_events(rng):
    ev = []
    if rng.random() < C.CLOUD_PROB:
        ev.append({"kind": "cloud", "start": rng.uniform(7, 16) * 60,
                   "dur": rng.uniform(*C.CLOUD_DUR_MIN), "depth": rng.uniform(*C.CLOUD_DEPTH)})
    if rng.random() < C.SPIKE_PROB:
        ev.append({"kind": "spike", "start": rng.uniform(6, 22) * 60, "dur": rng.uniform(*C.SPIKE_DUR_MIN),
                   "mag": rng.uniform(*C.SPIKE_MAG), "group": str(rng.choice(C.GROUPS))})
    return ev


def random_attack(kind, rng):
    channel = {"fdia": str(rng.choice(C.FDIA_CHANNELS)), "load": str(rng.choice(C.GROUPS))}.get(kind, "all")
    return Attack(kind, onset_min=round(rng.uniform(*C.ONSET_RANGE_H) * 60),
                  duration_min=round(rng.uniform(*C.DURATION_RANGE_MIN)),
                  intensity=round(float(rng.uniform(*C.ATTACK_RANGES[kind])), 3), channel=channel,
                  latency_min=round(rng.uniform(*C.LATENCY_RANGE_MIN)))


def features(df):
    """Telemetry the control centre can see, plus residuals against the digital twin."""
    expected = profiles.loads(df.t_min)                      # twin's load forecast (no events, no attacks)
    status = df[[f"status_{g}" for g in C.GROUPS]].astype(float).values
    f = pd.DataFrame({
        "hour": df.hour % 24,
        "soc_meas": df.soc_meas, "pv_meas_kw": df.pv_meas_kw, "load_meas_kw": df.meas_kw,
        "bat_kw": df.bat_kw, "pv_kw": df.pv_kw, "energized": df.energized.astype(int),
        "freq_hz": df.freq_hz.fillna(C.F_NOM),
        "v_min_pu": df[[c for c in df if c.startswith("v_")]].min(axis=1).fillna(0),
        **{f"meas_{g}": df[f"meas_{g}"] for g in C.GROUPS},
        **{f"status_{g}": df[f"status_{g}"].astype(int) for g in C.GROUPS},
        # residuals: what the twin expects vs. what the telemetry says
        "r_soc": df.soc_meas - df.soc_twin,
        "r_balance_kw": df.bat_kw + df.pv_kw - df.meas_kw,
        "r_load_kw": df.meas_kw - (expected.values * status).sum(axis=1) * df.energized,
        "r_command": sum((df[f"cmd_{g}"] != df[f"status_{g}"]).astype(int) for g in C.GROUPS),
        "r_pv_kw": np.maximum(0, np.minimum(df.pv_meas_kw, df.pv_limit_cmd) - df.pv_kw) * df.energized,
    })
    return f


def main():
    rng = np.random.default_rng(C.SEED)
    grid.load_model()
    rows, scen = [], []
    classes = ["normal"] + C.ATTACK_TYPES
    sid = 0
    for kind in classes:
        for _ in range(C.SCENARIOS_PER_CLASS):
            att = None if kind == "normal" else random_attack(kind, rng)
            ev = random_events(rng)
            df = twin.simulate(att, events=ev, seed=sid)
            df = df[df.t_min % C.LOG_STEP_MIN == 0].reset_index(drop=True)
            f = features(df)
            f.insert(0, "scenario", sid)
            f["t_min"] = df.t_min
            f["event"] = df.event
            f["label"] = df.label
            rows.append(f)
            scen.append({"scenario": sid, "kind": kind, "events": ";".join(e["kind"] for e in ev),
                         **({k: getattr(att, k) for k in ("onset_min", "duration_min", "intensity", "channel",
                                                          "latency_min")} if att else {}),
                         "end_min": att.end_min if att else np.nan})
            sid += 1
        print(f"{kind:<7} {C.SCENARIOS_PER_CLASS} scenarios done")
    data = pd.concat(rows, ignore_index=True)
    data.to_csv(C.out_dir(3) / "dataset.csv", index=False)
    pd.DataFrame(scen).to_csv(C.out_dir(3) / "scenarios.csv", index=False)
    print(f"\n{len(data)} rows, {data.scenario.nunique()} scenarios")
    print("Rows per label:\n" + data.label.value_counts().to_string())
    print("Rows with a benign disturbance:", (data.event != "").sum())


if __name__ == "__main__":
    main()
