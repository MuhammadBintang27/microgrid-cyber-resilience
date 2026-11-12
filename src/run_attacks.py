"""Stage 2: the four reference attacks on the sized design, each against the no-attack baseline.
    .venv\\Scripts\\python -m src.run_attacks
Writes results/stage2/<attack>.png and results/stage2/summary.csv.
"""
import pandas as pd
import config as C
from src import grid, metrics, plots, twin
from src.attacks import Attack

grid.load_model()
rows = []
for kind, p in C.SWEEP_ATTACKS.items():
    att = Attack(kind, latency_min=C.STATUS_QUO_LATENCY, **p)
    base = twin.simulate(days=C.IMPACT_DAYS, window=att)
    df = twin.simulate(att, days=C.IMPACT_DAYS)
    ind = metrics.indicators(df, base, att)
    rows.append({"attack": kind, "channel": att.channel, "intensity": att.intensity,
                 "onset_h": att.onset_min / 60, "effect_min": att.end_min - att.onset_min, **ind})
    plots.attack_run(df, base, att, C.out_dir(2) / f"{kind}.png",
                     f"Stage 2: {kind.upper()} attack (channel {att.channel}, intensity {att.intensity}, "
                     f"detected after {C.STATUS_QUO_LATENCY} min)")

summary = pd.DataFrame(rows)
summary.to_csv(C.out_dir(2) / "summary.csv", index=False)
cols = ["attack", "effect_min", "ens", "wens", "critical_downtime", "recovery", "trip", "financial"]
print(summary[cols].round(1).to_string(index=False))
