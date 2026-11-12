"""Stage 1 design screening: use the twin to size PV and battery.
    .venv\\Scripts\\python -m scripts.stage1_size_design

Every PV/battery pair in the config ranges runs the same typical day three times in a row,
with the controller active. A pair is feasible if no load is ever shed. The cheapest feasible
pair is the design; config.PV_KWP and config.BAT_KWH must match it (checked at the end).
Writes results/stage1/sizing.csv and results/stage1/sizing.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import config as C
from src import grid, twin
from src.plots import SERIES, limit, shade

DAYS = 3


def run(pv, bat):
    return twin.simulate(pv_kwp=pv, bat_kwh=bat, days=DAYS)


def main():
    grid.load_model()
    rows = []
    for pv in C.SIZING_PV_KWP:
        for bat in C.SIZING_BAT_KWH:
            df = run(pv, bat)
            ens = float((df.unserved_kw * df.dt_min / 60).sum())
            rows.append({"pv_kwp": pv, "bat_kwh": bat, "ens_kwh_3days": round(ens, 1),
                         "min_soc": round(df.soc.min(), 3), "feasible": ens == 0,
                         "cost_usd": pv * C.PV_COST_USD_PER_KWP + bat * C.BAT_COST_USD_PER_KWH})
    res = pd.DataFrame(rows)
    out = C.out_dir(1)
    res.to_csv(out / "sizing.csv", index=False)
    best = res[res.feasible].sort_values(["cost_usd", "bat_kwh"]).iloc[0]

    # SoC over three days: initial assumption vs. chosen design
    init, chosen = run(C.INITIAL_PV_KWP, C.INITIAL_BAT_KWH), run(best.pv_kwp, best.bat_kwh)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    shade(ax, init, init.unserved_kw > 0, "Initial design: load shed")
    ax.plot(init.hour, init.soc * 100, color=SERIES[1],
            label=f"Initial assumption: {C.INITIAL_PV_KWP:.0f} kWp, {C.INITIAL_BAT_KWH:.0f} kWh")
    ax.plot(chosen.hour, chosen.soc * 100, color=SERIES[0],
            label=f"Sized design: {best.pv_kwp:.0f} kWp, {best.bat_kwh:.0f} kWh")
    limit(ax, C.SOC_MIN * 100, f"minimum {C.SOC_MIN:.0%}")
    ax.set_ylim(0, 105)
    ax.set_ylabel("State of charge (%)")
    ax.set_xlabel("Hour")
    ax.set_xticks(range(0, 73, 12))
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
    fig.suptitle("Design screening: the same typical day repeated three times", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "sizing.png", dpi=130)
    plt.close(fig)

    e = lambda d, c: (d[c] * d.dt_min / 60).sum() / DAYS
    print(res.pivot(index="pv_kwp", columns="bat_kwh", values="ens_kwh_3days").to_string())
    print(f"\nInitial assumption {C.INITIAL_PV_KWP:.0f} kWp / {C.INITIAL_BAT_KWH:.0f} kWh: "
          f"{(init.unserved_kw * init.dt_min / 60).sum():.0f} kWh shed over {DAYS} days; "
          f"daily demand {e(init, 'demand_kw'):.0f} kWh, PV used {e(init, 'pv_kw'):.0f} kWh per day")
    print(f"Cheapest feasible design: {best.pv_kwp:.0f} kWp / {best.bat_kwh:.0f} kWh "
          f"(USD {best.cost_usd:,.0f}), min SoC {best.min_soc:.1%}")
    match = (best.pv_kwp, best.bat_kwh) == (C.PV_KWP, C.BAT_KWH)
    print("config.py design matches" if match else
          f"UPDATE config.py: PV_KWP = {best.pv_kwp:.1f}, BAT_KWH = {best.bat_kwh:.1f}")


if __name__ == "__main__":
    main()
