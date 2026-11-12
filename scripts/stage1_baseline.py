"""Stage 1: 24 h of normal operation with the proposal's design. Run from the project folder:
    .venv\\Scripts\\python -m scripts.stage1_baseline
"""
import config as C
from src import grid, plots, twin


def main():
    grid.load_model()
    df = twin.simulate()

    df.to_csv(C.out_dir(1) / "baseline.csv", index=False)
    plots.timeseries(df, C.out_dir(1) / "baseline.png", f"Stage 1: normal operation, 24 h (design {C.PV_KWP:.0f} kWp / {C.BAT_KWH:.0f} kWh, synthetic typical day)")
    plots.topology(df, *grid.topology(), C.out_dir(1) / "topology.png")

    vcols = [c for c in df if c.startswith("v_")]
    icols = [c for c in df if c.startswith("i_")]
    vmin, vmax = df[vcols].min().min(), df[vcols].max().max()
    ens = (df.unserved_kw * df.dt_min / 60).sum()
    ok = lambda b: "OK" if b else "VIOLATION"
    print(f"Voltage min/max     : {vmin:.4f} / {vmax:.4f} pu -> {ok(C.V_MIN <= vmin and vmax <= C.V_MAX)}")
    print(f"Frequency min/max   : {df.freq_hz.min():.2f} / {df.freq_hz.max():.2f} Hz -> "
          f"{ok(C.F_MIN <= df.freq_hz.min() and df.freq_hz.max() <= C.F_MAX)}")
    print(f"Cable loading       : max {df[icols].max().max():.0f}% -> {ok(df[icols].max().max() <= 100)}")
    print(f"Energy not served   : {ens:.2f} kWh -> {ok(ens == 0)}")
    print(f"SoC min/max/end     : {df.soc.min():.1%} / {df.soc.max():.1%} / {df.soc.iloc[-1]:.1%}")
    e = lambda col: (df[col] * df.dt_min / 60).sum()
    print(f"Energy: PV {e('pv_kw'):.0f} kWh (curtailed {e('pv_curtailed_kw'):.0f}), "
          f"load {e('served_kw'):.0f} kWh, network losses {e('losses_kw'):.1f} kWh")


if __name__ == "__main__":
    main()
