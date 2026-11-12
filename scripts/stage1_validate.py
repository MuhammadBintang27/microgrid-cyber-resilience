"""Validation of the prototype. Run from the project folder:
    .venv\\Scripts\\python -m scripts.stage1_validate

A-C validate the Stage 1 twin (physics only, no controller):
  A. internal consistency   B. hand calculation vs OpenDSS   C. plausibility and limits
D checks the controller, attack layer and scoring used in Stages 2-5.
Writes results/stage1/validation.csv and results/stage1/validation_3days.png.
"""
import numpy as np
import pandas as pd
import config as C
from src import grid, metrics, plots, profiles, twin
from src.attacks import Attack

rows = []


def check(group, name, ok, detail, warn_only=False):
    status = "PASS" if ok else ("WARNING" if warn_only else "FAIL")
    rows.append({"group": group, "check": name, "status": status, "detail": detail})


def main():
    grid.load_model()
    df = twin.simulate(control=False)
    e = lambda d, col: float((d[col] * d.dt_min / 60).sum())

    # ---------- A. internal consistency ----------
    res = df.pv_kw + df.bat_kw - df.served_kw - df.losses_kw
    check("A", "Power balance at every step: PV + battery = load + losses", res.abs().max() < 0.05,
          f"max error {res.abs().max():.4f} kW")
    de = np.where(df.bat_kw >= 0, df.bat_kw / C.BAT_EFF, df.bat_kw * C.BAT_EFF) * df.dt_min / 60
    err = np.abs(C.SOC_INIT - np.cumsum(de) / C.BAT_KWH - df.soc).max()
    check("A", "SoC equals the integral of battery power (recomputed)", err < 1e-9, f"max error {err:.2e}")
    check("A", f"SoC within {C.SOC_MIN:.0%}-{C.SOC_MAX:.0%}",
          df.soc.min() >= C.SOC_MIN - 1e-9 and df.soc.max() <= C.SOC_MAX + 1e-9, f"{df.soc.min():.1%} - {df.soc.max():.1%}")
    check("A", f"Battery power <= {C.BAT_KW:.0f} kW", df.bat_kw.abs().max() <= C.BAT_KW + 1e-6,
          f"max {df.bat_kw.abs().max():.1f} kW")

    # ---------- B. hand calculation vs OpenDSS ----------
    # Voltage drop ~ (R*P + X*Q) / V^2, checked at the household peak.
    k = df.demand_houses.idxmax()
    tanphi = np.tan(np.arccos(C.LOAD_PF))
    P_tot, P_h = df.served_kw[k] * 1e3, df.demand_houses[k] * 1e3
    R_t, X_t = grid.line_params("trunk")
    R_h, X_h = grid.line_params("l_houses")
    dv = (R_t * P_tot + X_t * P_tot * tanphi + R_h * P_h + X_h * P_h * tanphi) / 400 ** 2
    v_hand, v_dss = 1 - dv, df.v_houses_pu[k]
    check("B", "Household bus voltage: hand calculation vs OpenDSS", abs(v_hand - v_dss) < 0.003,
          f"hour {df.hour[k]:.1f}: hand {v_hand:.4f} pu, OpenDSS {v_dss:.4f} pu")
    I_t = P_tot / (np.sqrt(3) * 400 * C.LOAD_PF)
    I_h = P_h / (np.sqrt(3) * 400 * C.LOAD_PF)
    loss_hand = 3 * (I_t ** 2 * R_t + I_h ** 2 * R_h) / 1e3
    check("B", "Trunk + household cable losses: I^2R vs OpenDSS total", loss_hand <= df.losses_kw[k] * 1.05,
          f"hand {loss_hand:.2f} kW (2 cables) <= OpenDSS {df.losses_kw[k]:.2f} kW (all cables)")

    # ---------- C. plausibility and limits ----------
    vcols = [c for c in df if c.startswith("v_")]
    icols = [c for c in df if c.startswith("i_")]
    vmin, vmax = df[vcols].min().min(), df[vcols].max().max()
    check("C", f"Voltage {C.V_MIN}-{C.V_MAX} pu", C.V_MIN <= vmin and vmax <= C.V_MAX, f"{vmin:.4f} - {vmax:.4f} pu")
    check("C", f"Frequency indicator {C.F_MIN}-{C.F_MAX} Hz",
          C.F_MIN <= df.freq_hz.min() and df.freq_hz.max() <= C.F_MAX,
          f"{df.freq_hz.min():.2f} - {df.freq_hz.max():.2f} Hz")
    check("C", "Cable loading <= 100%", df[icols].max().max() <= 100, f"max {df[icols].max().max():.0f}%")
    ens = e(df, "unserved_kw")
    check("C", "No load lost over 24 h", ens == 0, f"ENS {ens:.2f} kWh")
    loss_pct = 100 * e(df, "losses_kw") / e(df, "served_kw")
    check("C", "Network losses plausible (0.5-5% of load)", 0.5 <= loss_pct <= 5, f"{loss_pct:.2f}%")
    t_day = np.arange(0, C.DAY_MIN, C.STEP_MIN)
    ghi = profiles.irradiance(t_day)
    psh = ghi.sum() * C.STEP_MIN / 60 / 1000
    check("C", f"Daily irradiation typical for Indonesia ({C.PSH_RANGE[0]}-{C.PSH_RANGE[1]} kWh/m2)",
          C.PSH_RANGE[0] <= psh <= C.PSH_RANGE[1], f"{psh:.2f} kWh/m2/day (clear-sky profile)", warn_only=True)
    n = C.VALIDATION_DAYS
    df_n = twin.simulate(control=False, days=n)
    ens_n = e(df_n, "unserved_kw")
    soc_end = [df_n.soc[df_n.t_min < (d + 1) * C.DAY_MIN].iloc[-1] for d in range(n)]
    check("C", f"Runs {n} days in a row without losing load", ens_n == 0,
          f"ENS {ens_n:.1f} kWh; end-of-day SoC: " + ", ".join(f"{s:.0%}" for s in soc_end), warn_only=True)
    pv_used, need = e(df, "pv_kw"), e(df, "served_kw") + e(df, "losses_kw")
    check("C", "PV energy used covers the daily need", pv_used >= need,
          f"PV used {pv_used:.0f} kWh vs load + losses {need:.0f} kWh (PV available {e(df, 'pv_avail_kw'):.0f} kWh)",
          warn_only=True)
    # Measured-data path: write the synthetic profile hourly to CSV, read it back, compare energy.
    C.DATA.mkdir(exist_ok=True)
    hour = np.arange(25)
    pd.DataFrame({"hour": hour, "ghi": profiles.irradiance(hour * 60) * (hour < 24)}).to_csv(
        C.DATA / "example_irradiance.csv", index=False)
    pd.DataFrame({"hour": hour, **profiles.loads(hour * 60)}).to_csv(C.DATA / "example_loads.csv", index=False)
    ref_ghi, ref_load = ghi.sum(), profiles.loads(t_day).values.sum()
    C.IRRADIANCE_CSV, C.LOAD_CSV = C.DATA / "example_irradiance.csv", C.DATA / "example_loads.csv"
    d_ghi = abs(profiles.irradiance(t_day).sum() / ref_ghi - 1)
    d_load = abs(profiles.loads(t_day).values.sum() / ref_load - 1)
    C.IRRADIANCE_CSV = C.LOAD_CSV = None
    check("C", "Measured-data CSV path (synthetic -> hourly CSV -> read back)", d_ghi < 0.03 and d_load < 0.03,
          f"energy difference: irradiance {d_ghi:.1%}, load {d_load:.1%} (hourly interpolation)")

    # ---------- D. controller, attacks and scoring (Stages 2-5) ----------
    a0 = Attack("fdia", 600, 240, 0.0, "soc")
    base = twin.simulate(window=a0)
    zero = twin.simulate(a0)
    diff = (zero.set_index("t_min").soc - base.set_index("t_min").soc).abs().max()
    check("D", "D1 Zero-intensity attack leaves the system unchanged", diff < 1e-12, f"max SoC difference {diff:.1e}")
    att = Attack("load", 18 * 60, 360, 25.0, "pump", latency_min=0, recovery_min=0)
    none = twin.simulate(att)
    check("D", "D2 Attack detected and recovered at once has no effect", (none.label == "normal").all(),
          f"steps labelled as attacked: {(none.label != 'normal').sum()}")
    low = twin.simulate(days=1, soc0=0.215)
    first_off = {g: low.t_min[~low[f"status_{g}"]].min() for g in C.GROUPS}
    order = [g for g in sorted(C.GROUPS, key=lambda g: (np.nan_to_num(first_off[g], nan=1e9), C.LOADS[g]["weight"]))]
    check("D", "D3 Controller sheds the least critical loads first", order[:2] == C.SHED_ORDER[:2],
          f"order shed from SoC 21.5%: {order}")
    b3 = twin.simulate(days=n)
    check("D", f"D4 With the controller, the design runs {n} days without shedding",
          e(b3, "unserved_kw") == 0, f"ENS {e(b3, 'unserved_kw'):.1f} kWh, min SoC {b3.soc.min():.1%}")
    for name, m in (("MCRI", C.MCRI_AHP), ("BIS", C.BIS_AHP)):
        w, cr = metrics.ahp_weights(m)
        check("D", f"D5 AHP weights for {name} are consistent (CR < 0.1)", cr < 0.1,
              f"CR {cr:.3f}; weights " + ", ".join(f"{x:.2f}" for x in w))

    # ---------- report ----------
    rep = pd.DataFrame(rows)
    rep.to_csv(C.out_dir(1) / "validation.csv", index=False)
    plots.timeseries(df_n, C.out_dir(1) / "validation_3days.png",
                     f"Validation: the same day repeated {n} times (energy sustainability)")
    for r in rows:
        print(f"[{r['status']:<7}] {r['group']}. {r['check']}\n{'':11}{r['detail']}")
    twin_rows = rep[rep.group != "D"]
    print(f"\nTwin (A-C): {(twin_rows.status == 'PASS').sum()} pass, {(twin_rows.status == 'WARNING').sum()} warning, "
          f"{(twin_rows.status == 'FAIL').sum()} fail | Stage 2-5 checks (D): "
          f"{(rep[rep.group == 'D'].status == 'PASS').sum()}/{(rep.group == 'D').sum()} pass")


if __name__ == "__main__":
    main()
