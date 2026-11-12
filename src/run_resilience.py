"""Stage 5: MCRI and BIS, the detection-latency sweep, and the decision model.
    .venv\\Scripts\\python -m src.run_resilience

1. Weights from the AHP pairwise matrices in config.py (consistency ratio must be < 0.1).
2. Latency sweep: each reference attack with detection latency 0-360 min -> ENS, MCRI, BIS.
3. Decision model: status quo (manual detection) vs. investing in AI detection (latency from
   Stage 4) or a bigger battery, ranked by BIS reduction per USD. Attack types are treated as
   equally likely [ASSUMPTION]. Attacks are ranked by BIS, and for comparison (RQ6) by MCRI alone.
4. Sensitivity: each AHP weight moved +/-20% (others rescaled); are the rankings stable?
Writes everything to results/stage5/.
"""
import json
import numpy as np
import pandas as pd
import config as C
from src import grid, metrics, plots, twin
from src.attacks import Attack


def run(kind, latency, bat_kwh=C.BAT_KWH):
    att = Attack(kind, latency_min=latency, **C.SWEEP_ATTACKS[kind])
    base = twin.simulate(bat_kwh=bat_kwh, days=C.IMPACT_DAYS, window=att)
    df = twin.simulate(att, bat_kwh=bat_kwh, days=C.IMPACT_DAYS)
    return metrics.indicators(df, base, att)


def scored(ind, wm, wb):
    return dict(ind, mcri=metrics.mcri(ind, wm), bis=metrics.bis(ind, wb))


def perturbed(w, i, factor):
    w = w.copy()
    w[i] *= factor
    return w / w.sum()


if __name__ == "__main__":
    grid.load_model()
    out = C.out_dir(5)

    # 1. AHP weights
    wm, cr_m = metrics.ahp_weights(C.MCRI_AHP)
    wb, cr_b = metrics.ahp_weights(C.BIS_AHP)
    assert cr_m < 0.1 and cr_b < 0.1, "AHP matrices are inconsistent; revise them in config.py"
    weights = pd.DataFrame([{"index": "MCRI", "criterion": k, "weight": w} for k, w in zip(C.MCRI_INDICATORS, wm)]
                           + [{"index": "BIS", "criterion": k, "weight": w} for k, w in zip(C.BIS_COMPONENTS, wb)])
    weights.to_csv(out / "ahp_weights.csv", index=False)

    # 2. latency sweep
    raw = {(k, L): run(k, L) for k in C.ATTACK_TYPES for L in C.SWEEP_LATENCIES}
    sweep = pd.DataFrame([{"attack": k, "latency_min": L, **scored(ind, wm, wb)} for (k, L), ind in raw.items()])
    sweep.to_csv(out / "latency_sweep.csv", index=False)
    plots.latency_sweep(sweep, out / "latency_sweep.png")

    # 3. decision model
    f = C.out_dir(4) / "latency.json"
    ai_lat = json.load(open(f))["latency_min"] if f.exists() else \
        {k: C.INVESTMENTS["ai_detection"]["fallback_latency_min"] for k in C.ATTACK_TYPES}
    options = {
        "status_quo": ({k: C.STATUS_QUO_LATENCY for k in C.ATTACK_TYPES}, C.BAT_KWH, 0.0),
        "ai_detection": (ai_lat, C.BAT_KWH, C.INVESTMENTS["ai_detection"]["cost_usd"]),
        "bigger_battery": ({k: C.STATUS_QUO_LATENCY for k in C.ATTACK_TYPES},
                           C.BAT_KWH + C.INVESTMENTS["bigger_battery"]["extra_kwh"],
                           C.INVESTMENTS["bigger_battery"]["cost_usd"]),
    }
    opt_raw = {name: {k: run(k, lat[k], bat) for k in C.ATTACK_TYPES} for name, (lat, bat, _) in options.items()}

    def option_table(wm, wb):
        rows = []
        for name, (lat, bat, cost) in options.items():
            s = [scored(opt_raw[name][k], wm, wb) for k in C.ATTACK_TYPES]
            rows.append({"option": name, "cost_usd": cost, "mean_mcri": np.mean([x["mcri"] for x in s]),
                         "mean_bis": np.mean([x["bis"] for x in s]), "mean_ens_kwh": np.mean([x["ens"] for x in s])})
        t = pd.DataFrame(rows)
        sq = t.loc[t.option == "status_quo", "mean_bis"].iloc[0]
        t["bis_reduction"] = sq - t.mean_bis
        t["bis_reduction_per_1000_usd"] = np.where(t.cost_usd > 0, 1000 * t.bis_reduction / t.cost_usd.replace(0, 1), np.nan)
        return t

    def attack_table(wm, wb):
        t = pd.DataFrame([{"attack": k, **scored(opt_raw["status_quo"][k], wm, wb)} for k in C.ATTACK_TYPES])
        return t.sort_values("bis", ascending=False)

    opts = option_table(wm, wb)
    attacks = attack_table(wm, wb)
    invest = opts[opts.cost_usd > 0].sort_values("bis_reduction_per_1000_usd", ascending=False)
    opts.to_csv(out / "decision_options.csv", index=False)
    attacks.to_csv(out / "attack_priority.csv", index=False)
    plots.risk_matrix(attacks, out / "risk_matrix.png")

    # 4. sensitivity of both rankings to the AHP weights
    sens = []
    technical = list(attacks.sort_values("mcri").attack)       # technical-only ranking: lowest MCRI first
    base_attack_rank, base_invest_rank = list(attacks.attack), list(invest.option)
    for idx_name, w0, names in (("MCRI", wm, C.MCRI_INDICATORS), ("BIS", wb, C.BIS_COMPONENTS)):
        for i, crit in enumerate(names):
            for factor in (1 - C.SENSITIVITY, 1 + C.SENSITIVITY):
                w = perturbed(w0, i, factor)
                a_wm, a_wb = (w, wb) if idx_name == "MCRI" else (wm, w)
                t = attack_table(a_wm, a_wb)
                a, m = list(t.attack), list(t.sort_values("mcri").attack)
                o = option_table(a_wm, a_wb)
                o = list(o[o.cost_usd > 0].sort_values("bis_reduction_per_1000_usd", ascending=False).option)
                sens.append({"index": idx_name, "criterion": crit, "factor": factor,
                             "attack_ranking": " > ".join(a), "attack_ranking_same": a == base_attack_rank,
                             "technical_ranking": " > ".join(m), "technical_ranking_same": m == technical,
                             "investment_ranking": " > ".join(o), "investment_ranking_same": o == base_invest_rank})
    sens = pd.DataFrame(sens)
    sens.to_csv(out / "sensitivity.csv", index=False)

    pd.set_option("display.width", 160)
    print(f"AHP weights (CR MCRI {cr_m:.3f}, BIS {cr_b:.3f}):\n{weights.round(3).to_string(index=False)}\n")
    print("Latency sweep (ENS kWh / MCRI / BIS):")
    print(sweep.pivot(index="latency_min", columns="attack", values=["ens", "mcri", "bis"]).round(2).to_string())
    print(f"\nAttack priority at status quo (manual detection after {C.STATUS_QUO_LATENCY} min):")
    print(attacks[["attack", "mcri", "bis", "ens", "critical_downtime", "financial"]].round(2).to_string(index=False))
    print(f"MCRI-BIS ranking: {' > '.join(base_attack_rank)}; technical-only (MCRI) ranking: {' > '.join(technical)}")
    print(f"\nInvestment options (AI detection latency from Stage 4: {ai_lat}):")
    print(opts.round(3).to_string(index=False))
    print(f"\nSensitivity (+/-{C.SENSITIVITY:.0%} per weight, {len(sens)} cases): attack ranking unchanged in "
          f"{sens.attack_ranking_same.sum()}, technical ranking unchanged in {sens.technical_ranking_same.sum()}, "
          f"investment ranking unchanged in {sens.investment_ranking_same.sum()}")
