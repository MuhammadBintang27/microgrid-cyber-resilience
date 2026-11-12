"""Impact indicators, the Microgrid Cyber-Resilience Index (MCRI), the Business Impact Score (BIS)
and AHP weighting. Indicators are measured against the same day without the attack."""
import numpy as np
import config as C


def ens_by_group(df):
    """Energy not served per group (kWh)."""
    return {g: float((df[f"unserved_{g}"] * df.dt_min / 60).sum()) for g in C.GROUPS}


def _max_vdev(df):
    v = df[[c for c in df if c.startswith("v_")]]
    return float((v - 1).abs().max().max()) if v.notna().any().any() else 0.0


def _loss_mask(df, base, tol=0.1):
    """Steps where the attacked run serves noticeably less than the baseline (kW tolerance)."""
    b = np.interp(df.t_min, base.t_min, base.unserved_kw)
    return (df.unserved_kw - b) > tol


def recovery_time(df, base, attack):
    """Minutes from the end of the attack effect until service is back to baseline."""
    lost = _loss_mask(df, base)
    if not lost.any():
        return 0.0
    last_end = float((df.t_min + df.dt_min)[lost].max())
    return max(0.0, last_end - attack.end_min)


def critical_downtime(df, base):
    """Minutes during which the clinic or the pump lost supply that the baseline still had."""
    crit = ["clinic", "pump"]
    b = {g: np.interp(df.t_min, base.t_min, base[f"unserved_{g}"]) for g in crit}
    down = np.zeros(len(df), bool)
    for g in crit:
        down |= (df[f"unserved_{g}"] - b[g]).values > 0.01
    return float(df.dt_min[down].sum())


def indicators(df, base, attack):
    """All MCRI indicators and BIS components for one attacked run."""
    ens, ens0 = ens_by_group(df), ens_by_group(base)
    d_ens = {g: max(0.0, ens[g] - ens0[g]) for g in C.GROUPS}
    peak = lambda x: float((x.unserved_kw / x.demand_kw.max()).max())
    rec = recovery_time(df, base, attack)
    trips = int((~df.energized).any() and base.energized.all())
    lost_any = bool(_loss_mask(df, base).any())
    cost = (sum(C.VOLL_USD_PER_KWH[g] * d_ens[g] for g in C.GROUPS)
            + lost_any * (C.RECOVERY_COST_FIXED + C.RECOVERY_COST_PER_H * rec / 60)
            + trips * C.BLACK_START_COST)
    return {
        # MCRI indicators
        "latency": min(attack.latency_min, attack.duration_min),
        "voltage": max(0.0, _max_vdev(df) - _max_vdev(base)),
        "imbalance": max(0.0, peak(df) - peak(base)),
        "ens": sum(d_ens.values()),
        "recovery": rec,
        # BIS components
        "wens": sum(C.LOADS[g]["weight"] * d_ens[g] for g in C.GROUPS),
        "critical_downtime": critical_downtime(df, base),
        "financial": cost,
        "trip": trips,
        **{f"ens_{g}": d_ens[g] for g in C.GROUPS},
    }


def ahp_weights(m):
    """Principal-eigenvector weights and consistency ratio of an AHP pairwise matrix."""
    vals, vecs = np.linalg.eig(m)
    i = np.argmax(vals.real)
    w = np.abs(vecs[:, i].real)
    w /= w.sum()
    n = len(m)
    ri = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32}[n]       # Saaty's random index
    cr = (vals[i].real - n) / (n - 1) / ri
    return w, cr


def _score(ind, names, ref, w):
    return float(sum(wi * min(ind[k] / ref[k], 1.0) for wi, k in zip(w, names)))


def mcri(ind, w):
    """0 to 1; 1 = no measurable degradation."""
    return 1.0 - _score(ind, C.MCRI_INDICATORS, C.MCRI_REF, w)


def bis(ind, w):
    """0 to 100; 100 = worst business impact on the reference scales."""
    return 100.0 * _score(ind, C.BIS_COMPONENTS, C.BIS_REF, w)
