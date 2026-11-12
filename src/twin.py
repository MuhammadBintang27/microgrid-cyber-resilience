"""Digital twin loop: OpenDSS solves the network; Python runs the battery, sensors,
controller and attack layer. Steps are 10 min, and 1 min inside attack windows."""
import numpy as np
import pandas as pd
import config as C
from src import grid, profiles
from src.attacks import tamper, commands_blocked, extra_load
from src.controller import Controller


def time_grid(days, attack):
    """Step start times (min) and step lengths (min)."""
    end = days * C.DAY_MIN
    t = set(range(0, end, C.STEP_MIN))
    if attack is not None:
        stop = min(end, attack.end_min + C.FINE_MARGIN_MIN)
        t.update(range(int(attack.onset_min), int(stop), C.FINE_STEP_MIN))
    t = np.array(sorted(t))
    return t, np.diff(np.append(t, end))


def simulate(attack=None, events=(), control=True, bat_kwh=C.BAT_KWH, pv_kwp=C.PV_KWP,
             days=1, soc0=C.SOC_INIT, seed=0, window=None):
    """Run the twin. Returns one row per step (see README for the column list).
    window: attack whose time window sets the 1-min steps (default: `attack`). Pass the attack
    with attack=None to get a no-attack baseline on the same time grid, so impacts compare fairly."""
    rng = np.random.default_rng(seed)
    t_all, dt_all = time_grid(days, window if window is not None else attack)
    pv_av = profiles.pv_kw(profiles.irradiance(t_all), pv_kwp)
    demand = profiles.loads(t_all)
    event = profiles.apply_events(pv_av, demand, t_all, list(events))

    ctrl = Controller() if control else None
    breakers, pv_limit = {g: True for g in C.GROUPS}, np.inf
    soc, soc_twin = soc0, soc0          # true state; the twin's own estimate from local inverter data
    tripped, t_trip = False, -1e9
    hist_t, hist_m, rows = [], [], []

    for k, (t, dt) in enumerate(zip(t_all, dt_all)):
        dt_h = dt / 60
        dem = demand.iloc[k].to_dict()
        for g, kw in extra_load(t, attack).items():
            dem[g] += kw

        # 1. remote telemetry (noisy), read before this step's control action
        on = 0.0 if tripped else 1.0
        meas_true = {
            "soc": soc + rng.normal(0, C.NOISE_SD["soc"]),
            "pv": pv_av[k] * (1 + rng.normal(0, C.NOISE_SD["pv"])),
            "loads": {g: dem[g] * breakers[g] * on * (1 + rng.normal(0, C.NOISE_SD["load"])) for g in C.GROUPS},
        }
        hist_t.append(t)
        hist_m.append(meas_true)
        meas = tamper(meas_true, t, attack, hist_t, hist_m)

        # 2. controller; a DoS attack stops its commands from reaching the field
        if ctrl:
            cmd, cmd_pv = ctrl.step(meas, t)
            if commands_blocked(t, attack):
                pv_limit = (1 - attack.intensity) * pv_kwp * C.PV_DERATE   # fallback on lost comms [ASSUMPTION]
            else:
                breakers, pv_limit = dict(cmd), cmd_pv
        else:
            cmd, cmd_pv = dict(breakers), pv_limit

        # 3. physics: restart after the delay if the battery still has energy (overload trip),
        #    or once PV alone can carry the connected load (empty-battery trip)
        if tripped and t - t_trip >= C.RESTART_DELAY_MIN and (
                soc > C.SOC_MIN + 0.02 or pv_av[k] >= sum(dem[g] * breakers[g] for g in C.GROUPS)):
            tripped = False
        res, pv_used, curtail, p_bat = None, 0.0, 0.0, 0.0
        if not tripped:
            served = {g: dem[g] * breakers[g] for g in C.GROUPS}
            pv_used = min(pv_av[k], pv_limit)
            res = grid.solve(pv_used, served)
            max_dis = min(C.BAT_KW, (soc - C.SOC_MIN) * bat_kwh * C.BAT_EFF / dt_h)
            max_chg = min(C.BAT_KW, (C.SOC_MAX - soc) * bat_kwh / C.BAT_EFF / dt_h)
            if res["p_bat"] < -max_chg:            # battery cannot absorb: inverter curtails PV
                curtail = min(pv_used, -res["p_bat"] - max_chg)
                pv_used -= curtail
                res = grid.solve(pv_used, served)
            if res["p_bat"] > max_dis + 1e-6:      # battery empty or inverter overloaded: trip
                tripped, t_trip, res = True, t, None
            else:
                p_bat = min(max(res["p_bat"], -max_chg), max_dis)
        if tripped:
            served, pv_used, curtail = {g: 0.0 for g in C.GROUPS}, 0.0, 0.0

        # 4. state of charge: true and the twin's estimate (deliberately different efficiency)
        soc += -p_bat * dt_h / (C.BAT_EFF * bat_kwh) if p_bat >= 0 else -p_bat * C.BAT_EFF * dt_h / bat_kwh
        soc_twin += -p_bat * dt_h / (C.TWIN_EFF * bat_kwh) if p_bat >= 0 else -p_bat * C.TWIN_EFF * dt_h / bat_kwh

        row = {
            "t_min": t, "dt_min": dt, "hour": t / 60,
            "label": attack.kind if attack is not None and attack.active(t) else "normal",
            "event": event[k], "energized": not tripped,
            "soc": soc, "soc_meas": meas["soc"], "soc_twin": soc_twin,
            "pv_avail_kw": pv_av[k], "pv_kw": pv_used, "pv_curtailed_kw": curtail,
            "pv_meas_kw": meas["pv"], "pv_limit_cmd": cmd_pv,
            "bat_kw": p_bat, "losses_kw": res["losses_kw"] if res else 0.0,
            "freq_hz": C.F_NOM - C.DROOP_HZ_PER_KW * p_bat if res else np.nan,
        }
        for g in C.GROUPS:
            row.update({f"demand_{g}": dem[g], f"served_{g}": served[g], f"unserved_{g}": dem[g] - served[g],
                        f"meas_{g}": meas["loads"][g], f"cmd_{g}": cmd[g], f"status_{g}": breakers[g]})
        if res:
            row.update({key: v for key, v in res.items() if key.startswith(("v_", "i_"))})
        rows.append(row)

    df = pd.DataFrame(rows)
    for kind in ("demand", "served", "unserved", "meas"):
        df[f"{kind}_kw"] = df[[f"{kind}_{g}" for g in C.GROUPS]].sum(axis=1)
    return df
