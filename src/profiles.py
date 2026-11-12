"""Irradiance (W/m2) and demand (kW) at any simulation minute. Synthetic, or read from CSV."""
import numpy as np
import pandas as pd
import config as C


def _hours(t_min):
    return (np.asarray(t_min, dtype=float) / 60) % 24


def _gauss(h, mu, sd):
    return np.exp(-0.5 * ((h - mu) / sd) ** 2)


def _from_csv(path, cols, h):
    """CSV with an 'hour' column (0-24, fractions allowed); linear interpolation."""
    df = pd.read_csv(path)
    return {c: np.interp(h, df["hour"], df[c]) for c in cols}


def irradiance(t_min):
    """Synthetic clear-sky profile: half sine between 06:00 and 18:00."""
    h = _hours(t_min)
    if C.IRRADIANCE_CSV:
        return _from_csv(C.IRRADIANCE_CSV, ["ghi"], h)["ghi"]
    return C.GHI_PEAK * np.clip(np.sin(np.pi * (h - 6) / 12), 0, None)


def pv_kw(ghi, kwp=C.PV_KWP):
    return kwp * ghi / 1000 * C.PV_DERATE


def loads(t_min):
    """Demand per group (kW). Curve shapes are [ASSUMPTION]."""
    h = _hours(t_min)
    if C.LOAD_CSV:
        return pd.DataFrame(_from_csv(C.LOAD_CSV, C.GROUPS, h))
    L = C.LOADS
    clinic = L["clinic"]["kw"] * (0.4 + 0.6 * np.clip(np.sin(np.pi * (h - 7) / 10), 0, None))
    pump = L["pump"]["kw"] * np.where(((h >= 6) & (h < 9)) | ((h >= 16) & (h < 19)), 1.0, 0.1)
    school = L["school"]["kw"] * np.where((h >= 7) & (h < 14), 1.0, 0.05)
    f = np.maximum(_gauss(h, 19, 2), 0.25 * _gauss(h, 6.5, 1.5))      # evening peak + morning bump
    houses = C.HOUSE_MIN_KW + (L["houses"]["kw"] - C.HOUSE_MIN_KW) * f
    return pd.DataFrame({"clinic": clinic, "pump": pump, "school": school, "houses": houses})


def apply_events(pv, demand, t_min, events):
    """Apply benign disturbances in place. events: list of dicts
    {kind: 'cloud', start, dur, depth} or {kind: 'spike', start, dur, mag, group}.
    Returns an array naming the active event at each time ('' when none)."""
    t = np.asarray(t_min)
    tag = np.full(len(t), "", dtype=object)
    for e in events:
        on = (t >= e["start"]) & (t < e["start"] + e["dur"])
        if e["kind"] == "cloud":
            pv[on] *= 1 - e["depth"]
        else:
            demand.loc[on, e["group"]] *= 1 + e["mag"]
        tag[on] = e["kind"]
    return tag
