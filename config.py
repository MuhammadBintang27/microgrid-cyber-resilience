"""All prototype parameters live here.

Every value tagged [ASSUMPTION] is a placeholder chosen for the proof of concept,
not validated field data. Change values here only; the code reads everything from this file.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
DSS_FILE = ROOT / "models" / "microgrid.dss"
RESULTS = ROOT / "results"
DATA = ROOT / "data"
SEED = 42


def out_dir(stage):
    """Results folder of one stage, e.g. results/stage2 (created if missing)."""
    p = RESULTS / f"stage{stage}"
    p.mkdir(parents=True, exist_ok=True)
    return p

# ---------------------------------------------------------------- time
DAY_MIN = 1440
STEP_MIN = 10            # normal simulation step (min)
FINE_STEP_MIN = 1        # step used inside attack windows (min)
FINE_MARGIN_MIN = 60     # keep 1-min steps this long after an attack ends
LOG_STEP_MIN = 10        # SCADA logging interval used for the dataset (min)

# ---------------------------------------------------------------- PV
# Design used in every stage. Sized by src/size_design.py from the initial assumption below:
# the cheapest PV + battery pair that runs three days in a row without shedding any load.
PV_KWP = 120.0
PV_DERATE = 0.85         # [ASSUMPTION] temperature, wiring and inverter losses
# [ASSUMPTION] peak of the synthetic profile, chosen so a day gives 5.0 kWh/m2,
# a typical daily average for Indonesia (a half-sine of 12 h gives peak x 7.64 h).
GHI_PEAK = 5.0 / 7.64 * 1000

# ---------------------------------------------------------------- battery
BAT_KWH = 400.0
BAT_KW = 60.0
SOC_INIT = 0.60
SOC_MIN, SOC_MAX = 0.20, 1.00
BAT_EFF = 0.95           # [ASSUMPTION] one-way efficiency (charge or discharge)

# ---------------------------------------------------------------- design sizing (src/size_design.py)
INITIAL_PV_KWP, INITIAL_BAT_KWH = 90.0, 250.0     # first assumption, before the twin checked it
SIZING_PV_KWP = range(90, 161, 10)
SIZING_BAT_KWH = range(250, 601, 50)
PV_COST_USD_PER_KWP = 900.0      # [ASSUMPTION] installed cost
BAT_COST_USD_PER_KWH = 450.0     # [ASSUMPTION] installed cost

# ---------------------------------------------------------------- loads
# kw = peak demand; weight = criticality (higher = more critical);
# shed_soc = state of charge below which the controller disconnects the group.
# Shedding order follows criticality: houses -> school -> pump -> clinic.
LOADS = {
    "clinic": {"kw": 5.0,  "weight": 5, "shed_soc": 0.205},
    "pump":   {"kw": 4.0,  "weight": 4, "shed_soc": 0.21},
    "school": {"kw": 6.0,  "weight": 3, "shed_soc": 0.22},
    "houses": {"kw": 30.0, "weight": 2, "shed_soc": 0.23},   # [ASSUMPTION] thresholds
}
GROUPS = list(LOADS)
SHED_ORDER = sorted(GROUPS, key=lambda g: LOADS[g]["weight"])     # least critical first
HOUSE_MIN_KW = 8.0       # households vary between 8 and 30 kW
LOAD_PF = 0.9            # [ASSUMPTION]

# ---------------------------------------------------------------- controller
RECONNECT_MARGIN = 0.05  # [ASSUMPTION] reconnect a group at shed_soc + margin (hysteresis)
CURTAIL_SOC = 0.95       # [ASSUMPTION] above this SoC the controller limits PV to the load
OVERLOAD_FRACTION = 0.95 # [ASSUMPTION] shed if (load - PV) would exceed this share of BAT_KW
RESTART_DELAY_MIN = 30   # [ASSUMPTION] inverter restart delay after a trip (blackout)

# ---------------------------------------------------------------- operating limits
V_MIN, V_MAX = 0.95, 1.05
F_NOM = 50.0
DROOP_HZ_PER_KW = 0.5 / BAT_KW   # [ASSUMPTION] 1% P-f droop of the grid-forming inverter
F_MIN, F_MAX = 49.5, 50.5        # [ASSUMPTION]

# ---------------------------------------------------------------- sensors
NOISE_SD = {"soc": 0.005, "load": 0.02, "pv": 0.02}   # [ASSUMPTION] sensor noise (abs SoC, relative kW)
TWIN_EFF = 0.93          # [ASSUMPTION] efficiency used by the twin estimator (deliberate model mismatch)

# ---------------------------------------------------------------- attacks
# intensity meaning per type:
#   fdia : SoC channel -> added to reported SoC; load channel -> fraction of load hidden
#   dos  : fraction of PV capacity lost when the PV inverter falls back on communication loss
#   load : extra kW the attacker switches on at one load bus
#   delay: age of the telemetry the controller receives (min)
ATTACK_TYPES = ["fdia", "dos", "load", "delay"]
ATTACK_RANGES = {        # [ASSUMPTION] ranges used to generate the dataset
    "fdia": (0.2, 0.5),
    "dos": (0.5, 1.0),
    "load": (10.0, 30.0),
    "delay": (30.0, 180.0),
}
FDIA_CHANNELS = ["soc", "load"]
ONSET_RANGE_H = (6, 18)
DURATION_RANGE_MIN = (60, 360)
LATENCY_RANGE_MIN = (30, 360)
RECOVERY_MIN = 30        # [ASSUMPTION] time to restore normal operation after detection

# ---------------------------------------------------------------- benign disturbances
CLOUD_PROB, CLOUD_DEPTH, CLOUD_DUR_MIN = 0.5, (0.3, 0.8), (10, 90)   # [ASSUMPTION]
SPIKE_PROB, SPIKE_MAG, SPIKE_DUR_MIN = 0.5, (0.3, 1.0), (10, 60)     # [ASSUMPTION]

# ---------------------------------------------------------------- dataset
SCENARIOS_PER_CLASS = 30   # normal + 4 attack types -> 150 one-day scenarios

# ---------------------------------------------------------------- MCRI (0-1, higher = more resilient)
MCRI_INDICATORS = ["latency", "voltage", "imbalance", "ens", "recovery"]
MCRI_REF = {"latency": 360, "voltage": 0.05, "imbalance": 1.0, "ens": 100.0, "recovery": 360}  # [ASSUMPTION]
# AHP pairwise comparisons, rows/cols in MCRI_INDICATORS order. [ASSUMPTION] placeholder until an expert survey.
MCRI_AHP = np.array([
    [1,   2,   1/2, 1/3, 1/2],
    [1/2, 1,   1/2, 1/4, 1/3],
    [2,   2,   1,   1/2, 1],
    [3,   4,   2,   1,   2],
    [2,   3,   1,   1/2, 1],
])

# ---------------------------------------------------------------- BIS (0-100, higher = worse impact)
BIS_COMPONENTS = ["wens", "critical_downtime", "recovery", "financial"]
BIS_REF = {"wens": 500.0, "critical_downtime": 360, "recovery": 360, "financial": 3000.0}  # [ASSUMPTION]
BIS_AHP = np.array([     # [ASSUMPTION] placeholder until an expert survey
    [1,   1/2, 3, 1],
    [2,   1,   3, 2],
    [1/3, 1/3, 1, 1/2],
    [1,   1/2, 2, 1],
])
VOLL_USD_PER_KWH = {"clinic": 20.0, "pump": 10.0, "school": 5.0, "houses": 2.0}   # [ASSUMPTION]
RECOVERY_COST_FIXED = 200.0      # [ASSUMPTION] USD per incident response
RECOVERY_COST_PER_H = 50.0       # [ASSUMPTION] USD per hour of recovery work
BLACK_START_COST = 300.0         # [ASSUMPTION] USD per inverter trip (site visit)

# ---------------------------------------------------------------- Stages 2 and 5
IMPACT_DAYS = 2          # attack on day 1; day 2 captures consequences that appear after midnight
SWEEP_LATENCIES = list(range(0, 361, 30))
SWEEP_ATTACKS = {        # [ASSUMPTION] reference scenarios for the latency sweep
    "fdia":  dict(onset_min=9 * 60,  duration_min=360, intensity=0.4,  channel="soc"),
    "dos":   dict(onset_min=9 * 60,  duration_min=360, intensity=1.0,  channel="all"),
    "load":  dict(onset_min=18 * 60, duration_min=360, intensity=25.0, channel="pump"),
    "delay": dict(onset_min=17 * 60, duration_min=360, intensity=120.0, channel="all"),
}
STATUS_QUO_LATENCY = 240        # [ASSUMPTION] without an AI detector, the operator notices after ~4 h
INVESTMENTS = {                 # [ASSUMPTION] options ranked by the decision model
    # latency taken from Stage 4 (results/stage4/latency.json); fallback_latency_min if Stage 4 was not run
    "ai_detection": {"cost_usd": 5000.0, "fallback_latency_min": 30},
    "bigger_battery": {"cost_usd": 20000.0, "extra_kwh": 100.0},
}
SENSITIVITY = 0.2                # +/-20% one-at-a-time weight perturbation

# ---------------------------------------------------------------- validation references
PSH_RANGE = (4.0, 5.5)   # [ASSUMPTION] typical daily irradiation in Indonesia (kWh/m2/day)
VALIDATION_DAYS = 3

# ---------------------------------------------------------------- optional measured data
IRRADIANCE_CSV = None    # e.g. DATA / "irradiance.csv"  (columns: hour, ghi in W/m2)
LOAD_CSV = None          # e.g. DATA / "loads.csv"       (columns: hour, clinic, pump, school, houses in kW)
