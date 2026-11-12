# Digital Twin-Based AI Cyber-Resilience and Business Impact Framework: prototype

Proof-of-concept code for the short PhD proposal. **It is not a validated model.** Every number tagged
`[ASSUMPTION]` in `config.py` and `models/microgrid.dss` is a placeholder chosen for the prototype.

## What maps to the proposal

| Proposal | Prototype | Results |
|---|---|---|
| A. Digital Twin (RQ1) | OpenDSS network + Python battery, sensors, controller; 10-min steps, 1-min inside attack windows | `results/stage1/` |
| Preliminary work: design sizing | `scripts/stage1_size_design.py` (initial 90 kWp / 250 kWh -> sized 120 kWp / 400 kWh) | `results/stage1/sizing.*` |
| Preliminary work (15 validation checks) | `scripts/stage1_validate.py`, groups A-C | `results/stage1/validation.csv` |
| B. Attack scenarios (RQ3) | FDIA, DoS, load manipulation, signal delay between sensors and controller; shedding by criticality | `results/stage2/` |
| Labelled multi-class dataset | 150 one-day scenarios with clouds and load spikes as benign disturbances | `results/stage3/` |
| C. AI-based detection (RQ2) | statistical baseline, Isolation Forest, Random Forest, twin-based residual detector | `results/stage4/` |
| D. MCRI, BIS, AHP, sensitivity, latency sweep (RQ4-RQ6) | `src/metrics.py`, `scripts/stage5_resilience.py` | `results/stage5/` |

Not in the prototype (planned for the PhD): LSTM, MATLAB/Simulink frequency dynamics, communication-network
co-simulation, calibration with field data, the web dashboard.

## Setup and run (Windows, from the project folder)

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

.venv\Scripts\python -m scripts.run_all     # every stage in order, about one minute
```
Or one stage at a time, in this order (Stage 4 reads the Stage 3 dataset, Stage 5 reads the Stage 4 latencies):
```
.venv\Scripts\python -m scripts.stage1_size_design   # size PV and battery with the twin
.venv\Scripts\python -m scripts.stage1_baseline      # 24 h normal operation, topology
.venv\Scripts\python -m scripts.stage1_validate      # twin checks (A-C) + Stage 2-5 checks (D)
.venv\Scripts\python -m scripts.stage2_attacks       # four reference attacks vs. baseline
.venv\Scripts\python -m scripts.stage3_dataset       # labelled dataset
.venv\Scripts\python -m scripts.stage4_detect        # detectors and metrics
.venv\Scripts\python -m scripts.stage5_resilience    # MCRI, BIS, latency sweep, decision model
```
Close `models/microgrid.dss` in the OpenDSS GUI first; the GUI can save an old copy over it.

## Code layout

```
config.py      every parameter, assumptions tagged [ASSUMPTION]
models/        OpenDSS network (microgrid.dss) and bus coordinates
data/          example CSV format for measured irradiance and load
src/           library: the twin, metrics and plots (imported, not run)
scripts/       pipeline: one script per stage, plus run_all.py
results/       outputs, one folder per stage
```

| File | Role |
|---|---|
| `config.py` | every parameter, assumptions tagged |
| `models/microgrid.dss`, `buscoords.csv` | 0.4 kV, 50 Hz, six-bus radial network |
| `src/grid.py` | OpenDSS wrapper: set PV and loads, solve, read voltages, currents, losses |
| `src/profiles.py` | synthetic irradiance and load profiles, optional CSV input, benign disturbances |
| `src/attacks.py` | attack definitions and where they act |
| `src/controller.py` | load shedding (houses -> school -> pump -> clinic) and PV limit |
| `src/twin.py` | the simulation loop |
| `src/metrics.py` | ENS, recovery time, critical downtime, MCRI, BIS, AHP |
| `src/plots.py` | all charts |

## How the twin works

- **Off-grid model.** The grid-forming battery inverter is the OpenDSS voltage source; the power it supplies is
  the battery power. Python tracks state of charge and the 60 kW limit, curtails PV when the battery cannot absorb
  it, and trips the inverter (blackout) when the battery is empty or overloaded. After a trip the inverter restarts
  after 30 min if the battery still has energy, or once PV alone can carry the connected load.
- **Sensors.** The controller receives noisy telemetry (SoC, PV, load per group). The attack layer sits between
  the sensors and the controller. The digital twin keeps its own SoC estimate from the local inverter power,
  using a slightly different efficiency on purpose, so its residuals are never perfectly zero.
- **Controller.** Sheds groups when SoC falls below their thresholds (least critical first), sheds for overload,
  reconnects with hysteresis, and limits PV to the load when SoC is above 95%. The battery is not dispatched; as
  the grid-forming source it balances the network automatically.
- **Attacks.** FDIA adds a bias to SoC or hides part of the load; DoS stops commands and the PV inverter falls back
  to reduced output; load manipulation switches on extra load at one bus; delay feeds the controller old telemetry.
  Each attack stops at `onset + min(duration, detection latency + recovery time)`.
- **Fair comparison.** Every attacked run is compared with a no-attack run on the same time grid
  (`simulate(window=attack)`), and impact runs last two days so night-time consequences are counted.

## Assumptions to know

- Profiles are synthetic; the irradiance curve is scaled to a typical Indonesian day of 5.0 kWh/m2. Measured data
  can be supplied as CSV (format in `data/example_*.csv`).
- **One design is used in every stage.** The initial assumption (90 kWp, 250 kWh) sheds load every night from day 2
  (demand about 468 kWh/day vs. 382 kWh of PV). `scripts/stage1_size_design.py` tests 64 PV/battery pairs over three repeated
  days and picks the cheapest one that never sheds load: **120 kWp with 400 kWh / 60 kW**. Unit costs are assumptions.
- MCRI and BIS reference scales, AHP matrices, VoLL and recovery costs are placeholders until an expert survey
  and field data are available.

## Current results (seed 42)

- **Stage 1:** design 120 kWp / 400 kWh; voltage 0.968-1.000 pu, SoC 37-96%, no load lost; hand calculation
  0.9693 pu vs OpenDSS 0.9683 pu; all 15 twin checks pass, and all 6 D checks pass.
- **Stage 2** (detected after 240 min): DoS loses 224 kWh and cuts the clinic or pump for 686 min the next night;
  FDIA loses 75 kWh; load manipulation trips the inverter; signal delay has no measurable impact here.
- **Stage 4:** Random Forest F1 0.95, twin residual detector F1 0.89, Isolation Forest 0.77, statistical baseline
  0.23, all at about 1% false alarms or less. RF and the twin detector find 31 of 32 attacked days, with a median
  latency of 3-7 min.
- **Stage 5:** losses start once latency passes 30-60 min (DoS) or 150-180 min (FDIA). At status quo (240 min)
  the priority is DoS > FDIA > load manipulation > delay; ranked by MCRI alone it would be load manipulation > DoS >
  FDIA > delay, because MCRI ignores which loads lose supply. AI detection lowers mean BIS from 45.9 to 3.3 for
  USD 5,000; a 100 kWh larger battery lowers it to 33.0 for USD 20,000. All three rankings hold under +/-20%
  changes to every AHP weight.

## Limitations

- OpenDSS is run in quasi-static time-series mode, which has no frequency dynamics (its separate dynamics mode is
  limited, mainly to generator models, and is not used here). Stability is represented by voltage deviation, power imbalance and
  a droop-based frequency indicator; dynamic checks in Simulink are planned for the PhD.
- Detection scores are optimistic: the data are synthetic, attacks are clean, and the twin and the "real" system
  share the same model apart from noise and one efficiency value.
- Detection latency is measured at the 10-minute logging resolution.
- Loads are balanced three-phase; no phase imbalance. The PV inverter has no Volt-VAR control.
- Communication networks are not simulated: DoS and delay are modelled as their effect on data and commands.
