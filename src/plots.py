"""Charts. Fixed categorical colour order that stays readable for colour-blind viewers."""
import matplotlib
matplotlib.use("Agg")                      # write PNG files only, no window
import matplotlib.pyplot as plt
import config as C

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
ALERT = "#d03b3b"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "font.size": 9, "legend.frameon": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left", "text.color": INK,
})


def limit(ax, y, text):
    """Dashed grey reference line with a label at the right end."""
    ax.axhline(y, ls="--", lw=1, color=MUTED)
    ax.annotate(text, (1, y), xycoords=("axes fraction", "data"), xytext=(-2, 2),
                textcoords="offset points", ha="right", va="bottom", color=MUTED, fontsize=8)


def shade(ax, df, mask, label=None, color=ALERT):
    ax.fill_between(df.hour, 0, 1, where=mask, transform=ax.get_xaxis_transform(),
                    color=color, alpha=0.12, lw=0, label=label, step="post")


def _hour_ticks(ax, df):
    end = df.hour.max()
    ax.set_xticks(range(0, int(end) + 2, 3 if end <= 24 else 12))
    ax.set_xlabel("Hour")


def timeseries(df, path, title):
    """Stage 1 overview: power balance, loads, SoC, voltages, frequency."""
    x = df.hour
    fig, ax = plt.subplots(5, 1, figsize=(10, 13), sharex=True)
    a = ax[0]
    a.set_title("Power balance")
    a.plot(x, df.pv_avail_kw, color=SERIES[0], lw=1, ls=":", label="PV available")
    a.plot(x, df.pv_kw, color=SERIES[0], label="PV used")
    a.plot(x, df.served_kw, color=SERIES[1], label="Load served")
    a.plot(x, df.bat_kw, color=SERIES[2], label="Battery (+ discharge)")
    a.axhline(0, color="#c3c2b7", lw=1)
    a.set_ylabel("kW")
    a = ax[1]
    a.set_title("Demand by load group (stacked)")
    a.stackplot(x, *[df[f"demand_{g}"] for g in C.GROUPS], labels=C.GROUPS,
                colors=SERIES[:len(C.GROUPS)], edgecolor="#fcfcfb", linewidth=0.8)
    a.set_ylabel("kW")
    a.legend(ncol=4, loc="lower right", bbox_to_anchor=(1, 1.0))
    a = ax[2]
    a.set_title("Battery state of charge")
    a.plot(x, df.soc * 100, color=SERIES[0])
    limit(a, C.SOC_MIN * 100, f"minimum {C.SOC_MIN:.0%}")
    a.set_ylabel("%")
    a.set_ylim(0, 105)
    a = ax[3]
    a.set_title("Bus voltage")
    for i, c in enumerate([c for c in df if c.startswith("v_")]):
        a.plot(x, df[c], color=SERIES[i % len(SERIES)], label=c[2:-3])
    limit(a, C.V_MIN, f"{C.V_MIN} pu")
    limit(a, C.V_MAX, f"{C.V_MAX} pu")
    a.set_ylabel("pu")
    a.set_ylim(C.V_MIN - 0.01, C.V_MAX + 0.01)
    a.legend(ncol=6, loc="lower right", bbox_to_anchor=(1, 1.0))
    a = ax[4]
    a.set_title("Frequency indicator (quasi-static P-f droop)")
    a.plot(x, df.freq_hz, color=SERIES[0])
    limit(a, C.F_MIN, f"{C.F_MIN} Hz")
    limit(a, C.F_MAX, f"{C.F_MAX} Hz")
    a.set_ylabel("Hz")
    _hour_ticks(a, df)
    out = df.unserved_kw > 0
    if out.any():
        for i, a in enumerate(ax):
            shade(a, df, out, "Load not served" if i == 0 else None)
    ax[0].legend(ncol=5, loc="lower right", bbox_to_anchor=(1, 1.0))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def attack_run(df, base, attack, path, title):
    """Stage 2: one attack against the no-attack baseline."""
    fig, ax = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
    act = df.label != "normal"
    a = ax[0]
    a.set_title("State of charge: true vs. what the controller sees")
    a.plot(base.hour, base.soc * 100, color=MUTED, lw=1.2, label="Baseline (no attack)")
    a.plot(df.hour, df.soc * 100, color=SERIES[0], label="True SoC")
    a.plot(df.hour, df.soc_meas * 100, color=SERIES[1], lw=1, label="Reported SoC")
    limit(a, C.SOC_MIN * 100, f"minimum {C.SOC_MIN:.0%}")
    a.set_ylabel("%")
    a.set_ylim(0, 105)
    a = ax[1]
    a.set_title("Load not served by group")
    a.stackplot(df.hour, *[df[f"unserved_{g}"] for g in C.GROUPS], labels=C.GROUPS,
                colors=SERIES[:len(C.GROUPS)], step="post")
    a.set_ylabel("kW")
    a.legend(ncol=4, loc="lower right", bbox_to_anchor=(1, 1.0))
    a = ax[2]
    a.set_title("PV: available vs. used")
    a.plot(df.hour, df.pv_avail_kw, color=SERIES[0], lw=1, ls=":", label="Available")
    a.plot(df.hour, df.pv_kw, color=SERIES[0], label="Used")
    a.set_ylabel("kW")
    _hour_ticks(a, df)
    for i, a in enumerate(ax):
        shade(a, df, act, "Attack active" if i == 0 else None, color=SERIES[3])
        shade(a, df, ~df.energized, "Blackout (inverter trip)" if i == 0 else None)
    ax[0].legend(ncol=5, loc="lower right", bbox_to_anchor=(1, 1.0))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def topology(df, buses, lines, path):
    """Network map: bus labels = lowest daily voltage, line labels = peak loading."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axis("off")
    for ln, (b1, b2) in lines.items():
        (x1, y1), (x2, y2) = buses[b1], buses[b2]
        pk = df[f"i_{ln}_pct"].max()
        ax.plot([x1, x2], [y1, y2], color=SERIES[0], lw=1.5 + pk / 10, solid_capstyle="round", zorder=1)
        ax.annotate(f"{ln}\n{pk:.0f}%", ((x1 + x2) / 2, (y1 + y2) / 2), ha="center", va="center",
                    fontsize=7.5, color=INK2, bbox=dict(boxstyle="round,pad=0.2", fc="#fcfcfb", ec="none"))
    roles = {"main": "Battery inverter + PV", "jn": "Junction"}
    for b, (x, y) in buses.items():
        ax.scatter(x, y, s=140, color=INK if b == "main" else "#fcfcfb", edgecolor=INK, lw=1.5, zorder=2)
        text = f"{roles.get(b, b.capitalize())}\n$V_{{min}}$ = {df[f'v_{b}_pu'].min():.3f} pu"
        if b in roles:
            ax.annotate(text, (x, y), xytext=(0, 14), textcoords="offset points", ha="center", fontsize=8.5)
        else:
            ax.annotate(text, (x, y), xytext=(14, 0), textcoords="offset points", va="center", fontsize=8.5)
    xs, ys = zip(*buses.values())
    ax.set_xlim(min(xs) - 60, max(xs) + 160)
    ax.set_ylim(min(ys) - 40, max(ys) + 60)
    ax.set_aspect("equal")
    fig.suptitle("Microgrid topology, 0.4 kV", x=0.02, ha="left", fontsize=12, fontweight="bold")
    ax.set_title("Line labels: peak current as % of rating; thicker = more loaded.\n"
                 "Schematic, not to scale (cable lengths are in microgrid.dss).",
                 fontsize=8.5, color=INK2, fontweight="normal")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def detection(res, by_kind, path):
    """Stage 4: F1 and false-alarm rate per detector, and median latency per attack type."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    names = list(res.detector)
    x = range(len(names))
    a = ax[0]
    a.bar([i - 0.2 for i in x], res.f1, width=0.4, color=SERIES[0], label="F1-score")
    a.bar([i + 0.2 for i in x], res.false_alarm_rate, width=0.4, color=SERIES[1], label="False-alarm rate")
    for i, (f, fa) in enumerate(zip(res.f1, res.false_alarm_rate)):
        a.annotate(f"{f:.2f}", (i - 0.2, f), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
        a.annotate(f"{fa:.3f}", (i + 0.2, fa), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
    a.set_xticks(list(x), [n.replace("_", " ") for n in names])
    a.set_ylim(0, 1.08)
    a.set_title("Row-level performance on the test scenarios")
    a.legend(loc="upper left")
    a = ax[1]
    w = 0.8 / len(names)
    for j, n in enumerate(names):
        s = by_kind[by_kind.detector == n].set_index("kind").reindex(C.ATTACK_TYPES)
        xs = [i + (j - (len(names) - 1) / 2) * w for i in range(len(C.ATTACK_TYPES))]
        a.bar(xs, s.median_min.fillna(0), width=w, color=SERIES[j], label=n.replace("_", " "))
        for xi, v in zip(xs, s.median_min):
            if v != v:      # NaN: no scenario of this type was detected
                a.annotate("missed", (xi, 0), xytext=(0, 3), textcoords="offset points", ha="center",
                           rotation=90, fontsize=7, color=INK2)
    a.set_xticks(range(len(C.ATTACK_TYPES)), [k.upper() for k in C.ATTACK_TYPES])
    a.set_ylabel("min")
    a.set_title("Median detection latency per attack type")
    a.legend(loc="upper left", fontsize=8)
    fig.suptitle("Stage 4: detection results", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def latency_sweep(sweep, path):
    """Stage 5: ENS, MCRI and BIS against detection latency, one line per attack type."""
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    for i, kind in enumerate(C.ATTACK_TYPES):
        s = sweep[sweep.attack == kind]
        for a, col in zip(ax, ["ens", "mcri", "bis"]):
            a.plot(s.latency_min, s[col], color=SERIES[i], marker="o", ms=4, label=kind.upper())
    for a, t, y in zip(ax, ["Energy not served", "MCRI (higher = more resilient)", "BIS (higher = worse)"],
                       ["kWh", "index 0-1", "score 0-100"]):
        a.set_title(t)
        a.set_ylabel(y)
        a.set_xlabel("Detection latency (min)")
        a.set_xticks(range(0, 361, 60))
    ax[1].set_ylim(0, 1.02)
    ax[0].legend(loc="upper left")
    fig.suptitle("Latency sweep: what each minute of detection delay costs", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def risk_matrix(table, path):
    """Stage 5 decision view: MCRI (x) against BIS (y), one point per attack type."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for r in table.itertuples():     # colour follows the attack type, as in the sweep chart
        ax.scatter(r.mcri, r.bis, s=90, color=SERIES[C.ATTACK_TYPES.index(r.attack)], edgecolor="#fcfcfb",
                   lw=2, zorder=3)
        ax.annotate(r.attack.upper(), (r.mcri, r.bis), xytext=(8, 4), textcoords="offset points", fontsize=9)
    ax.axvline(0.5, ls="--", lw=1, color=MUTED)
    ax.axhline(50, ls="--", lw=1, color=MUTED)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 102)
    ax.set_xlabel("MCRI (higher = more resilient)")
    ax.set_ylabel("BIS (higher = worse business impact)")
    ax.set_title("Priority: bottom-right is low risk, top-left is act first")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
