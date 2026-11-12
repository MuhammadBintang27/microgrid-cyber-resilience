"""Stage 4: attack detection on the Stage 3 dataset.
    .venv\\Scripts\\python -m scripts.stage4_detect

Detectors (all flag one logged row at a time):
  statistical   z-score of raw telemetry against normal operation (baseline)
  isolation     Isolation Forest trained on normal rows only (unsupervised)
  random_forest Random Forest trained on labelled rows (supervised, multi-class)
  twin_residual flags telemetry that disagrees with what the digital twin expects
Unsupervised thresholds are set so that about 1% of normal training rows raise an alarm.
LSTM is part of the PhD plan and not in this prototype.

Train/test split is by scenario, so no scenario appears in both.
Metrics: accuracy, precision, recall, F1, false-alarm rate (per row) and detection latency
(per attacked scenario: first alarm after onset, at the 10-minute logging resolution).
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
import config as C
from src import plots

RAW = ["hour", "soc_meas", "pv_meas_kw", "load_meas_kw", "bat_kw", "pv_kw", "energized", "freq_hz", "v_min_pu"] \
      + [f"meas_{g}" for g in C.GROUPS] + [f"status_{g}" for g in C.GROUPS]
RESID = ["r_soc", "r_balance_kw", "r_load_kw", "r_command", "r_pv_kw"]
FAR_TARGET = 0.99            # threshold percentile on normal training rows


def threshold_alarm(train_score, test_score):
    return test_score > np.percentile(train_score, 100 * FAR_TARGET)


def detection_latency(test, alarm, scen):
    """Minutes from onset to the first alarm while the attack is active (NaN if missed)."""
    out = []
    for sid, g in test.assign(alarm=alarm).groupby("scenario"):
        s = scen.loc[sid]
        if s.kind == "normal":
            continue
        hit = g[(g.t_min >= s.onset_min) & (g.t_min < s.end_min) & g.alarm]
        out.append({"scenario": sid, "kind": s.kind, "latency_min": hit.t_min.min() - s.onset_min if len(hit) else np.nan})
    return pd.DataFrame(out)


def main():
    data = pd.read_csv(C.out_dir(3) / "dataset.csv")
    scen = pd.read_csv(C.out_dir(3) / "scenarios.csv").set_index("scenario")
    split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=C.SEED)
    tr_i, te_i = next(split.split(data, groups=data.scenario))
    train, test = data.iloc[tr_i], data.iloc[te_i]
    normal_tr = train[train.label == "normal"]
    y_true = (test.label != "normal").values

    alarms = {}
    # 1. statistical baseline: largest z-score of raw telemetry
    mu, sd = normal_tr[RAW].mean(), normal_tr[RAW].std().replace(0, 1)
    z = lambda d: ((d[RAW] - mu) / sd).abs().max(axis=1)
    alarms["statistical"] = threshold_alarm(z(normal_tr), z(test)).values
    # 2. Isolation Forest on all features, trained on normal rows
    iso = IsolationForest(n_estimators=200, random_state=C.SEED).fit(normal_tr[RAW + RESID])
    alarms["isolation"] = threshold_alarm(-iso.score_samples(normal_tr[RAW + RESID]),
                                          -iso.score_samples(test[RAW + RESID]))
    # 3. Random Forest, supervised and multi-class
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=C.SEED, n_jobs=-1)
    rf.fit(train[RAW + RESID], train.label)
    rf_pred = rf.predict(test[RAW + RESID])
    alarms["random_forest"] = rf_pred != "normal"
    # 4. twin residuals: largest residual scaled by its spread in normal operation
    sc = normal_tr[RESID].abs().std().replace(0, 1)
    r = lambda d: (d[RESID].abs() / sc).max(axis=1)
    alarms["twin_residual"] = threshold_alarm(r(normal_tr), r(test)).values

    rows, lat_rows = [], []
    for name, a in alarms.items():
        p, rcl, f1, _ = precision_recall_fscore_support(y_true, a, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, a, labels=[False, True]).ravel()
        lat = detection_latency(test, a, scen)
        lat["detector"] = name
        lat_rows.append(lat)
        rows.append({"detector": name, "accuracy": accuracy_score(y_true, a), "precision": p, "recall": rcl,
                     "f1": f1, "false_alarm_rate": fp / (fp + tn),
                     "scenarios_detected": f"{lat.latency_min.notna().sum()}/{len(lat)}",
                     "median_latency_min": lat.latency_min.median()})
    res = pd.DataFrame(rows)
    lat_all = pd.concat(lat_rows)
    by_kind = lat_all.groupby(["detector", "kind"]).latency_min.agg(
        detected=lambda s: f"{s.notna().sum()}/{len(s)}", median_min="median").reset_index()

    res.to_csv(C.out_dir(4) / "detection.csv", index=False)
    by_kind.to_csv(C.out_dir(4) / "latency_by_attack.csv", index=False)
    labels = ["normal"] + C.ATTACK_TYPES
    cm = pd.DataFrame(confusion_matrix(test.label, rf_pred, labels=labels), index=labels, columns=labels)
    cm.to_csv(C.out_dir(4) / "rf_confusion.csv")
    with open(C.out_dir(4) / "rf_report.txt", "w") as fh:
        fh.write(classification_report(test.label, rf_pred, labels=labels, zero_division=0))
    # best detector's median latency per attack type feeds the Stage 5 decision model (missed -> 360 min)
    best = res.sort_values("f1", ascending=False).detector.iloc[0]
    lat_best = {k: float(np.nan_to_num(by_kind[(by_kind.detector == best) & (by_kind.kind == k)].median_min.iloc[0],
                                       nan=360)) for k in C.ATTACK_TYPES}
    json.dump({"detector": best, "latency_min": lat_best}, open(C.out_dir(4) / "latency.json", "w"), indent=1)
    plots.detection(res, by_kind, C.out_dir(4) / "detection.png")

    pd.set_option("display.width", 140)
    print(res.round(3).to_string(index=False))
    print("\nDetection by attack type (detected scenarios, median latency in min):")
    print(by_kind.pivot(index="detector", columns="kind", values="detected").to_string())
    print(by_kind.pivot(index="detector", columns="kind", values="median_min").round(0).to_string())
    print("\nRandom Forest confusion matrix (rows = true, columns = predicted):\n" + cm.to_string())
    print(f"\nBest F1: {best}; its latencies are used in Stage 5: {lat_best}")


if __name__ == "__main__":
    main()
