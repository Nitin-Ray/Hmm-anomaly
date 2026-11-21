"""Run full pipeline: select model -> threshold -> detect -> evaluate -> plots."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from detector import (WINDOW, StreamingDetector, fit_hmm, select_n_states, set_threshold,
                      window_scores, windows_to_points)

train = pd.read_csv("data/train_normal.csv")["sensor"].values
test = pd.read_csv("data/test_stream.csv")
x_test, y_test = test["sensor"].values, test["is_anomaly"].values

# 80/20 split of NORMAL data: train for EM, validation for threshold (avoids optimistic likelihoods)
cut = int(0.8 * len(train))
x_tr, x_val = train[:cut], train[cut:]

# 1) model selection
best_k, table = select_n_states(x_tr, x_val)
sel = pd.DataFrame(table).round(3)
print(sel.to_string(index=False), f"\n-> chosen n_states = {best_k} (lowest BIC)\n")
sel.to_csv("model_selection.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(sel.n_states, sel.BIC, "o-"); ax.axvline(best_k, ls="--", c="r", label=f"chosen = {best_k}")
ax.set(xlabel="Hidden states", ylabel="BIC (lower is better)", title="Model selection"); ax.legend()
fig.tight_layout(); fig.savefig("model_selection.png", dpi=150); plt.close(fig)

# 2) final model + threshold
model = fit_hmm(x_tr, best_k)
print("learned means:", np.sort(model.means_.ravel()).round(1))
threshold = set_threshold(model, x_val)
print(f"threshold (1st pct of normal window log-lik/obs): {threshold:.3f}")

# 3) batch sliding-window detection
scores = window_scores(model, x_test)
flags = windows_to_points(scores, threshold)

# 4) real-time streaming check (must match batch)
det = StreamingDetector(model, threshold)
stream_alarms = [det.update(v)[1] for v in x_test]
assert np.array_equal(np.array(stream_alarms), scores < threshold), "streaming != batch"
print("streaming detector matches batch: OK")

# 5) evaluation (window-level: a window is truly anomalous if it overlaps any fault)
valid = ~np.isnan(scores)
pred_w = scores[valid] < threshold
true_w = np.array([y_test[e - WINDOW + 1:e + 1].any() for e in np.where(valid)[0]])
tp = int((pred_w & true_w).sum()); fp = int((pred_w & ~true_w).sum()); fn = int((~pred_w & true_w).sum())
prec, rec = tp / (tp + fp), tp / (tp + fn)
f1 = 2 * prec * rec / (prec + rec)
events = {"spike 1": (600, 603), "spike 2": (1500, 1503), "drift": (2300, 2420),
          "noise burst": (3200, 3320), "stuck sensor": (4100, 4200)}
print(f"\nwindow-level  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}")
for name, (a, b) in events.items():
    print(f"  {name:13s}[{a}:{b}] detected = {flags[a:b].any()}")
json.dump({"n_states": best_k, "threshold": threshold, "precision": prec, "recall": rec, "f1": f1},
          open("results.json", "w"), indent=2)

# 6) visualization
fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.5]})
a1.plot(x_test, lw=0.7, c="steelblue", label="sensor reading")
a1.fill_between(np.arange(len(x_test)), x_test.min(), x_test.max(), where=flags, color="red", alpha=0.3, label="detected anomaly")
a1.scatter(np.where(y_test == 1)[0], x_test[y_test == 1], s=4, c="black", label="true fault", zorder=3)
a1.set(ylabel="Sensor value", title=f"HMM anomaly detection ({best_k} states, window={WINDOW})"); a1.legend(loc="upper right")
a2.plot(scores, lw=0.7, c="darkgreen"); a2.axhline(threshold, c="r", ls="--", label="threshold")
a2.set(xlabel="Time step", ylabel="Log-lik / obs"); a2.legend()
fig.tight_layout(); fig.savefig("anomaly_detection.png", dpi=150)
