"""Simulated machine-sensor data: 3 operating regimes (idle / normal / high-load) that switch as a Markov chain."""
import numpy as np
import pandas as pd

MEANS = np.array([20.0, 50.0, 80.0])
STDS = np.array([2.0, 3.0, 4.0])
TRANS = np.array([[0.98, 0.02, 0.00],
                  [0.01, 0.97, 0.02],
                  [0.00, 0.03, 0.97]])


def simulate_normal(n, rng):
    states = np.empty(n, dtype=int)
    states[0] = 1
    for t in range(1, n):
        states[t] = rng.choice(3, p=TRANS[states[t - 1]])
    return rng.normal(MEANS[states], STDS[states])


def simulate_test_stream(n, rng):
    """Normal stream + 3 injected fault types. Returns (signal, point-level labels)."""
    x = simulate_normal(n, rng)
    y = np.zeros(n, dtype=int)
    for s in (600, 1500):                       # sensor spikes
        x[s:s + 3] += 45; y[s:s + 3] = 1
    x[2300:2420] += np.linspace(0, 35, 120)     # gradual drift (bearing wear)
    y[2300:2420] = 1
    x[3200:3320] = rng.normal(x[3200:3320].mean(), 14, 120)  # noisy/vibrating burst
    y[3200:3320] = 1
    x[4100:4200] = 50.0                         # stuck sensor (flat-line)
    y[4100:4200] = 1
    return x, y


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    pd.DataFrame({"sensor": simulate_normal(20000, rng)}).to_csv("data/train_normal.csv", index=False)
    x, y = simulate_test_stream(5000, rng)
    pd.DataFrame({"sensor": x, "is_anomaly": y}).to_csv("data/test_stream.csv", index=False)
    print("saved data/train_normal.csv, data/test_stream.csv")
