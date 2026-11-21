"""HMM anomaly detector: model selection, threshold, batch + real-time sliding-window scoring."""
from collections import deque

import numpy as np
from hmmlearn.hmm import GaussianHMM

WINDOW = 50


def fit_hmm(x, n_states, seed=0, n_init=5):
    """Fit GaussianHMM; best of n_init random restarts (EM is sensitive to init)."""
    best, best_ll = None, -np.inf
    for i in range(n_init):
        m = GaussianHMM(n_states, covariance_type="diag", n_iter=200, tol=1e-4, random_state=seed + i)
        m.fit(x.reshape(-1, 1))
        ll = m.score(x.reshape(-1, 1))
        if ll > best_ll:
            best, best_ll = m, ll
    return best


def bic(model, x):
    k = model.n_components
    n_params = k * k - 1 + 2 * k          # transitions + start + means + variances
    return -2 * model.score(x.reshape(-1, 1)) + n_params * np.log(len(x))


def select_n_states(x_train, x_val, candidates=range(1, 7)):
    """Return (best_k, table). Best = lowest BIC (penalises complexity)."""
    rows = []
    for k in candidates:
        m = fit_hmm(x_train, k)
        rows.append({"n_states": k, "BIC": bic(m, x_train),
                     "val_loglik_per_obs": m.score(x_val.reshape(-1, 1)) / len(x_val)})
    best = min(rows, key=lambda r: r["BIC"])["n_states"]
    return best, rows


def window_scores(model, x, window=WINDOW):
    """Per-observation log-likelihood of every sliding window (stride 1). score[i] -> window ending at i."""
    scores = np.full(len(x), np.nan)
    for end in range(window, len(x) + 1):
        w = x[end - window:end].reshape(-1, 1)
        scores[end - 1] = model.score(w) / window
    return scores


def set_threshold(model, x_val, window=WINDOW, percentile=1.0):
    """Threshold = low percentile of window scores on held-out NORMAL data (~1% false-alarm rate)."""
    s = window_scores(model, x_val, window)
    return float(np.nanpercentile(s, percentile))


class StreamingDetector:
    """Real-time: feed one reading at a time; returns (score, is_anomaly) once the window is full."""

    def __init__(self, model, threshold, window=WINDOW):
        self.model, self.threshold, self.window = model, threshold, window
        self.buf = deque(maxlen=window)

    def update(self, value):
        self.buf.append(value)
        if len(self.buf) < self.window:
            return None, False
        score = self.model.score(np.array(self.buf).reshape(-1, 1)) / self.window
        return score, score < self.threshold


def windows_to_points(scores, threshold, window=WINDOW):
    """Mark every point inside a flagged window as anomalous."""
    flags = np.zeros(len(scores), dtype=bool)
    for end in np.where(scores < threshold)[0]:
        flags[end - window + 1:end + 1] = True
    return flags
