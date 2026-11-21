# Anomaly Detector with Hidden Markov Models

Gaussian HMM (hmmlearn) trained on normal machine-sensor data; a sliding window scores the live stream and flags low-likelihood windows.

## Run
```bash
pip install -r requirements.txt
python data_gen.py   # simulates data/train_normal.csv (20,000 pts) + data/test_stream.csv (5,000 pts, labelled faults)
python main.py       # model selection -> threshold -> detection -> metrics -> plots
```
Outputs: `anomaly_detection.png`, `model_selection.png`, `model_selection.csv`, `results.json`.
To use your own data, replace the two CSVs (column `sensor`; test file also `is_anomaly`).

## Method
1. **Data**: normal data split 80/20 (train / validation). Validation stays *normal-only* and is used only for the threshold.
2. **Model selection**: Gaussian HMM (diagonal cov.), 1–6 hidden states, 5 random EM restarts each, chosen by **BIC**.

| States | BIC | Val log-lik/obs |
|---|---|---|
| 1 | 144,871 | -4.492 |
| 2 | 119,549 | -3.768 |
| **3** | **85,376** | -2.696 |
| 4 | 85,451 | -2.696 |
| 5 | 85,555 | -2.696 |
| 6 | 85,673 | -2.697 |

   **Chosen: 3 states.** Likelihood jumps sharply up to 3 states then plateaus (no gain on held-out data), while BIC rises after 3 (extra states only add parameters). 3 states also has a physical meaning: idle / normal / high load (learned means ≈ 20, 50, 80).
3. **Threshold**: window = 50 readings, stride 1, score = log-likelihood / window length. Threshold = **1st percentile** of scores on held-out normal windows (≈1% expected false-alarm rate) = **-4.446**. Held-out data is used because likelihood on training data is optimistically high.
4. **Real-time**: `StreamingDetector.update(x)` takes one reading at a time (deque buffer) and returns `(score, is_anomaly)`. `main.py` asserts it gives identical results to batch scoring.
5. **Visualization**: `anomaly_detection.png` — raw signal, detected regions (red), true faults (black), and the window score vs. threshold.

## Results (injected faults: 2 spikes, gradual drift, noisy burst, stuck sensor)
Window-level (a window is truly anomalous if it overlaps a fault): **precision 0.907, recall 0.445, F1 0.597**.

| Fault | Detected |
|---|---|
| Spikes (x2) | Yes |
| Gradual drift | Yes |
| Noisy/vibrating burst | Yes |
| Stuck sensor (flat-line) | **No** |

## Limitations / future work
- **Stuck sensor missed**: a flat line at a normal level actually has *higher* likelihood under the Gaussian emission model, so a likelihood threshold can't see it. Fix: add a rolling-std feature (multivariate HMM) or a separate variance check.
- A few false alarms occur around regime switches; lowering the percentile (e.g. 0.5) trades recall for fewer false alarms.
- Data is simulated (no dataset was provided); the pipeline works unchanged on real CSVs.
