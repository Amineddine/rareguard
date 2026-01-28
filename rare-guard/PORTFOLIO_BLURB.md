## RareGuard — Conformal Transformer-based Log Anomaly Detection (Cybersecurity)
Built an end-to-end detection system for rare, high-impact cybersecurity incidents in log streams.

- Implemented self-supervised Transformer next-event prediction on log templates to learn baseline behavior without requiring attack labels.
- Added split conformal calibration to produce statistically grounded alert thresholds and p-values at a configurable false-alarm rate.
- Shipped a streaming detector with per-host context windows, a CLI for training/calibration/detection, and a FastAPI service for real-time scoring.
- Included a synthetic log generator to demonstrate detection of rare injected attack patterns in a fully reproducible pipeline.

Tech: Python, PyTorch, FastAPI, NumPy, sequence modeling, conformal prediction.
