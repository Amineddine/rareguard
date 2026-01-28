# RareGuard

**RareGuard** is a cybersecurity-focused, self-supervised log anomaly detection system designed to detect **rare, high-impact security events** with **statistically calibrated false-alarm control**.

It learns normal log behavior using a Transformer model and flags anomalous sequences using **conformal prediction**, producing interpretable **p-values** instead of arbitrary anomaly scores.

> Built for real-world SOC constraints: scarce labels, rare incidents, streaming logs, and alert fatigue.
## Why RareGuard?

Security incidents are rare by definition, yet most anomaly detectors:
- rely on arbitrary thresholds
- produce uncalibrated scores
- flood SOC teams with false positives
- fail silently when data drifts

RareGuard addresses this by combining:
- self-supervised sequence modeling
- uncertainty-aware detection
- statistically grounded alert thresholds
