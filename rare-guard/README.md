# RareGuard 🔐
**RareGuard** is a portfolio-ready, end-to-end cybersecurity project that detects **rare, high-impact security incidents** in log streams using:

- **Self-supervised sequence modeling** (Transformer next-event prediction on normal logs)
- **Conformal calibration** (statistically grounded thresholds with a target false-alarm rate)
- **Streaming detection** with per-host context windows
- **Actionable explanations** (top predicted events vs observed event, context window, p-value)

This is designed to be practical for SOC / detection engineering workflows: it works even when true attacks are **rare** and labels are limited.

---

## What you get
- ✅ Train a Transformer on normal logs (no attack labels required)
- ✅ Calibrate an alert threshold with split conformal prediction
- ✅ Score logs in batch or streaming mode
- ✅ Run a FastAPI service to score logs in real-time
- ✅ Generate synthetic demo logs to show the system end-to-end

---

## Repository layout
```
rare-guard/
  src/rare_guard/         # library code
  scripts/                # runnable scripts
  data/                   # sample & synthetic data
  artifacts/              # saved model + vocab + calibration
```

---

## Quickstart (demo you can run locally)

### 1) Create a venv and install deps
```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### 2) Generate synthetic logs (normal + injected "attack-like" rare patterns)
```bash
python scripts/generate_synthetic_logs.py --out data/synth.jsonl --days 2 --hosts 5 --attack_rate 0.002
```

### 3) Train on normal logs
```bash
python -m rare_guard.cli train \
  --log data/synth.jsonl \
  --out artifacts \
  --seq-len 24 \
  --epochs 3
```

### 4) Calibrate alerts (choose desired false-alarm rate alpha)
```bash
python -m rare_guard.cli calibrate \
  --artifacts artifacts \
  --alpha 0.01
```

### 5) Detect anomalies and print alerts
```bash
python -m rare_guard.cli detect \
  --log data/synth.jsonl \
  --artifacts artifacts \
  --show 20
```

### 6) Run the API (real-time scoring)
```bash
uvicorn rare_guard.api:app --reload --port 8000
```

Then POST a log event:
```bash
curl -X POST "http://127.0.0.1:8000/score" \
  -H "Content-Type: application/json" \
  -d '{"host":"host-1","timestamp":"2026-01-01T00:00:00Z","message":"Failed password for invalid user admin from 10.1.2.3 port 22 ssh2"}'
```

---

## How the algorithm works (high level)

### A) Templateize log messages
Raw log messages contain variable fields (IPs, ports, hashes). RareGuard converts messages to **templates**:
- `Failed password for invalid user <*> from <IP> port <*> ssh2`

Templates become tokens.

### B) Learn normal behavior via next-event prediction
For each host, we build sequences of templates and train a Transformer to predict the next template:
- Input: last *N* templates
- Target: next template

The model learns “what usually happens next” for normal operations.

### C) Score surprise and calibrate with conformal prediction
At detection time, each new event gets a **nonconformity score**:
- `score = -log P(observed_event | context)`

Using a held-out set of normal data, we compute a threshold **T** such that:
- With probability ≈ `1 - alpha`, normal events have score ≤ T

That lets you tune false-alarm rate in a principled way.

### D) Output p-values + explanations
RareGuard returns:
- anomaly score
- conformal p-value
- is_anomaly boolean
- top predicted events vs the observed event
- context window of templates

---

## Why this is portfolio-worthy
- Uses modern ML (Transformers) but stays practical for logs
- Handles **rare events** without needing large labeled attack corpora
- Includes calibration (a common missing piece in many “anomaly detector” repos)
- Includes an API service and CLI (real-world engineering)

---

## Notes & next upgrades (if you want to extend)
- Add Drain3 / structured parsing
- Add per-tenant / per-service models
- Add online learning with safe rollback
- Add richer explainability (attention visualization, SHAP on embeddings)
- Integrate with Kafka / SIEM pipelines

---

## License
MIT. See `LICENSE`.
