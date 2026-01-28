# Architecture

RareGuard is designed as a simple but realistic detection-engineering pipeline.

## Data flow

1) **Raw logs** (JSONL, or your own connector)
2) **Templateizer**
   - Replace variable fields (IPs, ports, hashes, UUIDs, numbers, paths) with stable placeholders
   - Output: a *template string* per event (token)

3) **Sequence builder**
   - Group by host (or user / container / service)
   - Build sliding windows of length `seq_len`

4) **Model (Transformer)**
   - Self-supervised next-event prediction trained only on normal data
   - Output: `P(event | context)`

5) **Nonconformity score**
   - `score = -log P(observed_event | context)`

6) **Conformal calibration**
   - Using held-out normal data, compute threshold `T` for target `alpha` false-alarm rate
   - Output: threshold + p-values

7) **Streaming detector**
   - Maintains per-host context window
   - Scores each incoming event
   - If `score > T` -> alert

8) **Interfaces**
   - CLI (train, calibrate, detect)
   - FastAPI (`POST /score`) for real-time scoring

## Mermaid diagram

```mermaid
flowchart LR
  A[JSONL Logs] --> B[Templateizer]
  B --> C[Vocabulary + Token IDs]
  C --> D[Sequence Builder]
  D --> E[Transformer Next-Event Model]
  E --> F[Score: -log P(observed|context)]
  F --> G[Conformal Calibration]
  G --> H[Threshold + p-values]
  H --> I[Streaming Detector]
  I --> J[CLI Alerts]
  I --> K[FastAPI /score]
```

## Production-hardening ideas

- Use structured fields (host, service, pid, user, program) as additional model inputs
- Train separate models per service / environment (dev/prod)
- Add drift monitoring (embedding drift, alert rate, calibration drift)
- Integrate with Kafka, Fluent Bit, or a SIEM forwarder
