# Threat scenarios RareGuard is good at (portfolio framing)

RareGuard is aimed at **rare, high-impact, behavior-breaking** events that appear as *unexpected sequences* in logs.

Examples:
- SSH brute force / password spraying spikes
- Suspicious auth flows (unexpected failures followed by privilege escalation)
- Web probing and enumeration (sudden 404 patterns to admin endpoints)
- Lateral movement indicators (new host-to-host connection patterns)
- Unexpected service restarts + configuration changes
- Sudden execution of unusual scheduled tasks or binaries

It is NOT a signature-based IDS replacement.
It complements signatures by catching "unknown unknowns" and novel sequences.

## Why sequence modeling helps
Many attacks are not one log line — they are a **chain**:
- scan → auth failures → successful login → privilege escalation → persistence

RareGuard learns the normal chain patterns and flags rare deviations.
