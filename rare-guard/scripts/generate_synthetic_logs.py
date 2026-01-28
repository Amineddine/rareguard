#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Simple synthetic log generator to make this repo runnable without external datasets.
# It emits "normal" sequences plus rare injected "attack-like" events.

NORMAL_TEMPLATES = [
    "Accepted publickey for {user} from {ip} port {port} ssh2",
    "session opened for user {user} by (uid=0)",
    "session closed for user {user}",
    "CRON[{pid}]: (root) CMD ({cmd})",
    "sudo: {user} : TTY=pts/0 ; PWD=/home/{user} ; USER=root ; COMMAND={cmd}",
    "sshd[{pid}]: Received disconnect from {ip}: 11: disconnected by user",
    "systemd: Started Daily apt download activities.",
    "systemd: Stopped User Manager for UID {uid}.",
    "nginx: {ip} - - \"GET /health HTTP/1.1\" 200 {bytes}",
    "kernel: eth0: link up, 1000 Mbps, full-duplex",
]

RARE_ATTACK_TEMPLATES = [
    "Failed password for invalid user {user} from {ip} port {port} ssh2",
    "sshd[{pid}]: error: maximum authentication attempts exceeded for invalid user {user} from {ip} port {port} ssh2",
    "sudo: {user} : user NOT in sudoers ; TTY=pts/1 ; PWD=/tmp ; USER=root ; COMMAND=/bin/bash",
    "nginx: {ip} - - \"GET /wp-admin HTTP/1.1\" 404 {bytes}",
    "kernel: IN=eth0 OUT= MAC={mac} SRC={ip} DST={ip2} LEN=60 TOS=0x00 PREC=0x00 TTL=64 ID={id} DF PROTO=TCP SPT={port} DPT=22 WINDOW=29200 RES=0x00 SYN URGP=0",
]

USERS = ["alice", "bob", "carol", "dave", "eve", "mallory", "admin", "root"]
CMDS = ["apt update", "apt upgrade -y", "backup.sh", "python3 job.py", "tar -czf /var/tmp/a.tgz /var/log"]
PORTS = [22, 2222, 443, 80, 8080]

def rand_ip():
    return f"10.{random.randint(0, 10)}.{random.randint(0, 255)}.{random.randint(1, 254)}"

def rand_mac():
    return ":".join(f"{random.randint(0,255):02x}" for _ in range(6))

def render(t: str):
    return t.format(
        user=random.choice(USERS),
        ip=rand_ip(),
        ip2=rand_ip(),
        port=random.choice(PORTS),
        pid=random.randint(100, 9999),
        uid=random.randint(1000, 2000),
        bytes=random.randint(200, 5000),
        cmd=random.choice(CMDS),
        mac=rand_mac(),
        id=random.randint(1, 50000),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--days", type=int, default=2, help="How many days of logs to generate")
    ap.add_argument("--hosts", type=int, default=5)
    ap.add_argument("--attack-rate", type=float, default=0.002, help="Probability per event of rare injected attack pattern")
    ap.add_argument("--events-per-host-per-day", type=int, default=2500)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    start = datetime.now(timezone.utc) - timedelta(days=args.days)

    rng = random.Random(7)
    random.seed(7)

    with out.open("w", encoding="utf-8") as f:
        for h in range(args.hosts):
            host = f"host-{h+1}"
            ts = start
            total = args.days * args.events_per_host_per_day

            for i in range(total):
                ts = ts + timedelta(seconds=rng.randint(5, 60))

                is_attack = rng.random() < args.attack_rate
                if is_attack:
                    msg = render(rng.choice(RARE_ATTACK_TEMPLATES))
                    label = "attack"
                else:
                    msg = render(rng.choice(NORMAL_TEMPLATES))
                    label = "normal"

                obj = {
                    "timestamp": ts.isoformat().replace("+00:00", "Z"),
                    "host": host,
                    "message": msg,
                    "label": label,
                }
                f.write(json.dumps(obj) + "\n")

    print(f"Wrote {out.resolve()}")

if __name__ == "__main__":
    main()
