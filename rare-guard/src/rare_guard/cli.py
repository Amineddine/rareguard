from __future__ import annotations

import argparse
from pathlib import Path

from .train import train_model
from .calibrate import calibrate_from_saved_split
from .detector import RareGuardDetector
from .data import read_jsonl


def cmd_train(args: argparse.Namespace) -> None:
    train_model(
        log_path=args.log,
        out_dir=args.out,
        seq_len=args.seq_len,
        max_vocab=args.max_vocab,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )


def cmd_calibrate(args: argparse.Namespace) -> None:
    calibrate_from_saved_split(
        artifacts_dir=args.artifacts,
        alpha=args.alpha,
        device=args.device,
    )


def cmd_detect(args: argparse.Namespace) -> None:
    det = RareGuardDetector(args.artifacts, device=args.device)
    shown = 0
    for e in read_jsonl(args.log):
        r = det.score_event(host=e.host, message=e.message, top_k=args.top_k)
        if r.is_anomaly:
            print("=" * 80)
            print(f"ALERT host={r.host} score={r.score:.4f} p={r.p_value:.4f}")
            print(f"template: {r.template}")
            print("top predictions:")
            for t, p in r.topk:
                print(f"  - {p:.4f}  {t}")
            print("context (last N templates):")
            for c in r.context[-10:]:
                print(f"  {c}")
            shown += 1
            if args.show is not None and shown >= args.show:
                break


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rare-guard", description="RareGuard: conformal transformer log anomaly detection")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train the Transformer on normal logs")
    p_train.add_argument("--log", required=True, help="Path to JSONL logs")
    p_train.add_argument("--out", required=True, help="Artifacts output directory")
    p_train.add_argument("--seq-len", type=int, default=24)
    p_train.add_argument("--max-vocab", type=int, default=50000)
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--batch-size", type=int, default=256)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--seed", type=int, default=7)
    p_train.add_argument("--device", default=None)
    p_train.set_defaults(func=cmd_train)

    p_cal = sub.add_parser("calibrate", help="Calibrate conformal threshold on held-out normal data")
    p_cal.add_argument("--artifacts", required=True, help="Artifacts directory (from train)")
    p_cal.add_argument("--alpha", type=float, default=0.01, help="Target false alarm rate (approx.)")
    p_cal.add_argument("--device", default=None)
    p_cal.set_defaults(func=cmd_calibrate)

    p_det = sub.add_parser("detect", help="Run anomaly detection over a log file")
    p_det.add_argument("--log", required=True, help="Path to JSONL logs")
    p_det.add_argument("--artifacts", required=True, help="Artifacts directory (train+calibrate)")
    p_det.add_argument("--top-k", type=int, default=5)
    p_det.add_argument("--show", type=int, default=20, help="Stop after printing N alerts (optional)")
    p_det.add_argument("--device", default=None)
    p_det.set_defaults(func=cmd_detect)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
