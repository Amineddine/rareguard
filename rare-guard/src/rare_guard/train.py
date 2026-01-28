from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data import Vocab, group_by_host, read_jsonl, build_sequences_for_next_event
from .model import LogTransformer, ModelConfig


class SeqDataset(Dataset):
    def __init__(self, X: List[List[int]], y: List[int]):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)


def split_train_cal(X: List[List[int]], y: List[int], cal_frac: float = 0.2, seed: int = 7):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)
    rng.shuffle(idx)
    cal_n = int(n * cal_frac)
    cal_idx = idx[:cal_n]
    tr_idx = idx[cal_n:]

    X_tr = [X[i] for i in tr_idx]
    y_tr = [y[i] for i in tr_idx]
    X_cal = [X[i] for i in cal_idx]
    y_cal = [y[i] for i in cal_idx]
    return (X_tr, y_tr), (X_cal, y_cal)


def train_model(
    log_path: str,
    out_dir: str,
    seq_len: int = 24,
    max_vocab: int = 50000,
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 3e-4,
    seed: int = 7,
    device: str | None = None,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    events = list(read_jsonl(log_path))
    by_host = group_by_host(events)

    # Build vocab from *normal* templates only
    templates = [e.template for e in events if e.label in (None, "normal")]
    vocab = Vocab.build(templates, max_size=max_vocab)

    X, y = build_sequences_for_next_event(by_host, vocab=vocab, seq_len=seq_len, only_normal=True)
    if len(y) < 100:
        raise RuntimeError("Not enough training data. Provide more logs or reduce seq_len.")

    (X_tr, y_tr), (X_cal, y_cal) = split_train_cal(X, y, cal_frac=0.2, seed=seed)

    # Save vocab
    (out / "vocab.json").write_text(json.dumps(vocab.to_json(), indent=2), encoding="utf-8")

    cfg = ModelConfig(
        vocab_size=len(vocab.id_to_token),
        seq_len=seq_len,
        pad_id=vocab.token_to_id["<PAD>"],
    )
    (out / "model_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    model = LogTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    tr_ds = SeqDataset(X_tr, y_tr)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        count = 0
        pbar = tqdm(tr_dl, desc=f"epoch {ep}/{epochs}")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * xb.size(0)
            count += xb.size(0)
            pbar.set_postfix(loss=total / max(count, 1))

    torch.save(model.state_dict(), out / "model.pt")

    # Save calibration split to disk so calibrate step can reproduce scores if desired
    np.save(out / "cal_split_X.npy", np.array(X_cal, dtype=np.int64))
    np.save(out / "cal_split_y.npy", np.array(y_cal, dtype=np.int64))

    print(f"Saved artifacts to: {out.resolve()}")
