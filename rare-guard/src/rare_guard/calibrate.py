from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from .calibration import CalibrationArtifact, conformal_threshold, save_calibration
from .data import Vocab
from .model import LogTransformer, ModelConfig, nll_score_from_logits


@torch.no_grad()
def calibrate_from_saved_split(
    artifacts_dir: str,
    alpha: float = 0.01,
    device: Optional[str] = None,
) -> None:
    art = Path(artifacts_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocab.from_json(json.loads((art / "vocab.json").read_text(encoding="utf-8")))
    cfg = ModelConfig(**json.loads((art / "model_config.json").read_text(encoding="utf-8")))

    model = LogTransformer(cfg).to(device)
    model.load_state_dict(torch.load(art / "model.pt", map_location=device))
    model.eval()

    X = np.load(art / "cal_split_X.npy")
    y = np.load(art / "cal_split_y.npy")
    n = int(y.shape[0])

    scores = []
    bs = 512
    for i in tqdm(range(0, n, bs), desc="calibrating"):
        xb = torch.tensor(X[i : i + bs], dtype=torch.long, device=device)
        yb = torch.tensor(y[i : i + bs], dtype=torch.long, device=device)
        logits = model(xb)
        nll = nll_score_from_logits(logits, yb)
        scores.append(nll.detach().cpu().numpy())

    scores = np.concatenate(scores, axis=0)
    thr = conformal_threshold(scores, alpha=alpha)

    cal = CalibrationArtifact(alpha=float(alpha), threshold=float(thr), n_cal=int(scores.shape[0]))
    save_calibration(art, scores, cal)

    print(f"Calibration complete. alpha={alpha} threshold={thr:.4f} n_cal={scores.shape[0]}")
