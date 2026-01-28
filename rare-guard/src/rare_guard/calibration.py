from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    """
    Split conformal threshold.

    We choose the (1 - alpha) quantile with the +1 correction:
    q = ceil((n + 1) * (1 - alpha)) / n

    See e.g., Romano et al. / standard split conformal classification practice.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1 (exclusive).")
    scores = np.asarray(scores, dtype=float)
    n = scores.shape[0]
    if n < 30:
        # Still works, but coverage is noisier; warn in README / logs.
        pass
    k = int(np.ceil((n + 1) * (1.0 - alpha)))  # 1..n+1
    k = min(max(k, 1), n)
    # numpy quantile uses interpolation; we want kth order statistic.
    thr = float(np.partition(scores, k - 1)[k - 1])
    return thr


def conformal_p_value(score: float, cal_scores: np.ndarray) -> float:
    """
    Compute conformal p-value:
      p = (1 + #{i: cal_scores[i] >= score}) / (n + 1)
    """
    cal_scores = np.asarray(cal_scores, dtype=float)
    n = cal_scores.shape[0]
    ge = int(np.sum(cal_scores >= float(score)))
    return float((1 + ge) / (n + 1))


@dataclass
class CalibrationArtifact:
    alpha: float
    threshold: float
    n_cal: int

    def to_json(self) -> dict:
        return {"alpha": self.alpha, "threshold": self.threshold, "n_cal": self.n_cal}

    @classmethod
    def from_json(cls, obj: dict) -> "CalibrationArtifact":
        return cls(alpha=float(obj["alpha"]), threshold=float(obj["threshold"]), n_cal=int(obj["n_cal"]))


def save_calibration(out_dir: str | Path, cal_scores: np.ndarray, cal: CalibrationArtifact) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "calibration_scores.npy", np.asarray(cal_scores, dtype=float))
    (out / "calibration.json").write_text(json.dumps(cal.to_json(), indent=2), encoding="utf-8")


def load_calibration(artifacts_dir: str | Path) -> Tuple[np.ndarray, CalibrationArtifact]:
    p = Path(artifacts_dir)
    scores = np.load(p / "calibration_scores.npy")
    cal = CalibrationArtifact.from_json(json.loads((p / "calibration.json").read_text(encoding="utf-8")))
    return scores, cal
