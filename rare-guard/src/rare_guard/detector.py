from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .calibration import conformal_p_value, load_calibration
from .data import LogEvent, Vocab, templateize
from .model import LogTransformer, ModelConfig, nll_score_from_logits


@dataclass
class DetectionResult:
    host: str
    template: str
    score: float
    p_value: float
    is_anomaly: bool
    topk: List[Tuple[str, float]]
    context: List[str]


@dataclass
class StreamState:
    context_ids: List[int] = field(default_factory=list)


class RareGuardDetector:
    """
    Streaming detector that maintains per-host context and scores each incoming event.
    """

    def __init__(self, artifacts_dir: str | Path, device: Optional[str] = None):
        self.artifacts_dir = Path(artifacts_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load vocab
        vocab_obj = __import__("json").loads((self.artifacts_dir / "vocab.json").read_text(encoding="utf-8"))
        self.vocab = Vocab.from_json(vocab_obj)

        # Load model config
        cfg_obj = __import__("json").loads((self.artifacts_dir / "model_config.json").read_text(encoding="utf-8"))
        self.cfg = ModelConfig(**cfg_obj)

        # Load model weights
        self.model = LogTransformer(self.cfg).to(self.device)
        self.model.load_state_dict(torch.load(self.artifacts_dir / "model.pt", map_location=self.device))
        self.model.eval()

        # Load calibration
        self.cal_scores, self.cal = load_calibration(self.artifacts_dir)

        self.states: Dict[str, StreamState] = {}

    def reset(self, host: str) -> None:
        self.states.pop(host, None)

    def _get_state(self, host: str) -> StreamState:
        if host not in self.states:
            self.states[host] = StreamState(context_ids=[self.vocab.token_to_id["<START>"]] * self.cfg.seq_len)
        return self.states[host]

    @torch.no_grad()
    def score_event(self, host: str, message: str, top_k: int = 5) -> DetectionResult:
        state = self._get_state(host)

        templ = templateize(message)
        token_id = self.vocab.encode(templ)

        # Context window is the previous seq_len events (already stored)
        x = torch.tensor([state.context_ids[-self.cfg.seq_len :]], dtype=torch.long, device=self.device)
        logits = self.model(x)  # (1, vocab)
        target = torch.tensor([token_id], dtype=torch.long, device=self.device)
        nll = nll_score_from_logits(logits, target)[0].item()

        # Conformal p-value & thresholding
        pval = conformal_p_value(nll, self.cal_scores)
        is_anom = bool(nll > self.cal.threshold)

        # Top-k predictions for explanation
        probs = torch.softmax(logits[0], dim=-1)
        topv, topi = torch.topk(probs, k=min(top_k, probs.shape[0]))
        topk = [(self.vocab.decode(int(i)), float(v)) for v, i in zip(topv.cpu().tolist(), topi.cpu().tolist())]

        # Context templates for readability
        ctx_templates = [self.vocab.decode(i) for i in state.context_ids[-self.cfg.seq_len :]]

        # Update context (append current token)
        state.context_ids.append(token_id)

        return DetectionResult(
            host=host,
            template=templ,
            score=float(nll),
            p_value=float(pval),
            is_anomaly=is_anom,
            topk=topk,
            context=ctx_templates,
        )
