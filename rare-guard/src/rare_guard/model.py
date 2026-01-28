from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int = 24
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.1
    pad_id: int = 0


class LogTransformer(nn.Module):
    """
    Transformer encoder for next-event prediction.

    Input: token ids for context window (batch, seq_len)
    Output: logits for next token (batch, vocab_size)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos = PositionalEncoding(cfg.d_model, max_len=max(512, cfg.seq_len + 8), dropout=cfg.dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        mask = (x == self.cfg.pad_id)  # (batch, seq_len) - True where padding
        h = self.embed(x)  # (batch, seq_len, d_model)
        h = self.pos(h)
        h = self.encoder(h, src_key_padding_mask=mask)
        h = self.norm(h)

        # Use the final position as summary of context
        # (you can also pool; keeping it simple & stable)
        last = h[:, -1, :]  # (batch, d_model)
        logits = self.head(last)  # (batch, vocab_size)
        return logits


def nll_score_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute per-sample negative log-likelihood score.
    logits: (batch, vocab_size)
    target: (batch,)
    returns: (batch,)
    """
    logp = torch.log_softmax(logits, dim=-1)
    idx = torch.arange(target.shape[0], device=target.device)
    nll = -logp[idx, target]
    return nll
