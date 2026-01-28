from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b|\b[0-9a-fA-F]{16,}\b")
UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b")
NUM_RE = re.compile(r"\b\d+\b")
PATH_RE = re.compile(r"(/[^ \t\n\r\f\v]+)+")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

DEFAULT_SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<START>"]

def templateize(message: str) -> str:
    """
    Convert a raw log message into a more stable template by replacing variable fields.

    This is intentionally lightweight (portfolio-friendly). In production, you'd often use Drain3
    or vendor-specific structured fields.
    """
    msg = message.strip()
    msg = EMAIL_RE.sub("<EMAIL>", msg)
    msg = IP_RE.sub("<IP>", msg)
    msg = UUID_RE.sub("<UUID>", msg)
    msg = HEX_RE.sub("<HEX>", msg)
    msg = PATH_RE.sub("<PATH>", msg)
    msg = NUM_RE.sub("<NUM>", msg)
    # Collapse multiple spaces
    msg = re.sub(r"\s+", " ", msg)
    return msg


@dataclass(frozen=True)
class LogEvent:
    timestamp: datetime
    host: str
    message: str
    label: Optional[str] = None  # optional: "normal" / "attack" / etc.

    @property
    def template(self) -> str:
        return templateize(self.message)


def read_jsonl(path: str) -> Iterator[LogEvent]:
    """Read JSONL with fields: timestamp (ISO), host, message, optional label."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            ts = obj.get("timestamp")
            if isinstance(ts, (int, float)):
                # unix seconds
                dt = datetime.fromtimestamp(float(ts))
            else:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            yield LogEvent(
                timestamp=dt,
                host=str(obj["host"]),
                message=str(obj["message"]),
                label=obj.get("label"),
            )


@dataclass
class Vocab:
    token_to_id: Dict[str, int]
    id_to_token: List[str]

    @classmethod
    def build(cls, templates: Iterable[str], max_size: int = 50000, special_tokens: Optional[List[str]] = None) -> "Vocab":
        if special_tokens is None:
            special_tokens = list(DEFAULT_SPECIAL_TOKENS)

        counts: Dict[str, int] = {}
        for t in templates:
            counts[t] = counts.get(t, 0) + 1

        # sort by frequency
        items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        tokens = special_tokens + [t for (t, _) in items[: max(0, max_size - len(special_tokens))]]

        token_to_id = {t: i for i, t in enumerate(tokens)}
        id_to_token = tokens
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)

    def encode(self, template: str) -> int:
        return self.token_to_id.get(template, self.token_to_id["<UNK>"])

    def decode(self, idx: int) -> str:
        if 0 <= idx < len(self.id_to_token):
            return self.id_to_token[idx]
        return "<UNK>"

    def to_json(self) -> Dict:
        return {"token_to_id": self.token_to_id, "id_to_token": self.id_to_token}

    @classmethod
    def from_json(cls, obj: Dict) -> "Vocab":
        return cls(token_to_id=dict(obj["token_to_id"]), id_to_token=list(obj["id_to_token"]))


def group_by_host(events: Iterable[LogEvent]) -> Dict[str, List[LogEvent]]:
    out: Dict[str, List[LogEvent]] = {}
    for e in events:
        out.setdefault(e.host, []).append(e)
    for host in out:
        out[host].sort(key=lambda x: x.timestamp)
    return out


def build_sequences_for_next_event(
    events_by_host: Dict[str, List[LogEvent]],
    vocab: Vocab,
    seq_len: int,
    only_normal: bool = True,
) -> Tuple[List[List[int]], List[int]]:
    """
    Build (X, y) pairs for next-event prediction.
    X: list of token id sequences length seq_len
    y: next token id
    """
    X: List[List[int]] = []
    y: List[int] = []

    pad_id = vocab.token_to_id["<PAD>"]
    start_id = vocab.token_to_id["<START>"]

    for host, events in events_by_host.items():
        # optionally filter out non-normal labels (synthetic demo injects "attack")
        filtered = [e for e in events if (not only_normal) or (e.label in (None, "normal"))]
        tokens = [vocab.encode(e.template) for e in filtered]

        # prepend start tokens so early steps can be trained too
        tokens = [start_id] * seq_len + tokens

        for i in range(seq_len, len(tokens) - 1):
            ctx = tokens[i - seq_len : i]
            nxt = tokens[i]
            # ctx is always length seq_len
            X.append(ctx)
            y.append(nxt)

    return X, y
