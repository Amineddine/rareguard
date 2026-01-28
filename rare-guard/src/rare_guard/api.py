from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import FastAPI

from .detector import RareGuardDetector

app = FastAPI(title="RareGuard API", version="0.1.0")

# NOTE: set this env var or change default path for your deployment.
import os
DEFAULT_ARTIFACTS_DIR = os.getenv("RAREGUARD_ARTIFACTS_DIR", "artifacts")

detector: RareGuardDetector | None = None


class ScoreRequest(BaseModel):
    host: str = Field(..., description="Host identifier (context is per-host)")
    timestamp: str | None = Field(None, description="Optional ISO timestamp")
    message: str = Field(..., description="Raw log message")


class ScoreResponse(BaseModel):
    host: str
    template: str
    score: float
    p_value: float
    is_anomaly: bool
    topk: list[tuple[str, float]]
    context: list[str]


@app.on_event("startup")
def _startup():
    global detector
    detector = RareGuardDetector(DEFAULT_ARTIFACTS_DIR)


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    assert detector is not None
    r = detector.score_event(host=req.host, message=req.message, top_k=5)
    return ScoreResponse(
        host=r.host,
        template=r.template,
        score=r.score,
        p_value=r.p_value,
        is_anomaly=r.is_anomaly,
        topk=r.topk,
        context=r.context,
    )


@app.post("/reset/{host}")
def reset(host: str):
    assert detector is not None
    detector.reset(host)
    return {"ok": True, "host": host}
