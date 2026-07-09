"""Chapter 7 — Module 7: Stochastic Calculus in the Hands."""
from fastapi import APIRouter
from ..engines import mfmf_engine_ch07 as eng
from ..common import jsonable
from ..models import ItoRequest

router = APIRouter(prefix="/api/ch07", tags=["ch07"])


@router.post("/integral")
def integral(req: ItoRequest):
    """E1 — the W dW integral: left vs midpoint, gap T/2."""
    kw = {}
    for f in ("T", "steps", "seed"):
        v = getattr(req, f)
        if v is not None:
            kw[f] = v
    return jsonable(eng.E1_wdw(**kw))


@router.post("/ito")
def ito(req: ItoRequest):
    """E2 — Ito's formula; classical rule fails by the QV term."""
    kw = {}
    for f in ("T", "steps", "seed"):
        v = getattr(req, f)
        if v is not None:
            kw[f] = v
    return jsonable(eng.E2_ito_verify(**kw))


@router.post("/girsanov")
def girsanov(req: ItoRequest):
    """E3 — Girsanov: Q-drift equals r, volatility unchanged."""
    kw = {}
    for f in ("r", "sigma", "mu", "theta", "T", "steps", "seed"):
        v = getattr(req, f)
        if v is not None:
            kw[f] = v
    return jsonable(eng.E3_girsanov(**kw))


@router.post("/hedge_error")
def hedge_error(req: ItoRequest):
    """E4 — the 1/sqrt(n) hedging-error law."""
    kw = {}
    for f in ("drift", "sigma", "r", "T", "seed"):
        v = getattr(req, f)
        if v is not None:
            kw[f] = v
    return jsonable(eng.E4_hedge_error(**kw))


@router.get("/validation")
def validation():
    return jsonable(eng.validation_checks())
