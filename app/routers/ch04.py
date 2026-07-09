"""Chapter 4 — Module 4: State Prices, Completeness, and Bounds."""
from fastapi import APIRouter
from ..engines import mfmf_engine_ch04 as eng
from ..common import jsonable
from ..models import StatePriceRequest

router = APIRouter(prefix="/api/ch04", tags=["ch04"])


@router.post("/segment")
def segment(req: StatePriceRequest):
    """E1 — the state-price segment and the stake's bounds."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E1_segment(**kw))


@router.post("/complete")
def complete(req: StatePriceRequest):
    """E2 — complete the market with the recession claim."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E2_complete(**kw))


@router.post("/three_ways")
def three_ways(req: StatePriceRequest):
    """E3 — price the stake three ways (psi, q, m)."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E3_stake_three_ways(**kw))


@router.post("/hj")
def hansen_jagannathan(req: StatePriceRequest):
    """E4 — the Hansen-Jagannathan wedge."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E4_hansen_jagannathan(**kw))


@router.get("/validation")
def validation():
    return jsonable(eng.validation_checks())
