"""Chapter 5 — Module 5: Fair Games, Measure Change, and Dynamic Hedging."""
from fastapi import APIRouter
from ..engines import mfmf_engine_ch05 as eng
from ..common import jsonable
from ..models import MartingaleRequest

router = APIRouter(prefix="/api/ch05", tags=["ch05"])


@router.post("/fair_game")
def fair_game(req: MartingaleRequest):
    """E1 — predictable rules average to zero; a peek buys a slope."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E1_fair_game(**kw))


@router.post("/measure_change")
def measure_change(req: MartingaleRequest):
    """E2 — the density process; risk-neutral vs ZH pricing agree."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E2_measure_change(**kw))


@router.post("/hedge")
def hedge(req: MartingaleRequest):
    """E3 — the (V, Delta) hedging lattice."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E3_hedge(**kw))


@router.get("/validation")
def validation():
    return jsonable(eng.validation_checks())
