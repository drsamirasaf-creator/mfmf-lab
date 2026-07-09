"""Chapter 8 — Module 8: The Pricing Machine."""
from fastapi import APIRouter
from ..engines import mfmf_engine_ch08 as eng
from ..common import jsonable
from ..models import PricingRequest

router = APIRouter(prefix="/api/ch08", tags=["ch08"])


@router.post("/collar")
def collar(req: PricingRequest):
    """E1 — the collar three ways (closed form / PDE / Monte Carlo)."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E1_collar(**kw))


@router.post("/greeks")
def greeks(req: PricingRequest):
    """E2 — the Greeks and the jump-day P&L."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E2_greeks(**kw))


@router.post("/smile")
def smile(req: PricingRequest):
    """E3 — reprice the collar under a volatility skew."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E3_smile(**kw))


@router.post("/barrier")
def barrier(req: PricingRequest):
    """E4 — barrier knockout; continuous vs daily monitoring."""
    kw = {}
    if req.B is not None:
        kw["B"] = req.B
    if req.seed is not None:
        kw["seed"] = req.seed
    return jsonable(eng.E4_barrier(**kw))


@router.get("/validation")
def validation():
    return jsonable(eng.validation_checks())
