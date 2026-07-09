"""Chapter 1 — Module 1: The State-Space & No-Arbitrage Sandbox."""
from fastapi import APIRouter
from ..engines import mfmf_engine_ch01 as eng
from ..engines.mfmf_engine_ch01 import MiniMarket
from ..common import jsonable, overrides
from ..models import MarketRequest

router = APIRouter(prefix="/api/ch01", tags=["ch01"])


def _market(req: MarketRequest) -> MiniMarket:
    kw = overrides(req, exclude={"K", "gamma", "seed"})
    return MiniMarket(**kw)


@router.post("/sandbox")
def sandbox(req: MarketRequest):
    """E1 — from a single number to a valuation distribution."""
    mkt = _market(req)
    K = req.K if req.K is not None else 105.0
    return jsonable(eng.E1_number_to_distribution(mkt=mkt, K=K))


@router.post("/arbitrage")
def arbitrage(req: MarketRequest):
    """E2 — sweep the traded price and watch the arbitrage region."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E2_arbitrage_sweep(**kw))


@router.post("/p_vs_q")
def p_vs_q(req: MarketRequest):
    """E3 — the physical measure P versus the risk-neutral measure Q."""
    kw = {}
    if req.gamma is not None:
        kw["gamma"] = req.gamma
    if req.seed is not None:
        kw["seed"] = req.seed
    return jsonable(eng.E3_P_vs_Q(**kw))


@router.post("/third_state")
def third_state(req: MarketRequest):
    """E4 — the incomplete third-state market and its pricing bound."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E4_third_state_bound(**kw))


@router.get("/validation")
def validation():
    return jsonable(eng.validation_checks())
