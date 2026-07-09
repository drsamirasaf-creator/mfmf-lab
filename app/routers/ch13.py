"""Chapter 13 — Module 13: The Risk Office."""
from fastapi import APIRouter
from ..engines import mfmf_engine_ch13 as eng
from ..common import jsonable
from ..models import RiskRequest

router = APIRouter(prefix="/api/ch13", tags=["ch13"])


@router.post("/subadditivity")
def subadditivity(req: RiskRequest):
    """E1 — the two-bond (-4, 98) VaR violation; ES coherent."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E1_var_subadditivity(**kw))


@router.post("/measure")
def measure(req: RiskRequest):
    """E2 — VaR and ES across confidence levels; the RU minimum."""
    kw = {}
    if req.portfolio_sd is not None:
        kw["portfolio_sd"] = req.portfolio_sd
    if req.seed is not None:
        kw["seed"] = req.seed
    return jsonable(eng.E2_measure(**kw))


@router.post("/robustness")
def robustness(req: RiskRequest):
    """E3 — ES over an ambiguity box."""
    kw = {}
    if req.base_sd is not None:
        kw["base_sd"] = req.base_sd
    if req.seed is not None:
        kw["seed"] = req.seed
    return jsonable(eng.E3_robustness(**kw))


@router.post("/backtest")
def backtest(req: RiskRequest):
    """E4 — backtesting and detection power."""
    kw = {}
    if req.understate is not None:
        kw["understate"] = req.understate
    if req.seed is not None:
        kw["seed"] = req.seed
    return jsonable(eng.E4_backtest(**kw))


@router.get("/validation")
def validation():
    return jsonable(eng.validation_checks())
