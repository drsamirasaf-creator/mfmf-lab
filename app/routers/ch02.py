"""Chapter 2 — Module 2: Probability and Distribution Lab."""
from fastapi import APIRouter
from ..engines import mfmf_engine_ch02 as eng
from ..common import jsonable
from ..models import DistributionRequest

router = APIRouter(prefix="/api/ch02", tags=["ch02"])


@router.post("/distribution")
def distribution(req: DistributionRequest):
    """E1 — two analysts (normal vs Student-t) and the tail crossing."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E1_two_analysts(**kw))


@router.post("/rolling_df")
def rolling_df(req: DistributionRequest):
    """E2 — the unstable rolling-window degrees of freedom."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E2_rolling_df(**kw))


@router.post("/copula")
def copula(req: DistributionRequest):
    """E3 — Gaussian vs t4 copula joint-tail counts."""
    kw = {}
    if req.rho is not None:
        kw["rho"] = req.rho
    if req.q is not None:
        kw["q"] = req.q
    if req.seed is not None:
        kw["seed"] = req.seed
    return jsonable(eng.E3_copula_switch(**kw))


@router.get("/validation")
def validation():
    return jsonable(eng.validation_checks())
