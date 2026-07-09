"""Chapter 14 — Module 14: The Market Itself (capstone)."""
from fastapi import APIRouter
from ..engines import mfmf_engine_ch14 as eng
from ..common import jsonable
from ..models import EquilibriumRequest

router = APIRouter(prefix="/api/ch14", tags=["ch14"])


@router.post("/equilibrium")
def equilibrium(req: EquilibriumRequest):
    """E1 — reverse-engineer (gamma, beta) = (1.41, 1.026)."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E1_equilibrium(**kw))


@router.post("/kyle")
def kyle(req: EquilibriumRequest):
    """E2 — the Kyle auction: lambda = 0.125, market depth 8."""
    kw = {}
    if req.Sigma0 is not None:
        kw["Sigma0"] = req.Sigma0
    if req.sigma_u is not None:
        kw["sigma_u"] = req.sigma_u
    if req.seed is not None:
        kw["seed"] = req.seed
    return jsonable(eng.E2_kyle(**kw))


@router.post("/spiral")
def spiral(req: EquilibriumRequest):
    """E3 — the margin spiral and the k -> 1 regime change."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E3_margin_spiral(**kw))


@router.post("/capstone")
def capstone(req: EquilibriumRequest):
    """E4 — the agenda as one problem family."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E4_capstone(**kw))


@router.get("/validation")
def validation():
    return jsonable(eng.validation_checks())
