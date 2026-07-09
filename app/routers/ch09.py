"""Chapter 9 — Module 9: The Allocation Laboratory."""
from fastapi import APIRouter
from ..engines import mfmf_engine_ch09 as eng
from ..common import jsonable
from ..models import AllocationRequest

router = APIRouter(prefix="/api/ch09", tags=["ch09"])


@router.post("/tangency")
def tangency(req: AllocationRequest):
    """E1 — the tangency portfolio and the Sharpe gap."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E1_tangency(**kw))


@router.post("/fragility")
def fragility(req: AllocationRequest):
    """E2 — estimation fragility and the shrinkage defense."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E2_fragility(**kw))


@router.post("/merton")
def merton(req: AllocationRequest):
    """E3 — Merton's constant weight and implied gamma."""
    kw = {}
    if req.gamma is not None:
        kw["gamma"] = req.gamma
    if req.seed is not None:
        kw["seed"] = req.seed
    return jsonable(eng.E3_merton(**kw))


@router.post("/kelly")
def kelly(req: AllocationRequest):
    """E4 — the Kelly parabola and the growth ceiling."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E4_kelly(**kw))


@router.get("/validation")
def validation():
    return jsonable(eng.validation_checks())
