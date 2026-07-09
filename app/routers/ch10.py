"""Chapter 10 — Module 10: The Control Room."""
from fastapi import APIRouter
from ..engines import mfmf_engine_ch10 as eng
from ..common import jsonable
from ..models import ControlRequest

router = APIRouter(prefix="/api/ch10", tags=["ch10"])


@router.post("/bellman")
def bellman(req: ControlRequest):
    """E1 — the Bellman recursion on a consumption toy."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E1_bellman(**kw))


@router.post("/verification")
def verification(req: ControlRequest):
    """E2 — the HJB deficit; bleed rates vs the flat optimum."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E2_verification(**kw))


@router.post("/spending")
def spending(req: ControlRequest):
    """E3 — the (gamma, rho) -> nu* spending surface."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E3_spending_surface(**kw))


@router.post("/execution")
def execution(req: ControlRequest):
    """E4 — optimal execution and the shortfall frontier."""
    kw = {}
    if req.X0 is not None:
        kw["X0"] = req.X0
    if req.seed is not None:
        kw["seed"] = req.seed
    return jsonable(eng.E4_execution(**kw))


@router.get("/validation")
def validation():
    return jsonable(eng.validation_checks())
