"""Chapter 3 — Module 3: Information and Conditional Expectation Simulator."""
from fastapi import APIRouter
from ..engines import mfmf_engine_ch03 as eng
from ..common import jsonable
from ..models import InfoRequest

router = APIRouter(prefix="/api/ch03", tags=["ch03"])


@router.post("/tower")
def tower(req: InfoRequest):
    """E1 — the tower property verified numerically."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    if req.T is not None:
        kw["T"] = req.T
    return jsonable(eng.E1_tower(**kw))


@router.post("/clairvoyance")
def clairvoyance(req: InfoRequest):
    """E2 — the clairvoyance detector rejects a look-ahead rule."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    if req.T is not None:
        kw["T"] = req.T
    return jsonable(eng.E2_clairvoyance(**kw))


@router.post("/total_variance")
def total_variance(req: InfoRequest):
    """E3 — the law of total variance / smoothing."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E3_total_variance(**kw))


@router.get("/validation")
def validation():
    return jsonable(eng.validation_checks())
