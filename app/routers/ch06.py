"""Chapter 6 — Module 6: Paths, Quadratic Variation, and Jumps."""
from fastapi import APIRouter
from ..engines import mfmf_engine_ch06 as eng
from ..common import jsonable
from ..models import PathRequest

router = APIRouter(prefix="/api/ch06", tags=["ch06"])


@router.post("/donsker")
def donsker(req: PathRequest):
    """E1 — Donsker convergence; the collar tends to -2.76."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E1_donsker_collar(**kw))


@router.post("/realized_variance")
def realized_variance(req: PathRequest):
    """E2 — realized variance across sampling frequencies."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E2_realized_variance(**kw))


@router.post("/reflection")
def reflection(req: PathRequest):
    """E3 — the reflection principle by Monte Carlo (target 0.556)."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E3_reflection_mc(**kw))


@router.post("/jumps")
def jumps(req: PathRequest):
    """E4 — the Poisson-jump overlay and the tail odometer."""
    kw = {}
    for f in ("lam", "muJ", "sigJ", "seed"):
        v = getattr(req, f)
        if v is not None:
            kw[f] = v
    return jsonable(eng.E4_jump_odometer(**kw))


@router.get("/validation")
def validation():
    return jsonable(eng.validation_checks())
