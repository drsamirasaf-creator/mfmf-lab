"""Chapter 12 — Module 12: The Learning Machine."""
from fastapi import APIRouter
from ..engines import mfmf_engine_ch12 as eng
from ..common import jsonable
from ..models import FilterRequest

router = APIRouter(prefix="/api/ch12", tags=["ch12"])


@router.post("/fusion")
def fusion(req: FilterRequest):
    """E1 — the Gaussian update ($151.5M +/- $3.3M nowcast)."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E1_nowcast(**kw))


@router.post("/kalman")
def kalman(req: FilterRequest):
    """E2 — the Kalman filter and innovation whiteness."""
    kw = {}
    for f in ("a", "q", "sigma_v", "seed"):
        v = getattr(req, f)
        if v is not None:
            kw[f] = v
    return jsonable(eng.E2_kalman(**kw))


@router.post("/desmooth")
def desmooth(req: FilterRequest):
    """E3 — de-smoothing: recover theta = 0.8."""
    kw = {}
    for f in ("theta", "true_vol", "seed"):
        v = getattr(req, f)
        if v is not None:
            kw[f] = v
    return jsonable(eng.E3_desmooth(**kw))


@router.post("/drift")
def drift(req: FilterRequest):
    """E4 — learning the drift over the horizon."""
    kw = {}
    for f in ("s0", "sigma", "years", "seed"):
        v = getattr(req, f)
        if v is not None:
            kw[f] = v
    return jsonable(eng.E4_drift(**kw))


@router.get("/validation")
def validation():
    return jsonable(eng.validation_checks())
