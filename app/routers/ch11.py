"""Chapter 11 — Module 11: The Timing Desk."""
from fastapi import APIRouter
from ..engines import mfmf_engine_ch11 as eng
from ..common import jsonable
from ..models import StoppingRequest

router = APIRouter(prefix="/api/ch11", tags=["ch11"])


@router.post("/american")
def american(req: StoppingRequest):
    """E1 — the American 90-put (1.826) and its boundary."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E1_american_put(**kw))


@router.post("/dividend_call")
def dividend_call(req: StoppingRequest):
    """E2 — the call's exercise region appears as dividends rise."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E2_dividend_call(**kw))


@router.post("/pasting")
def pasting(req: StoppingRequest):
    """E3 — smooth pasting; value maximized at S* = 66.0."""
    kw = {"seed": req.seed} if req.seed is not None else {}
    return jsonable(eng.E3_smooth_pasting(**kw))


@router.post("/platform")
def platform(req: StoppingRequest):
    """E4 — the platform: beta1, hurdle, option value, NPV."""
    kw = {}
    for f in ("V0", "I", "sigma", "delta", "seed"):
        v = getattr(req, f)
        if v is not None:
            kw[f] = v
    return jsonable(eng.E4_platform(**kw))


@router.get("/validation")
def validation():
    return jsonable(eng.validation_checks())
