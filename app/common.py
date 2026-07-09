"""Shared helpers for the MFMF Laboratory API.

Every engine returns plain Python / numpy objects; `jsonable` makes them
safe for FastAPI's JSON encoder (numpy scalars -> Python scalars, arrays
-> lists, booleans from numpy -> bool, etc.). `pick` applies request
overrides on top of engine defaults so the frontend may send partial
parameter sets.
"""
from __future__ import annotations
import math
import numpy as np


def jsonable(obj):
    """Recursively convert numpy / non-finite values into JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, bool):
        return obj
    return obj


def overrides(request, exclude: set[str] | None = None) -> dict:
    """Return the request's set (non-None) fields as kwargs for an engine
    call. Pydantic models expose .model_dump(); we drop None so the engine
    keeps its own defaults for anything the frontend didn't send."""
    exclude = exclude or set()
    data = request.model_dump() if hasattr(request, "model_dump") else dict(request)
    return {k: v for k, v in data.items() if v is not None and k not in exclude}
