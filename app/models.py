"""Typed request models for the MFMF Laboratory API.

Each chapter family has its own request schema. All fields are optional
with the frontend sending whatever the panel exposes; the engine supplies
book-calibrated defaults for anything omitted. Seeds are explicit integers
following the 2026CCNN convention.
"""
from __future__ import annotations
from pydantic import BaseModel
from typing import Optional


# ---------------- Part I ----------------
class MarketRequest(BaseModel):
    """Ch1 sandbox / Ch4 & Ch5 state-price markets."""
    S0: Optional[float] = None
    u: Optional[float] = None
    d: Optional[float] = None
    R: Optional[float] = None
    p: Optional[float] = None
    K: Optional[float] = None
    gamma: Optional[float] = None
    seed: Optional[int] = None


class DistributionRequest(BaseModel):
    """Ch2 distribution / copula lab."""
    mu: Optional[float] = None
    sigma: Optional[float] = None
    nu: Optional[int] = None
    threshold: Optional[float] = None
    rho: Optional[float] = None
    q: Optional[float] = None
    seed: Optional[int] = None


class InfoRequest(BaseModel):
    """Ch3 information / conditional expectation."""
    u: Optional[float] = None
    d: Optional[float] = None
    p: Optional[float] = None
    T: Optional[int] = None
    seed: Optional[int] = None


class StatePriceRequest(BaseModel):
    """Ch4 state prices, completeness, bounds."""
    R: Optional[float] = None
    recession_price: Optional[float] = None
    seed: Optional[int] = None


class MartingaleRequest(BaseModel):
    """Ch5 martingales, measure change, hedging."""
    u: Optional[float] = None
    d: Optional[float] = None
    R: Optional[float] = None
    p: Optional[float] = None
    K: Optional[float] = None
    seed: Optional[int] = None


# ---------------- Part II ----------------
class PathRequest(BaseModel):
    """Ch6 paths, quadratic variation, jumps."""
    sigma: Optional[float] = None
    barrier: Optional[float] = None
    lam: Optional[float] = None
    muJ: Optional[float] = None
    sigJ: Optional[float] = None
    seed: Optional[int] = None


class ItoRequest(BaseModel):
    """Ch7 Ito calculus, Girsanov, hedging."""
    r: Optional[float] = None
    sigma: Optional[float] = None
    mu: Optional[float] = None
    theta: Optional[float] = None
    drift: Optional[float] = None
    T: Optional[float] = None
    steps: Optional[int] = None
    seed: Optional[int] = None


class PricingRequest(BaseModel):
    """Ch8 Black-Scholes, Greeks, smile, barriers."""
    S0: Optional[float] = None
    sigma: Optional[float] = None
    r: Optional[float] = None
    T: Optional[float] = None
    Kp: Optional[float] = None
    Kc: Optional[float] = None
    B: Optional[float] = None
    seed: Optional[int] = None


# ---------------- Part III ----------------
class AllocationRequest(BaseModel):
    """Ch9 portfolio choice, Merton, Kelly."""
    gamma: Optional[float] = None
    rf: Optional[float] = None
    mu_risky: Optional[float] = None
    sigma: Optional[float] = None
    seed: Optional[int] = None


class ControlRequest(BaseModel):
    """Ch10 stochastic control, HJB, spending, execution."""
    gamma: Optional[float] = None
    rho: Optional[float] = None
    mu: Optional[float] = None
    rf: Optional[float] = None
    sigma: Optional[float] = None
    X0: Optional[float] = None
    seed: Optional[int] = None


class StoppingRequest(BaseModel):
    """Ch11 optimal stopping, American options, real options."""
    K: Optional[float] = None
    sigma: Optional[float] = None
    r: Optional[float] = None
    delta: Optional[float] = None
    S0: Optional[float] = None
    V0: Optional[float] = None
    I: Optional[float] = None
    seed: Optional[int] = None


class FilterRequest(BaseModel):
    """Ch12 filtering, Kalman, de-smoothing."""
    prior_mean: Optional[float] = None
    prior_sd: Optional[float] = None
    signal_mean: Optional[float] = None
    signal_sd: Optional[float] = None
    a: Optional[float] = None
    q: Optional[float] = None
    sigma_v: Optional[float] = None
    theta: Optional[float] = None
    true_vol: Optional[float] = None
    s0: Optional[float] = None
    sigma: Optional[float] = None
    years: Optional[int] = None
    seed: Optional[int] = None


# ---------------- Part IV ----------------
class RiskRequest(BaseModel):
    """Ch13 risk measures, VaR/ES, robustness, backtesting."""
    portfolio_sd: Optional[float] = None
    base_sd: Optional[float] = None
    understate: Optional[float] = None
    seed: Optional[int] = None


class EquilibriumRequest(BaseModel):
    """Ch14 equilibrium, Kyle, margin spiral."""
    gamma: Optional[float] = None
    beta: Optional[float] = None
    Sigma0: Optional[float] = None
    sigma_u: Optional[float] = None
    seed: Optional[int] = None
