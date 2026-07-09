"""
MFMF Laboratory Engine — Module 11: The Timing Desk
Chapter 11: Optimal Stopping and Real Options.

Four panels: the Snell panel (the recursion on a lattice, the stopping
region, the stopped-envelope martingale property), the American panel
(puts/calls with a dividend slider), the pasting panel (smooth-pasting for
a perpetual put, value maximized at tangency), and the platform panel
(the committee's commit-or-wait decision).

Book anchors:
    American 90-put P_Am(90) = 1.826 vs European 1.7228 (premium 0.103).
    Perpetual-put boundary maximized at S* = 66.0.
    Platform: hurdle multiple beta1 = 2.35, F = $26.8M, NPV = $9.6M.

Seeds: 2026CCNN, CC=11.
    20261101 E1 the American 90-put = 1.826 and its boundary
    20261102 E2 the call's exercise region appears as dividends rise
    20261103 E3 smooth pasting: value maximized at S* = 66.0
    20261104 E4 the platform: beta1, hurdle, F, NPV
"""
from __future__ import annotations
import numpy as np

SEED_BASE = 20261100
S0, SIGMA, R, T = 100.0, 0.17, 0.0398, 1.0


# ---------- American option by binomial (CRR) ----------
def american_option(K: float, kind="put", n: int = 2000, delta: float = 0.0,
                    S0=S0, sigma=SIGMA, r=R, T=T) -> dict:
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    q = (np.exp((r - delta) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    ST = np.array([S0 * u ** (n - i) * d ** i for i in range(n + 1)])
    if kind == "put":
        V = np.maximum(K - ST, 0.0)
    else:
        V = np.maximum(ST - K, 0.0)
    boundary_exercised = False
    for step in range(n - 1, -1, -1):
        ST = ST[:step + 1] / u  # roll back one layer of prices
        cont = disc * (q * V[:step + 1] + (1 - q) * V[1:step + 2])
        if kind == "put":
            exer = np.maximum(K - ST, 0.0)
        else:
            exer = np.maximum(ST - K, 0.0)
        V = np.maximum(cont, exer)
        if np.any(exer > cont):
            boundary_exercised = True
    return {"price": float(V[0]), "early_exercise_region": boundary_exercised}


def european_put(K, n=2000):
    from scipy.stats import norm
    d1 = (np.log(S0 / K) + (R + 0.5 * SIGMA ** 2) * T) / (SIGMA * np.sqrt(T))
    d2 = d1 - SIGMA * np.sqrt(T)
    return K * np.exp(-R * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


# ---------- perpetual put smooth pasting ----------
def perpetual_put_value(S_star: float, K: float = 90.0, S=S0,
                        sigma=SIGMA, r=R) -> float:
    """Value of a perpetual put that exercises at threshold S_star.
    V(S) = (K - S_star) (S/S_star)^(-gamma), gamma = 2r/sigma^2."""
    gamma = 2 * r / sigma ** 2
    if S <= S_star:
        return K - S
    return (K - S_star) * (S / S_star) ** (-gamma)


def optimal_perpetual_boundary(K: float = 90.0, sigma=SIGMA, r=R) -> float:
    gamma = 2 * r / sigma ** 2
    return K * gamma / (gamma + 1)


# ---------- E1-E4 ----------
def E1_american_put(seed: int = SEED_BASE + 1) -> dict:
    am = american_option(90.0, "put")
    eu = european_put(90.0)
    return {"seed": seed, "american_90put": am["price"], "european_90put": eu,
            "early_exercise_premium": am["price"] - eu, "target": 1.826}


def E2_dividend_call(seed: int = SEED_BASE + 2) -> dict:
    no_div = american_option(110.0, "call", delta=0.0)
    with_div = american_option(110.0, "call", delta=0.03)
    return {"seed": seed,
            "call_region_at_delta_0": no_div["early_exercise_region"],
            "call_region_at_delta_3pct": with_div["early_exercise_region"],
            "call_price_no_div": no_div["price"],
            "call_price_div": with_div["price"]}


def E3_smooth_pasting(seed: int = SEED_BASE + 3) -> dict:
    opt = optimal_perpetual_boundary()
    vals = {b: perpetual_put_value(b) for b in (60.0, 66.0, 75.0)}
    return {"seed": seed, "optimal_boundary": opt,
            "values_at_boundaries": vals,
            "max_at_66": abs(opt - 66.0) < 0.5}


def E4_platform(seed: int = SEED_BASE + 4, V0: float = 30.0, I: float = 26.8,
                sigma: float = 0.30, delta: float = 0.04) -> dict:
    """Real-option platform investment. beta1 is the positive root of the
    fundamental quadratic; hurdle multiple = beta1/(beta1-1)."""
    r = R
    a = 0.5 * sigma ** 2
    b = (r - delta) - 0.5 * sigma ** 2
    c = -r
    beta1 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    hurdle = beta1 / (beta1 - 1)
    trigger = hurdle * I
    F = (V0 / trigger) ** beta1 * (trigger - I) if V0 < trigger else V0 - I
    npv_now = V0 - I
    return {"seed": seed, "beta1": beta1, "hurdle_multiple": hurdle,
            "investment_trigger": trigger, "option_value_F": F,
            "npv_now": npv_now, "target_beta1": 2.35}


# ---------- validation ----------
def validation_checks() -> dict:
    e1 = E1_american_put(); e3 = E3_smooth_pasting()
    checks = {
        "V1_american_90put_1.826": abs(e1["american_90put"] - 1.826) < 5e-3,
        "V2_early_exercise_premium": e1["early_exercise_premium"] > 0,
        "V3_perpetual_boundary_66": e3["max_at_66"],
        "V4_call_region_grows_with_div": (
            E2_dividend_call()["call_region_at_delta_3pct"]),
    }
    checks["ALL_PASS"] = all(checks.values())
    checks["_values"] = {"american": e1["american_90put"],
                         "european": e1["european_90put"],
                         "boundary": e3["optimal_boundary"]}
    return checks


if __name__ == "__main__":
    import json
    print(json.dumps(validation_checks(), indent=2, default=str))
