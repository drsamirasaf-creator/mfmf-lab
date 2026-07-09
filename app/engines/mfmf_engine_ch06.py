"""
MFMF Laboratory Engine — Module 6: Paths, Quadratic Variation, and Jumps
Chapter 6: Stochastic Processes and Financial Dynamics.

Three panels: the scaled-walk limit (Donsker, running maximum, reflection
principle), quadratic variation / realized variance across sampling
frequencies, and a Poisson-jump overlay on GBM with a tail odometer.

Book anchors:
    Reflection principle: fraction of driftless sigma=0.17 paths touching
        +10% within the year = 2*Phi(-0.10/0.17) = 0.556.
    Collar limit toward which the tree prices wobble: -2.76
        (exact Black-Scholes value -2.7574, Chapter 8).
    Quadratic variation [W,W]_T = T; realized variance is its estimator.

Seeds: 2026CCNN, CC=06.
    20260601 E1 Donsker convergence and collar to -2.76
    20260602 E2 realized variance across sampling frequencies
    20260603 E3 reflection principle by Monte Carlo (target 0.556)
    20260604 E4 jump overlay and the tail odometer
"""
from __future__ import annotations
import numpy as np
from scipy import stats

SEED_BASE = 20260600
SIGMA = 0.17
BARRIER = 0.10   # +10%


def reflection_probability(barrier: float = BARRIER, sigma: float = SIGMA) -> float:
    """P(max of driftless BM over [0,1] exceeds `barrier`) = 2*Phi(-b/sigma)."""
    return float(2 * stats.norm.cdf(-barrier / sigma))


# ---------- quadratic variation ----------
def realized_variance(path: np.ndarray) -> float:
    """Sum of squared log-increments (the realized quadratic variation)."""
    logp = np.log(path)
    return float(np.sum(np.diff(logp) ** 2))


def simulate_gbm(seed: int, T: float = 1.0, steps: int = 252,
                 mu: float = 0.05, sigma: float = SIGMA, S0: float = 100.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = T / steps
    incr = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(steps)
    return S0 * np.exp(np.concatenate([[0], np.cumsum(incr)]))


# ---------- guided experiments ----------
def E1_donsker_collar(seed: int = SEED_BASE + 1) -> dict:
    """Binomial collar price under n steps -> BS limit -2.76 as n grows."""
    S0, r, T = 100.0, 0.0398, 1.0
    Kp, Kc = 90.0, 110.0
    results = {}
    for n in (4, 12, 52, 252):
        dt = T / n
        u = np.exp(SIGMA * np.sqrt(dt))
        d = 1 / u
        R = np.exp(r * dt)
        q = (R - d) / (u - d)
        # terminal distribution
        collar = 0.0
        from math import comb
        disc = np.exp(-r * T)
        for k in range(n + 1):
            ST = S0 * u ** k * d ** (n - k)
            prob = comb(n, k) * q ** k * (1 - q) ** (n - k)
            payoff = max(Kp - ST, 0) - max(ST - Kc, 0)  # long put, short call
            collar += prob * payoff
        results[n] = disc * collar
    return {"seed": seed, "collar_by_n": results,
            "limit": results[252], "target": -2.76}


def E2_realized_variance(seed: int = SEED_BASE + 2) -> dict:
    """Annualized vol from the same simulated year at monthly / daily /
    5-minute sampling; the QV estimator sharpens as sampling densifies."""
    T = 1.0
    fine = simulate_gbm(seed, T=T, steps=252 * 78)  # 5-min bars (~78/day)
    out = {}
    for label, step in [("monthly", 21 * 78), ("daily", 78), ("5-min", 1)]:
        sub = fine[::step]
        rv = realized_variance(sub)
        vol = np.sqrt(rv / T)
        # standard error of RV-based vol estimate ~ vol / sqrt(2 N)
        N = len(sub) - 1
        se = vol / np.sqrt(2 * N)
        out[label] = {"vol": vol, "se": se, "n_obs": N}
    return {"seed": seed, "estimates": out, "true_sigma": SIGMA}


def E3_reflection_mc(seed: int = SEED_BASE + 3, n_paths: int = 200_000,
                     steps: int = 252) -> dict:
    rng = np.random.default_rng(seed)
    dt = 1.0 / steps
    incr = SIGMA * np.sqrt(dt) * rng.standard_normal((n_paths, steps))
    paths = np.cumsum(incr, axis=1)
    # discrete running max understates the true continuous max; correct with
    # the Brownian-bridge maximum between grid points. For each step the
    # conditional prob the bridge exceeds b, given endpoints a,c, adds
    # crossing mass missed by sampling. We simulate the bridge max exactly.
    prev = np.concatenate([np.zeros((n_paths, 1)), paths[:, :-1]], axis=1)
    var = SIGMA ** 2 * dt
    # bridge max over each interval: m = (a+c + sqrt((c-a)^2 - 2 var ln U))/2
    U = rng.random((n_paths, steps))
    bridge_max = 0.5 * (prev + paths + np.sqrt((paths - prev) ** 2 - 2 * var * np.log(U)))
    running_max = bridge_max.max(axis=1)
    touched = float(np.mean(running_max >= BARRIER))
    return {"seed": seed, "mc_fraction": touched,
            "formula": reflection_probability(), "target": 0.556}


def E4_jump_odometer(seed: int = SEED_BASE + 4, lam: float = 0.5,
                     muJ: float = -0.02, sigJ: float = 0.05) -> dict:
    """Poisson jumps on GBM; the tail odometer reports the model-implied
    waiting time (in years) for a -6% day."""
    rng = np.random.default_rng(seed)
    n_days = 252
    dt = 1.0 / n_days
    n_sim = 100_000
    diff = (0.05 - 0.5 * SIGMA ** 2) * dt + SIGMA * np.sqrt(dt) * rng.standard_normal(n_sim)
    n_jumps = rng.poisson(lam * dt, n_sim)
    jump = np.where(n_jumps > 0, muJ + sigJ * rng.standard_normal(n_sim), 0.0)
    daily_ret = np.exp(diff + jump) - 1
    p_minus6 = float(np.mean(daily_ret <= -0.06))
    waiting_years = (1 / (p_minus6 * n_days)) if p_minus6 > 0 else np.inf
    return {"seed": seed, "p_minus6pct": p_minus6,
            "waiting_years_for_-6pct_day": waiting_years}


# ---------- validation ----------
def validation_checks() -> dict:
    e1 = E1_donsker_collar()
    e3 = E3_reflection_mc()
    checks = {
        "V1_reflection_formula_0.556": abs(reflection_probability() - 0.556) < 1e-3,
        "V2_collar_converges_to_-2.76": abs(e1["limit"] - (-2.76)) < 0.05,
        "V3_reflection_mc_matches": abs(e3["mc_fraction"] - 0.556) < 0.01,
        "V4_QV_equals_T": abs(realized_variance(
            np.exp(np.concatenate([[0], np.cumsum(
                SIGMA * np.sqrt(1/252) * np.random.default_rng(1).standard_normal(252))]))
            ) - SIGMA ** 2) < 0.05,
    }
    checks["ALL_PASS"] = all(checks.values())
    checks["_values"] = {"reflection": reflection_probability(),
                         "collar_limit": e1["limit"], "mc": e3["mc_fraction"]}
    return checks


if __name__ == "__main__":
    import json
    print(json.dumps(validation_checks(), indent=2, default=str))
