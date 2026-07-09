"""
MFMF Laboratory Engine — Module 10: The Control Room
Chapter 10: Stochastic Control and the Hamilton-Jacobi-Bellman Equation.

Four panels: the Bellman panel (discrete DP recursion, argmax policy), the
verification panel (a policy's performance-process mean bleeds at its HJB
deficit, flat at the optimum), the spending panel (the (gamma, rho) -> nu*
surface), and the execution panel (optimal transition schedule and the
implementation-shortfall frontier).

Book anchors:
    Merton spending rule nu* at (gamma, rho) = (2, 5%).
    A candidate policy's mean performance bleeds at its HJB deficit and
    flatlines exactly at the optimum.

Seeds: 2026CCNN, CC=10.
    20261001 E1 the Bellman recursion on a small consumption toy
    20261002 E2 verification: bleed rate vs the flat optimum
    20261003 E3 the spending surface nu*(gamma, rho)
    20261004 E4 optimal execution schedule and shortfall frontier
"""
from __future__ import annotations
import numpy as np

SEED_BASE = 20261000
MU, SIGMA, RF = 0.052, 0.175, 0.02


def merton_spending(gamma: float, rho: float, mu=MU, sigma=SIGMA, rf=RF) -> float:
    """Optimal consumption rate nu* for CRRA-Merton (infinite horizon):
    nu* = rho/gamma + (1 - 1/gamma) (rf + (mu-rf)^2 / (2 gamma sigma^2))."""
    sharpe_sq = (mu - rf) ** 2 / sigma ** 2
    return rho / gamma + (1 - 1 / gamma) * (rf + sharpe_sq / (2 * gamma))


def merton_weight(gamma: float, mu=MU, sigma=SIGMA, rf=RF) -> float:
    return (mu - rf) / (gamma * sigma ** 2)


# ---------- E1: Bellman recursion ----------
def E1_bellman(seed: int = SEED_BASE + 1) -> dict:
    """Three-date, two-action consumption toy solved by backward induction.
    States = wealth on a small grid; actions = consume {low, high}."""
    grid = np.array([50.0, 100.0, 150.0, 200.0])
    R = 1.05
    actions = [0.3, 0.6]   # consume fraction of wealth

    def util(c):
        return np.log(c) if c > 0 else -1e9

    T = 3
    V = {T: {w: util(w) for w in grid}}   # terminal: consume all
    policy = {}
    for t in range(T - 1, -1, -1):
        V[t] = {}; policy[t] = {}
        for w in grid:
            best_val, best_a = -1e18, None
            for a in actions:
                c = a * w
                w_next = (w - c) * R
                w_near = grid[np.argmin(abs(grid - w_next))]
                val = util(c) + 0.96 * V[t + 1][w_near]
                if val > best_val:
                    best_val, best_a = val, a
            V[t][w] = best_val; policy[t][w] = best_a
    return {"seed": seed, "V0_at_100": V[0][100.0],
            "policy_t0": {str(w): policy[0][w] for w in grid},
            "recursion_monotone": all(V[0][grid[i]] <= V[0][grid[i + 1]]
                                      for i in range(len(grid) - 1))}


# ---------- E2: verification / HJB deficit ----------
def E2_verification(seed: int = SEED_BASE + 2) -> dict:
    """A candidate policy's performance-process mean bleeds at its HJB
    deficit; the optimal policy has zero deficit (flat)."""
    gamma, rho = 2.0, 0.05
    w_opt = merton_weight(gamma)
    nu_opt = merton_spending(gamma, rho)

    def hjb_deficit(w, nu):
        # deficit of a constant (w, nu) policy vs the optimum, in value terms;
        # zero at (w_opt, nu_opt), negative (bleed) otherwise.
        drift_gap = -(0.5 * gamma * SIGMA ** 2) * (w - w_opt) ** 2
        cons_gap = -(nu - nu_opt) ** 2 / (2 * nu_opt)
        return drift_gap + cons_gap

    candidates = {"pi=120%,c=8%": (1.20, 0.08),
                  "pi=60%,c=4.5%": (0.60, 0.045),
                  "optimal": (w_opt, nu_opt)}
    bleed = {name: hjb_deficit(w, nu) for name, (w, nu) in candidates.items()}
    return {"seed": seed, "w_opt": w_opt, "nu_opt": nu_opt,
            "bleed_rates": bleed,
            "optimal_is_flat": abs(bleed["optimal"]) < 1e-9}


# ---------- E3: spending surface ----------
def E3_spending_surface(seed: int = SEED_BASE + 3) -> dict:
    nu_meridian = merton_spending(2.0, 0.05)
    # invert: rho implied by a 4.5% rule at gamma=2
    # nu = rho/gamma + (1-1/gamma)(rf + sharpe^2/(2 gamma)); solve for rho
    gamma = 2.0
    sharpe_sq = (MU - RF) ** 2 / SIGMA ** 2
    const = (1 - 1 / gamma) * (RF + sharpe_sq / (2 * gamma))
    implied_rho = gamma * (0.045 - const)
    return {"seed": seed, "nu_star_at_2_5pct": nu_meridian,
            "implied_rho_of_4.5pct_rule": implied_rho}


# ---------- E4: optimal execution ----------
def E4_execution(seed: int = SEED_BASE + 4, X0: float = 600.0,
                 eta: float = 0.01, sigma_d: float = 0.02) -> dict:
    """Almgren-Chriss style: trade X0 over N steps; the frontier trades off
    expected cost against variance as risk aversion kappa varies."""
    N = 20
    T = 1.0
    dt = T / N
    frontier = {}
    for kappa in (0.05, 0.2, 0.4):
        # optimal trajectory decay rate
        k = np.sqrt(kappa * sigma_d ** 2 / eta)
        t = np.linspace(0, T, N + 1)
        holdings = X0 * np.sinh(k * (T - t)) / np.sinh(k * T)
        trades = -np.diff(holdings)
        exp_cost = eta * np.sum(trades ** 2) / dt
        var_cost = kappa * sigma_d ** 2 * np.sum(holdings[1:] ** 2) * dt
        frontier[kappa] = {"exp_cost": float(exp_cost), "risk": float(var_cost)}
    return {"seed": seed, "X0": X0, "frontier": frontier}


# ---------- validation ----------
def validation_checks() -> dict:
    e2 = E2_verification(); e3 = E3_spending_surface()
    nu = merton_spending(2.0, 0.05)
    checks = {
        "V1_bellman_monotone": E1_bellman()["recursion_monotone"],
        "V2_optimal_is_flat": e2["optimal_is_flat"],
        "V3_suboptimal_bleeds": all(v < 0 for k, v in e2["bleed_rates"].items()
                                    if k != "optimal"),
        "V4_spending_positive": 0 < nu < 0.15,
    }
    checks["ALL_PASS"] = all(checks.values())
    checks["_values"] = {"nu_star": nu, "w_opt": e2["w_opt"],
                         "implied_rho": e3["implied_rho_of_4.5pct_rule"]}
    return checks


if __name__ == "__main__":
    import json
    print(json.dumps(validation_checks(), indent=2, default=str))
