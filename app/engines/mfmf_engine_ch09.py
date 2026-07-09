"""
MFMF Laboratory Engine — Module 9: The Allocation Laboratory
Chapter 9: Portfolio Choice and Dynamic Optimization.

Four panels: the frontier (mean-variance frontier, CML, tangency, Sharpe
gap), fragility (estimation error and the shrinkage / no-short / 1-over-N
defenses), Merton (the constant-weight dynamic solution and implied gamma),
and growth (the Kelly parabola and wealth fans).

Book anchors:
    Tangency weights 16.5 / 37.0 / 46.5 (%).
    Merton: gamma = 1.74 gives the 60% risky weight; horizon is irrelevant.

Seeds: 2026CCNN, CC=09.
    20260901 E1 tangency weights and the 60/40 Sharpe gap
    20260902 E2 estimation fragility and the defenses
    20260903 E3 Merton constant weight and implied gamma
    20260904 E4 the Kelly parabola and wealth fans
"""
from __future__ import annotations
import numpy as np

SEED_BASE = 20260900

# Three-asset capital market assumptions calibrated to yield the book's
# tangency weights 16.5 / 37.0 / 46.5.
MU = np.array([0.00922, 0.04872, 0.095])      # excess returns over rf
RF = 0.02
COV = np.array([[0.0064, 0.0012, 0.0010],
                [0.0012, 0.0225, 0.0040],
                [0.0010, 0.0040, 0.0400]])


def tangency_weights(mu=MU, cov=COV):
    inv = np.linalg.inv(cov)
    raw = inv @ mu
    return raw / raw.sum()


def frontier_sharpe(mu=MU, cov=COV):
    w = tangency_weights(mu, cov)
    port_ret = w @ mu
    port_vol = np.sqrt(w @ cov @ w)
    return port_ret / port_vol


def portfolio_sharpe(w, mu=MU, cov=COV):
    w = np.asarray(w, float)
    return (w @ mu) / np.sqrt(w @ cov @ w)


# ---------- guided experiments ----------
def E1_tangency(seed: int = SEED_BASE + 1) -> dict:
    w = tangency_weights()
    tan_sharpe = frontier_sharpe()
    # a 60/40-style policy across the three sleeves for comparison
    w6040 = np.array([0.30, 0.30, 0.40])
    gap = tan_sharpe - portfolio_sharpe(w6040)
    return {"seed": seed, "tangency_weights_pct": (w * 100).round(1).tolist(),
            "tangency_sharpe": tan_sharpe,
            "policy_sharpe": portfolio_sharpe(w6040),
            "sharpe_gap": gap}


def E2_fragility(seed: int = SEED_BASE + 2, n_resample: int = 500,
                 n_obs: int = 120) -> dict:
    """Resample means, recompute tangency weights, and show the cloud;
    shrinkage toward the grand mean tames it."""
    rng = np.random.default_rng(seed)
    base = tangency_weights()
    raw_spread, shrunk_spread = [], []
    for _ in range(n_resample):
        noisy_mu = MU + rng.multivariate_normal(np.zeros(3), COV / n_obs)
        w_raw = tangency_weights(noisy_mu)
        # shrink means toward the cross-sectional average
        shrunk_mu = 0.5 * noisy_mu + 0.5 * noisy_mu.mean()
        w_shr = tangency_weights(shrunk_mu)
        raw_spread.append(w_raw.std())
        shrunk_spread.append(w_shr.std())
    return {"seed": seed, "base_weights_pct": (base * 100).round(1).tolist(),
            "raw_weight_dispersion": float(np.mean(raw_spread)),
            "shrunk_weight_dispersion": float(np.mean(shrunk_spread)),
            "shrinkage_tames": float(np.mean(shrunk_spread)) < float(np.mean(raw_spread))}


def E3_merton(seed: int = SEED_BASE + 3, gamma: float = 1.74) -> dict:
    """Merton constant weight w* = (mu-rf)/(gamma sigma^2) for the single
    risky asset with the book's calibration giving 60% at gamma=1.74."""
    mu_risky, sigma = 0.052, 0.175
    w_star = (mu_risky - RF) / (gamma * sigma ** 2)
    # invert: gamma implied by a stated 60% weight
    implied_gamma = (mu_risky - RF) / (0.60 * sigma ** 2)
    return {"seed": seed, "gamma": gamma, "merton_weight": w_star,
            "implied_gamma_for_60pct": implied_gamma,
            "horizon_irrelevant": True}


def E4_kelly(seed: int = SEED_BASE + 4, n_sim: int = 20_000) -> dict:
    """Growth rate parabola g(w) = w*(mu-rf) - 0.5 w^2 sigma^2; Kelly at
    the vertex, half-Kelly and twice-Kelly marked."""
    mu_risky, sigma = 0.052, 0.175
    excess = mu_risky - RF

    def g(w):
        return w * excess - 0.5 * w ** 2 * sigma ** 2
    kelly = excess / sigma ** 2
    return {"seed": seed, "kelly_fraction": kelly,
            "g_at_60pct": g(0.60), "g_at_kelly": g(kelly),
            "g_at_twice_kelly": g(2 * kelly),
            "twice_kelly_zero_growth": abs(g(2 * kelly)) < 1e-9}


# ---------- validation ----------
def validation_checks() -> dict:
    e1 = E1_tangency(); e3 = E3_merton(); e4 = E4_kelly()
    w = e1["tangency_weights_pct"]
    checks = {
        "V1_tangency_16.5_37_46.5": (abs(w[0] - 16.5) < 1.0 and abs(w[1] - 37.0) < 1.5
                                     and abs(w[2] - 46.5) < 1.5),
        "V2_sharpe_gap_positive": e1["sharpe_gap"] > 0,
        "V3_merton_60pct_at_gamma_1.74": abs(e3["merton_weight"] - 0.60) < 0.02,
        "V4_twice_kelly_zero_growth": e4["twice_kelly_zero_growth"],
    }
    checks["ALL_PASS"] = all(checks.values())
    checks["_values"] = {"tangency": w, "sharpe_gap": e1["sharpe_gap"],
                         "merton_w": e3["merton_weight"]}
    return checks


if __name__ == "__main__":
    import json
    print(json.dumps(validation_checks(), indent=2, default=str))
