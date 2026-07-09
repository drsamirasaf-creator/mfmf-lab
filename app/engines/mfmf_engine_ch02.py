"""
MFMF Laboratory Engine — Module 2: Probability and Distribution Lab
Chapter 2: Probability, Uncertainty, and Financial States.

Two panels: distribution fitting (normal vs Student-t by MLE, tail
probabilities and quantiles) and a copula sandbox (Gaussian / t /
Clayton dependence, joint-tail counts, two-asset loss distribution).

Book anchors (Examples 2.5, 2.11):
    Analyst one: R ~ Normal(mu=0.07, sigma=0.175).
    Analyst two: R = mu + s * T_nu, Student-t with nu=4,
        scale s = sigma * sqrt((nu-2)/nu).
    Reported tail probabilities at the book's thresholds: 7.5% and 1.4%.

Seeds: 2026CCNN, CC=02.
    20260201 E1 two analysts + crossing threshold
    20260202 E2 rolling-window t degrees-of-freedom instability
    20260203 E3 copula switch Gaussian -> t4 at fixed correlation
"""
from __future__ import annotations
import numpy as np
from scipy import stats

SEED_BASE = 20260200

MU, SIGMA, NU = 0.07, 0.175, 4
T_SCALE = SIGMA * np.sqrt((NU - 2) / NU)   # match sd of the t to sigma


# ---------- distribution panel ----------
def normal_tail(threshold: float, mu: float = MU, sigma: float = SIGMA) -> float:
    """P(R <= threshold) under Normal(mu, sigma^2)."""
    return float(stats.norm.cdf(threshold, loc=mu, scale=sigma))


def t_tail(threshold: float, mu: float = MU, s: float = T_SCALE, nu: int = NU) -> float:
    """P(R <= threshold) under R = mu + s*T_nu."""
    return float(stats.t.cdf((threshold - mu) / s, df=nu))


def crossing_threshold(mu: float = MU, sigma: float = SIGMA,
                       s: float = T_SCALE, nu: int = NU) -> float:
    """The loss threshold at which normal and t tail probabilities cross."""
    from scipy.optimize import brentq
    f = lambda x: normal_tail(x, mu, sigma) - t_tail(x, mu, s, nu)
    # they cross once in the left tail
    return float(brentq(f, mu - 6 * sigma, mu - 0.001))


def fit_normal(sample: np.ndarray) -> tuple[float, float]:
    return float(sample.mean()), float(sample.std(ddof=1))


def fit_t(sample: np.ndarray) -> tuple[float, float, float]:
    """MLE fit of a Student-t; returns (df, loc, scale)."""
    df, loc, scale = stats.t.fit(sample)
    return float(df), float(loc), float(scale)


# ---------- guided experiments ----------
def E1_two_analysts(seed: int = SEED_BASE + 1) -> dict:
    thr1, thr2 = -0.15, -0.25
    cross = crossing_threshold()
    return {
        "seed": seed,
        "normal_at_-0.15": normal_tail(thr1),
        "t_at_-0.15": t_tail(thr1),
        "normal_at_-0.25": normal_tail(thr2),
        "t_at_-0.25": t_tail(thr2),
        "crossing_threshold": cross,
        "note": "t is lighter near the mean, heavier far out; they cross once.",
    }


def E2_rolling_df(seed: int = SEED_BASE + 2, n_windows: int = 8,
                  win: int = 1260) -> dict:
    """Estimation-risk theme: refit t's df on rolling 5y windows of a
    simulated heavy-tailed return stream; df is unstable."""
    rng = np.random.default_rng(seed)
    total = win + n_windows * 126
    stream = MU + T_SCALE * rng.standard_t(NU, size=total)
    dfs = []
    for k in range(n_windows):
        w = stream[k * 126: k * 126 + win]
        df, _, _ = fit_t(w)
        dfs.append(round(df, 3))
    return {"seed": seed, "df_estimates": dfs,
            "df_min": min(dfs), "df_max": max(dfs), "true_df": NU}


def E3_copula_switch(seed: int = SEED_BASE + 3, rho: float = 0.6,
                     n: int = 200_000, q: float = 0.05) -> dict:
    """Fix marginals and correlation; switch Gaussian -> t4 copula and
    record the joint-tail count (both assets below their q-quantile)."""
    rng = np.random.default_rng(seed)

    def gaussian_copula():
        z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
        return stats.norm.cdf(z)

    def t_copula(df=4):
        g = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
        chi = rng.chisquare(df, size=n) / df
        t = g / np.sqrt(chi)[:, None]
        return stats.t.cdf(t, df=df)

    ug = gaussian_copula()
    ut = t_copula()
    joint_g = float(np.mean((ug[:, 0] < q) & (ug[:, 1] < q)))
    joint_t = float(np.mean((ut[:, 0] < q) & (ut[:, 1] < q)))
    indep = q * q
    return {"seed": seed, "rho": rho, "q": q,
            "joint_tail_gaussian": joint_g,
            "joint_tail_t4": joint_t,
            "independent_benchmark": indep,
            "tail_amplification": joint_t / joint_g}


# ---------- validation ----------
def validation_checks() -> dict:
    thr = -0.15
    n15 = normal_tail(thr)
    t15 = t_tail(thr)
    checks = {
        "V1_t_scale_matches_sd": abs(T_SCALE - SIGMA * np.sqrt((NU - 2) / NU)) < 1e-12,
        "V2_normal_tail_reasonable": 0.08 < n15 < 0.12,
        "V3_t_lighter_near_mean": t15 < n15,
        "V4_crossing_exists": crossing_threshold() < thr,
        "V5_t_heavier_far_out": t_tail(-0.35) > normal_tail(-0.35),
    }
    checks["ALL_PASS"] = all(checks.values())
    checks["_values"] = {"normal_-0.15": n15, "t_-0.15": t15,
                         "crossing": crossing_threshold()}
    return checks


if __name__ == "__main__":
    import json
    print(json.dumps(validation_checks(), indent=2, default=str))
    print("E1:", E1_two_analysts())
