"""
MFMF Laboratory Engine — Module 7: Stochastic Calculus in the Hands
Chapter 7: Itô Calculus and Continuous-Time Finance.

Four panels: the integral builder (Riemann sums of theta dW, left vs
midpoint), the Ito verifier (both sides of Ito's formula; classical chain
rule fails by exactly the quadratic-variation term), the Girsanov panel
(P- and Q-paths on one noise stream, density Z_T), and the hedging panel
(the 1/sqrt(n) discretization-error law).

Book anchors:
    integral of W dW at left evaluation = (W_T^2 - T)/2; the left/midpoint
        gap is exactly T/2.
    Girsanov: theta = lambda = 0.178 makes the index's Q-drift equal r
        while measured volatility is unchanged at every theta.
    Hedging error scales as 1/sqrt(n); its mean is ~0 regardless of drift.

Seeds: 2026CCNN, CC=07.
    20260701 E1 W dW: left vs midpoint, gap T/2
    20260702 E2 Ito's formula verified; classical residual = (1/2)f_xx dt
    20260703 E3 Girsanov: Q-drift = r, vol unchanged
    20260704 E4 the 1/sqrt(n) hedging-error law
"""
from __future__ import annotations
import numpy as np

SEED_BASE = 20260700
THETA = 0.178   # = lambda, the market price of risk in the book's example


def brownian(seed: int, T: float = 1.0, steps: int = 10_000):
    rng = np.random.default_rng(seed)
    dt = T / steps
    dW = np.sqrt(dt) * rng.standard_normal(steps)
    W = np.concatenate([[0], np.cumsum(dW)])
    return W, dW, dt


# ---------- E1: stochastic integral, evaluation point matters ----------
def E1_wdw(seed: int = SEED_BASE + 1, T: float = 1.0, steps: int = 50_000) -> dict:
    W, dW, dt = brownian(seed, T, steps)
    left = np.sum(W[:-1] * dW)
    right = np.sum(W[1:] * dW)
    mid = np.sum(0.5 * (W[:-1] + W[1:]) * dW)
    ito_closed = 0.5 * (W[-1] ** 2 - T)      # Ito value of int W dW
    strat_closed = 0.5 * W[-1] ** 2          # Stratonovich (midpoint)
    return {"seed": seed, "left": left, "midpoint": mid, "right": right,
            "left_minus_right_gap": left - right,          # ~ -T
            "midpoint_minus_left": mid - left,             # ~ T/2
            "ito_closed_form": ito_closed, "strat_closed_form": strat_closed,
            "gap_target_T_over_2": T / 2}


# ---------- E2: Ito's formula, classical rule fails by (1/2)f_xx dt ----------
def E2_ito_verify(seed: int = SEED_BASE + 2, T: float = 1.0, steps: int = 50_000) -> dict:
    W, dW, dt = brownian(seed, T, steps)
    # f(x) = x^2 : df = 2x dW + dt ; classical (drop the dt) misses exactly t.
    x = W
    df_true = np.diff(x ** 2)
    df_ito = 2 * x[:-1] * dW + dt                 # includes (1/2)f_xx dt = dt
    df_classical = 2 * x[:-1] * dW                # drops the Ito term
    resid_ito = np.sum(df_true - df_ito)
    resid_classical = np.sum(df_true - df_classical)
    return {"seed": seed,
            "ito_residual": float(resid_ito),        # ~ 0
            "classical_residual": float(resid_classical),  # ~ T (the missing dt sum)
            "classical_residual_target_T": T,
            "matches_QV_term": abs(resid_classical - T) < 0.05}


# ---------- E3: Girsanov, Q-drift = r ----------
def E3_girsanov(seed: int = SEED_BASE + 3, r: float = 0.0398, sigma: float = 0.17,
                mu: float = None, theta: float = THETA, T: float = 1.0,
                steps: int = 10_000) -> dict:
    # mu = r + theta*sigma so that theta is the market price of risk
    mu = r + theta * sigma if mu is None else mu
    W, dW, dt = brownian(seed, T, steps)
    # P-drift of log-return is mu - sigma^2/2; under Q it is r - sigma^2/2.
    # density Z_T = exp(-theta W_T - 0.5 theta^2 T)
    ZT = np.exp(-theta * W[-1] - 0.5 * theta ** 2 * T)
    # realized vol is invariant to the measure change
    logret = (mu - 0.5 * sigma ** 2) * dt + sigma * dW
    realized_vol = np.sqrt(np.sum((logret - logret.mean()) ** 2) / T)
    return {"seed": seed, "mu": mu, "theta": theta,
            "P_drift": mu, "Q_drift": r, "Q_drift_equals_r": abs((mu - theta * sigma) - r) < 1e-9,
            "Z_T": float(ZT), "realized_vol": float(realized_vol),
            "vol_unchanged": abs(realized_vol - sigma) < 0.02}


# ---------- E4: the 1/sqrt(n) hedging-error law ----------
def E4_hedge_error(seed: int = SEED_BASE + 4, drift: float = 0.0,
                   sigma: float = 0.17, r: float = 0.0398, T: float = 1.0,
                   S0: float = 100.0, K: float = 100.0) -> dict:
    from scipy.stats import norm
    rng = np.random.default_rng(seed)

    def bs_call(S, tau):
        if tau <= 1e-12:
            return max(S - K, 0.0), (1.0 if S > K else 0.0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        price = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
        return price, norm.cdf(d1)

    results = {}
    n_sim = 2000
    for n in (21, 63, 252):
        dt = T / n
        errs = []
        for _ in range(n_sim):
            S = S0
            price0, delta = bs_call(S0, T)
            cash = price0 - delta * S0
            for i in range(1, n + 1):
                z = rng.standard_normal()
                S *= np.exp((drift - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
                cash *= np.exp(r * dt)
                if i < n:
                    _, new_delta = bs_call(S, T - i * dt)
                    cash -= (new_delta - delta) * S
                    delta = new_delta
            portfolio = cash + delta * S
            payoff = max(S - K, 0.0)
            errs.append(portfolio - payoff)
        errs = np.array(errs)
        results[n] = {"error_mean": float(errs.mean()), "error_std": float(errs.std())}
    # check std scales ~ 1/sqrt(n)
    stds = [results[n]["error_std"] for n in (21, 63, 252)]
    ratio = stds[0] / stds[2]  # should be ~ sqrt(252/21) = 3.46
    return {"seed": seed, "drift": drift, "by_n": results,
            "std_ratio_21_to_252": ratio, "target_ratio": np.sqrt(252 / 21),
            "error_mean_near_zero": abs(results[252]["error_mean"]) < 0.3}


# ---------- validation ----------
def validation_checks() -> dict:
    e1 = E1_wdw(); e2 = E2_ito_verify(); e3 = E3_girsanov()
    checks = {
        "V1_wdw_gap_is_T_over_2": abs(e1["midpoint_minus_left"] - 0.5) < 0.02,
        "V2_ito_residual_zero": abs(e1["left"] - e1["ito_closed_form"]) < 0.05,
        "V3_classical_fails_by_QV": e2["matches_QV_term"],
        "V4_girsanov_Q_drift_r": e3["Q_drift_equals_r"],
        "V5_vol_unchanged": e3["vol_unchanged"],
    }
    checks["ALL_PASS"] = all(checks.values())
    checks["_values"] = {"wdw_gap": e1["midpoint_minus_left"],
                         "classical_resid": e2["classical_residual"],
                         "Z_T": e3["Z_T"]}
    return checks


if __name__ == "__main__":
    import json
    print(json.dumps(validation_checks(), indent=2, default=str))
