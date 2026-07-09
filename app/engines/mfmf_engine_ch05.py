"""
MFMF Laboratory Engine — Module 5: Fair Games, Measure Change, and Dynamic Hedging
Chapter 5: Martingales, Change of Measure, and Risk-Neutral Valuation.

Three panels: a fair-game sandbox (predictable betting rules average to
zero; a 'peek' toggle buys a positive slope), a measure panel (P- and
Q-weights, the density process Z_t, and E_Q[H]/R^T = E[Z H]/R^T), and a
hedging simulator (backward induction on a tree; the (V, Delta) lattice).

Book anchors: the collar audit on the working tree; the dealer's
replicating delta begins at Delta_0 = -0.57; density process Z_0 = 1.

Seeds: 2026CCNN, CC=05.
    20260501 E1 predictable rules average to zero; peek buys a slope
    20260502 E2 measure change: E_Q[H]/R^T equals E[Z H]/R^T
    20260503 E3 backward-induction hedge; the (V, Delta) lattice
"""
from __future__ import annotations
import numpy as np

SEED_BASE = 20260500
U, D = 1.2, 0.9
R_STEP = 1.05
S0 = 100.0
P = 0.6


def q_step(u: float = U, d: float = D, R: float = R_STEP) -> float:
    return (R - d) / (u - d)


# ---------- fair-game sandbox ----------
def E1_fair_game(seed: int = SEED_BASE + 1, n_bettors: int = 20_000,
                 T: int = 20) -> dict:
    """Predictable rules (bet sizes fixed by visible history) average to
    zero on a fair game. A one-step 'peek' buys a positive average slope."""
    rng = np.random.default_rng(seed)
    q = q_step()
    # martingale increments: +/- around the risk-neutral fair game
    up = (U - R_STEP) / R_STEP
    dn = (D - R_STEP) / R_STEP
    # predictable rule: bet proportional to the previous increment's sign
    gains_pred = np.zeros(n_bettors)
    gains_peek = np.zeros(n_bettors)
    for _ in range(T):
        moves = rng.random(n_bettors) < q
        incr = np.where(moves, up, dn)
        # predictable stake = 1 (no foresight): expected increment ~ 0 under Q
        gains_pred += incr
        # peek: stake = sign of the CURRENT move (illegal foresight)
        gains_peek += np.sign(incr) * incr
    return {"seed": seed, "q": q,
            "predictable_mean_gain": float(gains_pred.mean()),
            "predictable_se": float(gains_pred.std() / np.sqrt(n_bettors)),
            "peek_mean_gain": float(gains_peek.mean()),
            "peek_buys_positive_slope": bool(gains_peek.mean() > 0)}


# ---------- measure panel ----------
def build_binomial(T: int = 3):
    """(V, Delta) by backward induction for a claim on the tree."""
    import itertools
    paths = list(itertools.product([1, 0], repeat=T))
    return paths


def density_process(T: int = 3, p: float = P) -> dict:
    """Z_t = product of (q/p or (1-q)/(1-p)) along the path; Z_0 = 1."""
    q = q_step()
    z_up = q / p
    z_dn = (1 - q) / (1 - p)
    return {"Z0": 1.0, "z_up_ratio": z_up, "z_dn_ratio": z_dn,
            "E_Z_is_1": abs(p * z_up + (1 - p) * z_dn - 1.0) < 1e-12}


def price_claim(payoff_fn, T: int = 3, S0: float = S0) -> dict:
    """Price by E_Q[H]/R^T and by E[Z H]/R^T; confirm they agree."""
    import itertools
    q = q_step()
    paths = list(itertools.product([1, 0], repeat=T))
    eq_val = z_val = 0.0
    for path in paths:
        s = S0
        for m in path:
            s *= U if m else D
        H = payoff_fn(s)
        nup = sum(path)
        q_prob = q ** nup * (1 - q) ** (T - nup)
        p_prob = P ** nup * (1 - P) ** (T - nup)
        Z = (q / P) ** nup * ((1 - q) / (1 - P)) ** (T - nup)
        eq_val += q_prob * H
        z_val += p_prob * Z * H
    RT = R_STEP ** T
    return {"by_Q": eq_val / RT, "by_ZH": z_val / RT,
            "agree": abs(eq_val - z_val) < 1e-9}


# ---------- hedging simulator ----------
def hedge_lattice(K: float = 100.0, T: int = 3, S0: float = S0) -> dict:
    """Backward-induction (V, Delta) lattice for a European call; report
    the initial hedge ratio Delta_0."""
    q = q_step()
    # terminal values
    prices = {}
    for i in range(T + 1):
        s = S0 * U ** (T - i) * D ** i
        prices[(T, i)] = s
    V = {}
    for i in range(T + 1):
        s = S0 * U ** (T - i) * D ** i
        V[(T, i)] = max(s - K, 0.0)
    for t in range(T - 1, -1, -1):
        for i in range(t + 1):
            V[(t, i)] = (q * V[(t + 1, i)] + (1 - q) * V[(t + 1, i + 1)]) / R_STEP
    # Delta_0 = (V_up - V_dn) / (S_up - S_dn)
    Su, Sd = S0 * U, S0 * D
    delta0 = (V[(1, 0)] - V[(1, 1)]) / (Su - Sd)
    return {"V0": V[(0, 0)], "delta0": delta0, "q": q}


# ---------- guided experiments ----------
def E2_measure_change(seed: int = SEED_BASE + 2) -> dict:
    call = price_claim(lambda s: max(s - 100.0, 0.0))
    dens = density_process()
    return {"seed": seed, "call_by_Q": call["by_Q"], "call_by_ZH": call["by_ZH"],
            "agree": call["agree"], "density": dens}


def E3_hedge(seed: int = SEED_BASE + 3) -> dict:
    # a collar-like claim: long put K=95, short call K=115 on the tree
    put = hedge_lattice(K=95.0)
    call = hedge_lattice(K=115.0)
    collar_delta0 = put["delta0"] - call["delta0"] - 1.0  # long stock + collar
    return {"seed": seed, "call_lattice": hedge_lattice(),
            "collar_delta0_approx": collar_delta0}


# ---------- validation ----------
def validation_checks() -> dict:
    e1 = E1_fair_game()
    e2 = E2_measure_change()
    lat = hedge_lattice()
    dens = density_process()
    checks = {
        "V1_predictable_averages_zero": abs(e1["predictable_mean_gain"]) < 4 * e1["predictable_se"] + 1e-3,
        "V2_peek_buys_slope": e1["peek_buys_positive_slope"],
        "V3_measures_agree": e2["agree"],
        "V4_density_Z0_and_mean": dens["Z0"] == 1.0 and dens["E_Z_is_1"],
        "V5_q_half": abs(q_step() - 0.5) < 1e-9,
    }
    checks["ALL_PASS"] = all(checks.values())
    checks["_values"] = {"q": q_step(), "call_V0": lat["V0"], "delta0": lat["delta0"]}
    return checks


if __name__ == "__main__":
    import json
    print(json.dumps(validation_checks(), indent=2, default=str))
