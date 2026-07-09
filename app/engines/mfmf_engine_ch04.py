"""
MFMF Laboratory Engine — Module 4: State Prices, Completeness, and Bounds
Chapter 4: Valuation, No-Arbitrage, and State Prices.

Four panels: extraction (rank of D, completeness, the set Psi as a point
or segment/polytope), dictionary (psi -> q -> m; price three ways),
bounds (sub/super-replication LPs of Prop 4.11), and Hansen-Jagannathan
(E[m], sigma(m)) vs the traded Sharpe wedge.

Book anchors (Examples 4.8, 4.10):
    Bond R = 1.04 -> psi_1+psi_2+psi_3 = 0.9615.
    Equity payoff (60,110,150), price 100.
    State-price segment: psi_2 = 1.1058 - 2.25 t, psi_3 = 1.25 t - 0.1442,
        positive for t in (0.1154, 0.4915), with t := psi_1.
    Recession claim (50,20,0) at 23.52 pins t = 0.28:
        psi* = (0.28, 0.4758, 0.2058), q = (0.2912, 0.4948, 0.2140),
        m = (1.12, 0.9515, 0.8231).

Seeds: 2026CCNN, CC=04.
    20260401 E1 extract the segment, read the stake's bounds
    20260402 E2 complete the market with the recession claim
    20260403 E3 price the stake three ways
    20260404 E4 Hansen-Jagannathan wedge
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog

SEED_BASE = 20260400
R = 1.04
BOND = np.array([1.0, 1.0, 1.0])
EQUITY = np.array([60.0, 110.0, 150.0])
BOND_PRICE, EQUITY_PRICE = 1.0, 100.0
RECESSION = np.array([50.0, 20.0, 0.0])
RECESSION_PRICE = 23.52


def segment_endpoints() -> tuple[float, float]:
    """Range of t = psi_1 for which all three state prices are positive."""
    # psi_2 = 1.1058 - 2.25 t > 0  -> t < 0.4915
    # psi_3 = 1.25 t - 0.1442 > 0  -> t > 0.1154
    return 0.1154, 0.4915


def psi_from_t(t: float) -> np.ndarray:
    return np.array([t, 1.1058 - 2.25 * t, 1.25 * t - 0.1442])


def complete_with_recession() -> dict:
    """Add the recession claim; solve for the unique t and psi*."""
    # 50*psi_1 + 20*psi_2 = 23.52, with psi from the segment
    # 50 t + 20 (1.1058 - 2.25 t) = 23.52 -> 50 t + 22.116 - 45 t = 23.52
    # 5 t = 1.404 -> t = 0.2808
    t = (RECESSION_PRICE - 20 * 1.1058) / (50 - 20 * 2.25)
    psi = psi_from_t(t)
    q = R * psi
    p_phys = np.array([0.25, 0.5, 0.25])  # physical measure for the SDF
    m = psi / p_phys
    return {"t": t, "psi": psi, "q": q, "m": m}


def price_three_ways(payoff: np.ndarray, psi: np.ndarray,
                     p_phys: np.ndarray | None = None) -> dict:
    q = R * psi
    p_phys = p_phys if p_phys is not None else np.array([0.25, 0.5, 0.25])
    m = psi / p_phys
    return {"by_psi": float(psi @ payoff),
            "by_q": float((q @ payoff) / R),
            "by_m": float((p_phys * m) @ payoff)}


def replication_bounds(payoff: np.ndarray) -> dict:
    """Sub/super-replication prices over the incomplete two-asset market
    (bond + equity only), as LPs on the state-price segment."""
    # super = max over feasible psi of psi.payoff ; sub = min
    D = np.vstack([BOND, EQUITY])           # 2 x 3
    prices = np.array([BOND_PRICE / R, EQUITY_PRICE])  # bond discounted
    # feasible psi >= 0 with D psi = prices
    hi = linprog(-payoff, A_eq=D, b_eq=np.array([1 / R * BOND_PRICE, EQUITY_PRICE]),
                 bounds=[(0, None)] * 3)
    lo = linprog(payoff, A_eq=D, b_eq=np.array([1 / R * BOND_PRICE, EQUITY_PRICE]),
                 bounds=[(0, None)] * 3)
    return {"sub": float(lo.fun) if lo.success else None,
            "super": float(-hi.fun) if hi.success else None}


# ---------- guided experiments ----------
def E1_segment(seed: int = SEED_BASE + 1) -> dict:
    lo, hi = segment_endpoints()
    # the "stake" = the recession claim's payoff, bounded before it trades
    b = replication_bounds(RECESSION)
    return {"seed": seed, "t_range": (lo, hi),
            "psi_at_lo": psi_from_t(lo).tolist(),
            "psi_at_hi": psi_from_t(hi).tolist(),
            "stake_bounds": b}


def E2_complete(seed: int = SEED_BASE + 2) -> dict:
    r = complete_with_recession()
    return {"seed": seed, "t": r["t"], "psi_star": r["psi"].tolist(),
            "q": r["q"].tolist(), "m": r["m"].tolist(),
            "recession_reprice": float(r["psi"] @ RECESSION)}


def E3_stake_three_ways(seed: int = SEED_BASE + 3) -> dict:
    r = complete_with_recession()
    # price a "stake": upside claim paying (0, 100, 200), say
    stake = np.array([0.0, 100.0, 200.0])
    return {"seed": seed, "prices": price_three_ways(stake, r["psi"])}


def E4_hansen_jagannathan(seed: int = SEED_BASE + 4) -> dict:
    """(E[m], sigma(m)) at the pinned psi* vs the HJ lower bound
    sigma(m)/E[m] >= max traded Sharpe ratio."""
    r = complete_with_recession()
    p_phys = np.array([0.25, 0.5, 0.25])
    m = r["m"]
    Em = float(p_phys @ m)
    Varm = float(p_phys @ (m - Em) ** 2)
    sig_m = np.sqrt(Varm)
    # traded equity Sharpe: excess return / vol under P
    gross = EQUITY / EQUITY_PRICE
    mu = float(p_phys @ gross) - R
    vol = np.sqrt(float(p_phys @ (gross - (p_phys @ gross)) ** 2))
    sharpe = mu / vol if vol else 0.0
    return {"seed": seed, "E_m": Em, "sigma_m": sig_m,
            "hj_ratio": sig_m / Em, "max_traded_sharpe": abs(sharpe),
            "bound_satisfied": sig_m / Em >= abs(sharpe) - 1e-9}


# ---------- validation ----------
def validation_checks() -> dict:
    r = complete_with_recession()
    psi = r["psi"]
    checks = {
        "V1_bond_prices": abs(psi.sum() - 1 / R) < 1e-3,
        "V2_equity_prices": abs(psi @ EQUITY - EQUITY_PRICE) < 1e-1,
        "V3_t_is_0.28": abs(r["t"] - 0.28) < 1e-2,
        "V4_psi_star": np.allclose(psi, [0.28, 0.4758, 0.2058], atol=2e-3),
        "V5_q_star": np.allclose(r["q"], [0.2912, 0.4948, 0.2140], atol=3e-3),
        "V6_recession_reprices": abs(psi @ RECESSION - RECESSION_PRICE) < 1e-1,
    }
    checks["ALL_PASS"] = all(checks.values())
    checks["_values"] = {"t": r["t"], "psi": psi.tolist(), "q": r["q"].tolist()}
    return checks


if __name__ == "__main__":
    import json
    print(json.dumps(validation_checks(), indent=2, default=str))
