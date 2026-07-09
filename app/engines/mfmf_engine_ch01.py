"""
MFMF Laboratory Engine — Module 1: Probability and State Space Explorer
=======================================================================
Chapter 1: The Mathematical Architecture of Modern Finance.

One-period, two-state complete market. This module is the shared engine
behind the Chapter 1 notebook, Excel workbook, and webapp dashboard.
All three consume this engine with the book's seeds (2026CCNN), so their
numbers agree by construction.

Book running example (Section 1.3):
    S0 = 100, u = 1.2, d = 0.9, R = 1.05, p = 0.6, call strike K = 105.
Reference results the engine must reproduce:
    state price psi_u = 0.4762, risk-neutral prob q = 0.5,
    call price C0 = 7.1429.

Seed convention: 2026CCNN, CC = chapter (01), NN = experiment/panel.
    20260101  E1  running example / from number to distribution
    20260102  E2  arbitrage-alarm sweep in R
    20260103  E3  P-vs-Q: allocation swings, prices do not
    20260104  E4  third state, no third asset -> price becomes a bound
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

SEED_BASE = 20260100  # chapter 1


@dataclass
class MiniMarket:
    """Complete one-period two-state market of Section 1.3."""
    S0: float = 100.0
    u: float = 1.2
    d: float = 0.9
    R: float = 1.05
    p: float = 0.6        # physical probability of the up state
    B0: float = 1.0

    # ---- primitives ----
    @property
    def Su(self) -> float:
        return self.S0 * self.u

    @property
    def Sd(self) -> float:
        return self.S0 * self.d

    def no_arbitrage(self) -> bool:
        """Theorem 1.6: a state-price vector exists iff d < R < u."""
        return self.d < self.R < self.u

    # ---- state prices (psi), risk-neutral (q), SDF (m) ----
    def state_prices(self) -> tuple[float, float]:
        """Solve the 2x2 replication/pricing system for (psi_u, psi_d).

        Bond:  psi_u * R      + psi_d * R      = 1
        Stock: psi_u * S0*u   + psi_d * S0*d   = S0
        """
        A = np.array([[self.R, self.R],
                      [self.Su, self.Sd]], dtype=float)
        b = np.array([self.B0, self.S0], dtype=float)
        psi = np.linalg.solve(A, b)
        return float(psi[0]), float(psi[1])

    def risk_neutral(self) -> tuple[float, float]:
        """q(omega) = R * psi(omega). Also q = (R - d)/(u - d)."""
        psi_u, psi_d = self.state_prices()
        return self.R * psi_u, self.R * psi_d

    def sdf(self) -> tuple[float, float]:
        """m(omega) = psi(omega) / P(omega)."""
        psi_u, psi_d = self.state_prices()
        return psi_u / self.p, psi_d / (1.0 - self.p)

    # ---- pricing ----
    def price(self, payoff_u: float, payoff_d: float) -> float:
        """Arbitrage-free price of a claim by state prices."""
        psi_u, psi_d = self.state_prices()
        return psi_u * payoff_u + psi_d * payoff_d

    def price_three_ways(self, payoff_u: float, payoff_d: float) -> dict:
        """Same price in the three languages of Table 1.2."""
        psi_u, psi_d = self.state_prices()
        q_u, q_d = self.risk_neutral()
        m_u, m_d = self.sdf()
        by_psi = psi_u * payoff_u + psi_d * payoff_d
        by_q = (q_u * payoff_u + q_d * payoff_d) / self.R
        by_m = self.p * m_u * payoff_u + (1 - self.p) * m_d * payoff_d
        return {"state_prices": by_psi, "risk_neutral": by_q, "sdf": by_m}

    def replicating_portfolio(self, payoff_u: float, payoff_d: float) -> tuple[float, float]:
        """(a shares of stock, b units of bond) that replicate the claim."""
        a = (payoff_u - payoff_d) / (self.Su - self.Sd)
        b = (payoff_u - a * self.Su) / self.R
        return a, b

    def call(self, K: float) -> tuple[float, float]:
        return max(self.Su - K, 0.0), max(self.Sd - K, 0.0)

    def put(self, K: float) -> tuple[float, float]:
        return max(K - self.Su, 0.0), max(K - self.Sd, 0.0)

    # ---- arbitrage witness (Module 1 alarm) ----
    def arbitrage_portfolio(self):
        """When d < R < u fails, return an explicit arbitrage portfolio.

        Returns None if the market is arbitrage-free.
        """
        if self.no_arbitrage():
            return None
        # If R <= d: borrow bond, buy stock. If R >= u: short stock, buy bond.
        if self.R <= self.d:
            # buy 1 share, short S0 bonds: cost 0 today, >=0 payoff, >0 somewhere
            a, bonds = 1.0, -self.S0 / self.B0
        else:  # R >= u
            a, bonds = -1.0, self.S0 / self.B0
        v0 = a * self.S0 + bonds * self.B0
        vu = a * self.Su + bonds * self.B0 * self.R
        vd = a * self.Sd + bonds * self.B0 * self.R
        return {"shares": a, "bonds": bonds, "V0": v0, "Vu": vu, "Vd": vd}


# ---------- guided experiments (seeded) ----------
def E1_number_to_distribution(mkt: MiniMarket | None = None, K: float = 105.0,
                              n: int = 100_000, seed: int = SEED_BASE + 1) -> dict:
    """E1: the call's price is a number; its *payoff* is a distribution.
    Monte-Carlo the payoff under P and confirm the price equals the
    state-price valuation (not the discounted physical mean)."""
    mkt = mkt or MiniMarket()
    rng = np.random.default_rng(seed)
    cu, cd = mkt.call(K)
    draws = rng.random(n) < mkt.p
    payoff = np.where(draws, cu, cd)
    q_u, q_d = mkt.risk_neutral()
    return {
        "seed": seed,
        "price_state": mkt.price(cu, cd),
        "payoff_mean_P": float(payoff.mean()),
        "discounted_mean_P": float(payoff.mean() / mkt.R),
        "price_riskneutral": (q_u * cu + q_d * cd) / mkt.R,
        "payoff_std_P": float(payoff.std()),
        "cu": cu, "cd": cd,
    }


def E2_arbitrage_sweep(seed: int = SEED_BASE + 2, n_grid: int = 61) -> dict:
    """E2: sweep R across [0.85, 1.25]; record psi_u and where the alarm fires."""
    Rs = np.linspace(0.85, 1.25, n_grid)
    psi_u, alarms = [], []
    for R in Rs:
        m = MiniMarket(R=float(R))
        if m.no_arbitrage():
            pu, _ = m.state_prices()
            psi_u.append(pu); alarms.append(False)
        else:
            psi_u.append(np.nan); alarms.append(True)
    return {"seed": seed, "R": Rs.tolist(), "psi_u": psi_u,
            "alarm": alarms, "d": 0.9, "u": 1.2}


def E3_P_vs_Q(gamma: float = 3.0, seed: int = SEED_BASE + 3, n_grid: int = 41) -> dict:
    """E3: vary p in (0,1); prices are fixed but the optimal stock
    allocation alpha* swings, crossing zero exactly at p = q."""
    mkt = MiniMarket()
    q_u, _ = mkt.risk_neutral()
    ps = np.linspace(0.05, 0.95, n_grid)
    # log-utility optimal fraction in the risky asset (illustrative closed form):
    # alpha* > 0 iff p > q; equals 0 at p = q.
    Ru, Rd = mkt.u / mkt.R, mkt.d / mkt.R  # gross risky returns relative to bond
    alpha = []
    for p in ps:
        # log-utility: maximize p ln(1+a(Ru-1)) + (1-p) ln(1+a(Rd-1))
        num = p * (Ru - 1) * (1 - Rd) + (1 - p) * (Rd - 1) * (1 - Ru)
        den = -(Ru - 1) * (Rd - 1)
        a = num / den if den != 0 else 0.0
        alpha.append(a)
    return {"seed": seed, "p": ps.tolist(), "alpha_star": alpha,
            "q": q_u, "price_fixed": mkt.price(*mkt.call(105))}


def E4_third_state_bound(seed: int = SEED_BASE + 4) -> dict:
    """E4: add a third state with no third asset. The unique price of a
    non-replicable claim dissolves into an interval [lower, upper]."""
    S0, R = 100.0, 1.05
    payoffs_stock = np.array([120.0, 100.0, 90.0])   # 3 states, 2 assets
    payoffs_bond = np.array([R, R, R])
    # A digital claim paying 1 only in the (new) middle state is NOT in the
    # span of {bond, stock}, so its no-arbitrage price is an interval.
    claim = np.array([0.0, 1.0, 0.0])
    # state prices psi >= 0 with A^T psi = prices; feasible set is a segment.
    # Solve LP bounds: min/max psi.claim s.t. psi>=0, R*sum(psi)=1, stock reprices.
    from scipy.optimize import linprog
    Aeq = np.vstack([payoffs_bond, payoffs_stock])
    beq = np.array([1.0, S0])
    lo = linprog(claim, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * 3)
    hi = linprog(-claim, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * 3)
    return {"seed": seed,
            "lower": float(lo.fun) if lo.success else None,
            "upper": float(-hi.fun) if hi.success else None,
            "note": "unique price -> interval once the market is incomplete"}


# ---------- validation checks (book) ----------
def validation_checks() -> dict:
    """The book's Module 1 validation checks. All must pass."""
    m = MiniMarket()
    psi_u, psi_d = m.state_prices()
    q_u, q_d = m.risk_neutral()
    cu, cd = m.call(105.0)
    C0 = m.price(cu, cd)
    three = m.price_three_ways(cu, cd)
    checks = {
        "V1_no_arbitrage": m.no_arbitrage() is True,
        "V2_psi_u": abs(psi_u - 0.47619) < 1e-4,
        "V3_q_half": abs(q_u - 0.5) < 1e-9,
        "V4_three_languages_agree":
            abs(three["state_prices"] - three["risk_neutral"]) < 1e-9 and
            abs(three["state_prices"] - three["sdf"]) < 1e-9,
        "V5_call_price": abs(C0 - 7.142857) < 1e-4,
        "V6_bond_prices_to_1": abs(m.price(1, 1) - 1 / m.R) < 1e-12,
    }
    checks["ALL_PASS"] = all(checks.values())
    checks["_values"] = {"psi_u": psi_u, "psi_d": psi_d, "q_u": q_u,
                         "C0": C0, "cu": cu, "cd": cd}
    return checks


if __name__ == "__main__":
    import json
    print("Validation:", json.dumps(validation_checks(), indent=2, default=str))
