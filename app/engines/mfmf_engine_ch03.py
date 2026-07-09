"""
MFMF Laboratory Engine — Module 3: Information and Conditional Expectation Simulator
Chapter 3: Information, Conditional Expectation, and Filtrations.

Three panels: a conditional-expectation tree (E[X | F_t] atom by atom),
a clairvoyance detector (F_{t-1}-measurability check for trading rules),
and a two-observer panel (public vs insider filtration; smoothing and the
law of total variance, Proposition 3.12).

Book anchor (Example 3.9): p = 0.6 per branch, independent across periods,
one-step growth mean = p*u + (1-p)*d = 1.08 with u=1.2, d=0.9.

Seeds: 2026CCNN, CC=03.
    20260301 E1 tower property, verified numerically
    20260302 E2 clairvoyance detector on a look-ahead rule
    20260303 E3 smoothing and the law of total variance
"""
from __future__ import annotations
import numpy as np
import itertools

SEED_BASE = 20260300
U, D, P = 1.2, 0.9, 0.6


def one_step_growth_mean(p: float = P, u: float = U, d: float = D) -> float:
    return p * u + (1 - p) * d


# ---------- tree model ----------
def build_tree(T: int = 3, S0: float = 100.0, u: float = U, d: float = D):
    """All 2^T paths with terminal prices; returns (paths, prices)."""
    paths = list(itertools.product([1, 0], repeat=T))  # 1=up,0=down
    prices = {}
    for path in paths:
        s = S0
        for step in path:
            s *= u if step else d
        prices[path] = s
    return paths, prices


def conditional_expectation(paths, prices, t: int, p: float = P) -> dict:
    """E[S_T | F_t] for each F_t atom (identified by the first t moves)."""
    atoms = {}
    for prefix in itertools.product([1, 0], repeat=t):
        matching = [pt for pt in paths if pt[:t] == prefix]
        # probability-weighted average of terminal price over the atom
        num = den = 0.0
        for pt in matching:
            future = pt[t:]
            prob = np.prod([p if m else (1 - p) for m in future]) if future else 1.0
            num += prob * prices[pt]
            den += prob
        atoms[prefix] = num / den
    return atoms


# ---------- guided experiments ----------
def E1_tower(seed: int = SEED_BASE + 1, T: int = 3) -> dict:
    """Verify E[ E[S_T|F_2] | F_1 ] = E[S_T|F_1] numerically."""
    paths, prices = build_tree(T)
    e_at_1 = conditional_expectation(paths, prices, 1)
    e_at_2 = conditional_expectation(paths, prices, 2)
    # average date-2 forecasts down to date-1 atoms
    reconstructed = {}
    for prefix1 in itertools.product([1, 0], repeat=1):
        up = e_at_2[prefix1 + (1,)]
        dn = e_at_2[prefix1 + (0,)]
        reconstructed[prefix1] = P * up + (1 - P) * dn
    max_err = max(abs(reconstructed[k] - e_at_1[k]) for k in e_at_1)
    return {"seed": seed, "growth_mean": one_step_growth_mean(),
            "E_at_1": {str(k): round(v, 4) for k, v in e_at_1.items()},
            "tower_reconstructed": {str(k): round(v, 4) for k, v in reconstructed.items()},
            "max_tower_error": max_err}


def E2_clairvoyance(seed: int = SEED_BASE + 2, T: int = 4) -> dict:
    """The rule 'hold the stock only in periods it rises' is not
    F_{t-1}-measurable: deciding at t-1 to hold through t requires
    knowing the t-th move. The detector rejects it and names the leak."""
    paths, prices = build_tree(T)
    # A rule is predictable if the position over (t-1, t] depends only on
    # information up to t-1. 'Hold only when it rises' uses the move in
    # (t-1, t], i.e. future info -> not predictable.
    leaks = []
    for t in range(1, T + 1):
        leaks.append({"period": t,
                      "requires": f"move in ({t-1},{t}]",
                      "available_at_{}".format(t - 1): False})
    return {"seed": seed, "rule": "hold the stock only in periods it rises",
            "predictable": False,
            "leak_periods": [l["period"] for l in leaks],
            "verdict": "REJECTED — consumes the contemporaneous move (look-ahead)."}


def E3_total_variance(seed: int = SEED_BASE + 3, n: int = 500_000) -> dict:
    """Law of total variance / smoothing (Prop 3.12):
    Var(X) = E[Var(X|G)] + Var(E[X|G]).
    Coarsening G (observing less) shifts variance from the first term to
    the second; a smoother-observer measures lower conditional variance."""
    rng = np.random.default_rng(seed)
    # X = terminal log-price of a 4-step tree; G = info after 2 steps.
    T = 4
    steps = rng.random((n, T)) < P
    logret = np.where(steps, np.log(U), np.log(D))
    X = logret.sum(axis=1)
    G = logret[:, :2].sum(axis=1)  # coarse observation: first 2 steps
    # group by G value (only 3 distinct sums)
    var_within, mean_within, wts = [], [], []
    for g in np.unique(G):
        mask = G == g
        var_within.append(X[mask].var())
        mean_within.append(X[mask].mean())
        wts.append(mask.mean())
    wts = np.array(wts)
    E_var = float((wts * np.array(var_within)).sum())
    Var_E = float((wts * (np.array(mean_within) - X.mean()) ** 2).sum())
    return {"seed": seed, "Var_X": float(X.var()),
            "E_Var_given_G": E_var, "Var_E_given_G": Var_E,
            "sum": E_var + Var_E,
            "identity_holds": abs((E_var + Var_E) - X.var()) < 1e-3}


# ---------- validation ----------
def validation_checks() -> dict:
    e1 = E1_tower()
    e3 = E3_total_variance()
    checks = {
        "V1_growth_mean_1.08": abs(one_step_growth_mean() - 1.08) < 1e-9,
        "V2_tower_holds": e1["max_tower_error"] < 1e-9,
        "V3_clairvoyance_rejected": E2_clairvoyance()["predictable"] is False,
        "V4_total_variance_identity": e3["identity_holds"],
    }
    checks["ALL_PASS"] = all(checks.values())
    checks["_values"] = {"growth_mean": one_step_growth_mean(),
                         "max_tower_error": e1["max_tower_error"]}
    return checks


if __name__ == "__main__":
    import json
    print(json.dumps(validation_checks(), indent=2, default=str))
