"""
MFMF Laboratory Engine — Module 12: The Learning Machine
Chapter 12: Filtering, Learning, and Hidden States.

Four panels: the update panel (Gaussian prior x signal -> posterior), the
Kalman panel (state, estimate, band, innovation whiteness), the de-smoothing
panel (AR estimate of theta, unsmooth marks), and the drift panel (learning
the drift; plug-in vs filtered weights).

Book anchors (Proposition 12.1 / Example 12.x):
    prior N($153M, $4M^2), signal N($148M, $6M^2) fuse to a nowcast of
    $151.5M with a +/- $3.3M (one-sd) band, weight 0.69 on the mark.
    De-smoothing recovers theta = 0.8 from an 18% (6% reported) mark series.

Seeds: 2026CCNN, CC=12.
    20261201 E1 the $151.5M +/- $3.3M nowcast
    20261202 E2 the Kalman filter and innovation whiteness
    20261203 E3 de-smoothing: recover theta = 0.8
    20261204 E4 learning the drift; plug-in vs filtered
"""
from __future__ import annotations
import numpy as np

SEED_BASE = 20261200


# ---------- E1: Gaussian fusion ----------
def gaussian_fuse(prior_mean, prior_sd, signal_mean, signal_sd):
    """Precision-weighted product of two Gaussians (Proposition 12.1)."""
    pp, ps = 1 / prior_sd ** 2, 1 / signal_sd ** 2
    post_var = 1 / (pp + ps)
    post_mean = post_var * (pp * prior_mean + ps * signal_mean)
    weight_on_prior = pp / (pp + ps)
    return post_mean, np.sqrt(post_var), weight_on_prior


def E1_nowcast(seed: int = SEED_BASE + 1) -> dict:
    mean, sd, w_prior = gaussian_fuse(153.0, 4.0, 148.0, 6.0)
    return {"seed": seed, "nowcast_mean": mean, "nowcast_sd": sd,
            "weight_on_prior": w_prior, "weight_on_signal": 1 - w_prior,
            "target_mean": 151.5, "target_sd": 3.3}


# ---------- E2: Kalman filter ----------
def kalman_filter(y, a, q, sigma_v, x0=0.0, p0=1.0):
    """Scalar Kalman filter: x_t = a x_{t-1} + w, y_t = x_t + v."""
    n = len(y)
    xhat = np.zeros(n); P = np.zeros(n); innov = np.zeros(n); gain = np.zeros(n)
    x_prev, p_prev = x0, p0
    for t in range(n):
        x_pred = a * x_prev
        p_pred = a ** 2 * p_prev + q
        S = p_pred + sigma_v ** 2
        K = p_pred / S
        innov[t] = y[t] - x_pred
        xhat[t] = x_pred + K * innov[t]
        P[t] = (1 - K) * p_pred
        gain[t] = K
        x_prev, p_prev = xhat[t], P[t]
    return {"xhat": xhat, "P": P, "innov": innov, "gain": gain}


def E2_kalman(seed: int = SEED_BASE + 2, n: int = 500,
              a: float = 0.95, q: float = 0.02, sigma_v: float = 0.20) -> dict:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = a * x[t - 1] + np.sqrt(q) * rng.standard_normal()
    y = x + sigma_v * rng.standard_normal(n)
    # correctly specified
    kf = kalman_filter(y, a, q, sigma_v)
    innov = kf["innov"][10:]
    ac1 = np.corrcoef(innov[:-1], innov[1:])[0, 1]
    # mis-specified sigma_v (x4): innovations become autocorrelated
    kf_bad = kalman_filter(y, a, q, sigma_v * 4)
    ib = kf_bad["innov"][10:]
    ac1_bad = np.corrcoef(ib[:-1], ib[1:])[0, 1]
    rmse = np.sqrt(np.mean((kf["xhat"] - x) ** 2))
    return {"seed": seed, "rmse": float(rmse),
            "innov_autocorr_correct": float(ac1),
            "innov_autocorr_misspec": float(ac1_bad),
            "whiteness_ok": abs(ac1) < 0.1}


# ---------- E3: de-smoothing ----------
def E3_desmooth(seed: int = SEED_BASE + 3, theta: float = 0.8,
                true_vol: float = 0.18, n: int = 2000) -> dict:
    """Reported r_t = (1-theta) r*_t + theta r_{t-1}; recover theta by AR(1)
    regression and unsmooth. Reported vol ~ 6% understates the true 18%."""
    rng = np.random.default_rng(seed)
    true_ret = true_vol * rng.standard_normal(n)
    reported = np.zeros(n)
    for t in range(1, n):
        reported[t] = (1 - theta) * true_ret[t] + theta * reported[t - 1]
    # AR(1) on reported gives theta_hat
    theta_hat = np.corrcoef(reported[:-1], reported[1:])[0, 1]
    unsmoothed = np.zeros(n)
    for t in range(1, n):
        unsmoothed[t] = (reported[t] - theta_hat * reported[t - 1]) / (1 - theta_hat)
    return {"seed": seed, "theta_true": theta, "theta_hat": float(theta_hat),
            "reported_vol": float(reported.std()),
            "unsmoothed_vol": float(unsmoothed.std()),
            "true_vol": true_vol,
            "recovers_theta": abs(theta_hat - theta) < 0.08}


# ---------- E4: learning the drift ----------
def E4_drift(seed: int = SEED_BASE + 4, s0: float = 0.02, sigma: float = 0.175,
             years: int = 20) -> dict:
    """Prior sd s0 on the drift; the data weight grows with the horizon.
    tau0 = prior precision; posterior weight on data = T/sigma^2 / (1/s0^2 + T/sigma^2)."""
    tau0 = 1 / s0 ** 2
    data_prec = years / sigma ** 2
    data_weight = data_prec / (tau0 + data_prec)
    return {"seed": seed, "tau0": tau0, "data_weight_20y": data_weight,
            "prior_weight_20y": 1 - data_weight}


# ---------- validation ----------
def validation_checks() -> dict:
    e1 = E1_nowcast(); e2 = E2_kalman(); e3 = E3_desmooth()
    checks = {
        "V1_nowcast_151.5": abs(e1["nowcast_mean"] - 151.5) < 0.1,
        "V2_nowcast_sd_3.3": abs(e1["nowcast_sd"] - 3.3) < 0.1,
        "V3_weight_on_prior_0.69": abs(e1["weight_on_prior"] - 0.69) < 0.01,
        "V4_kalman_innovation_white": e2["whiteness_ok"],
        "V5_desmooth_recovers_theta": e3["recovers_theta"],
    }
    checks["ALL_PASS"] = all(checks.values())
    checks["_values"] = {"nowcast": e1["nowcast_mean"], "sd": e1["nowcast_sd"],
                         "w_prior": e1["weight_on_prior"],
                         "theta_hat": e3["theta_hat"]}
    return checks


if __name__ == "__main__":
    import json
    print(json.dumps(validation_checks(), indent=2, default=str))
