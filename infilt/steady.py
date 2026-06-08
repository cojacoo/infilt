"""
Steady-state detection for infiltration rate time series.

Primary strategy: rolling CV scan (backward from end of series).
q_ss = mean of first stable window found; q_ss_se = std of ALL
masked points (every time step within q_ss ± tol).

Horton f = fc + b·exp(−λt) is always fitted for diagnostics
(conv_frac, t95_s, R²) but is NOT used for q_ss unless the CV scan
fails to find a stable window (series genuinely not converged).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class SteadyStateResult:
    """
    Steady-state detection results.

    Attributes
    ----------
    q_ss : float
        Steady-state flux [cm s⁻¹].
        CV-scan mean when a stable window is found; Horton fc otherwise.
    q_ss_se : float
        Standard deviation of all masked (steady-state) measurements
        [cm s⁻¹].  Represents flux variability at steady state.
    mask : ndarray of bool
        All time steps where |rate − q_ss| ≤ q_ss_se.
        Used for visualisation.
    conv_frac : float
        Fraction of measurements in the steady-state band = mask.sum()/n.
        1.0 when series is flat throughout.
    t95_s : float
        Horton-derived time to 95 % convergence [s].
        inf when Horton fit is poor or series is already converged.
    method : str
        'cv_window' | 'horton' | 'tail_mean'
    horton_fc : float
        Horton asymptote fc [cm s⁻¹] (diagnostic only).
    horton_r2 : float
        Horton fit R² (diagnostic only).
    """
    q_ss: float
    q_ss_se: float
    mask: np.ndarray = field(repr=False)
    conv_frac: float
    t95_s: float
    method: str
    horton_fc: float
    horton_r2: float


def _fit_horton(
    time_s: np.ndarray,
    rate_cm_s: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Fit Horton f = fc + b·exp(−λt).  Returns (fc, lam, r2, t95_s).
    Used for diagnostics only; not for q_ss when CV scan succeeds.
    """
    n = len(rate_cm_s)
    tail = slice(max(0, int(0.8 * n)), None)
    fc_init = float(np.mean(rate_cm_s[tail]))
    b_init  = max(float(rate_cm_s[0] - fc_init), 1e-12)
    lam_init = 1.0 / max(float(time_s[-1]) / 3.0, 1.0)

    def _h(t, fc, b, lam):
        return fc + b * np.exp(-lam * t)

    try:
        popt, pcov = curve_fit(
            _h, time_s, rate_cm_s,
            p0=[fc_init, b_init, lam_init],
            bounds=([0.0, 0.0, 1e-8], [np.inf, np.inf, np.inf]),
            maxfev=4_000,
        )
        fc, b, lam = float(popt[0]), float(popt[1]), float(popt[2])
        ss_res = float(np.sum((rate_cm_s - _h(time_s, fc, b, lam)) ** 2))
        ss_tot = float(np.sum((rate_cm_s - np.mean(rate_cm_s)) ** 2))
        r2    = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        t95_s = float(-np.log(0.05) / lam)
        return fc, lam, r2, t95_s
    except RuntimeError:
        return fc_init, lam_init, 0.0, float('inf')


def detect_steady_state(
    time_s: np.ndarray,
    rate_cm_s: np.ndarray,
    window: int = 4,
    cv_threshold: float = 0.05,
) -> SteadyStateResult:
    """
    Identify steady-state flux and quantify convergence.

    Algorithm
    ---------
    1. Rolling CV scan (backward): first window with CV ≤ cv_threshold
       defines q_ss = window mean, std_cv = window std.
    2. Mask: ALL time steps where |rate − q_ss| ≤ max(std_cv, 2 % of q_ss).
       This highlights every measurement consistent with steady state,
       not just the discovery window.
    3. q_ss_se = std of all masked points.
    4. conv_frac = mask.sum() / n  (fraction of series at steady state).
    5. Horton fit for t95_s diagnostic (always; not used for q_ss).
    6. If no CV window found: Horton fc used as q_ss (series not converged
       within measurement window).  Warns to extend measurement.

    Parameters
    ----------
    time_s : ndarray
        Time [s].
    rate_cm_s : ndarray
        Infiltration rate [cm s⁻¹].
    window : int
        Rolling window width (default 4).
    cv_threshold : float
        Maximum CV for steady-state criterion (default 0.05 = 5 %).

    Returns
    -------
    SteadyStateResult
    """
    time_s    = np.asarray(time_s,    dtype=float)
    rate_cm_s = np.asarray(rate_cm_s, dtype=float)
    n = len(rate_cm_s)

    # ── Horton fit (diagnostics) ──────────────────────────────────────────
    fc_h, lam_h, r2_h, t95_h = _fit_horton(time_s, rate_cm_s)
    tail_mean = float(np.mean(rate_cm_s[max(0, int(0.8 * n)):]))

    # ── Primary: rolling CV scan ──────────────────────────────────────────
    q_ss   = None
    std_cv = None
    for start in range(n - window, -1, -1):
        win = rate_cm_s[start:start + window]
        mu  = float(np.mean(win))
        if mu <= 0:
            continue
        sd = float(np.std(win))
        if sd / mu <= cv_threshold:
            q_ss   = mu
            std_cv = sd
            break

    method = 'cv_window'

    if q_ss is None:
        # ── Fallback A: Horton (if fit is sensible and fc not degenerate) ──
        # Degenerate: fc collapses to ~0 when data is already flat.
        # Guard: fc must be ≥ 30 % of tail mean.
        if r2_h >= 0.5 and fc_h >= 0.3 * tail_mean and fc_h > 0:
            q_ss   = float(fc_h)
            std_cv = None
            method = 'horton'
            warnings.warn(
                f"No stable CV window found; using Horton asymptote "
                f"fc={q_ss*36000:.2f} mm/h (R²={r2_h:.3f}).  "
                "Extend measurement duration for a direct steady-state reading.",
                stacklevel=2,
            )
        else:
            # ── Fallback B: tail mean ──────────────────────────────────────
            tail_sl = slice(max(0, int(0.8 * n)), None)
            q_ss    = float(np.mean(rate_cm_s[tail_sl]))
            std_cv  = float(np.std(rate_cm_s[tail_sl]))
            method  = 'tail_mean'
            warnings.warn(
                "Steady-state detection fell back to tail mean "
                f"q_ss={q_ss*36000:.2f} mm/h.  "
                "Data may not have reached steady state.",
                stacklevel=2,
            )

    # ── Mask: all points within the steady-state band ────────────────────
    tol = max(std_cv if std_cv is not None else 0.0,
              0.02 * q_ss if q_ss > 0 else 0.0)
    if tol == 0.0:
        tol = float(np.std(rate_cm_s)) * 0.5 or 1e-10

    mask = np.abs(rate_cm_s - q_ss) <= tol
    if not mask.any():
        mask[max(0, n - window):] = True

    # ── q_ss_se = std of all steady-state measurements ────────────────────
    q_ss_se = float(np.std(rate_cm_s[mask])) if mask.sum() > 1 else tol

    # ── conv_frac: fraction of series at steady state ─────────────────────
    conv_frac = float(mask.sum()) / n

    # ── t95_s: Horton diagnostic — reliable only when Horton fit is good ──
    t95_s = t95_h if (r2_h >= 0.5 and fc_h >= 0.3 * tail_mean) else float('inf')

    return SteadyStateResult(
        q_ss=float(q_ss),
        q_ss_se=float(q_ss_se),
        mask=mask,
        conv_frac=conv_frac,
        t95_s=t95_s,
        method=method,
        horton_fc=float(fc_h),
        horton_r2=r2_h,
    )
