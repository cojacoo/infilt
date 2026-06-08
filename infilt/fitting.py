"""
Fit Philip (1957) two-term infiltration equation to infiltrometer data.

I = C₁ √t + C₂ t

Five fitting strategies
-----------------------
OLS  — ordinary least squares (default)
NLS  — nonlinear least squares with OLS warm start
DL   — differentiated linearisation (Vandervaere et al. 2000)
SU   — Su (2025) generalised Philip: I = A·t + F·t^β, β free
ML   — Mittag-Leffler / Guo et al. (2026): f = fc + b·E_α(−λt^α)

References
----------
Philip (1957) Soil Science 84:257–264.
Vandervaere et al. (2000) SSSAJ 64:1272–1284.
Su (2025) Scientific Reports 15:20396.
Guo et al. (2026) J. Hydrology 674:135443.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gamma as _gamma, gammaln as _gammaln
from scipy.integrate import cumulative_trapezoid as _cumtrapz


@dataclass
class PhilipFit:
    """Results of a Philip two-term infiltration fit."""

    C1: float
    """Sorptivity-related coefficient C₁ [cm s⁻½]."""

    C2: float
    """Gravity-related coefficient C₂ [cm s⁻¹]."""

    r2: float
    """Coefficient of determination R²."""

    rmse: float
    """Root-mean-squared error [cm]."""

    method: str
    """Fitting method: 'ols', 'nls', 'dl', 'su', or 'ml'."""

    I_fit: np.ndarray = field(repr=False)
    """Fitted cumulative infiltration at each observed time [cm]."""

    beta: float = 0.5
    """Fractional exponent β/α.  0.5 for OLS/NLS/DL; free for 'su'/'ml'."""

    lambda_s: float = 0.0
    """ML timescale λ [s^{−α}].  Meaningful only for method='ml'."""

    degenerate: bool = False
    """True when Su/ML fit collapsed and OLS was substituted."""

    @property
    def valid(self) -> bool:
        """True when C₂ > 0 (physically meaningful gravity term)."""
        return self.C2 > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r2(y_obs: np.ndarray, y_fit: np.ndarray) -> float:
    ss_res = np.sum((y_obs - y_fit) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ---------------------------------------------------------------------------
# Individual fitting routines
# ---------------------------------------------------------------------------

def fit_philip_ols(time_s: np.ndarray, I_cm: np.ndarray) -> PhilipFit:
    """OLS fit I = C₁√t + C₂t (intercept forced through origin)."""
    sqrt_t = np.sqrt(time_s)
    X = np.column_stack([sqrt_t, time_s])
    coeffs, *_ = np.linalg.lstsq(X, I_cm, rcond=None)
    C1, C2 = float(coeffs[0]), float(coeffs[1])
    I_fit = C1 * sqrt_t + C2 * time_s
    return PhilipFit(
        C1=C1, C2=C2,
        r2=_r2(I_cm, I_fit),
        rmse=float(np.sqrt(np.mean((I_cm - I_fit) ** 2))),
        method='ols', I_fit=I_fit,
    )


def fit_philip_nls(
    time_s: np.ndarray,
    I_cm: np.ndarray,
    p0: tuple[float, float] | None = None,
) -> PhilipFit:
    """NLS fit via scipy.optimize.curve_fit, warm-started from OLS."""
    def _philip(t, C1, C2):
        return C1 * np.sqrt(t) + C2 * t

    if p0 is None:
        ols = fit_philip_ols(time_s, I_cm)
        p0 = (max(ols.C1, 1e-8), ols.C2)

    mask = time_s > 0
    try:
        popt, _ = curve_fit(
            _philip, time_s[mask], I_cm[mask],
            p0=p0, maxfev=10_000,
            bounds=([0.0, -np.inf], [np.inf, np.inf]),
        )
        C1, C2 = float(popt[0]), float(popt[1])
    except RuntimeError:
        C1, C2 = p0

    sqrt_t = np.sqrt(time_s)
    I_fit = C1 * sqrt_t + C2 * time_s
    return PhilipFit(
        C1=C1, C2=C2,
        r2=_r2(I_cm, I_fit),
        rmse=float(np.sqrt(np.mean((I_cm - I_fit) ** 2))),
        method='nls', I_fit=I_fit,
    )


def fit_philip_dl(time_s: np.ndarray, I_cm: np.ndarray) -> PhilipFit:
    """
    Differentiated Linearization (Vandervaere et al. 2000).

    Fits dI/d(√t) = C₁ + 2 C₂ √t.  Recommended for fine-textured soils.
    """
    if len(time_s) < 5:
        raise ValueError("DL method needs ≥ 5 data points.")

    sqrt_t = np.sqrt(time_s)
    dI_dsqrtt = np.gradient(I_cm, sqrt_t)
    x = sqrt_t[1:-1]
    y = dI_dsqrtt[1:-1]
    X = np.column_stack([np.ones_like(x), x])
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    C1 = float(coeffs[0])
    C2 = float(coeffs[1]) / 2.0

    I_fit = C1 * sqrt_t + C2 * time_s
    return PhilipFit(
        C1=C1, C2=C2,
        r2=_r2(I_cm, I_fit),
        rmse=float(np.sqrt(np.mean((I_cm - I_fit) ** 2))),
        method='dl', I_fit=I_fit,
    )


def _ea1_hybrid(alpha: float, x_arr: np.ndarray) -> np.ndarray:
    """E_{alpha,1}(-x) for x ≥ 0.  Hybrid series/asymptotic evaluation."""
    x = np.asarray(x_arr, dtype=float)
    if alpha >= 0.999:
        return np.exp(-x)

    x_cross = alpha * 6.0
    result = np.empty_like(x)
    sm = x <= x_cross
    la = ~sm

    if sm.any():
        xs = x[sm]
        s = np.ones_like(xs)
        sign = -1.0
        for k in range(1, 200):
            lx = k * np.log(np.where(xs > 0, xs, 1e-300)) - _gammaln(alpha * k + 1)
            term = sign * np.exp(np.where(xs > 0, lx, -np.inf))
            s += term
            sign *= -1
            if k > 8 and np.max(np.abs(term)) < 1e-13 * (np.max(np.abs(s)) + 1e-300):
                break
        result[sm] = np.clip(s, 0.0, None)

    if la.any():
        result[la] = np.clip(1.0 / (x[la] * _gamma(1.0 - alpha)), 0.0, None)

    return result


def fit_philip_ml(time_s: np.ndarray, I_cm: np.ndarray) -> PhilipFit:
    """
    Mittag-Leffler model — Guo et al. (2026).

    Rate form:  f(t) = fc + b · E_{α}(−λ t^α)
    Cumulative: I(t) = fc·t + b · ∫₀ᵗ E_{α,1}(−λ τ^α) dτ

    PhilipFit mapping: C₂=fc, C₁=b, beta=α, lambda_s=λ.
    """
    _N_FINE = 300

    def _I_model(t_arr, fc, b, lam, alpha):
        t_max = t_arr[-1]
        t_fine = np.linspace(0, t_max, _N_FINE)
        x_fine = lam * np.where(t_fine > 0, t_fine ** alpha, 0.0)
        ea1_fine = _ea1_hybrid(alpha, x_fine)
        I_cap = b * _cumtrapz(ea1_fine, t_fine, initial=0.0) + fc * t_fine
        return np.interp(t_arr, t_fine, I_cap)

    ols = fit_philip_ols(time_s, I_cm)
    alpha0 = 0.7
    t_mid = max(time_s[len(time_s) // 2], 1.0)
    lam0 = 1.0 / t_mid ** alpha0
    p0 = [max(ols.C2, 1e-9), max(ols.C1, 1e-9), lam0, alpha0]

    mask = time_s > 0
    degen = False
    try:
        popt, _ = curve_fit(
            _I_model, time_s[mask], I_cm[mask],
            p0=p0,
            bounds=([0.0, 0.0, 1e-8, 0.05], [np.inf, np.inf, 1e4, 1.0]),
            maxfev=8_000, method='trf', x_scale='jac',
        )
        fc, b, lam, alpha = [float(v) for v in popt]
    except RuntimeError:
        fc, b, lam, alpha = float(ols.C2), float(ols.C1), lam0, alpha0
        degen = True

    I_fit = _I_model(time_s, fc, b, lam, alpha)
    return PhilipFit(
        C1=b, C2=fc, beta=alpha, lambda_s=lam,
        r2=_r2(I_cm, I_fit),
        rmse=float(np.sqrt(np.mean((I_cm - I_fit) ** 2))),
        method='ml', I_fit=I_fit, degenerate=degen,
    )


def fit_philip_su(time_s: np.ndarray, I_cm: np.ndarray) -> PhilipFit:
    """
    Su (2025) generalised Philip: I = A·t + F·t^β, β free.

    PhilipFit mapping: C₂=A (gravity→K), C₁=F (capillary), beta=β.
    β ≈ 0.5 → standard Philip; β<0.5 → sub-diffusion; β→1 → gravity-dominated.
    """
    def _su(t, A, F, beta):
        return A * t + F * t ** beta

    ols = fit_philip_ols(time_s, I_cm)
    p0 = [max(ols.C2, 1e-9), max(ols.C1, 1e-9), 0.5]

    mask = time_s > 0
    degen = False
    try:
        popt, _ = curve_fit(
            _su, time_s[mask], I_cm[mask],
            p0=p0,
            bounds=([0.0, 0.0, 0.05], [np.inf, np.inf, 1.0]),
            maxfev=10_000,
        )
        A, F, beta = float(popt[0]), float(popt[1]), float(popt[2])
    except RuntimeError:
        A, F, beta = float(ols.C2), float(ols.C1), 0.5
        degen = True

    if A < max(ols.C2 * 0.01, 1e-12):
        warnings.warn(
            "Su fit degenerate: gravity term A → 0.  "
            "Run too short to separate A from F·t^β.  "
            f"Fitted β={beta:.3f} not reliable.  Falling back to OLS K.",
            stacklevel=3,
        )
        return PhilipFit(
            C1=ols.C1, C2=ols.C2, beta=beta,
            r2=ols.r2, rmse=ols.rmse,
            method='su', I_fit=ols.I_fit, degenerate=True,
        )

    I_fit = np.where(time_s > 0, _su(time_s, A, F, beta), 0.0)
    return PhilipFit(
        C1=F, C2=A, beta=beta,
        r2=_r2(I_cm, I_fit),
        rmse=float(np.sqrt(np.mean((I_cm - I_fit) ** 2))),
        method='su', I_fit=I_fit, degenerate=degen,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_METHODS = {
    'ols': fit_philip_ols,
    'nls': fit_philip_nls,
    'dl':  fit_philip_dl,
    'su':  fit_philip_su,
    'ml':  fit_philip_ml,
}


def fit_philip(time_s, I_cm, method: str = 'ols') -> PhilipFit:
    """
    Fit Philip (1957) two-term equation I = C₁√t + C₂t.

    Parameters
    ----------
    time_s : array-like
        Elapsed time [s], starting at 0.
    I_cm : array-like
        Cumulative infiltration [cm], starting at 0.
    method : {'ols', 'nls', 'dl', 'su', 'ml'}
        Fitting strategy.

    Returns
    -------
    PhilipFit
    """
    time_s = np.asarray(time_s, dtype=float)
    I_cm = np.asarray(I_cm, dtype=float)

    if method not in _METHODS:
        raise ValueError(f"method must be one of {list(_METHODS)}; got '{method}'")
    if len(time_s) < 3:
        raise ValueError("At least 3 time–infiltration pairs required.")

    return _METHODS[method](time_s, I_cm)
