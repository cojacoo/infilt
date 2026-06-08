"""
Theory for mini-disk tension infiltrometer K estimation.

Implements the Zhang (1997) semi-empirical approach and the Dohnal et al.
(2010) extension for soils with wide pore-size distributions (n < 1.35).

Physical setup
--------------
A disk infiltrometer maintains a prescribed negative pressure head h₀ < 0
(suction) at the soil surface via a Mariotte bottle.  Transient cumulative
infiltration I(t) beneath the disk follows Philip's (1957) two-term
approximation:

    I = C₁ √t + C₂ t                                           (1)

where C₁ [cm s⁻½] is dominated by capillary sorptivity and C₂ [cm s⁻¹]
by gravity (hydraulic conductivity).

Zhang (1997) approach
---------------------
Zhang related C₁ and C₂ to the near-saturated hydraulic conductivity K(h₀)
through empirical coefficients A₁ and A₂:

    C₁ = S · A₁     →   S = C₁ / A₁                          (2)
    C₂ = K · A₂     →   K = C₂ / A₂                          (3)

where

    A₁ = 1.4 b^½ (Δθ)^¼ exp(3(n−1.9) α h₀) / (α r₀)^½       (4)

    A₂ = 11.65 (n^0.1 − 1) exp(d₂ (n−1.9) α h₀) / (α r₀)^0.91 (5)

    d₂ = 2.92  for n ≥ 1.9
    d₂ = 7.5   for n < 1.9

with b = 0.55, Δθ = θ(h₀) − θᵢ (volumetric difference), h₀ the pressure
head at the disk surface (negative, [cm]), r₀ the disk radius [cm], α and n
the van Genuchten retention parameters [1/cm, −].

Sign convention
---------------
h₀ is the **signed pressure head** (negative for unsaturated conditions):
    h₀ = −1  cm  corresponds to 1 cm suction (10 mm suction setting)
    h₀ = −3  cm  corresponds to 3 cm suction (30 mm suction setting)
    h₀ =  0  cm  corresponds to ponded / zero-suction conditions

The user-facing ``suction_mm`` parameter (always ≥ 0) is converted to
``h0_cm = −suction_mm / 10`` inside :mod:`minidisk.analysis`.

Dohnal et al. (2010) extension for n < 1.35
--------------------------------------------
For soils with n < 1.35 (fine-textured soils, Cambisols), Eq. (5) yields
large errors because lateral capillary forces dominate.  Dohnal et al.
recalibrated the formula for minidisk-scale experiments (disk area ≈ 15 cm²,
|h₀| ≤ 6 cm):

    K = C₂ (α r₀)^0.60 /
        [11.65 (n^0.82 − 1) exp(34.65 (n − 1.19) α h₀)]       (6)

Average relative error ≈ 10 % (max 31 % UNSODA, 28 % Cambisols).

At h₀ = 0 (ponded), exp(·) = 1 in all formulas; A₂ and the Dohnal formula
reduce to their respective saturated forms.

References
----------
Zhang, R. (1997). Determination of soil sorptivity and hydraulic conductivity
    from the disk infiltrometer. SSSAJ 61:1024–1030.
Dohnal, M., Dusek, J., Vogel, T. (2010). Improving hydraulic conductivity
    estimates from minidisk infiltrometer measurements for soils with wide
    pore-size distributions. SSSAJ 74:804–811.
"""

from __future__ import annotations

import numpy as np

# Zhang's (1997) b constant (lies in [½, π/4]; default 0.55)
_B_ZHANG = 0.55


def compute_theta_vg(
    h0_cm: float,
    alpha: float,
    n: float,
    thr: float,
    ths: float,
) -> float:
    """
    Van Genuchten water content at pressure head h₀ [cm].

    θ(h₀) = θᵣ + (θₛ − θᵣ) · [1 + (α |h₀|)ⁿ]^(−m)

    At h₀ ≥ 0 (ponded or saturated) returns θₛ.

    Parameters
    ----------
    h0_cm : float
        Pressure head [cm] — **negative** for unsaturated conditions
        (e.g. −2 for 2 cm suction).  Use 0 for ponded.
    alpha, n : float
        Van Genuchten parameters [1/cm, −].
    thr, ths : float
        Residual and saturated volumetric water content [−].
    """
    if h0_cm >= 0.0:
        return float(ths)
    m = 1.0 - 1.0 / n
    se = (1.0 + (alpha * abs(h0_cm)) ** n) ** (-m)
    return float(thr + (ths - thr) * se)


def compute_A2(
    alpha: float,
    n: float,
    r0_cm: float,
    h0_cm: float,
) -> float:
    """
    Zhang (1997) A₂ coefficient.  K(h₀) = C₂ / A₂.

    Uses the two-branch formula (Eqs. 8–9 as cited by Dohnal et al. 2010):
    - n ≥ 1.9 : d₂ = 2.92
    - n < 1.9  : d₂ = 7.5

    Valid for n ≥ 1.35.  For n < 1.35 use :func:`compute_K_dohnal`.

    Parameters
    ----------
    alpha : float
        VG α [1/cm].
    n : float
        VG n [−].
    r0_cm : float
        Disk radius [cm].
    h0_cm : float
        Pressure head [cm] — **negative** for unsaturated conditions.

    Returns
    -------
    float
        A₂ [−].

    Notes
    -----
    For h₀ = 0 (ponded) both branches give the same result because the
    exponential collapses to 1.
    """
    ar0 = alpha * r0_cm
    d2 = 2.92 if n >= 1.9 else 7.5
    return (11.65 * (n ** 0.1 - 1.0)
            * np.exp(d2 * (n - 1.9) * alpha * h0_cm)
            / ar0 ** 0.91)


def compute_K_zhang(
    C2: float,
    alpha: float,
    n: float,
    r0_cm: float,
    h0_cm: float,
) -> float:
    """
    K(h₀) via Zhang (1997): K = C₂ / A₂.  Valid for n ≥ 1.35.

    Parameters
    ----------
    C2 : float
        Gravity term from Philip fit [cm s⁻¹].
    alpha, n : float
        VG parameters [1/cm, −].
    r0_cm : float
        Disk radius [cm].
    h0_cm : float
        Pressure head [cm] — negative for unsaturated conditions.

    Returns
    -------
    float
        K(h₀) [cm s⁻¹].
    """
    return C2 / compute_A2(alpha, n, r0_cm, h0_cm)


def compute_K_dohnal(
    C2: float,
    alpha: float,
    n: float,
    r0_cm: float,
    h0_cm: float,
) -> float:
    """
    K(h₀) via Dohnal et al. (2010) Eq. 20.  Optimised for n < 1.35.

    Calibrated for minidisk geometry (disk area ≈ 15–20 cm²) and
    |h₀| ≤ 6 cm:

        K = C₂ (α r₀)^0.60 /
            [11.65 (n^0.82 − 1) exp(34.65 (n − 1.19) α h₀)]

    Parameters
    ----------
    C2 : float
        Gravity term from Philip fit [cm s⁻¹].
    alpha, n : float
        VG parameters [1/cm, −].
    r0_cm : float
        Disk radius [cm].
    h0_cm : float
        Pressure head [cm] — negative for unsaturated conditions.

    Returns
    -------
    float
        K(h₀) [cm s⁻¹].
    """
    ar0 = alpha * r0_cm
    return (C2 * ar0 ** 0.60
            / (11.65 * (n ** 0.82 - 1.0)
               * np.exp(34.65 * (n - 1.19) * alpha * h0_cm)))


def compute_K(
    C2: float,
    alpha: float,
    n: float,
    r0_cm: float,
    h0_cm: float,
) -> tuple[float, str]:
    """
    Unsaturated hydraulic conductivity K(h₀) with automatic formula selection.

    Selection rule:
    - n < 1.35  → Dohnal et al. (2010) Eq. 20  (``'dohnal2010'``)
    - n ≥ 1.35  → Zhang (1997) Eqs. 8–9         (``'zhang1997'``)

    Parameters
    ----------
    C2 : float
        Gravity term from Philip fit [cm s⁻¹].
    alpha : float
        VG α [1/cm].
    n : float
        VG n [−].
    r0_cm : float
        Disk radius [cm].
    h0_cm : float
        Pressure head [cm] — **negative** for unsaturated (e.g. −2 for 2 cm
        suction); 0 for ponded conditions.

    Returns
    -------
    K : float
        Near-saturated hydraulic conductivity [cm s⁻¹].
    method : str
        Formula used: ``'zhang1997'`` or ``'dohnal2010'``.

    Raises
    ------
    ValueError
        If C₂ ≤ 0 (physically impossible).
    """
    if C2 <= 0:
        raise ValueError(
            f"C₂ = {C2:.4e} ≤ 0: physically impossible.  "
            "Possible causes: shallow flow-restricting layer, instrument "
            "movement, or run too short.  Check raw data."
        )

    if n < 1.35:
        K = compute_K_dohnal(C2, alpha, n, r0_cm, h0_cm)
        method = 'dohnal2010'
    else:
        K = compute_K_zhang(C2, alpha, n, r0_cm, h0_cm)
        method = 'zhang1997'

    return float(K), method


def compute_A1(
    alpha: float,
    n: float,
    r0_cm: float,
    h0_cm: float,
    delta_theta: float,
) -> float:
    """
    Zhang (1997) A₁ coefficient.  Sorptivity S = C₁ / A₁.

    A₁ = 1.4 b^½ (Δθ)^¼ exp(3 (n−1.9) α h₀) / (α r₀)^½

    Parameters
    ----------
    alpha, n : float
        VG parameters [1/cm, −].
    r0_cm : float
        Disk radius [cm].
    h0_cm : float
        Pressure head [cm] — negative for unsaturated conditions.
    delta_theta : float
        θ(h₀) − θᵢ [−].  Must be > 0.

    Returns
    -------
    float
        A₁ coefficient [s^½].
    """
    if delta_theta <= 0:
        raise ValueError("delta_theta = θ(h₀) − θᵢ must be > 0")
    ar0 = alpha * r0_cm
    return (1.4 * _B_ZHANG ** 0.5
            * delta_theta ** 0.25
            * np.exp(3.0 * (n - 1.9) * alpha * h0_cm)
            / ar0 ** 0.5)


def fit_gardner_K(
    h_arr_cm,
    K_arr_cm_s,
) -> tuple[float, float]:
    """
    Fit Gardner model K(h) = Ks · exp(αG · h) via OLS on log scale.

    Linearises as  ln K = ln Ks + αG · h  and performs ordinary least squares.
    More accurate than pairwise estimates when ≥ 3 measurement tensions are
    available.

    Parameters
    ----------
    h_arr_cm : array-like
        Pressure heads h [cm] — negative for unsaturated conditions.
    K_arr_cm_s : array-like
        Hydraulic conductivity K(h) [cm s⁻¹] at each tension.

    Returns
    -------
    Ks : float
        Saturated hydraulic conductivity [cm s⁻¹] — OLS extrapolation to h = 0.
    alpha_G : float
        Gardner sorptive number αG [1/cm].  Positive (K decreases with suction).
    """
    h = np.asarray(h_arr_cm, dtype=float)
    K = np.asarray(K_arr_cm_s, dtype=float)
    A = np.column_stack([np.ones_like(h), h])
    coeffs, *_ = np.linalg.lstsq(A, np.log(K), rcond=None)
    return float(np.exp(coeffs[0])), float(coeffs[1])


def compute_gardner_alpha(
    h_arr_cm,
    K_arr_cm_s,
) -> float:
    """
    Estimate Gardner sorptive number αG from log-linear K(h) relationship.

    Wrapper around :func:`fit_gardner_K` returning only αG.

    Parameters
    ----------
    h_arr_cm : array-like
        Pressure heads h [cm] — negative for unsaturated conditions.
    K_arr_cm_s : array-like
        Hydraulic conductivity K(h) [cm s⁻¹] corresponding to each h.

    Returns
    -------
    float
        Gardner αG [1/cm].  Positive; larger values → K declines steeply with
        increasing suction (coarse-textured soils).
    """
    _, alpha_G = fit_gardner_K(h_arr_cm, K_arr_cm_s)
    return alpha_G


def compute_K_wooding(
    q_ss_cm_s: float,
    r0_cm: float,
    gardner_alpha_cm: float,
) -> float:
    """
    Unsaturated K(h₀) via Wooding (1968) steady-state radial-flow formula.

    At steady state, flow from a circular source of radius r₀ is:

        q_ss = K(h₀) · (1 + 4 / (π · r₀ · αG))

    Solving for K:

        K(h₀) = q_ss / (1 + 4 / (π · r₀ · αG))

    Parameters
    ----------
    q_ss_cm_s : float
        Steady-state infiltration flux [cm s⁻¹].
    r0_cm : float
        Hood (or disk) radius [cm].
    gardner_alpha_cm : float
        Gardner sorptive number αG [1/cm].

    Returns
    -------
    float
        K(h₀) [cm s⁻¹].

    References
    ----------
    Wooding, R.A. (1968). Steady infiltration from a shallow circular pond.
        Water Resources Research 4:1259–1273.
    """
    wooding_factor = 1.0 + 4.0 / (np.pi * r0_cm * gardner_alpha_cm)
    return float(q_ss_cm_s / wooding_factor)


def tabulate_A2(
    alpha: float,
    n: float,
    r0_cm: float,
    suctions_cm: list[float] | None = None,
) -> dict[float, float]:
    """
    Tabulate A₂ for a set of suction magnitudes.

    Useful for cross-checking against METER manual Table 2.

    Parameters
    ----------
    alpha, n : float
        VG parameters.
    r0_cm : float
        Disk radius [cm].
    suctions_cm : list of float, optional
        Suction magnitudes [cm] to evaluate (positive values).
        Defaults to [0, 1, 2, 3, 4, 5, 6].

    Returns
    -------
    dict
        ``{suction_cm: A₂}`` mapping.
    """
    if suctions_cm is None:
        suctions_cm = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    return {h: compute_A2(alpha, n, r0_cm, -h) for h in suctions_cm}


# ---------------------------------------------------------------------------
# Mualem-van Genuchten K(h)
# ---------------------------------------------------------------------------

def compute_K_vg_mualem(
    h0_cm: float,
    Ks: float,
    alpha: float,
    n: float,
    L: float = 0.5,
) -> float:
    """
    Mualem-van Genuchten K(h) [cm s⁻¹].

    K(h) = Ks · Se^L · [1 − (1 − Se^(1/m))^m]²,   m = 1 − 1/n

    Parameters
    ----------
    h0_cm : float
        Pressure head [cm] — negative for unsaturated (e.g. −2 for 2 cm suction).
    Ks : float
        Saturated hydraulic conductivity [cm s⁻¹].
    alpha : float
        VG α [1/cm].
    n : float
        VG n [−]; must be > 1.
    L : float
        Mualem pore-connectivity parameter (default 0.5).
    """
    if h0_cm >= 0.0:
        return float(Ks)
    m = 1.0 - 1.0 / n
    Se = (1.0 + (alpha * abs(h0_cm)) ** n) ** (-m)
    return float(Ks * Se ** L * (1.0 - (1.0 - Se ** (1.0 / m)) ** m) ** 2)


def fit_vg_mualem_K(
    h_arr_cm,
    K_arr_cm_s,
    L: float = 0.5,
) -> tuple[float, float, float]:
    """
    Fit Mualem-van Genuchten K(h) to measured (h₀, K) pairs via nonlinear LS.

    Parameters
    ----------
    h_arr_cm : array-like
        Pressure heads [cm] — negative for unsaturated.
    K_arr_cm_s : array-like
        K(h₀) [cm s⁻¹].
    L : float
        Mualem connectivity (default 0.5).

    Returns
    -------
    Ks : float  [cm s⁻¹]
    alpha : float  [1/cm]
    n : float  [−]
    """
    from scipy.optimize import curve_fit
    h = np.asarray(h_arr_cm, dtype=float)
    K = np.asarray(K_arr_cm_s, dtype=float)

    def _model(h_vec, Ks, alpha, n):
        return np.array([compute_K_vg_mualem(hi, Ks, alpha, n, L) for hi in h_vec])

    Ks0 = float(K[np.argmax(h)])   # K at least-negative h (closest to saturation)
    p0 = [Ks0, 0.05, 2.0]
    bounds = ([0.0, 1e-4, 1.01], [np.inf, 10.0, 20.0])
    popt, _ = curve_fit(_model, h, K, p0=p0, bounds=bounds, maxfev=10_000)
    return float(popt[0]), float(popt[1]), float(popt[2])


# ---------------------------------------------------------------------------
# Mualem-Kosugi K(h)
# ---------------------------------------------------------------------------

def compute_K_kosugi(
    h0_cm: float,
    Ks: float,
    hm: float,
    sigma: float,
    L: float = 0.5,
) -> float:
    """
    Mualem-Kosugi K(h) [cm s⁻¹] per Kosugi (1994) / HYDRUS-1D formulation.

    Se(h) = ½ erfc[(ln(|h|/hm)) / (σ√2)]
    Kr(h) = Se^L · {½ erfc[(ln|h|/hm)/(σ√2) − σ/√2]}² / {½ erfc(−σ/√2)}²

    'Ks' is a fitting parameter; may differ slightly from the true saturated K
    because the Mualem connectivity integral does not normalise to 1 at h = 0
    (see HYDRUS-1D manual §2.3, Kosugi 1994 Eq. 9).

    Parameters
    ----------
    h0_cm : float
        Pressure head [cm] — negative for unsaturated.
    Ks : float
        Fitting Ks [cm s⁻¹].
    hm : float
        Median suction head [cm] — positive (Se(hm) = 0.5).
    sigma : float
        Log-standard deviation of pore-size distribution [−]; > 0.
    L : float
        Mualem connectivity (default 0.5).
    """
    from scipy.special import erfc as _erfc
    if h0_cm >= 0.0:
        return float(Ks)
    h_abs = abs(h0_cm)
    s2 = np.sqrt(2.0)
    w = np.log(h_abs / hm) / (sigma * s2)
    Se = 0.5 * _erfc(w)
    num = 0.5 * _erfc(w - sigma / s2)
    den = 0.5 * _erfc(-sigma / s2)
    Kr = Se ** L * (num / den) ** 2
    return float(Ks * Kr)


def fit_kosugi_K(
    h_arr_cm,
    K_arr_cm_s,
    L: float = 0.5,
) -> tuple[float, float, float]:
    """
    Fit Mualem-Kosugi K(h) to measured (h₀, K) pairs via nonlinear LS.

    Parameters
    ----------
    h_arr_cm : array-like
        Pressure heads [cm] — negative for unsaturated.
    K_arr_cm_s : array-like
        K(h₀) [cm s⁻¹].
    L : float
        Mualem connectivity (default 0.5).

    Returns
    -------
    Ks : float  [cm s⁻¹]
    hm : float  median suction head [cm]
    sigma : float  log-std of pore-size distribution [−]
    """
    from scipy.optimize import curve_fit
    h = np.asarray(h_arr_cm, dtype=float)
    K = np.asarray(K_arr_cm_s, dtype=float)

    def _model(h_vec, Ks, hm, sigma):
        return np.array([compute_K_kosugi(hi, Ks, hm, sigma, L) for hi in h_vec])

    Ks0 = float(K[np.argmax(h)])
    hm0 = float(np.exp(np.mean(np.log(np.abs(h)))))   # geometric mean of |h|
    p0 = [Ks0, hm0, 1.2]
    bounds = ([0.0, 1e-3, 0.1], [np.inf, 1_000.0, 5.0])
    popt, _ = curve_fit(_model, h, K, p0=p0, bounds=bounds, maxfev=10_000)
    return float(popt[0]), float(popt[1]), float(popt[2])
