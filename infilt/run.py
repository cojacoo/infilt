"""
Unified single-tension infiltration run.

Both mini-disk and hood produce cumulative infiltration I(t) from different
raw signals; the same Philip fitting chain and geometry correction (A₂) apply.
Only the geometric parameters differ.

Signal types
------------
'volume' : Mariotte bottle [mL].  I(t) = ΔV / A_disk.
'level'  : Supply cylinder water level [mm].
           I(t) = Δh · A_reservoir / (10 · A_disk).

Device presets
--------------
MINIDISK_STUDENT  r₀ = 25 mm,  signal_type='volume'
MINIDISK_METER    r₀ = 22.5 mm, signal_type='volume'
HOOD_IL2700       r₀ = 124 mm, signal_type='level', reservoir_area_cm2=23.0
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .fitting import fit_philip, PhilipFit
from .steady import detect_steady_state, SteadyStateResult
from .theory import compute_A2, compute_K, compute_theta_vg, compute_A1
from .soil_db import get_vg_params

# ---------------------------------------------------------------------------
# Device presets
# ---------------------------------------------------------------------------

MINIDISK_STUDENT: dict = dict(disk_radius_mm=25.0,  signal_type='volume')
MINIDISK_METER:   dict = dict(disk_radius_mm=22.5,  signal_type='volume')
HOOD_IL2700:      dict = dict(disk_radius_mm=124.0, signal_type='level',
                               reservoir_area_cm2=23.0)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class InfiltrationResult:
    """
    Results from a single-tension run (any instrument).

    K values are stored internally in [cm s⁻¹].
    Properties expose [m s⁻¹] and [mm h⁻¹].
    """

    # Metadata
    site: Optional[str]
    suction_mm: float
    disk_radius_cm: float
    signal_type: str
    reservoir_area_cm2: Optional[float]

    # Soil params
    alpha: float
    n: float
    soil_texture: Optional[str]
    db: str

    # Philip fits
    ols: PhilipFit
    su: PhilipFit
    ml: PhilipFit
    fit_formula: str          # 'zhang1997' | 'dohnal2010'

    # K from Philip [cm s⁻¹]
    K_ols_cm_s: float
    K_su_cm_s: float
    K_ml_cm_s: Optional[float]   # None when ml degenerate

    # Wooding steady-state (set by Campaign)
    ss: Optional[SteadyStateResult] = None
    K_wooding_cm_s: Optional[float] = None
    gardner_alpha_cm: Optional[float] = None

    # Optional sorptivity
    sorptivity_cm_s05: Optional[float] = None

    # Flags
    flags: list[str] = field(default_factory=list)

    # Raw arrays
    time_s:    Optional[np.ndarray] = field(repr=False, default=None)
    I_cm_obs:  Optional[np.ndarray] = field(repr=False, default=None)
    rate_cm_s: Optional[np.ndarray] = field(repr=False, default=None)

    # ── Unit-conversion properties ────────────────────────────────────────

    @property
    def h0_cm(self) -> float:
        return -self.suction_mm / 10.0

    @property
    def disk_area_cm2(self) -> float:
        return np.pi * self.disk_radius_cm ** 2

    @staticmethod
    def _to(k_cm_s: Optional[float]):
        if k_cm_s is None:
            return None, None
        return float(k_cm_s * 1e-2), float(k_cm_s * 36_000)

    @property
    def K_ols_ms(self): return self.K_ols_cm_s * 1e-2
    @property
    def K_ols_mmh(self): return self.K_ols_cm_s * 36_000
    @property
    def K_su_ms(self): return self.K_su_cm_s * 1e-2
    @property
    def K_su_mmh(self): return self.K_su_cm_s * 36_000
    @property
    def K_ml_ms(self): return self.K_ml_cm_s * 1e-2 if self.K_ml_cm_s else None
    @property
    def K_ml_mmh(self): return self.K_ml_cm_s * 36_000 if self.K_ml_cm_s else None
    @property
    def K_wooding_ms(self): return self.K_wooding_cm_s * 1e-2 if self.K_wooding_cm_s else None
    @property
    def K_wooding_mmh(self): return self.K_wooding_cm_s * 36_000 if self.K_wooding_cm_s else None

    @property
    def K_primary_cm_s(self) -> Optional[float]:
        """Wooding K if available (from Campaign), else OLS Philip K."""
        return self.K_wooding_cm_s if self.K_wooding_cm_s else self.K_ols_cm_s

    @property
    def K_primary_mmh(self) -> Optional[float]:
        k = self.K_primary_cm_s
        return k * 36_000 if k else None

    def __str__(self) -> str:
        _f = lambda k: f'{k*36000:.2f} mm/h  ({k*1e-2:.3e} m/s)' if k else 'n/a'
        lines = [
            '─── Infiltration Result ───',
            f'  Site       : {self.site or "—"}',
            f'  Suction    : {self.suction_mm:.0f} mm  (h₀ = {self.h0_cm:.2f} cm)',
            f'  Instrument : {self.signal_type}',
            f'  VG α/n     : '
            + (f'{self.alpha:.4f} / {self.n:.3f}' if self.alpha is not None else 'n/a'),
            f'  K_Wooding  : {_f(self.K_wooding_cm_s)}',
            f'  K_OLS      : {_f(self.K_ols_cm_s)}',
            f'  K_Su       : {_f(self.K_su_cm_s)}',
            f'  K_ML       : {_f(self.K_ml_cm_s)}',
        ]
        if self.ss:
            lines += [
                f'  q_ss       : {self.ss.q_ss:.3e} ± {self.ss.q_ss_se:.1e} cm/s'
                f'  ({self.ss.q_ss*36000:.2f} ± {self.ss.q_ss_se*36000:.2f} mm/h)',
                f'  Convergence: {self.ss.conv_frac:.0%}  '
                f'(t₉₅={self.ss.t95_s:.0f} s, Horton R²={self.ss.horton_r2:.3f})',
            ]
        if self.flags:
            lines.append(f'  ⚠ Flags    : {", ".join(self.flags)}')
        lines.append('─' * 32)
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class InfiltrationRun:
    """
    Single-tension run from any tension infiltrometer.

    Parameters
    ----------
    time_s : array-like
        Elapsed time [s].
    signal : array-like
        Water volume [mL]  (signal_type='volume')
        or water level [mm] (signal_type='level').
        Must be monotonically decreasing.
    suction_mm : float
        Applied suction head [mm] (≥ 0).
    signal_type : {'volume', 'level'}
        'volume' for mini-disk Mariotte bottle; 'level' for hood supply cylinder.
    disk_radius_mm : float
        Radius of the infiltration contact area [mm].
        25 mm = student mini-disk; 22.5 mm = METER mini-disk; 124 mm = IL-2700.
    reservoir_area_cm2 : float, optional
        Supply reservoir cross-sectional area [cm²].
        Required for signal_type='level'.  Hood IL-2700 ≈ 23.0 cm².
    soil_texture : str, optional
        USDA texture class (e.g. 'loam', 'silt_loam').
        Provide either this or both alpha + n.
    alpha, n : float, optional
        Van Genuchten α [1/cm] and n [−].
    thr, ths : float, optional
        Residual / saturated water content for sorptivity calculation.
    theta_initial : float, optional
        Initial water content for sorptivity calculation.
    db : str
        Soil database: 'carsel_parrish' (default) or 'rosetta'.
    site : str, optional
        Site label (metadata).

    Examples
    --------
    Mini-disk student device::

        run = InfiltrationRun(time_s, volume_mL, suction_mm=30,
                              **MINIDISK_STUDENT, soil_texture='loam')

    Hood IL-2700::

        run = InfiltrationRun(time_s, level_mm, suction_mm=30, **HOOD_IL2700)
    """

    def __init__(
        self,
        time_s,
        signal,
        suction_mm: float = 30.0,
        signal_type: str = 'volume',
        disk_radius_mm: float = 25.0,
        reservoir_area_cm2: Optional[float] = None,
        soil_texture: Optional[str] = None,
        alpha: Optional[float] = None,
        n: Optional[float] = None,
        thr: Optional[float] = None,
        ths: Optional[float] = None,
        theta_initial: Optional[float] = None,
        db: str = 'carsel_parrish',
        site: Optional[str] = None,
    ):
        self.time_s = np.asarray(time_s, dtype=float)
        self.signal = np.asarray(signal, dtype=float)
        self.suction_mm = float(suction_mm)
        self.signal_type = signal_type
        self.r0_cm = float(disk_radius_mm) / 10.0
        self.reservoir_area_cm2 = (
            float(reservoir_area_cm2) if reservoir_area_cm2 is not None else None
        )
        self.theta_initial = theta_initial
        self.db = db
        self.site = site

        if signal_type not in ('volume', 'level'):
            raise ValueError("signal_type must be 'volume' or 'level'.")
        if signal_type == 'level' and self.reservoir_area_cm2 is None:
            raise ValueError(
                "reservoir_area_cm2 is required for signal_type='level'."
            )

        if soil_texture is not None:
            params = get_vg_params(soil_texture, db=db)
            self.alpha: Optional[float] = params['alpha']
            self.n:     Optional[float] = params['n']
            self.thr = params.get('thr') if thr is None else thr
            self.ths = params.get('ths') if ths is None else ths
            self.soil_texture = soil_texture
        elif alpha is not None and n is not None:
            self.alpha = float(alpha)
            self.n = float(n)
            self.thr = thr
            self.ths = ths
            self.soil_texture = None
        else:
            # No VG params — Philip K will be skipped; Wooding still works.
            # Campaign supplies first-pass VG params via alpha_vg / n_vg.
            self.alpha = None
            self.n = None
            self.thr = thr
            self.ths = ths
            self.soil_texture = None

        self._validate()

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def h0_cm(self) -> float:
        return -self.suction_mm / 10.0

    @property
    def disk_area_cm2(self) -> float:
        return np.pi * self.r0_cm ** 2

    # ── Signal processing ──────────────────────────────────────────────────

    def cumulative_infiltration(self) -> np.ndarray:
        """Cumulative infiltration I(t) [cm] regardless of instrument type."""
        if self.signal_type == 'volume':
            delta = self.signal[0] - self.signal      # mL = cm³
            return delta / self.disk_area_cm2
        else:
            delta_h = self.signal[0] - self.signal    # mm
            return delta_h * self.reservoir_area_cm2 / (10.0 * self.disk_area_cm2)

    def infiltration_rate(self, ols_window: int = 5) -> np.ndarray:
        """
        Infiltration rate dI/dt [cm s⁻¹] via sliding OLS differentiation.

        Central OLS on ols_window consecutive points gives far better noise
        rejection than point-by-point numerical differences.
        """
        I = self.cumulative_infiltration()
        n = len(self.time_s)
        half = max(1, ols_window // 2)
        rate = np.zeros(n)
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, lo + ols_window)
            lo = max(0, hi - ols_window)
            t_w = self.time_s[lo:hi]
            I_w = I[lo:hi]
            if len(t_w) < 2:
                continue
            A_mat = np.column_stack([np.ones_like(t_w), t_w])
            coeffs, *_ = np.linalg.lstsq(A_mat, I_w, rcond=None)
            rate[i] = float(coeffs[1])
        return rate

    # ── Analysis ───────────────────────────────────────────────────────────

    def run(
        self,
        alpha_vg: Optional[float] = None,
        n_vg: Optional[float] = None,
    ) -> InfiltrationResult:
        """
        Philip transient analysis (OLS + Su + ML) with A₂ geometry correction.

        Parameters
        ----------
        alpha_vg, n_vg : float, optional
            Override VG params for A₂.  Campaign passes first-pass fitted values
            here to break the chicken-and-egg dependency.
        """
        I_cm = self.cumulative_infiltration()
        rate = self.infiltration_rate()

        # Own VG params take priority; campaign fallback when run has none.
        a = self.alpha if self.alpha is not None else alpha_vg
        n = self.n     if self.n     is not None else n_vg

        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            ols = fit_philip(self.time_s, I_cm, method='ols')
            su  = fit_philip(self.time_s, I_cm, method='su')
            ml  = fit_philip(self.time_s, I_cm, method='ml')

        flags: list[str] = []
        formula = 'n/a'
        K_ols = K_su = 0.0
        K_ml: Optional[float] = None

        if a is not None and n is not None:
            A2 = compute_A2(a, n, self.r0_cm, self.h0_cm)
            _, formula = compute_K(max(ols.C2, 1e-20), a, n, self.r0_cm, self.h0_cm)
            K_ols = float(ols.C2 / A2) if ols.C2 > 0 else 0.0
            K_su  = float(su.C2  / A2) if su.C2  > 0 else 0.0
            K_ml  = float(ml.C2  / A2) if (ml.C2 > 0 and not ml.degenerate) else None
        else:
            flags.append('NO_VG_PARAMS')

        if ols.r2 < 0.95:
            flags.append(f'LOW_R2:{ols.r2:.3f}')
        if ols.C2 <= 0:
            flags.append('NEGATIVE_K')
        if su.degenerate:
            flags.append('SU_DEGEN')
        if ml.degenerate:
            flags.append('ML_DEGEN')
        if not su.degenerate and abs(su.beta - 0.5) > 0.1:
            flags.append(f'ANOM_β:{su.beta:.3f}')

        sorptivity = None
        if (self.thr is not None and self.ths is not None
                and self.theta_initial is not None):
            theta_h0 = compute_theta_vg(self.h0_cm, a, n, self.thr, self.ths)
            delta_theta = theta_h0 - self.theta_initial
            if delta_theta > 0:
                A1 = compute_A1(a, n, self.r0_cm, self.h0_cm, delta_theta)
                if A1 > 0:
                    sorptivity = float(ols.C1 / A1)

        return InfiltrationResult(
            site=self.site,
            suction_mm=self.suction_mm,
            disk_radius_cm=self.r0_cm,
            signal_type=self.signal_type,
            reservoir_area_cm2=self.reservoir_area_cm2,
            alpha=self.alpha,
            n=self.n,
            soil_texture=self.soil_texture,
            db=self.db,
            ols=ols, su=su, ml=ml,
            fit_formula=formula,
            K_ols_cm_s=K_ols,
            K_su_cm_s=K_su,
            K_ml_cm_s=K_ml,
            sorptivity_cm_s05=sorptivity,
            flags=flags,
            time_s=self.time_s,
            I_cm_obs=I_cm,
            rate_cm_s=rate,
        )

    # ── Internal ───────────────────────────────────────────────────────────

    def _validate(self) -> None:
        if len(self.time_s) != len(self.signal):
            raise ValueError("time_s and signal must have the same length.")
        if len(self.time_s) < 3:
            raise ValueError("At least 3 time–signal pairs required.")
        if not np.all(np.diff(self.time_s) >= 0):
            raise ValueError("time_s must be non-decreasing.")
        if self.suction_mm < 0:
            raise ValueError("suction_mm must be ≥ 0.")
        if not np.all(np.diff(self.signal) <= 0.5):
            warnings.warn(
                "signal is not monotonically decreasing — check for "
                "instrument disturbance or reading errors.",
                stacklevel=3,
            )
