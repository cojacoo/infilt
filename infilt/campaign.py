"""
Multi-tension, multi-instrument campaign analysis.

Pipeline
--------
1. Philip OLS on each run → first-pass K(h) pairs.
2. Mualem-VG fit on first-pass pairs → α_fp, n_fp for A₂.
3. Wooding steady-state inversion (iterative) on all runs with level signal.
4. Re-run Philip on all runs with corrected A₂ (two-pass).
5. K(h) fits: Gardner + Mualem-VG + Mualem-Kosugi on primary K(h) pairs.

Primary K per run
-----------------
Wooding q_ss-based K when available (large r₀, level signal);
Philip OLS K otherwise.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .run import InfiltrationRun, InfiltrationResult
from .steady import detect_steady_state, SteadyStateResult
from .theory import (
    compute_A2, compute_K_wooding, compute_gardner_alpha,
    fit_gardner_K, fit_vg_mualem_K, fit_kosugi_K,
    compute_K_vg_mualem, compute_K_kosugi,
)


# ---------------------------------------------------------------------------
# Wooding inversion helpers
# ---------------------------------------------------------------------------

def _wooding_inversion(
    q_ss_arr: np.ndarray,
    h_arr_cm: np.ndarray,
    r0_cm: float,
) -> tuple[np.ndarray, float]:
    """Iterative Wooding inversion → K(h) sorted by ascending h + αG."""
    q = np.asarray(q_ss_arr, dtype=float)
    h = np.asarray(h_arr_cm, dtype=float)
    idx = np.argsort(h)
    h_s, q_s = h[idx], q[idx]

    K = q_s.copy()
    for _ in range(30):
        if np.any(K <= 0):
            break
        aG = compute_gardner_alpha(h_s, K)
        if aG <= 0:
            break
        K_new = np.array([compute_K_wooding(qi, r0_cm, aG) for qi in q_s])
        if np.max(np.abs(K_new - K) / (K + 1e-15)) < 1e-6:
            K = K_new
            break
        K = K_new

    aG = float(compute_gardner_alpha(h_s, K))
    return K, aG


# ---------------------------------------------------------------------------
# K(h) fits dataclass
# ---------------------------------------------------------------------------

@dataclass
class KhFit:
    """Fitted K(h) model parameters (three models)."""
    # Gardner: K = Ks · exp(αG · h)
    Ks_g: float
    aG: float
    # Mualem-van Genuchten
    Ks_vg: float
    alpha_vg: float
    n_vg: float
    # Mualem-Kosugi
    Ks_ko: float
    hm_ko: float
    sigma_ko: float

    # plotting arrays (set by Campaign)
    h_plot: Optional[np.ndarray] = field(repr=False, default=None)
    K_g_plot:  Optional[np.ndarray] = field(repr=False, default=None)
    K_vg_plot: Optional[np.ndarray] = field(repr=False, default=None)
    K_ko_plot: Optional[np.ndarray] = field(repr=False, default=None)


# ---------------------------------------------------------------------------
# Campaign result
# ---------------------------------------------------------------------------

@dataclass
class CampaignResult:
    """Full results for one site campaign."""

    site: Optional[str]
    results: list[InfiltrationResult]
    kh: Optional[KhFit]
    h_arr: np.ndarray       # pressure heads used for K(h) fit [cm]
    K_arr: np.ndarray       # primary K [cm s⁻¹]
    alpha_fp: float         # first-pass VG α used for A₂
    n_fp: float             # first-pass VG n used for A₂

    # ── Comprehensive table ────────────────────────────────────────────────

    def table(self, width: int = 112) -> str:
        """Comprehensive results table (all methods per run)."""
        na = '     n/a'

        def _kf(k: Optional[float]) -> str:
            return f'{k * 36_000:>8.1f}' if (k and k > 0) else na

        lines = [
            f'CAMPAIGN RESULTS — {self.site or "unnamed"}',
            '═' * width,
            (f'  {"h₀[cm]":>7}  {"signal":>7}  {"suct[mm]":>8}'
             f'  {"Wooding":>8}  {"OLS":>8}  {"Su":>8}  {"ML":>8}'
             f'  {"β_Su":>6}  {"R²_OLS":>7}  {"@SS%":>5}  flags'),
            (f'  {"":>7}  {"":>7}  {"":>8}'
             f'  {"[mm/h]":>8}  {"[mm/h]":>8}  {"[mm/h]":>8}  {"[mm/h]":>8}'
             f'  {"":>6}  {"":>7}  {"":>6}'),
            '─' * width,
        ]

        for res in sorted(self.results, key=lambda r: r.h0_cm):
            conv_str = (f'{res.ss.conv_frac:>4.0%}' if res.ss else '  n/a')
            flag_str = ('✓' if not res.flags
                        else '⚠ ' + '  '.join(res.flags))
            lines.append(
                f'  {res.h0_cm:>7.2f}  {res.signal_type:>7}  {res.suction_mm:>8.0f}'
                f'  {_kf(res.K_wooding_cm_s)}'
                f'  {_kf(res.K_ols_cm_s)}'
                f'  {_kf(res.K_su_cm_s)}'
                f'  {_kf(res.K_ml_cm_s)}'
                f'  {res.su.beta:>6.3f}  {res.ols.r2:>7.4f}  {conv_str}  '
                + flag_str
            )

        lines += [
            '─' * width,
            (f'  Philip A₂: first-pass VG α={self.alpha_fp:.4f} cm⁻¹'
             f'  n={self.n_fp:.3f}'),
        ]

        level_with_ss = [r for r in self.results
                         if r.signal_type == 'level' and r.ss is not None]
        if level_with_ss:
            lines += [
                '',
                'STEADY-STATE FLUX (CV-scan mean; Horton fc as fallback)',
                '─' * 72,
                (f'  {"h₀[cm]":>7}  {"q_ss [mm/h]":>12}  {"±std [mm/h]":>12}'
                 f'  {"method":>10}  {"@SS%":>5}  {"t₉₅[s]":>7}  {"Horton R²":>9}'),
                '─' * 72,
            ]
            for r in sorted(level_with_ss, key=lambda x: x.h0_cm):
                t95 = f'{r.ss.t95_s:>7.0f}' if np.isfinite(r.ss.t95_s) else '    inf'
                lines.append(
                    f'  {r.h0_cm:>7.2f}  {r.ss.q_ss*36000:>12.3f}'
                    f'  {r.ss.q_ss_se*36000:>12.3f}'
                    f'  {r.ss.method:>10}'
                    f'  {r.ss.conv_frac:>4.0%}'
                    f'  {t95}'
                    f'  {r.ss.horton_r2:>9.4f}'
                )
            lines.append('─' * 72)

        if self.kh:
            lines += [
                '',
                f'K(h) FITS — {len(self.h_arr)} valid run(s)',
                '═' * width,
                (f'  {"Model":<16}  {"Ks [m/s]":>12}  {"Ks [mm/h]":>10}'
                 f'  {"param1":>22}  {"param2":>20}'),
                (f'  {"Gardner":<16}  {self.kh.Ks_g*1e-2:>12.3e}'
                 f'  {self.kh.Ks_g*36000:>10.2f}'
                 f'  {"αG = "+f"{self.kh.aG:.4f} cm⁻¹":>22}'),
                (f'  {"Mualem-VG":<16}  {self.kh.Ks_vg*1e-2:>12.3e}'
                 f'  {self.kh.Ks_vg*36000:>10.2f}'
                 f'  {"α = "+f"{self.kh.alpha_vg:.4f} cm⁻¹":>22}'
                 f'  {"n = "+f"{self.kh.n_vg:.4f}":>20}'),
                (f'  {"Mualem-Kosugi":<16}  {self.kh.Ks_ko*1e-2:>12.3e}'
                 f'  {self.kh.Ks_ko*36000:>10.2f}'
                 f'  {"hm = "+f"{self.kh.hm_ko:.4f} cm":>22}'
                 f'  {"σ = "+f"{self.kh.sigma_ko:.4f}":>20}'),
                '',
                (f'  {"h₀[cm]":>8}  {"K_prim[mm/h]":>13}'
                 f'  {"res_Gardner":>13}  {"res_VG":>10}  {"res_Kosugi":>12}'
                 f'  (mm/h)'),
                '─' * 64,
            ]
            for h0, K in zip(self.h_arr, self.K_arr):
                K_g  = self.kh.Ks_g  * np.exp(self.kh.aG * h0)
                K_vg = compute_K_vg_mualem(h0, self.kh.Ks_vg,
                                            self.kh.alpha_vg, self.kh.n_vg)
                K_ko = compute_K_kosugi(h0, self.kh.Ks_ko,
                                         self.kh.hm_ko, self.kh.sigma_ko)
                lines.append(
                    f'  {h0:>8.2f}  {K*36000:>13.1f}'
                    f'  {(K-K_g)*36000:>+13.1f}  {(K-K_vg)*36000:>+10.1f}'
                    f'  {(K-K_ko)*36000:>+12.1f}'
                )
        else:
            lines.append('  Not enough valid runs for K(h) fitting (need ≥ 2).')

        n_flagged = sum(1 for r in self.results if r.flags)
        lines.append(
            f'  ⚠ {n_flagged} run(s) flagged.' if n_flagged else '  All runs clean.'
        )
        return '\n'.join(lines)

    # ── Plotly figure ──────────────────────────────────────────────────────

    def figure(self):
        """
        Plotly figure with two panels.

        Panel A : raw signals per tension
          - level runs : infiltration rate q [mm/h] vs time [min]
          - volume runs: cumulative I [cm] vs √t [min^½] with OLS fit
        Panel B : K(h) curves — Gardner / Mualem-VG / Mualem-Kosugi
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError as e:
            raise ImportError("Install plotly: pip install plotly") from e

        PALETTE = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        ]

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(
                                'A)  Raw signals per tension',
                                'B)  K(h) — Gardner / Mualem-VG / Mualem-Kosugi',
                            ))
        # Explicit domains: A 0–40 %, gap 40–65 %, B 65–90 %, legend 91–100 %
        fig.update_xaxes(domain=[0.00, 0.40], row=1, col=1)
        fig.update_xaxes(domain=[0.65, 0.90], row=1, col=2)
        fig.update_layout(
            template='none',
            title_text=f'Tension Infiltrometer Campaign — {self.site or ""}',
            title_font_size=14,
            height=500,
            legend=dict(orientation='v', x=0.91, y=1.0),
        )

        sorted_res = sorted(self.results, key=lambda r: r.h0_cm)

        # ── Panel A ────────────────────────────────────────────────────────
        for i, res in enumerate(sorted_res):
            col = PALETTE[i % len(PALETTE)]
            label = f'{res.suction_mm:.0f} mm ({res.signal_type})'

            if res.signal_type == 'level' and res.rate_cm_s is not None:
                t_min = res.time_s / 60.0
                q_mmh = res.rate_cm_s * 36_000
                fig.add_trace(
                    go.Scatter(
                        x=t_min, y=q_mmh,
                        mode='lines', name=label,
                        line=dict(color=col, width=2),
                        legendgroup=label,
                    ),
                    row=1, col=1,
                )
                if res.ss is not None and res.ss.mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=t_min[res.ss.mask],
                            y=q_mmh[res.ss.mask],
                            mode='markers',
                            marker=dict(color=col, size=9, symbol='circle'),
                            name=f'{label} (SS)',
                            legendgroup=label,
                            showlegend=False,
                        ),
                        row=1, col=1,
                    )
                if res.ss is not None:
                    q_ss_mmh = res.ss.q_ss * 36_000
                    se_mmh   = res.ss.q_ss_se * 36_000
                    K_w_mmh  = (res.K_wooding_cm_s * 36_000
                                if res.K_wooding_cm_s else None)
                    ann = (
                        f'q = {q_ss_mmh:.2f} ± {se_mmh:.2f} mm/h'
                        f'<br>K = {K_w_mmh:.2f} mm/h'
                        if K_w_mmh
                        else f'q = {q_ss_mmh:.2f} ± {se_mmh:.2f} mm/h'
                    )
                    fig.add_hrect(
                        y0=q_ss_mmh - se_mmh,
                        y1=q_ss_mmh + se_mmh,
                        fillcolor=col, opacity=0.12,
                        line_width=0,
                        row=1, col=1,
                    )
                    fig.add_hline(
                        y=q_ss_mmh,
                        line_dash='dash', line_color=col,
                        opacity=0.7, row=1, col=1,
                        annotation_text=ann,
                        annotation_font_size=9,
                        annotation_position='right',
                    )
            elif res.signal_type == 'volume' and res.I_cm_obs is not None:
                sqt = np.sqrt(np.maximum(res.time_s, 0)) / 60.0
                fig.add_trace(
                    go.Scatter(
                        x=sqt, y=res.I_cm_obs,
                        mode='markers', name=label,
                        marker=dict(color=col, size=8, symbol='square'),
                        legendgroup=label,
                        yaxis='y2',
                    ),
                    row=1, col=1,
                )
                if res.ols.I_fit is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=sqt, y=res.ols.I_fit,
                            mode='lines', name=f'{label} OLS fit',
                            line=dict(color=col, width=1.5, dash='dash'),
                            legendgroup=label, showlegend=False,
                            yaxis='y2',
                        ),
                        row=1, col=1,
                    )

        has_level  = any(r.signal_type == 'level'  for r in sorted_res)
        has_volume = any(r.signal_type == 'volume' for r in sorted_res)
        if has_level and has_volume:
            x_label_a = 'Time [min]  (level)  /  √t [min^½]  (volume)'
        elif has_volume:
            x_label_a = '√t [min^½]'
        else:
            x_label_a = 'Time [min]'
        fig.update_xaxes(title_text=x_label_a, row=1, col=1)
        fig.update_yaxes(title_text='Rate q [mm/h]', row=1, col=1)

        # ── Panel B ────────────────────────────────────────────────────────
        if self.kh is not None and self.kh.h_plot is not None:
            h_pl = self.kh.h_plot
            fig.add_trace(
                go.Scatter(
                    x=h_pl, y=self.kh.K_g_plot * 36_000,
                    mode='lines', name=f'Gardner  αG={self.kh.aG:.3f}',
                    line=dict(color='#8c564b', width=2),
                ),
                row=1, col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=h_pl, y=self.kh.K_vg_plot * 36_000,
                    mode='lines',
                    name=f'Mualem-VG  α={self.kh.alpha_vg:.3f}  n={self.kh.n_vg:.2f}',
                    line=dict(color='#ff7f0e', width=2, dash='dash'),
                ),
                row=1, col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=h_pl, y=self.kh.K_ko_plot * 36_000,
                    mode='lines',
                    name=f'Mualem-Ko  hm={self.kh.hm_ko:.2f}  σ={self.kh.sigma_ko:.2f}',
                    line=dict(color='#9467bd', width=2, dash='dot'),
                ),
                row=1, col=2,
            )

        # Same color per tension as Panel A (sorted_res order)
        color_by_h0 = {r.h0_cm: PALETTE[i % len(PALETTE)]
                       for i, r in enumerate(sorted_res)}

        for h0, K in zip(self.h_arr, self.K_arr):
            res = next((r for r in self.results
                        if abs(r.h0_cm - h0) < 0.01), None)
            col = color_by_h0.get(h0, PALETTE[0])
            sym = 'circle' if (res and res.signal_type == 'level') else 'square'
            lbl = f'{abs(h0)*10:.0f} mmWC'
            # σ_K = σ_q × (K/q_ss)  — Wooding inversion is linear in q_ss
            se_K_mmh = 0.0
            if (res and res.ss and res.K_wooding_cm_s
                    and res.ss.q_ss > 0 and res.ss.q_ss_se > 0):
                se_K_mmh = float(
                    res.ss.q_ss_se * 36_000
                    * (res.K_wooding_cm_s / res.ss.q_ss)
                )
            fig.add_trace(
                go.Scatter(
                    x=[abs(h0)], y=[K * 36_000],
                    mode='markers',
                    marker=dict(color=col, size=12, symbol=sym),
                    error_y=dict(
                        type='data', array=[se_K_mmh],
                        visible=se_K_mmh > 0,
                        color=col, thickness=1.5, width=6,
                    ),
                    name=lbl,
                    legendgroup=lbl,
                    showlegend=True,
                ),
                row=1, col=2,
            )

        # Ksat marker at h=0 from Gardner and VG models
        if self.kh is not None:
            ks_vals = [self.kh.Ks_g * 36_000, self.kh.Ks_vg * 36_000]
            ks_mean = float(np.mean(ks_vals))
            ks_lo   = float(min(ks_vals))
            ks_hi   = float(max(ks_vals))
            fig.add_trace(
                go.Scatter(
                    x=[0], y=[ks_mean],
                    mode='markers+text',
                    marker=dict(color='black', size=14, symbol='diamond'),
                    text=[f'Ksat≈{ks_mean:.1f}<br>[{ks_lo:.1f}–{ks_hi:.1f}]'],
                    textposition='top right',
                    textfont=dict(size=9),
                    name=f'Ksat = {ks_mean:.1f} mm/h',
                    showlegend=True,
                ),
                row=1, col=2,
            )

        fig.update_xaxes(title_text='Suction |h₀| [cm]', range=[-0.05, None],
                         row=1, col=2)
        fig.update_yaxes(title_text='K(h₀) [mm/h]', type='log', row=1, col=2)

        return fig


# ---------------------------------------------------------------------------
# Campaign class
# ---------------------------------------------------------------------------

class Campaign:
    """
    Multi-tension, multi-instrument campaign at one site.

    Parameters
    ----------
    runs : list of InfiltrationRun
        All measurement runs for this site.  Mix of signal types and
        instruments freely.
    site : str, optional
        Site identifier.

    Examples
    --------
    ::

        from infilt import InfiltrationRun, Campaign, HOOD_IL2700, MINIDISK_STUDENT

        runs = [
            InfiltrationRun(t30, level30, 30, **HOOD_IL2700),
            InfiltrationRun(t60, level60, 60, **HOOD_IL2700),
            InfiltrationRun(t90, level90, 90, **HOOD_IL2700),
            InfiltrationRun(t_md, vol_md, 30, **MINIDISK_STUDENT,
                            soil_texture='loam'),
        ]
        result = Campaign(runs, site='Field A').run()
        print(result.table())
        result.figure().show()
    """

    def __init__(
        self,
        runs: list[InfiltrationRun],
        site: Optional[str] = None,
    ):
        if not runs:
            raise ValueError("Campaign needs at least one run.")
        self.runs = runs
        self.site = site

    @classmethod
    def from_dataframe(
        cls,
        df,
        site: Optional[str] = None,
        hood_preset: Optional[dict] = None,
        minidisk_preset: Optional[dict] = None,
        **shared_kwargs,
    ) -> 'Campaign':
        """
        Build a Campaign from a tidy long-format DataFrame.

        Expected columns
        ----------------
        site         : str   — site identifier (used if ``site`` kwarg is None)
        instrument   : str   — 'hood' | 'minidisk'
        suction_mm   : float — applied suction [mm]
        signal_type  : str   — 'level' | 'volume'
        time_s       : float — elapsed time [s]
        signal       : float — level [mm] or volume [mL]

        Parameters
        ----------
        df : pandas.DataFrame
        site : str, optional
            Overrides df['site'] when supplied.
        hood_preset : dict, optional
            Device constants for hood runs (e.g. ``HOOD_IL2700``).
            ``signal_type`` key is ignored (taken from DataFrame).
        minidisk_preset : dict, optional
            Device constants for mini-disk runs (e.g. ``MINIDISK_STUDENT``).
        **shared_kwargs
            Passed to every InfiltrationRun (e.g. ``soil_texture='loam'``).

        Examples
        --------
        ::

            df = build_dataframe(parsed_runs)          # long format
            result = Campaign.from_dataframe(
                df, hood_preset=HOOD_IL2700,
            ).run()
        """
        if site is None and 'site' in df.columns:
            site = str(df['site'].iloc[0])

        runs: list[InfiltrationRun] = []
        for (suct, sig_type), grp in df.groupby(
                ['suction_mm', 'signal_type'], sort=True):
            grp = grp.sort_values('time_s')
            instrument = (str(grp['instrument'].iloc[0])
                          if 'instrument' in grp.columns else None)

            preset = {}
            if sig_type == 'level' and hood_preset:
                preset = {k: v for k, v in hood_preset.items()
                          if k != 'signal_type'}
            elif sig_type == 'volume' and minidisk_preset:
                preset = {k: v for k, v in minidisk_preset.items()
                          if k != 'signal_type'}

            run_site = (str(grp['site'].iloc[0])
                        if 'site' in grp.columns else site)
            runs.append(InfiltrationRun(
                time_s=grp['time_s'].to_numpy(dtype=float),
                signal=grp['signal'].to_numpy(dtype=float),
                suction_mm=float(suct),
                signal_type=str(sig_type),
                site=run_site,
                **preset,
                **shared_kwargs,
            ))

        return cls(runs, site=site)

    def run(self) -> CampaignResult:
        """Execute the full two-pass campaign analysis."""

        level_runs = [r for r in self.runs if r.signal_type == 'level']
        vol_runs   = [r for r in self.runs if r.signal_type == 'volume']

        # ── Step 1: q_ss + Wooding inversion for all level runs ───────────
        q_ss_list: list[float] = []
        h_ss_list: list[float] = []
        ss_list:   list        = []

        for run in level_runs:
            rate = run.infiltration_rate()
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                ss = detect_steady_state(run.time_s, rate)
            q_ss_list.append(ss.q_ss)
            h_ss_list.append(run.h0_cm)
            ss_list.append(ss)

        K_wooding_by_h: dict[float, float] = {}
        aG_wooding: Optional[float] = None
        if len(q_ss_list) >= 2:
            r0_cm = level_runs[0].r0_cm
            K_wood, aG_wooding = _wooding_inversion(
                np.array(q_ss_list), np.array(h_ss_list), r0_cm
            )
            h_sorted = np.sort(np.array(h_ss_list))
            K_wooding_by_h = {float(hv): float(kv)
                              for hv, kv in zip(h_sorted, K_wood)}

        # ── Step 2: first-pass Philip OLS for volume runs (own VG params) ─
        fp_vol = [r.run() for r in vol_runs]

        # ── Step 3: first-pass K(h) → VG fit for A₂ ──────────────────────
        h_fp_list: list[float] = []
        K_fp_list: list[float] = []

        for h, K in K_wooding_by_h.items():       # Wooding K — correct
            h_fp_list.append(h); K_fp_list.append(K)
        for fp in fp_vol:                          # Philip OLS K from volume
            if fp.K_ols_cm_s > 0 and 'NO_VG_PARAMS' not in fp.flags:
                h_fp_list.append(fp.h0_cm); K_fp_list.append(fp.K_ols_cm_s)

        h_fp = np.array(h_fp_list)
        K_fp = np.array(K_fp_list)
        alpha_fp, n_fp = 0.05, 2.0
        if len(h_fp) >= 3:
            try:
                _, alpha_fp, n_fp = fit_vg_mualem_K(h_fp, K_fp)
            except Exception:
                pass
        elif len(h_fp) >= 2:
            try:
                _, aG_fp = fit_gardner_K(h_fp, K_fp)
                alpha_fp = float(aG_fp) * 0.5
            except Exception:
                pass

        # ── Step 4: final Philip runs with corrected A₂ ───────────────────
        final_results: list[InfiltrationResult] = []

        # Level runs — Philip secondary (large r₀ makes Philip A₂ less reliable)
        for run, ss in zip(level_runs, ss_list):
            res = run.run(alpha_vg=alpha_fp, n_vg=n_fp)
            K_w = K_wooding_by_h.get(run.h0_cm)
            if K_w is None:
                res.flags.append('NO_WOODING_K')
            if ss.method != 'cv_window':
                res.flags.append(f'NO_SS_WINDOW:{ss.method}')
            res.ss = ss
            res.K_wooding_cm_s = K_w
            res.gardner_alpha_cm = aG_wooding
            final_results.append(res)

        # Volume runs — Philip primary
        for run in vol_runs:
            res = run.run(alpha_vg=alpha_fp, n_vg=n_fp)
            final_results.append(res)

        # Restore original order (interleaved level/volume)
        run_map = {id(r): res
                   for r, res in zip(level_runs + vol_runs,
                                     final_results[:len(level_runs)]
                                     + final_results[len(level_runs):])}
        final_results = [run_map[id(r)] for r in self.runs]

        # ── K(h) model fits ───────────────────────────────────────────────
        valid = [(r.h0_cm, r.K_primary_cm_s) for r in final_results
                 if r.K_primary_cm_s and r.K_primary_cm_s > 0
                 and 'NEGATIVE_K' not in r.flags
                 and 'NO_WOODING_K' not in r.flags]

        h_arr = np.array([h for h, _ in valid])
        K_arr = np.array([k for _, k in valid])
        kh: Optional[KhFit] = None

        if len(valid) >= 2:
            try:
                Ks_g, aG = fit_gardner_K(h_arr, K_arr)
                Ks_vg, a_vg, n_vg = fit_vg_mualem_K(h_arr, K_arr)
                Ks_ko, hm_ko, sig_ko = fit_kosugi_K(h_arr, K_arr)

                h_plot = np.linspace(0.0, max(abs(h_arr)) * 1.3, 400)
                K_g_pl  = Ks_g * np.exp(-aG * h_plot)
                K_vg_pl = np.array([
                    compute_K_vg_mualem(-h, Ks_vg, a_vg, n_vg) for h in h_plot
                ])
                K_ko_pl = np.array([
                    compute_K_kosugi(-h, Ks_ko, hm_ko, sig_ko) for h in h_plot
                ])

                kh = KhFit(
                    Ks_g=Ks_g, aG=aG,
                    Ks_vg=Ks_vg, alpha_vg=a_vg, n_vg=n_vg,
                    Ks_ko=Ks_ko, hm_ko=hm_ko, sigma_ko=sig_ko,
                    h_plot=h_plot,
                    K_g_plot=K_g_pl,
                    K_vg_plot=K_vg_pl,
                    K_ko_plot=K_ko_pl,
                )
            except Exception as exc:
                warnings.warn(f"K(h) fitting failed: {exc}", stacklevel=2)

        return CampaignResult(
            site=self.site,
            results=final_results,
            kh=kh,
            h_arr=h_arr,
            K_arr=K_arr,
            alpha_fp=alpha_fp,
            n_fp=n_fp,
        )
