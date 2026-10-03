"""
InFilt — Tension Infiltrometer Analysis
Streamlit app for field-data upload, analysis, reporting, and logging.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from infilt import (
    Campaign, InfiltrationRun,
    HOOD_IL2700, MINIDISK_STUDENT, MINIDISK_METER,
    list_textures, get_vg_params,
    REFERENCES, FOOTER, methods_markdown, methods_pdf_paragraphs, references_pdf,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_FILE = Path(__file__).parent / 'data' / 'results_log.csv'

INSTRUMENT_PRESETS: dict[str, dict] = {
    'UGT Hood Infiltrometer':             {**HOOD_IL2700},
    'Mini-disk TUBAF (50 mm)': {**MINIDISK_STUDENT},
    'Mini-disk METER (45 mm)':  {**MINIDISK_METER},
    'Custom':                   {'disk_radius_mm': 25.0, 'signal_type': 'volume'},
}

SOIL_TEXTURES = list_textures()

_A2_NAME = {'zhang1997': 'Zhang', 'dohnal2010': 'Dohnal'}

FOOTER_LINE = FOOTER  # from infilt.methods

# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------

def _generate_pdf(result, meta: dict, df: pd.DataFrame) -> bytes:
    """Produce a single-page A4 PDF report via reportlab."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import (
        Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    DARK  = HexColor('#2c3e50')
    MID   = HexColor('#34495e')
    LIGHT = HexColor('#bdc3c7')
    STRIPE = HexColor('#f5f5f5')
    WHITE = colors.white

    # Styles — all ASCII-safe (Helvetica is Latin-1 only)
    T = ParagraphStyle('T', fontName='Helvetica-Bold',   fontSize=12, spaceAfter=2*mm,
                       textColor=DARK, leading=16)
    H = ParagraphStyle('H', fontName='Helvetica-Bold',   fontSize=7.5,  spaceBefore=2*mm,
                       spaceAfter=1*mm, textColor=MID)
    B = ParagraphStyle('B', fontName='Helvetica',        fontSize=6.5,  leading=8.5, textColor=DARK)
    S = ParagraphStyle('S', fontName='Helvetica-Oblique',fontSize=6,leading=8,
                       textColor=HexColor('#555555'))
    LK= ParagraphStyle('LK',fontName='Helvetica',        fontSize=6,leading=8,
                       textColor=HexColor('#1a5276'))

    ts = TableStyle([
        ('BACKGROUND',     (0, 0), (-1, 0),  MID),
        ('TEXTCOLOR',      (0, 0), (-1, 0),  WHITE),
        ('FONTNAME',       (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',       (0, 0), (-1, -1), 7),
        ('BACKGROUND',     (0, 1), (-1, -1), WHITE),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, STRIPE]),
        ('TEXTCOLOR',      (0, 1), (-1, -1), DARK),
        ('GRID',           (0, 0), (-1, -1), 0.3, LIGHT),
        ('ALIGN',          (1, 0), (-1, -1), 'RIGHT'),
        ('VALIGN',         (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING',     (0, 0), (-1, -1), 1.2*mm),
        ('BOTTOMPADDING',  (0, 0), (-1, -1), 1.2*mm),
        ('LEFTPADDING',    (0, 0), (-1, -1), 1.5*mm),
        ('RIGHTPADDING',   (0, 0), (-1, -1), 1.5*mm),
    ])

    # Footer drawn on canvas at fixed position
    def _draw_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 6.5)
        canvas.setFillColor(HexColor('#888888'))
        canvas.line(20*mm, 14*mm, A4[0] - 20*mm, 14*mm)
        canvas.drawString(20*mm, 10*mm, FOOTER_LINE)
        canvas.restoreState()

    # Header cell style for two-line column labels (white text on dark background)
    HC = ParagraphStyle('HC', fontName='Helvetica-Bold', fontSize=6.5,
                        leading=8, textColor=WHITE, alignment=1)

    def _h(line1, line2=''):
        """Two-line table header cell."""
        return Paragraph(f'{line1}<br/>{line2}' if line2 else line1, HC)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=20*mm, leftMargin=20*mm,
                            topMargin=15*mm,   bottomMargin=20*mm)
    elems = []

    # ── Title + meta ─────────────────────────────────────────────────────────
    date_str = datetime.now().strftime('%d %b %Y, %H:%M')
    elems.append(Paragraph('InFilt — Tension Infiltrometer Analysis Report', T))
    tensions = sorted(df['suction_mm'].unique())
    instr    = df['instrument'].iloc[0] if 'instrument' in df.columns else '—'
    sig_type = df['signal_type'].iloc[0] if 'signal_type' in df.columns else '—'
    elems.append(Paragraph(
        f"<b>Site:</b> {meta.get('site','—')} &nbsp;"
        f"<b>Coords:</b> {meta.get('lat',0):.5f} N / {meta.get('lon',0):.5f} E &nbsp;"
        f"<b>Instrument:</b> {instr} ({sig_type}) &nbsp;"
        f"<b>Tensions:</b> {', '.join(str(int(t)) for t in tensions)} mmWC &nbsp;"
        f"<b>Generated:</b> {date_str}",
        B,
    ))
    elems.append(Spacer(1, 2*mm))

    # ── Combined q_ss + K(h) table ────────────────────────────────────────────
    elems.append(Paragraph('Steady-State Flux and Hydraulic Conductivity', H))
    comb_rows = [[
        _h('h0', '[cm]'),
        _h('Suct.', '[mm]'),
        _h('q_ss', '[mm/h]'),
        _h('+/-std', '[mm/h]'),
        _h('K_Wood', '[mm/h]'),
        _h('+/-sK', '[mm/h]'),
        _h('K_OLS', '[mm/h]'),
        _h('K_Su', '[mm/h]'),
        _h('R2', 'OLS'),
        _h('SS', '%'),
        _h('A2', ''),
    ]]
    for r in sorted(result.results, key=lambda x: x.h0_cm):
        se_K = 0.0
        if r.ss and r.K_wooding_cm_s and r.ss.q_ss > 0:
            se_K = r.ss.q_ss_se * 36000 * (r.K_wooding_cm_s / r.ss.q_ss)
        comb_rows.append([
            f'{r.h0_cm:.2f}',
            f'{r.suction_mm:.0f}',
            f'{r.ss.q_ss*36000:.2f}'    if r.ss else '—',
            f'{r.ss.q_ss_se*36000:.2f}' if r.ss else '—',
            f'{r.K_wooding_cm_s*36000:.2f}' if r.K_wooding_cm_s else '—',
            f'{se_K:.2f}' if se_K > 0 else '—',
            f'{r.K_ols_cm_s*36000:.2f}',
            f'{r.K_su_cm_s*36000:.2f}' if r.K_su_cm_s else '—',
            f'{r.ols.r2:.4f}',
            f'{r.ss.conv_frac:.0%}' if r.ss else '—',
            _A2_NAME.get(r.fit_formula, '—'),
        ])
    elems.append(Table(comb_rows, style=ts,
                       colWidths=[13*mm, 11*mm, 16*mm, 14*mm,
                                  16*mm, 13*mm, 16*mm, 16*mm, 13*mm, 10*mm, 14*mm]))
    elems.append(Spacer(1, 2*mm))

    # ── K(h) model fits — list ────────────────────────────────────────────────
    if result.kh:
        kh = result.kh
        na = 'n/a (needs >= 4 tensions)'
        elems.append(Paragraph('K(h) Model Fits', H))
        fit_lines = [
            f'<b>Gardner:</b>  Ks = {kh.Ks_g*36000:.2f} mm/h,  aG = {kh.aG:.4f} cm-1',
            f'<b>Mualem-VG:</b>  '
            + (f'Ks = {kh.Ks_vg*36000:.2f} mm/h,  a = {kh.alpha_vg:.4f} cm-1,'
               f'  n = {kh.n_vg:.3f}' if kh.Ks_vg is not None else na),
            f'<b>Mualem-Kosugi:</b>  '
            + (f'Ks = {kh.Ks_ko*36000:.2f} mm/h,  hm = {kh.hm_ko:.4f} cm,'
               f'  sigma = {kh.sigma_ko:.4f}' if kh.Ks_ko is not None else na),
            f'<b>Ksat estimate (Gardner + VG mean):</b>  {kh.Ks_est*36000:.2f} mm/h',
        ]
        for line in fit_lines:
            elems.append(Paragraph(line, B))
        elems.append(Spacer(1, 2*mm))

    # ── Figure ────────────────────────────────────────────────────────────────
    elems.append(Paragraph('Campaign Figure', H))
    try:
        fig = result.figure()
        fig_bytes = fig.to_image(format='png', width=1600, height=520, scale=2)
        img = Image(io.BytesIO(fig_bytes))
        img.drawWidth  = 170*mm
        img.drawHeight = 55*mm
        elems.append(img)
    except Exception:
        elems.append(Paragraph('[Figure unavailable — pip install kaleido]', S))
    elems.append(Spacer(1, 2*mm))

    # ── Methodology ──────────────────────────────────────────────────────────
    elems.append(Paragraph('Methodology', H))
    for heading, body_rl in methods_pdf_paragraphs():
        elems.append(Paragraph(f'<b>{heading}.</b>  {body_rl}', B))
        elems.append(Spacer(1, 1*mm))

    # ── References ───────────────────────────────────────────────────────────
    elems.append(Paragraph('References', H))
    for txt, doi in references_pdf():
        elems.append(Paragraph(
            f'{txt} <link href="{doi}" color="#1a5276">{doi}</link>',
            S,
        ))

    doc.build(elems, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Results log
# ---------------------------------------------------------------------------

def _append_to_log(result, meta: dict) -> None:
    rows = []
    for r in result.results:
        se_K = 0.0
        if r.ss and r.K_wooding_cm_s and r.ss.q_ss > 0:
            se_K = float(r.ss.q_ss_se * 36000 * (r.K_wooding_cm_s / r.ss.q_ss))
        rows.append({
            'timestamp':        datetime.now().isoformat(timespec='seconds'),
            'site':             meta.get('site', ''),
            'lat':              meta.get('lat', float('nan')),
            'lon':              meta.get('lon', float('nan')),
            'suction_mm':       r.suction_mm,
            'h0_cm':            r.h0_cm,
            'signal_type':      r.signal_type,
            'q_ss_mmh':         round(r.ss.q_ss * 36000, 3) if r.ss else None,
            'q_ss_se_mmh':      round(r.ss.q_ss_se * 36000, 3) if r.ss else None,
            'ss_method':        r.ss.method if r.ss else None,
            'ss_frac':          round(r.ss.conv_frac, 3) if r.ss else None,
            'K_wooding_mmh':    round(r.K_wooding_cm_s * 36000, 3) if r.K_wooding_cm_s else None,
            'sigma_K_mmh':      round(se_K, 3) if se_K else None,
            'K_ols_mmh':        round(r.K_ols_cm_s * 36000, 3),
            'K_su_mmh':         round(r.K_su_cm_s * 36000, 3) if r.K_su_cm_s else None,
            'r2_ols':           round(r.ols.r2, 4),
            'A2_formula':       r.fit_formula if r.fit_formula != 'n/a' else None,
            'vg_alpha':         r.alpha,
            'vg_n':             r.n,
            'Ks_gardner_mmh':   round(result.kh.Ks_g  * 36000, 3) if result.kh else None,
            'Ks_vg_mmh':        round(result.kh.Ks_vg * 36000, 3)
                                if result.kh and result.kh.Ks_vg is not None else None,
            'Ks_est_mmh':       round(result.kh.Ks_est * 36000, 3) if result.kh else None,
        })

    new_df = pd.DataFrame(rows)
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    if RESULTS_FILE.exists():
        existing = pd.read_csv(RESULTS_FILE)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(RESULTS_FILE, index=False)


# ---------------------------------------------------------------------------
# Data parsing helpers
# ---------------------------------------------------------------------------

def _parse_text(text: str) -> pd.DataFrame | None:
    """Try common separators; return DataFrame or None."""
    text = text.strip()
    if not text:
        return None
    for sep in [',', ';', '\t', r'\s+']:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, engine='python')
            if len(df.columns) >= 2 and len(df) >= 2:
                return df
        except Exception:
            pass
    return None


def _parse_upload(uploaded) -> pd.DataFrame | None:
    name = uploaded.name.lower()
    if name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(uploaded)
    content = uploaded.read().decode('latin-1')
    return _parse_text(content)


def _to_seconds(val) -> float | None:
    """Parse a time value into seconds.

    Accepts:
      - plain (possibly decimal) seconds, e.g. '4', '4.5', '4,5'
      - relative clock strings 'HH:MM:SS' or 'MM:SS', with optional
        decimal seconds, e.g. '00:00:01', '00:00:01.5', '01:30'
    Returns None if the value can't be parsed (row will be dropped).
    """
    if val is None:
        return None
    s = str(val).strip().replace(',', '.')
    if s in ('', 'nan', 'NaT', 'None'):
        return None
    if ':' in s:
        parts = s.split(':')
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            return None
        if len(nums) == 3:
            h, m, sec = nums
        elif len(nums) == 2:
            h, (m, sec) = 0.0, nums
        else:
            return None
        return h * 3600 + m * 60 + sec
    try:
        return float(s)
    except ValueError:
        return None


def _to_float(val) -> float | None:
    """Parse a signal value, tolerating trailing annotations like '10,3 (5 min)'."""
    if val is None:
        return None
    s = str(val).strip().replace(',', '.')
    if s in ('', 'nan', 'None'):
        return None
    s = s.split()[0]
    try:
        return float(s)
    except ValueError:
        return None


def _parse_multi_csv(uploaded) -> tuple[list[dict], list[str]]:
    """Parse a multi-run CSV with the following 4-row header structure:

    Row 1: site/run name  (repeated across two columns per tension)
    Row 2: soil texture   (repeated)
    Row 3: suction value  in cm (0 / 1 / 3 …)
    Row 4: column labels  ('Zeit', 'Wasserstand [cm *10]')
    Row 5+: data

    Returns (runs, warnings) where each run is a dict with keys:
        site, texture, df  (tidy DataFrame with suction_mm / time_s / signal columns)
    """
    content = uploaded.read().decode('latin-1')
    raw = pd.read_csv(io.StringIO(content), header=None, dtype=str)

    runs: list[dict] = []
    warnings_out: list[str] = []

    n_cols = len(raw.columns)

    # Walk through column pairs
    col_idx = 0
    while col_idx + 1 < n_cols:
        site_val    = str(raw.iloc[0, col_idx]).strip()
        texture_val = str(raw.iloc[1, col_idx]).strip()
        suction_raw = str(raw.iloc[2, col_idx]).strip()

        # Skip empty / filler columns
        if site_val in ('', 'nan') or suction_raw in ('', 'nan'):
            col_idx += 1
            continue

        # Convert suction: stored in cm, app expects mm
        try:
            suction_cm = float(suction_raw.replace(',', '.'))
            suction_mm = suction_cm * 10
        except ValueError:
            warnings_out.append(
                f"Column {col_idx}: could not parse suction '{suction_raw}' — skipped."
            )
            col_idx += 2
            continue

        # Extract the two data columns (time, signal)
        time_col   = raw.iloc[4:, col_idx].reset_index(drop=True)
        signal_col = raw.iloc[4:, col_idx + 1].reset_index(drop=True)

        # Parse time — supports HH:MM:SS, MM:SS, and plain (decimal) seconds
        times   = time_col.map(_to_seconds)
        signals = signal_col.map(_to_float)

        df_pair = pd.DataFrame({
            'suction_mm': suction_mm,
            'time_s':     times,
            'signal':     signals,
        }).dropna(subset=['time_s', 'signal'])

        # Warn only about rows that actually contained a value but failed to
        # parse (as opposed to blank padding at the end of a shorter column).
        def _had_content(v) -> bool:
            return str(v).strip().lower() not in ('', 'nan', 'none')

        n_bad = sum(
            1 for tv, sv, t, s in zip(time_col, signal_col, times, signals)
            if (_had_content(tv) or _had_content(sv)) and (t is None or s is None)
        )
        if n_bad > 0:
            warnings_out.append(
                f"Run '{site_val}' suction {suction_mm:.0f} mm: {n_bad} row(s) "
                "had an unparseable time or signal value and were skipped."
            )

        # Convert signal: stored as cm×10 (i.e. mm×10 → need mm)
        # Wasserstand [cm *10] means the value is level in units of 0.1 cm = 1 mm
        # so the raw number already equals mm directly (e.g. 14.8 → 14.8 mm)
        # No conversion needed — signal is already in mm (level in mm).

        if df_pair.empty:
            warnings_out.append(
                f"Run '{site_val}' suction {suction_mm:.0f} mm: no valid data rows — skipped."
            )
            col_idx += 2
            continue

        # Re-base time to 0 at first measurement
        df_pair['time_s'] = df_pair['time_s'] - df_pair['time_s'].iloc[0]

        # Find existing run for this site or create new one
        existing = next((r for r in runs if r['site'] == site_val), None)
        if existing is None:
            runs.append({
                'site':    site_val,
                'texture': texture_val,
                'df':      df_pair,
            })
        else:
            existing['df'] = pd.concat(
                [existing['df'], df_pair], ignore_index=True
            )

        col_idx += 2

    return runs, warnings_out


def _auto_map(cols: list[str]) -> tuple[str, str, str]:
    """Guess suction / time / signal column names."""
    def _pick(candidates, fallback_idx):
        for c in candidates:
            for col in cols:
                if c in col.lower():
                    return col
        return cols[min(fallback_idx, len(cols) - 1)]

    suction = _pick(['suction', 'tens', 'pressure', 'h'], 0)
    time    = _pick(['time', 'zeit', 't_s', 'sec'], 1)
    signal  = _pick(['signal', 'level', 'vol', 'height', 'mm', 'ml'], 2)
    return suction, time, signal


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title='InFilt — Tension Infiltrometer Analysis',
    page_icon='💧',
    layout='wide',
)

# ---------------------------------------------------------------------------
# Sidebar — instrument & site configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title('⚙️ Configuration')

    st.subheader('Site & Location')
    site_name = st.text_input('Site name / ID', 'Site_01')
    col_a, col_b = st.columns(2)
    lat = col_a.number_input('Latitude °N', value=0.0, format='%.6f', step=0.000001)
    lon = col_b.number_input('Longitude °E', value=0.0, format='%.6f', step=0.000001)

    st.divider()
    st.subheader('Instrument')
    instr_key = st.selectbox('Device', list(INSTRUMENT_PRESETS.keys()))
    preset_base = INSTRUMENT_PRESETS[instr_key].copy()

    disk_radius_mm = st.number_input(
        'Disk / hood radius [mm]',
        value=float(preset_base['disk_radius_mm']),
        min_value=1.0, step=0.5,
    )
    # signal_type selects the evaluation: 'level' → hood (Wooding steady state),
    # 'volume' → mini-disk (Philip transient). Fixed by the device preset.
    signal_type = preset_base['signal_type']
    if instr_key == 'Custom':
        evaluation = st.radio(
            'Evaluation',
            ['Mini-disk (Philip, transient)', 'Hood (Wooding, steady state)'],
            help='Mini-disk: reading = volume in the Mariotte tube. '
                 'Hood: reading = water level in the supply reservoir.',
        )
        signal_type = 'level' if evaluation.startswith('Hood') else 'volume'

    # Mini-disk read as a water level in the tube → mL via tube cross-section
    signal_factor = 1.0
    if signal_type == 'volume':
        md_reading = st.radio('Mini-disk reading',
                              ['Volume [mL]', 'Water level in tube'], horizontal=True,
                              help='What you wrote down in the field: the mL scale, '
                                   'or a water level that is converted via the tube cross-section.')
        if md_reading == 'Water level in tube':
            lc1, lc2 = st.columns(2)
            tube_area_cm2 = lc1.number_input(
                'Tube cross-section [cm²]', value=None, min_value=0.01,
                step=0.1, format='%.3f',
                help='Inner cross-section of the reservoir tube, π·(ID/2)². '
                     'Measure on the device.',
            )
            level_unit = lc2.selectbox('Level unit', ['mm', 'cm'])
            if tube_area_cm2 is None:
                st.error('Enter the tube cross-section to convert level → mL.')
                st.stop()
            signal_factor = tube_area_cm2 * (0.1 if level_unit == 'mm' else 1.0)
            st.caption(f'mL = level × {signal_factor:g}')

    suction_offset_cm = st.number_input(
        'Suction offset [cm]', value=0.0, step=0.1, format='%.2f',
        help='Added to every suction setting, e.g. 0.5 if setting "0" actually '
             'applies 0.5 cm suction at the disk. Device-specific — check calibration.',
    )
    reservoir_area_cm2: float | None = None
    if signal_type == 'level':
        reservoir_area_cm2 = st.number_input(
            'Reservoir cross-section [cm²]',
            value=float(preset_base.get('reservoir_area_cm2', 23.0)),
            min_value=0.1, step=0.1,
            help='Hood reading = water level [mm] in the supply reservoir.',
        )

    st.divider()
    st.subheader('Soil (for mini-disk A₂)')
    soil_mode = st.radio('VG params source', ['Texture class', 'Manual α, n'],
                          horizontal=True)
    soil_texture: str | None = None
    use_file_texture = False
    alpha_vg: float | None  = None
    n_vg: float | None      = None
    if soil_mode == 'Texture class':
        soil_texture = st.selectbox('USDA texture class', SOIL_TEXTURES,
                                     index=SOIL_TEXTURES.index('loam'))
        use_file_texture = st.checkbox(
            'Use per-site texture from multi-run file', value=True,
            help='Falls back to the class above if the file texture is unknown.',
        )
    else:
        c1, c2 = st.columns(2)
        alpha_vg = c1.number_input('α [1/cm]', value=0.036, format='%.4f', step=0.001)
        n_vg     = c2.number_input('n [−]',     value=1.56,  format='%.3f', step=0.01)


# ---------------------------------------------------------------------------
# Main — tabs
# ---------------------------------------------------------------------------

st.title('💧 InFilt — Tension Infiltrometer Analysis')
st.caption(
    'Workflow: set instrument, soil and site in the **sidebar** → load data in **Data Input** '
    '→ verify in **Data Check** → evaluate in **Results & Report** → compare saved runs in '
    '**History**. Sidebar settings apply live to all loaded data.'
)


def _intro(what: str, why: str, how: str) -> None:
    """Short 'what / why / how' brief at the top of a tab."""
    st.info(f'**What:** {what}  \n**Why:** {why}  \n**How:** {how}')

tab_input, tab_check, tab_results, tab_history, tab_methods = st.tabs(
    ['📋  Data Input', '🔍  Data Check', '📊  Results & Report', '📁  History', '📖  Methods']
)


def _run_key(df: pd.DataFrame) -> pd.Series:
    """'site @ suction' label per row (raw suction setting)."""
    return df['site'].astype(str) + ' @ ' + df['suction_mm'].map(lambda x: f'{x:g}') + ' mm'


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Apply current sidebar conversions and exclusions to raw loaded data.

    Done at use time (not load time) so changing a setting after loading
    always takes effect.
    """
    keep = ~_run_key(df).isin(st.session_state.get('excluded', []))
    return df[keep].assign(
        signal=df['signal'] * signal_factor,
        suction_mm=df['suction_mm'] + suction_offset_cm * 10,
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Data Input
# ══════════════════════════════════════════════════════════════════════════════

with tab_input:
    _intro(
        'Reads raw field readings (time and reservoir reading per suction step) into the app.',
        'All later steps work on one tidy table: site · suction · time · reading.',
        '**Upload file** or **Copy-paste** for one site in long format — columns `suction` '
        '[mm], `time` [s or HH:MM:SS], `signal` (hood: level [mm]; mini-disk: mL or tube '
        'level, see sidebar); then map the columns. **File with multiple Raw Data** for the '
        'wide field sheet with many sites; then click *Load ALL runs*.',
    )

    input_mode = st.radio(
        'Input method', ['📂 Upload file', '📋 Copy-paste table', "File with multiple Raw Data"],
        horizontal=True,
    )

    df_raw: pd.DataFrame | None = None

    if input_mode == '📂 Upload file':
        uploaded = st.file_uploader(
            'CSV, TSV, or Excel', type=['csv', 'txt', 'tsv', 'xlsx', 'xls']
        )
        if uploaded:
            df_raw = _parse_upload(uploaded)
            if df_raw is None:
                st.error('Could not parse file — check separator and encoding.')

    elif input_mode == "File with multiple Raw Data":
        st.markdown(
            '**Expected format:** 4-row header per column-pair — '
            '**Row 1:** site name · **Row 2:** soil texture · '
            '**Row 3:** suction [cm] · **Row 4:** `Zeit` / `Wasserstand [cm *10]`. '
            'Each site can have multiple tension pairs side-by-side.'
        )
        uploaded = st.file_uploader(
            'Multi-run CSV', type=['csv', 'txt', 'tsv']
        )
        if uploaded:
            with st.spinner('Parsing multi-run file …'):
                runs, parse_warnings = _parse_multi_csv(uploaded)

            for w in parse_warnings:
                st.warning(w)

            if not runs:
                st.error('No valid runs found — check the file format.')
            else:
                unique_names = sorted({r['site'] for r in runs})
                total_tensions = sum(r['df']['suction_mm'].nunique() for r in runs)
                st.success(
                    f'Found **{len(unique_names)} site(s)** with '
                    f'**{total_tensions} tension series** total.'
                )

                # Preview table of all runs
                preview_rows = []
                for r in runs:
                    tensions_list = sorted(r['df']['suction_mm'].unique())
                    preview_rows.append({
                        'Site': r['site'],
                        'Texture': r['texture'],
                        'Tensions [mmWC]': str([int(t) for t in tensions_list]),
                        'Rows': len(r['df']),
                    })
                st.dataframe(pd.DataFrame(preview_rows), width='stretch')

                if st.button('✅ Load ALL runs into analysis', type='primary'):
                    # Build a combined tidy df with all runs, tagged by site
                    all_dfs = []
                    for r in runs:
                        df_r = pd.DataFrame({
                            'site':        r['site'],
                            'instrument':  instr_key,
                            'suction_mm':  r['df']['suction_mm'],
                            'signal_type': signal_type,
                            'time_s':      r['df']['time_s'],
                            'signal':      r['df']['signal'],
                            'texture':     r['texture'],
                        })
                        all_dfs.append(df_r)
                    combined = pd.concat(all_dfs, ignore_index=True)
                    st.session_state['multi_runs']    = runs          # raw run list
                    st.session_state['multi_df_tidy'] = combined      # combined tidy df
                    st.session_state.pop('multi_results', None)       # clear old results
                    st.success(
                        f'✅ Loaded {len(unique_names)} runs ({len(combined)} rows total) '
                        f'→ switch to **Results & Report** tab to run the analysis.'
                    )

    else:
        pasted = st.text_area(
            'Paste table (CSV, TSV, or space-separated; first row = header)',
            height=220,
            placeholder=(
                'suction,time,signal\n'
                '9,0,200.10\n'
                '9,4,199.85\n'
                '26,0,201.30\n'
                '26,4,201.05\n'
                '...'
            ),
        )
        if pasted.strip():
            df_raw = _parse_text(pasted)
            if df_raw is None:
                st.error('Could not parse text — check format.')

    if df_raw is not None:
        st.markdown(f'**Detected:** {len(df_raw)} rows, columns: `{list(df_raw.columns)}`')

        # Column mapping
        st.markdown('#### Map columns')
        cols = list(df_raw.columns)
        auto_s, auto_t, auto_sig = _auto_map(cols)
        c1, c2, c3 = st.columns(3)
        suction_col = c1.selectbox('Suction [mmWC]', cols, index=cols.index(auto_s))
        time_col    = c2.selectbox('Time [s]',        cols, index=cols.index(auto_t))
        signal_col  = c3.selectbox('Signal',          cols, index=cols.index(auto_sig))

        try:
            time_parsed = pd.to_numeric(
                df_raw[time_col].map(_to_seconds), errors='coerce'
            )
            n_time_bad = df_raw[time_col].notna().sum() - time_parsed.notna().sum()
            if n_time_bad > 0:
                st.warning(
                    f'⚠️ {n_time_bad} row(s) had a time value that could not be '
                    'parsed (expected plain seconds or HH:MM:SS / MM:SS) and will '
                    'be dropped.'
                )

            df_tidy = pd.DataFrame({
                'site':        site_name,
                'instrument':  instr_key,
                'suction_mm':  pd.to_numeric(df_raw[suction_col], errors='coerce'),
                'signal_type': signal_type,
                'time_s':      time_parsed,
                'signal':      pd.to_numeric(df_raw[signal_col],  errors='coerce'),
            }).dropna(subset=['suction_mm', 'time_s', 'signal'])

            tensions = sorted(df_tidy['suction_mm'].unique())
            st.success(
                f'✅ {len(df_tidy)} rows — '
                f'{len(tensions)} tension(s): {[int(t) for t in tensions]} mmWC'
            )
            st.dataframe(df_tidy.head(12), width='stretch')
            st.session_state['df_tidy'] = df_tidy

        except Exception as exc:
            st.error(f'Column mapping error: {exc}')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1b — Data Check
# ══════════════════════════════════════════════════════════════════════════════

with tab_check:
    import plotly.graph_objects as go

    _intro(
        'Shows exactly what enters the analysis per site and suction — after time parsing, '
        'level → mL conversion and suction offset.',
        'Typos, wrong units or disturbed runs bias K; catch them before evaluating.',
        'Pick site and suction, check table and curve (reading should fall steadily). '
        'Put bad runs into **Exclude runs**, then (re-)run the analysis.',
    )

    frames = []
    if 'df_tidy' in st.session_state:
        frames.append(st.session_state['df_tidy'])
    if 'multi_df_tidy' in st.session_state:
        frames.append(st.session_state['multi_df_tidy'])

    if not frames:
        st.info('Load data in the **📋 Data Input** tab first.')
    else:
        raw_df = pd.concat(frames, ignore_index=True)
        st.multiselect(
            '🚫 Exclude runs from analysis', sorted(_run_key(raw_df).unique()),
            key='excluded',
            help='E.g. disturbed runs or instrument problems. Re-run the analysis afterwards.',
        )
        check_df = _prepare(raw_df)

        sites = sorted(check_df['site'].astype(str).unique())
        sel_site = st.selectbox('Site', sites, key='check_site')

        site_df = check_df[check_df['site'].astype(str) == sel_site]
        tensions = sorted(site_df['suction_mm'].unique())
        sel_suction = st.selectbox(
            'Suction level',
            tensions,
            format_func=lambda x: f'{int(x)} mmWC',
            key='check_suction',
        )

        sub = (
            site_df[site_df['suction_mm'] == sel_suction]
            .sort_values('time_s')
            .reset_index(drop=True)
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Rows loaded', len(sub))
        m2.metric(
            'Time range [s]',
            f"{sub['time_s'].min():.1f} – {sub['time_s'].max():.1f}"
            if not sub.empty else '—',
        )
        m3.metric(
            'Signal range',
            f"{sub['signal'].min():.3f} – {sub['signal'].max():.3f}"
            if not sub.empty else '—',
        )
        m4.metric(
            'Conversion applied',
            f"× {signal_factor:g}, +{suction_offset_cm:g} cm",
        )

        st.dataframe(
            sub[['time_s', 'signal']].rename(
                columns={'time_s': 'time [s]',
                         'signal': f'signal ({sub["signal_type"].iloc[0]})'
                                   if not sub.empty else 'signal'}
            ),
            width='stretch',
            height=320,
        )

        if not sub.empty:
            fig_check = go.Figure(go.Scatter(
                x=sub['time_s'], y=sub['signal'],
                mode='lines+markers', line=dict(color='#4C72B0'),
            ))
            fig_check.update_layout(
                title=f'{sel_site} — {int(sel_suction)} mmWC — raw signal vs. time',
                xaxis_title='time [s]',
                yaxis_title=f"signal ({sub['signal_type'].iloc[0]})",
                height=380,
            )
            st.plotly_chart(fig_check, width='stretch')

        with st.expander('📄 Show full loaded table for this site (all tensions)'):
            st.dataframe(
                site_df[['suction_mm', 'time_s', 'signal']]
                .sort_values(['suction_mm', 'time_s'])
                .reset_index(drop=True),
                width='stretch',
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Results & Report
# ══════════════════════════════════════════════════════════════════════════════

with tab_results:
    import plotly.graph_objects as go
    import plotly.express as px

    _intro(
        'Computes K(h₀) per suction (hood: Wooding from steady-state flux; mini-disk: Philip '
        'fit + Zhang/Dohnal A₂), fits K(h) models and estimates Ksat.',
        'K near saturation characterises macropore and matrix flow; Ksat is the h = 0 limit.',
        'Click **Run / Analyse ALL**, read the table and figure, check the ⚠ flags '
        '(e.g. `LOW_R2`, `FEW_PTS`, `NEGATIVE_K`), then download the PDF or **save** to the '
        'log. See the **Methods** tab for equations.',
    )

    has_single = 'df_tidy' in st.session_state
    has_multi  = 'multi_df_tidy' in st.session_state

    if not has_single and not has_multi:
        st.info('Load and map data in the **Data Input** tab first.')

    # ── shared helpers (defined inside the tab so they close over sidebar vars) ──

    def _make_preset() -> dict:
        p: dict = {'disk_radius_mm': disk_radius_mm, 'signal_type': signal_type}
        if reservoir_area_cm2 is not None:
            p['reservoir_area_cm2'] = reservoir_area_cm2
        return p

    def _make_shared_kw(df_in: pd.DataFrame) -> dict:
        kw: dict = {}
        if signal_type == 'volume':
            if soil_texture:
                kw['soil_texture'] = soil_texture
                if use_file_texture and 'texture' in df_in.columns:
                    tex = str(df_in['texture'].iloc[0])
                    try:
                        get_vg_params(tex)
                        kw['soil_texture'] = tex
                    except ValueError:
                        warnings.warn(f"Unknown texture '{tex}' in file — "
                                      f"using '{soil_texture}'.")
            elif alpha_vg and n_vg:
                kw['alpha'] = alpha_vg
                kw['n']     = n_vg
        return kw

    def _run_campaign(df_in: pd.DataFrame, site_label: str):
        """Run Campaign for one tidy df. Returns (result, warn_msgs)."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            shared_kw = _make_shared_kw(df_in)
            result = Campaign.from_dataframe(
                df_in,
                site=site_label,
                hood_preset=_make_preset()     if signal_type == 'level'  else None,
                minidisk_preset=_make_preset() if signal_type == 'volume' else None,
                **shared_kw,
            ).run()
        warn_msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        return result, warn_msgs

    def _summary_rows_for(site_label: str, r_obj) -> list[dict]:
        rows = []
        for r in r_obj.results:
            se_K = 0.0
            if r.ss and r.K_wooding_cm_s and r.ss.q_ss > 0:
                se_K = r.ss.q_ss_se * 36000 * (r.K_wooding_cm_s / r.ss.q_ss)
            rows.append({
                'Site':           site_label,
                'Suction [mmWC]': int(r.suction_mm),
                'q_ss [mm/h]':    round(r.ss.q_ss * 36000, 2) if r.ss else None,
                'K_Wood [mm/h]':  round(r.K_wooding_cm_s * 36000, 2) if r.K_wooding_cm_s else None,
                '±σK [mm/h]':     round(se_K, 2) if se_K > 0 else None,
                'K_OLS [mm/h]':   round(r.K_ols_cm_s * 36000, 2),
                'K_Su [mm/h]':    round(r.K_su_cm_s * 36000, 2) if r.K_su_cm_s else None,
                'R²':             round(r.ols.r2, 4),
                'A₂':             _A2_NAME.get(r.fit_formula),
                'SS%':            f'{r.ss.conv_frac:.0%}' if r.ss else None,
                'Ks_est [mm/h]':  round(r_obj.kh.Ks_est * 36000, 2)
                                  if r_obj.kh else None,
                'Flags':          ' '.join(r.flags) or '✓',
            })
        return rows

    def _table(rows: list[dict]) -> pd.DataFrame:
        """Rows → DataFrame without columns that are empty for every run
        (e.g. hood-only q_ss / Wooding columns for mini-disk data)."""
        return pd.DataFrame(rows).dropna(axis=1, how='all')

    def _kh_table(kh) -> pd.DataFrame:
        mmh = lambda k: round(k * 36000, 2) if k is not None else None
        return pd.DataFrame([
            {'Model': 'Gardner', 'Ks [mm/h]': mmh(kh.Ks_g),
             'Parameters': f'αG = {kh.aG:.4f} 1/cm'},
            {'Model': 'Mualem-VG', 'Ks [mm/h]': mmh(kh.Ks_vg),
             'Parameters': f'α = {kh.alpha_vg:.4f} 1/cm, n = {kh.n_vg:.3f}'
                           if kh.Ks_vg is not None else 'not fitted (needs ≥ 4 tensions)'},
            {'Model': 'Mualem-Kosugi', 'Ks [mm/h]': mmh(kh.Ks_ko),
             'Parameters': f'hm = {kh.hm_ko:.3f} cm, σ = {kh.sigma_ko:.3f}'
                           if kh.Ks_ko is not None else 'not fitted (needs ≥ 4 tensions)'},
            {'Model': 'Ksat estimate (Gardner + VG mean)', 'Ks [mm/h]': mmh(kh.Ks_est),
             'Parameters': ''},
        ])

    def _show_run_detail(site_label: str, result, meta: dict, df_site: pd.DataFrame,
                         warn_msgs: list[str], key_suffix: str):
        """Render the full detail view for one run: warnings, table, figure, actions."""
        for msg in warn_msgs:
            st.warning(msg)

        st.markdown(f'#### 📍 {site_label}')

        st.plotly_chart(result.figure(), width='stretch')
        st.caption(
            'A: infiltration rate per run. Hood: smoothed rate, steady-state band and '
            'Wooding K. Mini-disk: rate between readings (□), Philip rate (dashed) and its '
            'gravity asymptote C₂ (dotted), K = C₂/A₂. '
            'B: K per tension and fitted K(h) curves; ◆ = Ksat at h = 0.'
        )

        st.markdown('**Per tension**')
        st.dataframe(
            _table(_summary_rows_for(site_label, result)).drop(
                columns=['Site', 'Ks_est [mm/h]'], errors='ignore'),
            width='stretch', hide_index=True,
        )
        if result.kh:
            st.markdown('**K(h) fits and Ksat**')
            st.dataframe(_kh_table(result.kh), width='stretch', hide_index=True)
        else:
            st.info('K(h) fit needs ≥ 2 valid tensions.')
        with st.expander('Full text table (all methods, steady-state details, A₂ parameters)'):
            st.code(result.table(), language=None)

        st.divider()
        act1, act2 = st.columns(2)

        with act1:
            if st.button('📄 Generate PDF Report', key=f'pdf_{key_suffix}',
                         width='stretch'):
                with st.spinner('Rendering PDF …'):
                    try:
                        pdf_bytes = _generate_pdf(result, meta, df_site)
                        fname = (
                            f'infilt_{site_label.replace(" ","_")}_'
                            f'{datetime.now().strftime("%Y%m%d")}.pdf'
                        )
                        st.download_button(
                            '⬇ Download PDF', data=pdf_bytes,
                            file_name=fname, mime='application/pdf',
                            width='stretch', key=f'dl_pdf_{key_suffix}',
                        )
                    except Exception as exc:
                        st.error(f'PDF generation failed: {exc}')

        with act2:
            if st.button('💾 Save to results log', key=f'save_{key_suffix}',
                         width='stretch'):
                try:
                    _append_to_log(result, meta)
                    st.success(f'Saved → {RESULTS_FILE.relative_to(Path(__file__).parent)}')
                except Exception as exc:
                    st.error(f'Save failed: {exc}')

    # ════════════════════════════════════════════════════════════════════════
    # MULTI-RUN PATH
    # ════════════════════════════════════════════════════════════════════════
    if has_multi:
        st.subheader('Multi-Run Analysis')

        multi_df_tidy = _prepare(st.session_state['multi_df_tidy'])
        all_sites     = sorted(multi_df_tidy['site'].unique())

        st.caption(
            f'{len(all_sites)} sites loaded — '
            f'{len(multi_df_tidy)} rows total'
        )

        run_col, _ = st.columns([1, 4])
        if run_col.button('▶  Analyse ALL runs', type='primary', width='stretch'):
            multi_results: dict = {}
            bar = st.progress(0, text='Starting …')
            errors: list[str] = []
            for i, site_label in enumerate(all_sites):
                bar.progress(i / len(all_sites), text=f'Analysing {site_label} …')
                df_site = multi_df_tidy[multi_df_tidy['site'] == site_label].copy()
                try:
                    result, warn_msgs = _run_campaign(df_site, site_label)
                    multi_results[site_label] = {
                        'result':   result,
                        'warnings': warn_msgs,
                        'df':       df_site,
                        'meta':     {'site': site_label, 'lat': lat, 'lon': lon},
                    }
                except Exception as exc:
                    errors.append(f'**{site_label}:** {exc}')
            bar.progress(1.0, text='Done.')
            bar.empty()
            st.session_state['multi_results'] = multi_results
            for e in errors:
                st.error(e)
            n_ok = len(multi_results)
            st.success(f'✅ {n_ok}/{len(all_sites)} runs analysed successfully.')

        # ── Show results once analysis has been run ───────────────────────
        if st.session_state.get('multi_results'):
            multi_results = st.session_state['multi_results']
            done_sites    = sorted(multi_results.keys())

            st.divider()

            # ── 1. Summary table ──────────────────────────────────────────
            st.subheader('Summary — All Runs')
            all_summary_rows: list[dict] = []
            for s in done_sites:
                all_summary_rows.extend(
                    _summary_rows_for(s, multi_results[s]['result'])
                )
            summary_df = _table(all_summary_rows)
            st.dataframe(summary_df, width='stretch', height=300, hide_index=True)
            st.download_button(
                '⬇ Download summary CSV',
                data=summary_df.to_csv(index=False).encode(),
                file_name='infilt_multi_summary.csv',
                mime='text/csv',
            )

            st.divider()

            # ── 2. Cross-run comparison charts ────────────────────────────
            st.subheader('Cross-Run Comparison')

            # Ks_est per site (bar)
            ks_rows = [
                {'Site': s,
                 'Ks_est [mm/h]': round(multi_results[s]['result'].kh.Ks_est * 36000, 2)}
                for s in done_sites
                if multi_results[s]['result'].kh
            ]
            if ks_rows:
                fig_ks = px.bar(
                    pd.DataFrame(ks_rows), x='Site', y='Ks_est [mm/h]',
                    color='Site', text='Ks_est [mm/h]',
                    title='Ksat estimate (Gardner+VG mean) per site',
                    height=380,
                )
                fig_ks.update_traces(textposition='outside')
                fig_ks.update_layout(showlegend=False, xaxis_tickangle=-35)
                st.plotly_chart(fig_ks, width='stretch')

            # K_Wood vs suction — one line per site
            kw_rows = []
            for s in done_sites:
                for r in multi_results[s]['result'].results:
                    if r.K_wooding_cm_s:
                        kw_rows.append({
                            'Site': s,
                            'Suction [mmWC]': int(r.suction_mm),
                            'K_Wood [mm/h]': round(r.K_wooding_cm_s * 36000, 3),
                        })
            if kw_rows:
                fig_kw = px.line(
                    pd.DataFrame(kw_rows),
                    x='Suction [mmWC]', y='K_Wood [mm/h]',
                    color='Site', markers=True,
                    title='K Wooding vs. Suction — all sites',
                    height=400,
                )
                st.plotly_chart(fig_kw, width='stretch')

            st.divider()

            # ── 3. Per-run detail (selector) ──────────────────────────────
            st.subheader('Inspect Individual Run')
            sel_run = st.selectbox(
                'Select run', done_sites, key='multi_sel_run'
            )
            v = multi_results[sel_run]
            _show_run_detail(
                sel_run, v['result'], v['meta'], v['df'],
                v['warnings'], key_suffix=f'multi_{sel_run}'
            )

            st.divider()

            # ── 4. Save all ───────────────────────────────────────────────
            if st.button('💾 Save ALL runs to results log',
                         key='save_all_multi', width='content'):
                saved, failed = 0, []
                for s in done_sites:
                    try:
                        _append_to_log(multi_results[s]['result'], multi_results[s]['meta'])
                        saved += 1
                    except Exception as exc:
                        failed.append(f'{s}: {exc}')
                if saved:
                    st.success(f'Saved {saved} run(s) → '
                               f'{RESULTS_FILE.relative_to(Path(__file__).parent)}')
                for f in failed:
                    st.error(f)

    # ════════════════════════════════════════════════════════════════════════
    # SINGLE-RUN PATH
    # ════════════════════════════════════════════════════════════════════════
    elif has_single:
        df_tidy = _prepare(st.session_state['df_tidy'])

        run_col, _ = st.columns([1, 4])
        if run_col.button('▶  Run Analysis', type='primary', width='stretch'):
            with st.spinner('Running campaign analysis …'):
                try:
                    result, warn_msgs = _run_campaign(df_tidy, site_name)
                    st.session_state['result']   = result
                    st.session_state['warnings'] = warn_msgs
                    st.session_state['meta']     = {'site': site_name, 'lat': lat, 'lon': lon}
                    st.success('Analysis complete.')
                except Exception as exc:
                    st.error(f'Analysis failed: {exc}')

        if 'result' in st.session_state:
            _show_run_detail(
                st.session_state['meta']['site'],
                st.session_state['result'],
                st.session_state['meta'],
                df_tidy,
                st.session_state.get('warnings', []),
                key_suffix='single',
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — History
# ══════════════════════════════════════════════════════════════════════════════

with tab_history:
    import plotly.graph_objects as go
    import plotly.express as px

    _intro(
        f'Shows all runs saved to `{RESULTS_FILE.name}`.',
        'Compare sites, tensions and campaigns over time.',
        'Choose one site for detailed charts or *All sites* for cross-site comparison; '
        'download the table as CSV. Saving the same run twice adds duplicate rows.',
    )
    st.subheader('Results Log')

    if not RESULTS_FILE.exists():
        st.info('No results saved yet. Run an analysis and click "Save to results log".')
    else:
        log_df = pd.read_csv(RESULTS_FILE)

        st.markdown(
            f'`{RESULTS_FILE.name}` — **{len(log_df)} rows** · '
            f'**{log_df["site"].nunique()} site(s)** · '
            f'**{log_df["suction_mm"].nunique()} tension level(s)**'
        )

        # ── Site selector ─────────────────────────────────────────────────
        site_opts   = ['All sites'] + sorted(log_df['site'].unique().tolist())
        sel_h_site  = st.selectbox('Select site', site_opts, key='hist_site')
        view_df     = log_df if sel_h_site == 'All sites' else log_df[log_df['site'] == sel_h_site].copy()

        # ── Data table ────────────────────────────────────────────────────
        with st.expander('📋 Raw data table', expanded=(sel_h_site != 'All sites')):
            st.dataframe(view_df, width='stretch')
            st.download_button(
                '⬇ Download CSV',
                data=view_df.to_csv(index=False).encode(),
                file_name='infilt_results_log.csv',
                mime='text/csv',
                key='hist_dl',
            )

        st.divider()

        # ════════════════════════════════════════════════════════════════
        # SINGLE SITE — detailed charts
        # ════════════════════════════════════════════════════════════════
        if sel_h_site != 'All sites' and not view_df.empty:
            st.subheader(f'📊 Charts — {sel_h_site}')

            tensions_avail = sorted(view_df['suction_mm'].dropna().unique())

            # 1. Hydraulic conductivity vs suction
            k_methods = {
                'K_wooding_mmh': ('K Wooding', '#1f77b4'),
                'K_ols_mmh':     ('K OLS',     '#ff7f0e'),
                'K_su_mmh':      ('K Su',      '#2ca02c'),
            }
            k_present = [c for c in k_methods if c in view_df.columns]
            if k_present:
                fig_k = go.Figure()
                for col in k_present:
                    label, color = k_methods[col]
                    sub = view_df[['suction_mm', col]].dropna()
                    fig_k.add_trace(go.Scatter(
                        x=sub['suction_mm'], y=sub[col],
                        mode='lines+markers', name=label,
                        line=dict(color=color),
                        marker=dict(color=color, size=9),
                    ))
                fig_k.update_layout(
                    title='Hydraulic Conductivity vs. Suction',
                    xaxis_title='Suction [mmWC]',
                    yaxis_title='K [mm/h]',
                    legend_title='Method',
                    height=400,
                )
                st.plotly_chart(fig_k, width='stretch')

            # 2. q_ss bar chart with error bars
            if 'q_ss_mmh' in view_df.columns:
                sub_q = view_df[['suction_mm', 'q_ss_mmh', 'q_ss_se_mmh']].dropna(subset=['q_ss_mmh'])
                if not sub_q.empty:
                    fig_q = go.Figure(go.Bar(
                        x=sub_q['suction_mm'].astype(str) + ' mmWC',
                        y=sub_q['q_ss_mmh'],
                        error_y=dict(
                            type='data',
                            array=sub_q['q_ss_se_mmh'].fillna(0).tolist(),
                            visible=True,
                        ),
                        marker_color='#4C72B0',
                        text=sub_q['q_ss_mmh'].round(2),
                        textposition='outside',
                    ))
                    fig_q.update_layout(
                        title='Steady-State Flux (q_ss) per Suction Level',
                        xaxis_title='Suction',
                        yaxis_title='q_ss [mm/h]',
                        height=360,
                    )
                    st.plotly_chart(fig_q, width='stretch')

            # 3. Ksat estimates comparison
            ks_map = {
                'Ks_gardner_mmh': ('Ks Gardner', '#5599cc'),
                'Ks_vg_mmh':      ('Ks Mualem-VG', '#88bb44'),
                'Ks_est_mmh':     ('Ks estimate', '#ee8833'),
            }
            ks_present = [c for c in ks_map if c in view_df.columns]
            if ks_present:
                ks_vals  = {ks_map[c][0]: view_df[c].dropna().mean() for c in ks_present}
                ks_colors = [ks_map[c][1] for c in ks_present]
                fig_ks = go.Figure(go.Bar(
                    x=list(ks_vals.keys()),
                    y=list(ks_vals.values()),
                    marker_color=ks_colors,
                    text=[f'{v:.2f}' for v in ks_vals.values()],
                    textposition='outside',
                ))
                fig_ks.update_layout(
                    title='Ksat Estimates',
                    yaxis_title='Ks [mm/h]',
                    height=340,
                )
                st.plotly_chart(fig_ks, width='stretch')

            # 4. R² quality indicator per tension
            if 'r2_ols' in view_df.columns:
                sub_r2 = view_df[['suction_mm', 'r2_ols']].dropna()
                if not sub_r2.empty:
                    fig_r2 = go.Figure(go.Bar(
                        x=sub_r2['suction_mm'].astype(str) + ' mmWC',
                        y=sub_r2['r2_ols'],
                        marker_color=[
                            '#2ecc71' if v >= 0.95 else '#f39c12' if v >= 0.85 else '#e74c3c'
                            for v in sub_r2['r2_ols']
                        ],
                        text=sub_r2['r2_ols'].round(4),
                        textposition='outside',
                    ))
                    fig_r2.add_hline(y=0.95, line_dash='dash', line_color='green',
                                     annotation_text='R²=0.95')
                    fig_r2.update_layout(
                        title='OLS Fit Quality (R²) per Suction Level',
                        xaxis_title='Suction',
                        yaxis_title='R²',
                        yaxis_range=[0, 1.05],
                        height=320,
                    )
                    st.plotly_chart(fig_r2, width='stretch')

        # ════════════════════════════════════════════════════════════════
        # ALL SITES — cross-site comparison charts
        # ════════════════════════════════════════════════════════════════
        elif sel_h_site == 'All sites' and not view_df.empty:
            st.subheader('📊 Cross-Site Comparison')

            # 1. Ks_est per site
            if 'Ks_est_mmh' in view_df.columns:
                ks_site = (
                    view_df.dropna(subset=['Ks_est_mmh'])
                    .groupby('site')['Ks_est_mmh']
                    .mean()
                    .reset_index()
                    .rename(columns={'Ks_est_mmh': 'Ks_est [mm/h]'})
                    .sort_values('Ks_est [mm/h]', ascending=False)
                )
                if not ks_site.empty:
                    fig_ks_all = px.bar(
                        ks_site, x='site', y='Ks_est [mm/h]',
                        color='site', text='Ks_est [mm/h]',
                        title='Mean Ksat estimate per site',
                        height=400,
                    )
                    fig_ks_all.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                    fig_ks_all.update_layout(showlegend=False, xaxis_tickangle=-35)
                    st.plotly_chart(fig_ks_all, width='stretch')

            # 2. K_Wood vs suction — one line per site
            if 'K_wooding_mmh' in view_df.columns:
                sub_kw = view_df[['site', 'suction_mm', 'K_wooding_mmh']].dropna()
                if not sub_kw.empty:
                    fig_kw_all = px.line(
                        sub_kw.sort_values(['site', 'suction_mm']),
                        x='suction_mm', y='K_wooding_mmh',
                        color='site', markers=True,
                        title='K Wooding vs. Suction — all sites',
                        labels={'suction_mm': 'Suction [mmWC]',
                                'K_wooding_mmh': 'K Wooding [mm/h]'},
                        height=420,
                    )
                    st.plotly_chart(fig_kw_all, width='stretch')

            # 3. q_ss heatmap: sites × suctions
            if 'q_ss_mmh' in view_df.columns:
                pivot = (
                    view_df[['site', 'suction_mm', 'q_ss_mmh']]
                    .dropna()
                    .pivot_table(index='site', columns='suction_mm',
                                 values='q_ss_mmh', aggfunc='mean')
                )
                if not pivot.empty:
                    fig_hm = px.imshow(
                        pivot,
                        text_auto='.2f',
                        color_continuous_scale='Blues',
                        title='q_ss [mm/h] — Sites × Suctions',
                        labels={'x': 'Suction [mmWC]', 'y': 'Site',
                                'color': 'q_ss [mm/h]'},
                        aspect='auto',
                        height=max(300, 60 * len(pivot)),
                    )
                    st.plotly_chart(fig_hm, width='stretch')

            # 4. R² heatmap
            if 'r2_ols' in view_df.columns:
                pivot_r2 = (
                    view_df[['site', 'suction_mm', 'r2_ols']]
                    .dropna()
                    .pivot_table(index='site', columns='suction_mm',
                                 values='r2_ols', aggfunc='mean')
                )
                if not pivot_r2.empty:
                    fig_r2_all = px.imshow(
                        pivot_r2,
                        text_auto='.3f',
                        color_continuous_scale='RdYlGn',
                        range_color=[0.7, 1.0],
                        title='OLS R² — Sites × Suctions',
                        labels={'x': 'Suction [mmWC]', 'y': 'Site', 'color': 'R²'},
                        aspect='auto',
                        height=max(300, 60 * len(pivot_r2)),
                    )
                    st.plotly_chart(fig_r2_all, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Methods
# ══════════════════════════════════════════════════════════════════════════════

with tab_methods:
    _intro(
        'Equations, assumptions and references used by the evaluation.',
        'To judge and cite the results correctly.',
        'Read before interpreting K; the same text is included in the PDF report.',
    )
    st.markdown(methods_markdown())
    st.divider()
    st.caption(FOOTER)
