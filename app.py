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
    list_textures,
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
            f'{r.K_su_cm_s*36000:.2f}',
            f'{r.ols.r2:.4f}',
            f'{r.ss.conv_frac:.0%}' if r.ss else '—',
        ])
    elems.append(Table(comb_rows, style=ts,
                       colWidths=[14*mm, 12*mm, 18*mm, 15*mm,
                                  18*mm, 14*mm, 17*mm, 17*mm, 14*mm, 11*mm]))
    elems.append(Spacer(1, 2*mm))

    # ── K(h) model fits — list ────────────────────────────────────────────────
    if result.kh:
        ks_g   = result.kh.Ks_g  * 36000
        ks_vg  = result.kh.Ks_vg * 36000
        ks_ko  = result.kh.Ks_ko * 36000
        ks_est = (ks_g + ks_vg) / 2
        elems.append(Paragraph('K(h) Model Fits', H))
        fit_lines = [
            f'<b>Gardner:</b>  Ks = {ks_g:.2f} mm/h,  aG = {result.kh.aG:.4f} cm-1',
            f'<b>Mualem-VG:</b>  Ks = {ks_vg:.2f} mm/h,  a = {result.kh.alpha_vg:.4f} cm-1,'
            f'  n = {result.kh.n_vg:.3f}',
            f'<b>Mualem-Kosugi:</b>  Ks = {ks_ko:.2f} mm/h,  hm = {result.kh.hm_ko:.4f} cm,'
            f'  sigma = {result.kh.sigma_ko:.4f}',
            f'<b>Ksat estimate (Gardner + VG mean):</b>  {ks_est:.2f} mm/h',
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
            'K_su_mmh':         round(r.K_su_cm_s * 36000, 3),
            'r2_ols':           round(r.ols.r2, 4),
            'Ks_gardner_mmh':   round(result.kh.Ks_g  * 36000, 3) if result.kh else None,
            'Ks_vg_mmh':        round(result.kh.Ks_vg * 36000, 3) if result.kh else None,
            'Ks_est_mmh':       round((result.kh.Ks_g + result.kh.Ks_vg) / 2 * 36000, 3)
                                if result.kh else None,
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
    signal_type = st.selectbox(
        'Signal type',
        ['level', 'volume'],
        index=0 if preset_base.get('signal_type') == 'level' else 1,
    )
    reservoir_area_cm2: float | None = None
    if signal_type == 'level':
        reservoir_area_cm2 = st.number_input(
            'Reservoir cross-section [cm²]',
            value=float(preset_base.get('reservoir_area_cm2', 23.0)),
            min_value=0.1, step=0.1,
        )

    st.divider()
    st.subheader('Soil (for mini-disk A₂)')
    soil_mode = st.radio('VG params source', ['Texture class', 'Manual α, n'],
                          horizontal=True)
    soil_texture: str | None = None
    alpha_vg: float | None  = None
    n_vg: float | None      = None
    if soil_mode == 'Texture class':
        soil_texture = st.selectbox('USDA texture class', SOIL_TEXTURES,
                                     index=SOIL_TEXTURES.index('loam'))
    else:
        c1, c2 = st.columns(2)
        alpha_vg = c1.number_input('α [1/cm]', value=0.036, format='%.4f', step=0.001)
        n_vg     = c2.number_input('n [−]',     value=1.56,  format='%.3f', step=0.01)


# ---------------------------------------------------------------------------
# Main — tabs
# ---------------------------------------------------------------------------

st.title('💧 InFilt — Tension Infiltrometer Analysis')

tab_input, tab_results, tab_history, tab_methods = st.tabs(
    ['📋  Data Input', '📊  Results & Report', '📁  History', '📖  Methods']
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Data Input
# ══════════════════════════════════════════════════════════════════════════════

with tab_input:
    st.markdown(
        '**Expected columns:** `suction` [mmWC] · `time` [s] · `signal` '
        '(level [mm] for hood, volume [mL] for mini-disk).  '
        'Multiple tensions in one table (long format). Column names are flexible.'
    )

    input_mode = st.radio(
        'Input method', ['📂 Upload file', '📋 Copy-paste table'],
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
            df_tidy = pd.DataFrame({
                'site':        site_name,
                'instrument':  instr_key,
                'suction_mm':  pd.to_numeric(df_raw[suction_col], errors='coerce'),
                'signal_type': signal_type,
                'time_s':      pd.to_numeric(df_raw[time_col],    errors='coerce'),
                'signal':      pd.to_numeric(df_raw[signal_col],  errors='coerce'),
            }).dropna(subset=['suction_mm', 'time_s', 'signal'])

            tensions = sorted(df_tidy['suction_mm'].unique())
            st.success(
                f'✅ {len(df_tidy)} rows — '
                f'{len(tensions)} tension(s): {[int(t) for t in tensions]} mmWC'
            )
            st.dataframe(df_tidy.head(12), use_container_width=True)
            st.session_state['df_tidy'] = df_tidy

        except Exception as exc:
            st.error(f'Column mapping error: {exc}')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Results & Report
# ══════════════════════════════════════════════════════════════════════════════

with tab_results:
    if 'df_tidy' not in st.session_state:
        st.info('Load and map data in the **Data Input** tab first.')
    else:
        df_tidy = st.session_state['df_tidy']

        run_col, _ = st.columns([1, 4])
        if run_col.button('▶  Run Analysis', type='primary', use_container_width=True):
            preset_run: dict = {'disk_radius_mm': disk_radius_mm, 'signal_type': signal_type}
            if reservoir_area_cm2 is not None:
                preset_run['reservoir_area_cm2'] = reservoir_area_cm2

            shared_kw: dict = {}
            if signal_type == 'volume':
                if soil_texture:
                    shared_kw['soil_texture'] = soil_texture
                elif alpha_vg and n_vg:
                    shared_kw['alpha'] = alpha_vg
                    shared_kw['n']     = n_vg

            with st.spinner('Running campaign analysis …'):
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        result = Campaign.from_dataframe(
                            df_tidy,
                            site=site_name,
                            hood_preset=preset_run     if signal_type == 'level'  else None,
                            minidisk_preset=preset_run if signal_type == 'volume' else None,
                            **shared_kw,
                        ).run()

                    warn_msgs = [str(w.message) for w in caught
                                 if issubclass(w.category, UserWarning)]
                    st.session_state['result']   = result
                    st.session_state['warnings'] = warn_msgs
                    st.session_state['meta']     = {
                        'site': site_name, 'lat': lat, 'lon': lon,
                    }
                    st.success('Analysis complete.')
                except Exception as exc:
                    st.error(f'Analysis failed: {exc}')

        if 'result' in st.session_state:
            result = st.session_state['result']
            meta   = st.session_state['meta']

            # Warnings
            for msg in st.session_state.get('warnings', []):
                st.warning(msg)

            # ── Results table ─────────────────────────────────────────────
            st.subheader('Results Table')
            st.code(result.table(), language=None)

            # ── Figure ───────────────────────────────────────────────────
            st.subheader('Campaign Figure')
            fig = result.figure()
            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # ── Actions ──────────────────────────────────────────────────
            act1, act2 = st.columns(2)

            # PDF report
            with act1:
                if st.button('📄 Generate PDF Report', use_container_width=True):
                    with st.spinner('Rendering PDF …'):
                        try:
                            pdf_bytes = _generate_pdf(result, meta, df_tidy)
                            fname = (
                                f'infilt_{site_name.replace(" ","_")}_'
                                f'{datetime.now().strftime("%Y%m%d")}.pdf'
                            )
                            st.download_button(
                                '⬇ Download PDF',
                                data=pdf_bytes,
                                file_name=fname,
                                mime='application/pdf',
                                use_container_width=True,
                            )
                        except Exception as exc:
                            st.error(f'PDF generation failed: {exc}')

            # Save to log
            with act2:
                if st.button('💾 Save to results log', use_container_width=True):
                    try:
                        _append_to_log(result, meta)
                        st.success(f'Saved → {RESULTS_FILE.relative_to(Path(__file__).parent)}')
                    except Exception as exc:
                        st.error(f'Save failed: {exc}')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — History
# ══════════════════════════════════════════════════════════════════════════════

with tab_history:
    st.subheader('Results Log')

    if not RESULTS_FILE.exists():
        st.info('No results saved yet.  Run an analysis and click "Save to results log".')
    else:
        log_df = pd.read_csv(RESULTS_FILE)
        st.markdown(f'`{RESULTS_FILE.name}` — **{len(log_df)} entries** from '
                    f'{log_df["site"].nunique()} site(s)')

        # Filter by site
        sites = ['All'] + sorted(log_df['site'].unique())
        sel_site = st.selectbox('Filter by site', sites)
        if sel_site != 'All':
            log_df = log_df[log_df['site'] == sel_site]

        st.dataframe(log_df, use_container_width=True)

        # Download log
        csv_bytes = log_df.to_csv(index=False).encode()
        st.download_button(
            '⬇ Download filtered log (CSV)',
            data=csv_bytes,
            file_name='infilt_results_log.csv',
            mime='text/csv',
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Methods
# ══════════════════════════════════════════════════════════════════════════════

with tab_methods:
    st.markdown(methods_markdown())
    st.divider()
    st.caption(FOOTER)
