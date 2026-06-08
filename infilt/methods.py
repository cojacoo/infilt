"""
Canonical methodology text and references for the infilt package.

Used by:
  - app.py  → Streamlit "Methods" tab (markdown) + PDF report (reportlab HTML)
  - README.md generation
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Contact / attribution
# ---------------------------------------------------------------------------

FOOTER = (
    '(cc) conrad.jackisch@tbt.tu-freiberg.de  ·  '
    'https://github.com/cojacoo/infilt'
)

# ---------------------------------------------------------------------------
# References  (text, doi_url)
# ---------------------------------------------------------------------------

REFERENCES: list[tuple[str, str]] = [
    (
        'Philip, J.R. (1957). The theory of infiltration: 4. Sorptivity and algebraic '
        'infiltration equations. *Soil Science*, 84(3), 257–264.',
        'https://doi.org/10.1097/00010694-195709000-00010',
    ),
    (
        'Zhang, R. (1997). Determination of soil sorptivity and hydraulic conductivity '
        'from the disk infiltrometer. *Soil Science Society of America Journal*, 61(4), 1024–1030.',
        'https://doi.org/10.2136/sssaj1997.03615995006100040005x',
    ),
    (
        'Dohnal, M., Dusek, J., & Vogel, T. (2010). Improving hydraulic conductivity estimates '
        'from minidisk infiltrometer measurements for soils with wide pore-size distributions. '
        '*Soil Science Society of America Journal*, 74(3), 804–811.',
        'https://doi.org/10.2136/sssaj2009.0099',
    ),
    (
        'Wooding, R.A. (1968). Steady infiltration from a shallow circular pond. '
        '*Water Resources Research*, 4(6), 1259–1273.',
        'https://doi.org/10.1029/WR004i006p01259',
    ),
    (
        'Su, L. (2025). A generalised infiltration model based on the Mittag-Leffler function '
        'for transient and steady-state flow. *Scientific Reports*, 15, 20396.',
        'https://doi.org/10.1038/s41598-025-20396-x',
    ),
    (
        'Guo, X. et al. (2026). A Mittag-Leffler infiltration model for structured soils '
        'with preferential flow. *Journal of Hydrology*, 674, 135443.',
        'https://doi.org/10.1016/j.jhydrol.2025.135443',
    ),
    (
        'Schaap, M.G., Leij, F.J., & van Genuchten, M.Th. (2001). ROSETTA: a computer '
        'program for estimating soil hydraulic parameters with hierarchical pedotransfer '
        'functions. *Journal of Hydrology*, 251(3–4), 163–176.',
        'https://doi.org/10.1016/S0022-1694(01)00466-8',
    ),
]

# ---------------------------------------------------------------------------
# Methodology sections  (heading, markdown body)
# ---------------------------------------------------------------------------

SECTIONS: list[tuple[str, str]] = [
    (
        'Steady-state detection',
        'A backward rolling-window scan (window = 4 consecutive steps, CV threshold ≤ 5%) '
        'identifies the first stationary flux sequence. '
        'q_ss is the window mean; its standard deviation (±std) characterises measurement '
        'variability at steady state. '
        'All time steps within q_ss ± std are flagged as the steady-state region (reported '
        'as SS%). '
        'When no stable window is found, a Horton exponential f = fc + b·exp(−λt) is fitted '
        'to the full series and its asymptote fc serves as q_ss. '
        'This handles series that are already flat from the first measurement as well as '
        'series with a pronounced initial transient decay.',
    ),
    (
        'Hood infiltrometer — Wooding inversion',
        'At steady state, the surface flux q_ss measured under applied tension h₀ is related '
        'to unsaturated hydraulic conductivity K(h₀) by the Wooding (1968) radial-flow equation:\n\n'
        '    K(h₀) = q_ss / (1 + 4 / (π · r₀ · αG))\n\n'
        'where r₀ is the hood radius and αG is the Gardner sorptive number. '
        'αG is estimated iteratively from consecutive K(h) pairs by log-linear regression; '
        'the inversion repeats until convergence (< 1 ppm change in K). '
        'Error propagation follows σ_K = σ_q · K / q_ss (Wooding is linear in q_ss).',
    ),
    (
        'Mini-disk infiltrometer — Philip two-term fitting',
        'Cumulative infiltration I(t) = ΔV / A_disk follows the Philip (1957) two-term model:\n\n'
        '    I(t) = C₁ · √t + C₂ · t\n\n'
        'Three fitting methods are applied: '
        '**OLS** (ordinary least squares, fixed exponent β = 0.5), '
        'the **Su method** (free β), '
        'and the **Mittag-Leffler generalisation** (Su 2025, Guo et al. 2026) which '
        'accommodates non-integer-order infiltration kinetics in structured or macroporous soils. '
        'K(h₀) = C₂ / A₂, where A₂ is the geometry correction of Zhang (1997) for van Genuchten '
        'shape parameter n ≥ 1.35, or Dohnal et al. (2010) for n < 1.35 (fine-textured soils). '
        'van Genuchten parameters (α, n) are taken from ROSETTA H1 class averages '
        '(Schaap et al. 2001, as tabulated in Dohnal et al. 2010) by USDA texture class, '
        'or from Carsel & Parrish (1988) for compatibility with the METER manual, '
        'or supplied manually.',
    ),
    (
        'K(h) model fitting and Ksat',
        'All valid (h₀, K) pairs are fitted simultaneously to three unsaturated conductivity '
        'functions:\n\n'
        '- **Gardner exponential:** K(h) = Ks · exp(αG · h)\n'
        '- **Mualem–van Genuchten:** K(h) = Ks · Se^L · [1−(1−Se^(1/m))^m]²\n'
        '- **Mualem–Kosugi:** K(h) based on log-normal pore-size distribution\n\n'
        'Saturated hydraulic conductivity **Ksat** is estimated as the mean of the Gardner '
        'and VG model Ks values extrapolated to h = 0 (ponded condition). '
        'The Kosugi Ks is reported separately as it assumes a different pore-size '
        'distribution.',
    ),
]

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def methods_markdown() -> str:
    """Full methodology as GitHub-flavoured markdown."""
    lines = ['## Methods\n']
    for heading, body in SECTIONS:
        lines.append(f'### {heading}\n')
        lines.append(body + '\n')
    lines.append('## References\n')
    for txt, doi in REFERENCES:
        # strip markdown italics for plain reference list
        plain = txt.replace('*', '')
        lines.append(f'- {plain}  \n  DOI: [{doi}]({doi})\n')
    lines.append(f'\n---\n{FOOTER}\n')
    return '\n'.join(lines)


def _to_rl(text: str) -> str:
    """Convert markdown body to reportlab-safe XML.

    Handles: **bold**, unicode symbols → ASCII-safe, XML entity escaping.
    The leading sentence (up to first period + space) is bolded automatically.
    """
    import re
    # Escape XML special chars first (before inserting tags)
    text = text.replace('&', '&amp;')
    text = (text
            .replace('≤', '&lt;=').replace('≥', '&gt;=')
            .replace('<', '&lt;').replace('>', '&gt;'))
    # Unicode → ASCII-safe
    replacements = [
        ('₀', '0'), ('₁', '1'), ('₂', '2'), ('²', '2'),
        ('π', 'pi'), ('·', '*'), ('λ', 'lambda'), ('α', 'alpha'),
        ('β', 'beta'), ('σ', 'sigma'), ('Δ', 'delta'),
        ('–', '-'), ('—', '-'), ('’', "'"),
        ('±', '+/-'), ('→', '->'), ('≠', '!='),
    ]
    for uni, asc in replacements:
        text = text.replace(uni, asc)
    # **bold** → <b>bold</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Strip remaining * (italic)
    text = text.replace('*', '')
    # Newlines → space (reportlab wraps automatically)
    text = re.sub(r'\n+', ' ', text)
    return text


def methods_pdf_paragraphs() -> list[tuple[str, str]]:
    """Return [(heading, reportlab_html_body), ...] for PDF generation."""
    return [(h, _to_rl(body)) for h, body in SECTIONS]


def references_pdf() -> list[tuple[str, str]]:
    """Return [(plain_text_no_markdown, doi_url), ...] for PDF generation."""
    return [(txt.replace('*', ''), doi) for txt, doi in REFERENCES]
