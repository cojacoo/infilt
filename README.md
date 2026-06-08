# infilt — Tension Infiltrometer Analysis

Unified Python package for evaluation tension infiltrometer data.
Mini-Disk and Hood instruments alike yield **(h₀, K) pairs** via the same pipeline; only geometry differs. Includes the package, a streamlit app and a demo jupyter notebook.

## Quick start

```python
import pandas as pd
from infilt import Campaign, HOOD_IL2700, MINIDISK_STUDENT

# Long-format DataFrame: columns site, instrument, suction_mm, signal_type, time_s, signal
df = pd.read_csv('my_campaign.csv')

result = Campaign.from_dataframe(df, hood_preset=HOOD_IL2700).run()
print(result.table())
result.figure().show()
```

### Streamlit app

```bash
streamlit run app.py
```

Upload or paste a table with columns `suction`, `time`, `signal`, select instrument and
signal type, enter site coordinates, and run the full analysis.
Results are displayed interactively, exportable as a PDF report, and logged to
`data/results_log.csv`.

---

## Data format

Long-format (one row per time step):

| suction_mm | time_s | signal |
|------------|--------|--------|
| 9.0        | 0      | 200.1  |
| 9.0        | 4      | 199.8  |

`signal` units: level [mm] for hood, volume [mL] for mini-disk.

---

## Methods

### Steady-state detection

A backward rolling-window scan (window = 4 consecutive steps, CV threshold ≤ 5%)
identifies the first stationary flux sequence.
q_ss is the window mean; its standard deviation (±std) characterises measurement
variability at steady state.
All time steps within q_ss ± std are flagged as the steady-state region (SS%).
When no stable window is found, a Horton exponential f = fc + b·exp(−λt) is fitted
to the full series and its asymptote fc serves as q_ss.

### Wooding inversion (original Hood Infiltrometer interpretation)

At steady state, the surface flux q_ss measured under applied tension h₀ relates to
unsaturated hydraulic conductivity K(h₀) via the Wooding (1968) radial-flow equation:

```
K(h₀) = q_ss / (1 + 4 / (π · r₀ · αG))
```

where r₀ is the hood radius and αG is the Gardner sorptive number.
αG is estimated iteratively from consecutive K(h) pairs by log-linear regression until
convergence (< 1 ppm).
Error propagation: σ_K = σ_q · K / q_ss.

### Philip two-term fitting (original Mini-Disk  interpretation)

Cumulative infiltration I(t) = ΔV / A_disk follows the Philip (1957) two-term model:

```
I(t) = C₁ · √t + C₂ · t
```

Three fitting methods: **OLS** (β = 0.5 fixed), **Su method** (free β), and
**Mittag-Leffler generalisation** (Su 2025, Guo et al. 2026) for structured/macroporous soils.

K(h₀) = C₂ / A₂, where A₂ is the geometry correction of Zhang (1997) for n ≥ 1.35,
or Dohnal et al. (2010) for n < 1.35.
van Genuchten parameters (α, n) from ROSETTA (Schaap et al. 2001 in Dohnal et al. 2010) by texture class or manual entry.

### K(h) model fitting and Ksat

All valid (h₀, K) pairs are fitted to three models:

- **Gardner exponential:** K(h) = Ks · exp(αG · h)
- **Mualem–van Genuchten:** K(h) = Ks · Seˡ · [1−(1−Se^(1/m))^m]²
- **Mualem–Kosugi:** based on log-normal pore-size distribution

**Ksat** is estimated as the mean of the Gardner and VG model Ks values at h = 0.
Kosugi Ks is reported separately.

---

## References

- Philip, J.R. (1957). The theory of infiltration: 4. Sorptivity and algebraic infiltration equations. *Soil Science*, 84(3), 257–264. <https://doi.org/10.1097/00010694-195709000-00010>
- Zhang, R. (1997). Determination of soil sorptivity and hydraulic conductivity from the disk infiltrometer. *SSSAJ*, 61(4), 1024–1030. <https://doi.org/10.2136/sssaj1997.03615995006100040005x>
- Dohnal, M., Dusek, J., & Vogel, T. (2010). Improving hydraulic conductivity estimates from minidisk infiltrometer measurements for soils with wide pore-size distributions. *SSSAJ*, 74(3), 804–811. <https://doi.org/10.2136/sssaj2009.0099>
- Wooding, R.A. (1968). Steady infiltration from a shallow circular pond. *Water Resources Research*, 4(6), 1259–1273. <https://doi.org/10.1029/WR004i006p01259>
- Su, L. (2025). A generalised infiltration model based on the Mittag-Leffler function. *Scientific Reports*, 15, 20396. <https://doi.org/10.1038/s41598-025-20396-x>
- Guo, X. et al. (2026). A Mittag-Leffler infiltration model for structured soils. *Journal of Hydrology*, 674, 135443. <https://doi.org/10.1016/j.jhydrol.2025.135443>
- Schaap, M.G., Leij, F.J., & van Genuchten, M.Th. (2001). ROSETTA: a computer program for estimating soil hydraulic parameters with hierarchical pedotransfer functions. *Journal of Hydrology*, 251(3–4), 163–176. <https://doi.org/10.1016/S0022-1694(01)00466-8>

---

(cc) conrad.jackisch@tbt.tu-freiberg.de · <https://github.com/cojacoo/infilt>
