"""
Van Genuchten parameter databases for the 12 USDA texture classes.

Two databases:

- **Carsel & Parrish (1988)** — classic 12-class parameters used in the
  METER Mini Disk Infiltrometer manual (Table 2).
- **ROSETTA H1** (Schaap et al. 2001) — texture-class averages from ~2134
  samples; used in Dohnal et al. (2010) Table 1 for the verification study.

References
----------
Carsel, R.F. & Parrish, R.S. (1988). Developing joint probability
    distributions of soil water retention characteristics.
    Water Resources Research 24(5):755-769.
Schaap, M.G., Leij, F.J. & van Genuchten, M.Th. (2001). ROSETTA: a
    computer program for estimating soil hydraulic parameters with
    hierarchical pedotransfer functions. J. Hydrology 251:163-176.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Carsel & Parrish (1988)
# alpha [1/cm], n [-], thr [-], ths [-], ks [cm/h]
# ---------------------------------------------------------------------------
_CARSEL_PARRISH: dict[str, dict] = {
    'sand':            {'alpha': 0.145, 'n': 2.68, 'thr': 0.045, 'ths': 0.430, 'ks': 29.70},
    'loamy_sand':      {'alpha': 0.124, 'n': 2.28, 'thr': 0.057, 'ths': 0.410, 'ks': 14.59},
    'sandy_loam':      {'alpha': 0.075, 'n': 1.89, 'thr': 0.065, 'ths': 0.410, 'ks':  4.42},
    'loam':            {'alpha': 0.036, 'n': 1.56, 'thr': 0.078, 'ths': 0.430, 'ks':  1.04},
    'silt':            {'alpha': 0.016, 'n': 1.37, 'thr': 0.034, 'ths': 0.460, 'ks':  0.25},
    'silt_loam':       {'alpha': 0.020, 'n': 1.41, 'thr': 0.067, 'ths': 0.450, 'ks':  0.45},
    'sandy_clay_loam': {'alpha': 0.059, 'n': 1.48, 'thr': 0.100, 'ths': 0.390, 'ks':  1.31},
    'clay_loam':       {'alpha': 0.019, 'n': 1.31, 'thr': 0.095, 'ths': 0.410, 'ks':  0.26},
    'silty_clay_loam': {'alpha': 0.010, 'n': 1.23, 'thr': 0.089, 'ths': 0.430, 'ks':  0.07},
    'sandy_clay':      {'alpha': 0.027, 'n': 1.23, 'thr': 0.100, 'ths': 0.380, 'ks':  0.12},
    'silty_clay':      {'alpha': 0.005, 'n': 1.09, 'thr': 0.070, 'ths': 0.360, 'ks':  0.02},
    'clay':            {'alpha': 0.008, 'n': 1.09, 'thr': 0.068, 'ths': 0.380, 'ks':  0.20},
}

# ---------------------------------------------------------------------------
# ROSETTA H1 class averages — Schaap et al. (2001)
# Values from Dohnal et al. (2010) Table 1; ks in cm/h
# ---------------------------------------------------------------------------
_ROSETTA: dict[str, dict] = {
    'sand':            {'alpha': 0.035, 'n': 3.18, 'thr': 0.053, 'ths': 0.375, 'ks': 26.791},
    'loamy_sand':      {'alpha': 0.035, 'n': 1.75, 'thr': 0.049, 'ths': 0.390, 'ks':  4.380},
    'sandy_loam':      {'alpha': 0.027, 'n': 1.45, 'thr': 0.039, 'ths': 0.387, 'ks':  1.594},
    'loam':            {'alpha': 0.011, 'n': 1.47, 'thr': 0.061, 'ths': 0.399, 'ks':  0.502},
    'silt':            {'alpha': 0.007, 'n': 1.68, 'thr': 0.050, 'ths': 0.489, 'ks':  1.823},
    'silt_loam':       {'alpha': 0.005, 'n': 1.66, 'thr': 0.065, 'ths': 0.439, 'ks':  0.761},
    'sandy_clay_loam': {'alpha': 0.021, 'n': 1.33, 'thr': 0.063, 'ths': 0.384, 'ks':  0.550},
    'clay_loam':       {'alpha': 0.016, 'n': 1.41, 'thr': 0.079, 'ths': 0.442, 'ks':  0.341},
    'silty_clay_loam': {'alpha': 0.008, 'n': 1.52, 'thr': 0.090, 'ths': 0.482, 'ks':  0.463},
    'sandy_clay':      {'alpha': 0.033, 'n': 1.21, 'thr': 0.117, 'ths': 0.385, 'ks':  0.473},
    'silty_clay':      {'alpha': 0.016, 'n': 1.32, 'thr': 0.111, 'ths': 0.481, 'ks':  0.401},
    'clay':            {'alpha': 0.015, 'n': 1.25, 'thr': 0.098, 'ths': 0.459, 'ks':  0.615},
}

_ALIASES: dict[str, str] = {
    # abbreviations
    's':    'sand',
    'ls':   'loamy_sand',
    'sl':   'sandy_loam',
    'l':    'loam',
    'si':   'silt',
    'sil':  'silt_loam',
    'scl':  'sandy_clay_loam',
    'cl':   'clay_loam',
    'sicl': 'silty_clay_loam',
    'sc':   'sandy_clay',
    'sic':  'silty_clay',
    'c':    'clay',
    # natural language
    'loamy sand':       'loamy_sand',
    'sandy loam':       'sandy_loam',
    'silt loam':        'silt_loam',
    'sandy clay loam':  'sandy_clay_loam',
    'clay loam':        'clay_loam',
    'silty clay loam':  'silty_clay_loam',
    'sandy clay':       'sandy_clay',
    'silty clay':       'silty_clay',
}

DATABASES: dict[str, dict] = {
    'carsel_parrish': _CARSEL_PARRISH,
    'cp':             _CARSEL_PARRISH,
    'rosetta':        _ROSETTA,
}


def list_textures() -> list[str]:
    """Return sorted list of all available USDA texture class names."""
    return sorted(_CARSEL_PARRISH.keys())


def get_vg_params(soil_texture: str, db: str = 'carsel_parrish') -> dict:
    """
    Van Genuchten parameters for a USDA texture class.

    Parameters
    ----------
    soil_texture : str
        Texture class name or abbreviation (case-insensitive).
        E.g. 'silt_loam', 'sil', 'Silt Loam'.
    db : str
        ``'carsel_parrish'`` (default, matches METER manual) or ``'rosetta'``.

    Returns
    -------
    dict
        Keys: ``alpha`` [1/cm], ``n`` [-], ``thr`` [-], ``ths`` [-],
        ``ks`` [cm/h], ``m`` [-].

    Raises
    ------
    ValueError
        If texture or database name is not recognised.
    """
    if db not in DATABASES:
        raise ValueError(f"db must be one of {list(DATABASES)}; got '{db}'")

    database = DATABASES[db]
    key = _ALIASES.get(soil_texture.lower(),
                       soil_texture.lower().replace(' ', '_'))

    if key not in database:
        available = sorted(set(list(database) + list(_ALIASES)))
        raise ValueError(
            f"Texture '{soil_texture}' not found.\n"
            f"Available: {available}"
        )

    params = database[key].copy()
    params['m'] = 1.0 - 1.0 / params['n']
    return params
