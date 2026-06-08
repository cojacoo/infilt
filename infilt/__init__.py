"""
infilt — tension infiltrometer analysis
========================================

Unified package for mini-disk and hood infiltrometers.

Both instruments produce (h₀, K) pairs via the same Philip two-term equation
and A₂ geometry correction.  Only the geometric parameters differ.

Quick start
-----------
Mini-disk::

    from infilt import InfiltrationRun, Campaign, MINIDISK_STUDENT
    run = InfiltrationRun(time_s, volume_mL, suction_mm=30,
                          **MINIDISK_STUDENT, soil_texture='loam')
    result = Campaign([run]).run()
    print(result.table())

Hood IL-2700::

    from infilt import InfiltrationRun, Campaign, HOOD_IL2700
    runs = [
        InfiltrationRun(t30, level30, 30, **HOOD_IL2700),
        InfiltrationRun(t60, level60, 60, **HOOD_IL2700),
        InfiltrationRun(t90, level90, 90, **HOOD_IL2700),
    ]
    result = Campaign(runs, site='Field A').run()
    result.figure().show()

Device presets
--------------
MINIDISK_STUDENT  dict(disk_radius_mm=25.0,  signal_type='volume')
MINIDISK_METER    dict(disk_radius_mm=22.5,  signal_type='volume')
HOOD_IL2700       dict(disk_radius_mm=124.0, signal_type='level',
                       reservoir_area_cm2=23.0)

References
----------
Philip (1957) Soil Science 84:257–264.
Zhang (1997) SSSAJ 61:1024–1030.
Dohnal et al. (2010) SSSAJ 74:804–811.
Wooding (1968) Water Resources Research 4:1259–1273.
Su (2025) Scientific Reports 15:20396.
Guo et al. (2026) J. Hydrology 674:135443.
"""

from .run import (
    InfiltrationRun,
    InfiltrationResult,
    MINIDISK_STUDENT,
    MINIDISK_METER,
    HOOD_IL2700,
)
from .campaign import (
    Campaign,
    CampaignResult,
    KhFit,
)
from .fitting import fit_philip, PhilipFit
from .steady import detect_steady_state, SteadyStateResult
from .soil_db import get_vg_params, list_textures
from .theory import (
    compute_A1,
    compute_A2,
    compute_K,
    compute_K_dohnal,
    compute_K_zhang,
    compute_K_wooding,
    compute_K_vg_mualem,
    fit_vg_mualem_K,
    compute_K_kosugi,
    fit_kosugi_K,
    compute_gardner_alpha,
    fit_gardner_K,
    compute_theta_vg,
    tabulate_A2,
)

from .methods import (
    REFERENCES,
    SECTIONS,
    FOOTER,
    methods_markdown,
    methods_pdf_paragraphs,
    references_pdf,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "InfiltrationRun",
    "InfiltrationResult",
    "Campaign",
    "CampaignResult",
    "KhFit",
    # Device presets
    "MINIDISK_STUDENT",
    "MINIDISK_METER",
    "HOOD_IL2700",
    # Fitting
    "fit_philip",
    "PhilipFit",
    # Steady-state
    "detect_steady_state",
    "SteadyStateResult",
    # Theory — Philip/A₂
    "compute_K",
    "compute_K_zhang",
    "compute_K_dohnal",
    "compute_A1",
    "compute_A2",
    "tabulate_A2",
    "compute_theta_vg",
    # Theory — Wooding/Gardner
    "compute_K_wooding",
    "compute_gardner_alpha",
    "fit_gardner_K",
    # Theory — K(h) models
    "compute_K_vg_mualem",
    "fit_vg_mualem_K",
    "compute_K_kosugi",
    "fit_kosugi_K",
    # Soil database
    "get_vg_params",
    "list_textures",
]
