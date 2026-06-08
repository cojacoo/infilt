"""
Tests for the infilt package.

Reference values:
- METER Mini Disk Infiltrometer Manual Table 2 (A₂ values)
- METER Manual worked example (K = 3.53×10⁻⁴ cm s⁻¹)
- Dohnal et al. (2010) Table 2 (RE values for ZH method)
"""

import numpy as np
import pytest
import warnings

from infilt import (
    InfiltrationRun,
    Campaign,
    HOOD_IL2700,
    MINIDISK_STUDENT,
    MINIDISK_METER,
    compute_A2,
    compute_K,
    compute_K_dohnal,
    compute_K_wooding,
    compute_K_zhang,
    compute_gardner_alpha,
    compute_theta_vg,
    fit_philip,
    get_vg_params,
    tabulate_A2,
)
from infilt.steady import detect_steady_state


# ---------------------------------------------------------------------------
# Soil database
# ---------------------------------------------------------------------------

def test_get_vg_params_cp():
    p = get_vg_params("silt_loam")
    assert abs(p["alpha"] - 0.020) < 1e-6
    assert abs(p["n"] - 1.41) < 1e-6
    assert "m" in p


def test_get_vg_params_aliases():
    assert get_vg_params("sil") == get_vg_params("silt_loam")
    assert get_vg_params("Silt Loam") == get_vg_params("silt_loam")


def test_get_vg_params_rosetta():
    p = get_vg_params("sand", db="rosetta")
    assert abs(p["alpha"] - 0.035) < 1e-6
    assert abs(p["n"] - 3.18) < 1e-6


def test_get_vg_params_invalid():
    with pytest.raises(ValueError):
        get_vg_params("not_a_soil")


# ---------------------------------------------------------------------------
# Theory — A₂ table (METER Manual Table 2, r₀ = 2.25 cm)
# ---------------------------------------------------------------------------

class TestA2Table:
    CASES = [
        ("silt_loam",  0.020, 1.41, 2.0, 7.93),
        ("silt_loam",  0.020, 1.41, 1.0, 7.37),
        ("silt_loam",  0.020, 1.41, 3.0, 8.53),
        ("sand",       0.145, 2.68, 1.0, 2.40),
        ("loam",       0.036, 1.56, 2.0, 6.27),
        ("clay_loam",  0.019, 1.31, 1.0, 6.11),
    ]

    @pytest.mark.parametrize("name,alpha,n,h_s,expected", CASES)
    def test_a2(self, name, alpha, n, h_s, expected):
        A2 = compute_A2(alpha, n, r0_cm=2.25, h0_cm=-h_s)
        assert abs(A2 - expected) / expected < 0.01, (
            f"{name}: A₂={A2:.3f}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Theory — METER manual worked example
# ---------------------------------------------------------------------------

def test_meter_manual_example():
    """METER manual: silt loam, 2 cm suction, C₂ = 0.0028 → K = 3.53×10⁻⁴."""
    p = get_vg_params("silt_loam")
    K, method = compute_K(0.0028, p["alpha"], p["n"], r0_cm=2.25, h0_cm=-2.0)
    assert method == "zhang1997"
    assert abs(K - 3.53e-4) / 3.53e-4 < 0.01, f"K={K:.4e}"


def test_compute_K_negative_C2_raises():
    p = get_vg_params("silt_loam")
    with pytest.raises(ValueError, match="C₂"):
        compute_K(-1e-4, p["alpha"], p["n"], 2.25, -2.0)


def test_zhang_formula_for_n_ge_135():
    _, method = compute_K(1e-4, 0.020, 1.5, 2.5, -1.0)
    assert method == "zhang1997"


def test_dohnal_formula_for_n_lt_135():
    _, method = compute_K(1e-4, 0.010, 1.23, 2.5, -1.0)
    assert method == "dohnal2010"


def test_ponded_suction_both_formulas():
    K_zh, _ = compute_K(1e-4, 0.020, 1.5, 2.5, 0.0)
    K_do, _ = compute_K(1e-4, 0.010, 1.23, 2.5, 0.0)
    assert K_zh > 0
    assert K_do > 0


# ---------------------------------------------------------------------------
# Theory — theta VG
# ---------------------------------------------------------------------------

def test_theta_vg_saturated():
    assert compute_theta_vg(0.0, 0.02, 1.41, 0.067, 0.45) == pytest.approx(0.45)


def test_theta_vg_unsaturated():
    θ = compute_theta_vg(-2.0, 0.02, 1.41, 0.067, 0.45)
    assert 0.067 < θ < 0.45


# ---------------------------------------------------------------------------
# Fitting — Philip 2-term
# ---------------------------------------------------------------------------

SAMPLE_TIME = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300], dtype=float)
SAMPLE_VOL  = np.array([95, 89, 86, 83, 80, 77, 75, 73, 71, 69, 67], dtype=float)


def _I_cm(vol_mL, disk_radius_cm=2.25):
    A = np.pi * disk_radius_cm ** 2
    return (vol_mL[0] - vol_mL) / A


class TestPhilipFit:
    def test_ols_returns_positive_C2(self):
        fit = fit_philip(SAMPLE_TIME, _I_cm(SAMPLE_VOL), method="ols")
        assert fit.C2 > 0
        assert fit.r2 > 0.99

    def test_nls_close_to_ols(self):
        I = _I_cm(SAMPLE_VOL)
        ols = fit_philip(SAMPLE_TIME, I, method="ols")
        nls = fit_philip(SAMPLE_TIME, I, method="nls")
        assert abs(nls.C2 - ols.C2) / ols.C2 < 0.05

    def test_dl_returns_fit(self):
        fit = fit_philip(SAMPLE_TIME, _I_cm(SAMPLE_VOL), method="dl")
        assert fit.method == "dl"
        assert fit.r2 > 0.90

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            fit_philip(SAMPLE_TIME, _I_cm(SAMPLE_VOL), method="bogus")

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            fit_philip([0, 30], [0.1, 0.5], method="ols")

    def test_ols_beta_and_lambda_defaults(self):
        fit = fit_philip(SAMPLE_TIME, _I_cm(SAMPLE_VOL), method="ols")
        assert fit.beta == 0.5
        assert fit.lambda_s == 0.0

    def test_ml_returns_valid_fit(self):
        fit = fit_philip(SAMPLE_TIME, _I_cm(SAMPLE_VOL), method="ml")
        assert fit.method == "ml"
        assert fit.C2 > 0
        assert fit.C1 > 0
        assert 0.05 <= fit.beta <= 1.0
        assert fit.lambda_s > 0
        assert fit.r2 > 0.90

    def test_ml_horton_limit(self):
        t = np.linspace(0, 300, 50)
        fc, b, lam = 5e-5, 8e-4, 0.02
        I = fc * t + b * (1 - np.exp(-lam * t)) / lam
        fit = fit_philip(t, I, method="ml")
        assert fit.beta > 0.85, f"beta={fit.beta:.3f}"
        assert abs(fit.C2 - fc) / fc < 0.05

    def test_ml_lambda_s_nonzero(self):
        fit = fit_philip(SAMPLE_TIME, _I_cm(SAMPLE_VOL), method="ml")
        assert fit.lambda_s > 0


# ---------------------------------------------------------------------------
# InfiltrationRun — volume signal (mini-disk)
# ---------------------------------------------------------------------------

class TestInfiltrationRunVolume:
    def _run(self, **kw):
        defaults = dict(suction_mm=20, disk_radius_mm=22.5, soil_texture="silt_loam")
        defaults.update(kw)
        return InfiltrationRun(
            SAMPLE_TIME, SAMPLE_VOL, signal_type='volume', **defaults
        )

    def test_basic_run(self):
        res = self._run().run()
        assert res.K_ols_cm_s > 0
        assert res.ols.r2 > 0.99
        assert res.fit_formula == "zhang1997"

    def test_student_device_suctions(self):
        for suction in [0, 10, 30]:
            res = InfiltrationRun(
                SAMPLE_TIME, SAMPLE_VOL, suction_mm=suction,
                **MINIDISK_STUDENT, soil_texture="loam",
            ).run()
            assert res.K_ols_cm_s > 0

    def test_manual_vg_params(self):
        res = InfiltrationRun(
            SAMPLE_TIME, SAMPLE_VOL, suction_mm=10,
            signal_type='volume', alpha=0.020, n=1.41,
        ).run()
        assert res.soil_texture is None
        assert res.K_ols_cm_s > 0

    def test_dohnal_path_n_low(self):
        p = get_vg_params("silty_clay_loam")
        res = InfiltrationRun(
            SAMPLE_TIME, SAMPLE_VOL, suction_mm=10,
            signal_type='volume', alpha=p["alpha"], n=p["n"],
        ).run()
        assert res.fit_formula == "dohnal2010"

    def test_unit_conversion_properties(self):
        res = self._run().run()
        assert abs(res.K_ols_mmh - res.K_ols_cm_s * 36000) < 1e-10
        assert abs(res.K_ols_ms  - res.K_ols_cm_s * 1e-2)  < 1e-14

    def test_invalid_suction_raises(self):
        with pytest.raises(ValueError):
            InfiltrationRun(SAMPLE_TIME, SAMPLE_VOL, suction_mm=-5,
                            signal_type='volume', soil_texture="silt_loam")

    def test_no_vg_params_sets_flag(self):
        res = InfiltrationRun(
            SAMPLE_TIME, SAMPLE_VOL, suction_mm=10, signal_type='volume',
        ).run()
        assert 'NO_VG_PARAMS' in res.flags

    def test_ols_beta_and_lambda(self):
        res = self._run().run()
        assert res.ols.beta == 0.5
        assert res.ols.lambda_s == 0.0

    def test_ml_result_populated(self):
        res = self._run().run()
        assert res.ml.method == "ml"
        assert res.K_ml_cm_s is not None
        assert res.K_ml_cm_s > 0
        assert 0.05 <= res.ml.beta <= 1.0
        assert res.ml.lambda_s > 0

    def test_all_three_fits_computed(self):
        res = self._run().run()
        assert res.ols.C2 > 0
        assert res.su.C2  > 0
        assert res.ml.C2  > 0


# ---------------------------------------------------------------------------
# Steady-state detection (direct)
# ---------------------------------------------------------------------------

HOOD_TIME    = np.arange(0, 600 + 30, 30, dtype=float)


def _make_hood_level(
    time_s, q_ss_cm_s, hood_radius_cm=12.4,
    reservoir_area_cm2=23.0, noise_mm=0.05, rng_seed=0,
):
    rng = np.random.default_rng(rng_seed)
    hood_area = np.pi * hood_radius_cm ** 2
    rate_mm_s = q_ss_cm_s * 10.0 * hood_area / reservoir_area_cm2
    return 200.0 - rate_mm_s * time_s + rng.normal(0, noise_mm, len(time_s))


HOOD_LEVEL_30 = _make_hood_level(HOOD_TIME, 5e-4, rng_seed=1)
HOOD_LEVEL_60 = _make_hood_level(HOOD_TIME, 2e-4, rng_seed=2)
HOOD_LEVEL_90 = _make_hood_level(HOOD_TIME, 8e-5, rng_seed=3)


class TestSteadyStateDetection:
    def _rate(self, level, q_ss):
        run = InfiltrationRun(HOOD_TIME, level, suction_mm=30, **HOOD_IL2700)
        return run.infiltration_rate()

    def test_q_ss_positive(self):
        run = InfiltrationRun(HOOD_TIME, HOOD_LEVEL_30, suction_mm=30, **HOOD_IL2700)
        with warnings.catch_warnings(record=True): warnings.simplefilter('always')
        ss = detect_steady_state(HOOD_TIME, run.infiltration_rate())
        assert ss.q_ss > 0

    def test_q_ss_close_to_true(self):
        q_true = 5e-4
        run = InfiltrationRun(HOOD_TIME, HOOD_LEVEL_30, suction_mm=30, **HOOD_IL2700)
        with warnings.catch_warnings(record=True): warnings.simplefilter('always')
        ss = detect_steady_state(HOOD_TIME, run.infiltration_rate())
        assert abs(ss.q_ss - q_true) / q_true < 0.10

    def test_mask_has_true_values(self):
        run = InfiltrationRun(HOOD_TIME, HOOD_LEVEL_30, suction_mm=30, **HOOD_IL2700)
        with warnings.catch_warnings(record=True): warnings.simplefilter('always')
        ss = detect_steady_state(HOOD_TIME, run.infiltration_rate())
        assert ss.mask.any()

    def test_q_ss_se_positive(self):
        run = InfiltrationRun(HOOD_TIME, HOOD_LEVEL_30, suction_mm=30, **HOOD_IL2700)
        with warnings.catch_warnings(record=True): warnings.simplefilter('always')
        ss = detect_steady_state(HOOD_TIME, run.infiltration_rate())
        assert ss.q_ss_se >= 0

    def test_conv_frac_between_0_and_1(self):
        run = InfiltrationRun(HOOD_TIME, HOOD_LEVEL_30, suction_mm=30, **HOOD_IL2700)
        with warnings.catch_warnings(record=True): warnings.simplefilter('always')
        ss = detect_steady_state(HOOD_TIME, run.infiltration_rate())
        assert 0.0 <= ss.conv_frac <= 1.0


# ---------------------------------------------------------------------------
# Wooding theory
# ---------------------------------------------------------------------------

class TestWoodingTheory:
    def test_wooding_formula_exact(self):
        r0, alpha_G, K_true = 12.4, 0.04, 2e-4
        q_ss = K_true * (1 + 4.0 / (np.pi * r0 * alpha_G))
        K_est = compute_K_wooding(q_ss, r0, alpha_G)
        assert abs(K_est - K_true) / K_true < 1e-10

    def test_gardner_alpha_two_points(self):
        alpha_G_true = 0.05
        h = np.array([-9.0, -3.0])
        K = np.exp(alpha_G_true * h)
        alpha_G_est = compute_gardner_alpha(h, K)
        assert abs(alpha_G_est - alpha_G_true) < 1e-10


# ---------------------------------------------------------------------------
# Campaign — level runs (hood infiltrometer)
# ---------------------------------------------------------------------------

class TestCampaignLevel:
    def _runs(self):
        return [
            InfiltrationRun(HOOD_TIME, HOOD_LEVEL_90, suction_mm=90, **HOOD_IL2700),
            InfiltrationRun(HOOD_TIME, HOOD_LEVEL_60, suction_mm=60, **HOOD_IL2700),
            InfiltrationRun(HOOD_TIME, HOOD_LEVEL_30, suction_mm=30, **HOOD_IL2700),
        ]

    def test_returns_three_results(self):
        with warnings.catch_warnings(record=True): warnings.simplefilter('always')
        result = Campaign(self._runs()).run()
        assert len(result.results) == 3

    def test_all_have_K_wooding(self):
        with warnings.catch_warnings(record=True): warnings.simplefilter('always')
        result = Campaign(self._runs()).run()
        for r in result.results:
            assert r.K_wooding_cm_s is not None
            assert r.K_wooding_cm_s > 0

    def test_single_run_no_K_wooding(self):
        """Wooding inversion requires ≥ 2 tensions."""
        with warnings.catch_warnings(record=True): warnings.simplefilter('always')
        result = Campaign([
            InfiltrationRun(HOOD_TIME, HOOD_LEVEL_30, suction_mm=30, **HOOD_IL2700)
        ]).run()
        assert result.results[0].K_wooding_cm_s is None

    def test_K_increases_toward_saturation(self):
        with warnings.catch_warnings(record=True): warnings.simplefilter('always')
        result = Campaign(self._runs()).run()
        K_by_suction = {r.suction_mm: r.K_wooding_cm_s for r in result.results}
        assert K_by_suction[30] > K_by_suction[60] > K_by_suction[90]

    def test_ss_q_ss_set_on_results(self):
        with warnings.catch_warnings(record=True): warnings.simplefilter('always')
        result = Campaign(self._runs()).run()
        for r in result.results:
            assert r.ss is not None
            assert r.ss.q_ss > 0

    def test_sigma_K_propagated(self):
        with warnings.catch_warnings(record=True): warnings.simplefilter('always')
        result = Campaign(self._runs()).run()
        for r in result.results:
            if r.ss and r.K_wooding_cm_s and r.ss.q_ss > 0:
                se_K = r.ss.q_ss_se * (r.K_wooding_cm_s / r.ss.q_ss)
                assert se_K >= 0

    def test_kh_fit_computed(self):
        with warnings.catch_warnings(record=True): warnings.simplefilter('always')
        result = Campaign(self._runs()).run()
        assert result.kh is not None
        assert result.kh.Ks_g > 0
        assert result.kh.Ks_vg > 0

    def test_str_no_K_wooding_shows_na(self):
        with warnings.catch_warnings(record=True): warnings.simplefilter('always')
        result = Campaign([
            InfiltrationRun(HOOD_TIME, HOOD_LEVEL_30, suction_mm=30, **HOOD_IL2700)
        ]).run()
        s = str(result.results[0])
        assert 'n/a' in s

    def test_table_output_contains_wooding(self):
        with warnings.catch_warnings(record=True): warnings.simplefilter('always')
        result = Campaign(self._runs()).run()
        tbl = result.table()
        assert 'Wooding' in tbl
        assert 'q_ss' in tbl

    def test_from_dataframe_roundtrip(self):
        """Campaign.from_dataframe reproduces same K as direct Campaign."""
        import pandas as pd
        runs = self._runs()
        rows = []
        for r in runs:
            for t, s in zip(r.time_s, r.signal):
                rows.append({'suction_mm': r.suction_mm, 'time_s': t,
                             'signal': s, 'signal_type': 'level',
                             'site': 'test', 'instrument': 'hood'})
        df = pd.DataFrame(rows)
        with warnings.catch_warnings(record=True): warnings.simplefilter('always')
        r1 = Campaign(runs).run()

        with warnings.catch_warnings(record=True): warnings.simplefilter('always')
        r2 = Campaign.from_dataframe(df, hood_preset=HOOD_IL2700).run()

        K1 = sorted(r.K_wooding_cm_s for r in r1.results if r.K_wooding_cm_s)
        K2 = sorted(r.K_wooding_cm_s for r in r2.results if r.K_wooding_cm_s)
        for a, b in zip(K1, K2):
            assert abs(a - b) / a < 1e-10
