import math

from validation import DIAZ_LOBO, diaz_lobo_reproduction


def _b():
    return diaz_lobo_reproduction()


def test_water_vapor_is_consistent_with_ground_test_temperature():
    # Regression for the humidity override bug: previously T_air was set to
    # 34.6 C but the water-vapor mole fraction (and thus P_w and eps_sky) still
    # came from the 15 C US76 sea-level state.
    b = _b()
    p_w = b.modeled["water_vapor_pressure_hPa"]
    # Buck saturation pressure at 34.6 C is ~55 hPa; at 27.3% RH -> ~15 hPa.
    assert 14.0 < p_w < 16.0
    assert 0.75 < b.modeled["eps_sky"] < 0.83
    # Skinny air at 34.6 C, not the 15 C value (~4.65 hPa).
    assert p_w > 14.0


def test_surface_balance_residual_zero_for_both_materials():
    b = _b()
    for key in ("AAO_balance_W", "bare_balance_W"):
        assert abs(b.modeled[key]["residual_W"]) < 1e-9


def test_three_definitions_reported_and_distinct():
    b = _b()
    m = b.modeled
    assert "deltaT_contrast_C" in m and "deltaT_subambient_AAO_C" in m
    assert m["peak_cooling_at_ambient_W_m2"] != m["peak_cooling_at_surface_W_m2"]
    # The old code conflated contrast with the reported reduction; they differ.
    assert abs(m["deltaT_contrast_C"] - m["deltaT_subambient_AAO_C"]) > 1.0
    # Reported definitions are labelled clearly.
    assert "empty-box" in b.reported["deltaT_definition"]
    assert "surface" in b.reported["peak_cooling_definition"]


def test_wind_sweep_monotone_contrast():
    b = _b()
    sweep = b.modeled["wind_sweep"]
    contrasts = [sweep[k]["deltaT_contrast_C"] for k in ("0.5", "1", "2", "5")]
    assert all(a > c for a, c in zip(contrasts, contrasts[1:]))
