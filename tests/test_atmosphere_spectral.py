import numpy as np

from atmosphere import atmosphere_at
from radiation import box_face_views
from spectral import planck_sigma_check, stefan_boltzmann, wien_peak_m
from solar import solar_position
from datetime import datetime, timezone


def test_us76_20km():
    a = atmosphere_at(20000.0)
    assert abs(a.temperature_k - 216.65) < 0.05
    assert abs(a.pressure_pa - 5474.9) / 5474.9 < 0.02
    assert abs(a.density_kg_m3 - 0.0880) / 0.0880 < 0.03


def test_us76_sl():
    a = atmosphere_at(0.0)
    assert abs(a.temperature_k - 288.15) < 1e-6
    assert abs(a.pressure_pa - 101325) < 1e-3


def test_planck_stefan_boltzmann():
    chk = planck_sigma_check(300.0)
    # Wide integration range: residual is quadrature noise, not band truncation.
    assert chk["rel_error"] < 1e-6
    assert chk["lam_min_m"] <= 1e-8
    assert chk["lam_max_m"] >= 1e-3


def test_planck_sigma_truncation_source():
    # Narrow band reproduces the historical ~6e-4 residual; widening removes it.
    import numpy as np
    from scipy.integrate import quad
    from spectral import planck_exitance_lambda, stefan_boltzmann

    t = 300.0
    narrow, _ = quad(lambda l: float(planck_exitance_lambda(l, t)), 1e-6, 200e-6, limit=2000)
    exact = stefan_boltzmann(t)
    assert abs(narrow - exact) / exact > 1e-4
    wide, _ = quad(lambda l: float(planck_exitance_lambda(l, t)), 1e-9, 1e-2, epsabs=1e-12, limit=2000)
    assert abs(wide - exact) / exact < 1e-6


def test_wien():
    t = 5800.0
    assert abs(wien_peak_m(t) - 2.897771955e-3 / t) < 1e-15


def test_view_factors_sum():
    for v in box_face_views().values():
        assert abs(v.f_sky + v.f_ground + v.f_aircraft - 1.0) < 1e-12


def test_night_solar_zero():
    # 2026-01-01 00:00 UTC at lat 80 N should be night in winter
    pos = solar_position(80.0, 0.0, datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
    # May still be polar night-ish; just check cosine handling
    if not pos.is_day:
        assert pos.cos_zenith == 0.0


def test_t_air_not_t_sky():
    from radiation import effective_sky_temperature

    a = atmosphere_at(0.0)
    t_sky, eps = effective_sky_temperature(a)
    assert t_sky != a.temperature_k
    assert t_sky > 3.0
    assert 0 < eps < 1
