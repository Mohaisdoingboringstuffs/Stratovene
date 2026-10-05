from config import SimulationConfig
from enclosure import solve_steady
from materials import load_materials
from validation import diaz_lobo_reproduction


def test_energy_balance_flag_reasonable():
    cfg = SimulationConfig()
    cfg.environment.radiation_mode = "effective_sky"
    cfg.environment.altitude_m = 20000.0
    cfg.environment.irradiance_direct_w_m2 = 800.0
    mat = load_materials()["STRATOVANE_AAO_LITERATURE"]
    sol = solve_steady(cfg, mat)
    assert sol.relative_energy_error < 1e-3
    assert sol.q_electronics_w == cfg.electronics.heat_load_w


def test_night_no_solar():
    cfg = SimulationConfig()
    cfg.environment.radiation_mode = "effective_sky"
    cfg.environment.utc_hour = 3.0
    cfg.environment.month = 12
    cfg.environment.latitude_deg = 70.0
    cfg.environment.irradiance_direct_w_m2 = None
    mat = load_materials()["STRATOVANE_AAO_LITERATURE"]
    sol = solve_steady(cfg, mat, altitude_m=20000.0)
    # Polar winter night or not depending on equation; if night, solar=0
    if sol.day_night == "night":
        assert sol.q_solar_w == 0.0


def test_high_h_at_ground_vs_20km():
    cfg = SimulationConfig()
    cfg.environment.radiation_mode = "effective_sky"
    cfg.environment.wind_speed_m_s = 10.0
    cfg.environment.irradiance_direct_w_m2 = 0.0
    mat = load_materials()["WHITE_REFERENCE_PLACEHOLDER"]
    g = solve_steady(cfg, mat, altitude_m=0.0)
    s = solve_steady(cfg, mat, altitude_m=20000.0)
    assert g.h_w_m2_k > s.h_w_m2_k


def test_diaz_lobo_runs():
    b = diaz_lobo_reproduction()
    assert "deltaT_model_minus_reported_K" in b.discrepancy
    assert b.modeled["T_air_C"] == 34.6


def test_mass_not_zero():
    cfg = SimulationConfig()
    cfg.environment.radiation_mode = "effective_sky"
    mat = load_materials()["STRATOVANE_AAO_LITERATURE"]
    sol = solve_steady(cfg, mat, altitude_m=0.0)
    assert sol.mass.aluminum_kg > 0
    assert sol.mass.aao_kg > 0
    assert sol.mass.aao_kg < 0.05 * sol.mass.aluminum_kg
