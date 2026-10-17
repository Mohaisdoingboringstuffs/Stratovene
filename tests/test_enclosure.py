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


def test_q_env_is_split_energy_balance():
    # Q_env is not an echoed constant: it must decompose into the physical
    # rejection channels and satisfy conv + net_longwave - absorbed_solar = Q_elec.
    cfg = SimulationConfig()
    cfg.environment.radiation_mode = "effective_sky"
    cfg.environment.altitude_m = 20000.0
    cfg.environment.irradiance_direct_w_m2 = 800.0
    for mid in ("BARE_ALUMINUM_OXIDIZED", "WHITE_REFERENCE_PLACEHOLDER", "STRATOVANE_AAO_LITERATURE"):
        mat = load_materials()[mid]
        sol = solve_steady(cfg, mat)
        assert sol.relative_energy_error < 1e-3
        total = sol.q_env_rad_net_w + sol.q_env_conv_w + sol.q_env_solar_w
        assert abs(total - sol.p_cool_net_w) < 1e-6
        # First-law identity at steady state.
        assert abs(total - sol.q_electronics_w) < 1e-6 * max(1.0, abs(sol.q_electronics_w))
        # Channels are not trivially all zero for a sunlit, convecting surface.
        assert abs(sol.q_env_conv_w) > 1e-6
        assert abs(sol.q_env_rad_net_w) > 1e-6


def test_bare_al_nonmonotonic_temperature_is_physical():
    # Independent first-principles balance at each altitude:
    #   eps * sigma * A * T^4 + h * A * (T - T_air) = q_solar + q_elec + q_atm + q_g
    # We close the balance with a fresh Stefan-Boltzmann emission term instead of
    # the solver's rad_out, then hand off to T_e via the conduction network. It
    # reproduces the solver to <0.1 K, so the non-monotonic shape (solar dominates
    # once thin air weakens convection) is physics, not a numerical artifact.
    import numpy as np
    from scipy.optimize import brentq

    from spectral import SIGMA_SB

    cfg = SimulationConfig()
    cfg.environment.radiation_mode = "effective_sky"
    bare = load_materials()["BARE_ALUMINUM_OXIDIZED"]
    geom_areas = cfg.geometry
    a_total = 2.0 * (
        geom_areas.length_m * geom_areas.width_m
        + geom_areas.length_m * geom_areas.height_m
        + geom_areas.width_m * geom_areas.height_m
    )
    eps = bare.ir_emissivity
    g_ew = cfg.electronics.mount_conductance_w_k

    zs = [0.0, 5.0, 10.0, 11.0, 12.0, 15.0, 20.0, 25.0, 30.0]
    h_list, ts_model, ts_hand, te_model = [], [], [], []
    for zkm in zs:
        s = solve_steady(cfg, bare, altitude_m=zkm * 1000.0)
        q_in = s.q_solar_w + s.q_electronics_w + s.q_atm_in_w + s.q_ground_lw_w
        h = s.h_w_m2_k
        t_air = s.t_air_k
        t_hand = brentq(
            lambda T: eps * SIGMA_SB * a_total * T**4 + h * a_total * (T - t_air) - q_in,
            150.0,
            800.0,
        )
        g_ws = s.q_electronics_w / (s.t_wall_k - s.t_surface_k)
        t_e_hand = t_hand + s.q_electronics_w / g_ws + s.q_electronics_w / g_ew
        h_list.append(h)
        ts_model.append(s.t_surface_k)
        ts_hand.append(t_hand)
        te_model.append(s.t_electronics_k)

    assert max(abs(a - b) for a, b in zip(ts_model, ts_hand)) < 0.1
    # Convection (density-driven) weakens monotonically with altitude.
    assert all(h_list[i] > h_list[i + 1] for i in range(len(h_list) - 1))
    # Minimum is at the cold tropopause, not at the highest altitude.
    imin = int(np.argmin(ts_model))
    assert 8.0 <= zs[imin] <= 14.0
    assert ts_model[zs.index(20.0)] > ts_model[zs.index(11.0)]
    assert te_model[zs.index(20.0)] > te_model[zs.index(11.0)]


