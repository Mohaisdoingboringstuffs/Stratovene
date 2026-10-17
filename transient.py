"""Transient multi-node integration: C_i dT_i/dt = ΣQ.

Default 24 h, dt 30 s. Radiation in the marching loop uses the broadband
sky-emissivity model unless the user forces spectral (much slower).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import numpy as np

from atmosphere import atmosphere_at
from conduction import ALLOYS
from config import SimulationConfig
from convection import combined_h, q_conv
from enclosure import Geometry, _ground_temperature, _radiative_loads, _solar_on_faces, solve_steady
from materials import Material


@dataclass
class TransientResult:
    time_s: np.ndarray
    t_electronics_k: np.ndarray
    t_wall_k: np.ndarray
    t_air_k: np.ndarray
    g_solar_w_m2: np.ndarray
    q_elec_w: np.ndarray
    t_min_c: float
    t_max_c: float
    swing_k: float


def heat_schedule(cfg: SimulationConfig, is_day: bool) -> float:
    e = cfg.electronics
    if e.time_varying:
        return e.day_heat_load_w if is_day else e.night_heat_load_w
    return e.heat_load_w


def simulate_24h(cfg: SimulationConfig, material: Material, altitude_m: float | None = None) -> TransientResult:
    env = cfg.environment
    z = env.altitude_m if altitude_m is None else altitude_m
    geom = Geometry(cfg.geometry.length_m, cfg.geometry.width_m, cfg.geometry.height_m, cfg.geometry.wall_thickness_m)
    atm = atmosphere_at(z, env.relative_humidity_surface, env.stratospheric_h2o_ppmv, env.precipitable_water_mm)
    alloy = ALLOYS[cfg.alloy]
    m_wall = geom.wall_volume * alloy["rho_kg_m3"]
    c_wall = m_wall * alloy["cp_j_kg_k"]
    c_elec = cfg.electronics.mass_kg * cfg.electronics.cp_j_kg_k
    g_ew = cfg.electronics.mount_conductance_w_k
    t_g = _ground_temperature(cfg, atm)

    dt = cfg.transient.dt_s
    n = int(cfg.transient.duration_s / dt) + 1
    t_e = np.zeros(n)
    t_wall = np.zeros(n)
    t_air = np.full(n, atm.temperature_k)
    gsol = np.zeros(n)
    qe = np.zeros(n)
    times = np.arange(n) * dt

    start = datetime(env.year, env.month, env.day, tzinfo=timezone.utc)

    original_mode = env.radiation_mode
    env.radiation_mode = "effective_sky"  # speed; documented ASSUMPTION for marching

    # Spin-up by initializing the lumped nodes at the t=0 steady state instead of
    # a fixed offset above T_air. A fixed offset (T_air+15) is the *coldest*
    # point in the record and makes the reported minimum an initial-condition
    # artifact identical for every material. The steady state depends on the
    # material optics, so it does not erase the material signal.
    ss0 = solve_steady(cfg, material, altitude_m=z, dt=start)
    t_e[0] = ss0.t_electronics_k
    t_wall[0] = ss0.t_wall_k
    if cfg.transient.t_init_offset_k:
        # Explicit user perturbation is honored on top of the steady state so the
        # offset remains a user-visible knob, but it is no longer the default IC.
        t_e[0] += cfg.transient.t_init_offset_k
        t_wall[0] += cfg.transient.t_init_offset_k

    for i in range(n - 1):
        clock = start + timedelta(seconds=float(times[i]))
        solar = _solar_on_faces(cfg, geom, atm, material, clock)
        q_solar = solar["top"] + solar["bottom"] + solar["sides"]
        is_day = solar["day"] == "day"
        q_el = heat_schedule(cfg, is_day)
        rad = _radiative_loads(material, t_wall[i], atm, geom, cfg, t_g)
        conv = combined_h(atm, t_wall[i], env.wind_speed_m_s, geom.char_length)
        q_c = q_conv(conv.h_combined, geom.area_total, t_wall[i], atm.temperature_k)
        q_ew = g_ew * (t_e[i] - t_wall[i])
        
        dte = (q_el - q_ew) / c_elec
        dt_wall = (q_ew + q_solar + rad["q_atm"] + rad["q_g"] - rad["q_out"] - q_c) / c_wall
        
        t_e[i + 1] = t_e[i] + dt * dte
        t_wall[i + 1] = t_wall[i] + dt * dt_wall

        if t_e[i + 1] <= 0 or t_wall[i + 1] <= 0:
            raise ValueError(
                f"Non-positive temperature encountered at step {i+1} (t={times[i+1]}s). "
                f"T_e={t_e[i+1]:.2f}K, T_wall={t_wall[i+1]:.2f}K. "
                f"Heat terms: q_el={q_el:.1f}W, q_ew={q_ew:.1f}W, q_solar={q_solar:.1f}W, "
                f"q_atm={rad['q_atm']:.1f}W, q_g={rad['q_g']:.1f}W, q_out={rad['q_out']:.1f}W, q_conv={q_c:.1f}W"
            )
            
        gsol[i] = solar["g_direct_horizontal"] + solar["g_diffuse"]
        qe[i] = q_el

    env.radiation_mode = original_mode
    gsol[-1] = gsol[-2]
    qe[-1] = qe[-2]
    te_c = t_e - 273.15
    return TransientResult(
        times,
        t_e,
        t_wall,
        t_air,
        gsol,
        qe,
        float(te_c.min()),
        float(te_c.max()),
        float(te_c.max() - te_c.min()),
    )
