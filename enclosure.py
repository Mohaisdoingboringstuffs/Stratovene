"""Multi-node electronics enclosure thermal model.

Nodes
-----
1 electronics
2 aluminum wall (lumped)
3 external surface (thin treated skin; nearly isothermal with wall)
4 internal air (optional MODEL B)
Atmosphere and sky are boundary nodes, not solved capacitances.

Steady state is solved first. Transient uses C_i dT_i/dt = ΣQ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

import numpy as np
from scipy.optimize import root

from atmosphere import AtmosphereState, atmosphere_at
from conduction import ALLOYS, RHO_AL2O3_KG_M3, WallConduction, alloy_k
from config import SimulationConfig
from convection import combined_h, q_conv
from materials import Material
from parameters import ParameterLedger, Status
from radiation import (
    SIGMA_SB,
    FaceViews,
    box_face_views,
    broadband_face_budget,
    effective_sky_temperature,
    face_spectral_budget,
)
from solar import default_broadband, solar_position, TSI_W_M2
from spectral import SIGMA_SB as _SB

assert SIGMA_SB == _SB


@dataclass
class Geometry:
    length: float
    width: float
    height: float
    thickness: float

    @property
    def volume_ext(self) -> float:
        return self.length * self.width * self.height

    @property
    def area_top(self) -> float:
        return self.length * self.width

    @property
    def area_bottom(self) -> float:
        return self.length * self.width

    @property
    def area_sides(self) -> float:
        return 2.0 * (self.length + self.width) * self.height

    @property
    def area_total(self) -> float:
        return 2.0 * (self.length * self.width + self.length * self.height + self.width * self.height)

    @property
    def area_internal(self) -> float:
        li, wi, hi = self.length - 2 * self.thickness, self.width - 2 * self.thickness, self.height - 2 * self.thickness
        if min(li, wi, hi) <= 0:
            return self.area_total
        return 2.0 * (li * wi + li * hi + wi * hi)

    @property
    def wall_volume(self) -> float:
        return max(self.volume_ext - max(self.length - 2 * self.thickness, 0) * max(self.width - 2 * self.thickness, 0) * max(self.height - 2 * self.thickness, 0), 0.0)

    @property
    def char_length(self) -> float:
        return self.length


@dataclass
class MassBudget:
    aluminum_kg: float
    aao_kg: float
    coating_kg: float
    alternative_cooling_kg: float
    total_enclosure_kg: float
    notes: str


def mass_budget(geom: Geometry, alloy: str, material: Material, coating_kg: float = 0.0) -> MassBudget:
    rho = ALLOYS[alloy]["rho_kg_m3"]
    m_al = geom.wall_volume * rho
    m_aao = 0.0
    if material.aao_thickness_um and material.porosity is not None:
        t = material.aao_thickness_um * 1e-6
        phi = material.porosity
        rho_eff = RHO_AL2O3_KG_M3 * (1.0 - phi)
        m_aao = geom.area_total * t * rho_eff
    # Alternative active cooling hardware is USER INPUT; default empty.
    alt = 0.5  # kg, ASSUMED placeholder small TEC/fan pack for comparison only
    return MassBudget(
        m_al,
        m_aao,
        coating_kg,
        alt,
        m_al + m_aao + coating_kg,
        "AAO mass uses rho_Al2O3*(1-porosity)*thickness*area. "
        "Alternative cooling mass is an ASSUMED 0.5 kg placeholder, not a vendor quote. "
        "STRATOVANE is not zero-mass; the oxide is thin relative to the box.",
    )


@dataclass
class NodeSolution:
    t_electronics_k: float
    t_wall_k: float
    t_surface_k: float
    t_internal_air_k: float
    t_air_k: float
    t_sky_effective_k: float
    q_electronics_w: float
    q_cond_w: float
    q_conv_w: float
    q_solar_w: float
    q_atm_in_w: float
    q_ground_lw_w: float
    q_rad_out_w: float
    q_window_out_w: float
    q_space_w: float
    q_atm_absorbed_emitted_w: float
    p_cool_net_w: float
    q_env_rad_net_w: float
    q_env_conv_w: float
    q_env_solar_w: float
    h_w_m2_k: float
    energy_balance_error_w: float
    relative_energy_error: float
    flagged: bool
    solar_direct_w_m2: float
    solar_diffuse_w_m2: float
    day_night: str
    mass: MassBudget
    material_id: str
    altitude_m: float
    ledger: ParameterLedger = field(default_factory=ParameterLedger)

    @property
    def t_electronics_c(self) -> float:
        return self.t_electronics_k - 273.15

    @property
    def t_surface_c(self) -> float:
        return self.t_surface_k - 273.15

    @property
    def t_wall_c(self) -> float:
        return self.t_wall_k - 273.15


def _ground_temperature(cfg: SimulationConfig, atm: AtmosphereState) -> float:
    env = cfg.environment
    if env.earth_skin_temperature_k is not None:
        return env.earth_skin_temperature_k
    # Sea-level standard 288.15 K as Earth skin if unspecified. USER/STANDARD, not local T_air at altitude.
    return 288.15


def _earth_lw(cfg: SimulationConfig, t_g: float) -> float:
    if cfg.environment.earth_longwave_w_m2 is not None:
        return cfg.environment.earth_longwave_w_m2
    return SIGMA_SB * t_g**4  # ε_earth ≈ 1 ASSUMED unless user supplies OLR


def _solar_on_faces(
    cfg: SimulationConfig,
    geom: Geometry,
    atm: AtmosphereState,
    material: Material,
    dt: datetime,
) -> dict[str, float]:
    env = cfg.environment
    pos = solar_position(env.latitude_deg, env.longitude_deg, dt)
    g_dir_h, g_dif = default_broadband(
        atm.geometric_altitude_m,
        pos,
        env.irradiance_direct_w_m2,
        env.irradiance_diffuse_w_m2,
        env.sky_condition,
        env.cloud_fraction,
    )
    albedo = env.earth_albedo
    # Horizontal direct is G * cos(zenith). Beam on a tilted face:
    # top: same as horizontal if lid is zenith-facing
    # bottom: no direct if zenith sun (cos negative)
    # sides: approximate as vertical: I_v = I_beam * sin(zenith) / π  (isotropic azimuth average)
    beam_normal = g_dir_h / max(pos.cos_zenith, 1e-6) if pos.is_day else 0.0
    g_top_dir = g_dir_h
    g_bot_dir = 0.0
    g_side_dir = beam_normal * np.sin(np.deg2rad(pos.zenith_deg)) / np.pi if pos.is_day else 0.0
    g_top = g_top_dir + g_dif + 0.0
    g_bot = g_bot_dir + albedo * (g_dir_h + g_dif)
    g_side = g_side_dir + 0.5 * g_dif + 0.5 * albedo * (g_dir_h + g_dif)
    alpha = material.solar_absorptance
    q = {
        "top": alpha * g_top * geom.area_top,
        "bottom": alpha * g_bot * geom.area_bottom,
        "sides": alpha * g_side * geom.area_sides,
        "g_direct_horizontal": g_dir_h,
        "g_diffuse": g_dif,
        "day": "day" if pos.is_day else "night",
        "cos_zenith": pos.cos_zenith,
    }
    if env.radiating_mode == "top_only":
        q["bottom"] = 0.0
        q["sides"] = 0.0
    if not pos.is_day:
        q["top"] = 0.0
        q["bottom"] = 0.0
        q["sides"] = 0.0
        q["g_direct_horizontal"] = 0.0
        q["g_diffuse"] = 0.0
    return q


def _radiative_loads(
    material: Material,
    t_s: float,
    atm: AtmosphereState,
    geom: Geometry,
    cfg: SimulationConfig,
    t_g: float,
) -> dict[str, float]:
    views = box_face_views()
    env = cfg.environment
    eps_g = 0.95  # ASSUMED Earth IR emissivity unless user OLR provided
    fn = face_spectral_budget if env.radiation_mode == "spectral" else broadband_face_budget
    areas = {"top": geom.area_top, "bottom": geom.area_bottom, "sides": geom.area_sides}
    if env.radiating_mode == "top_only":
        areas = {"top": geom.area_top, "bottom": 0.0, "sides": 0.0}
    tot = {
        "q_out": 0.0,
        "q_atm": 0.0,
        "q_g": 0.0,
        "q_win": 0.0,
        "q_space": 0.0,
        "q_atm_abs": 0.0,
        "t_sky": 0.0,
    }
    for name, area in areas.items():
        if area <= 0:
            continue
        b = fn(
            material,
            t_s,
            atm,
            area,
            views[name],
            t_g,
            eps_g,
            env.sky_condition,
            env.cloud_fraction,
        )
        tot["q_out"] += b.q_rad_out
        tot["q_atm"] += b.q_atm_in
        tot["q_g"] += b.q_ground_in
        tot["q_win"] += b.q_window_out
        tot["q_space"] += b.q_escaping_to_space
        tot["q_atm_abs"] += b.q_absorbed_by_atmosphere
        tot["t_sky"] = b.t_sky_effective
    if env.earth_longwave_w_m2 is not None:
        # Replace modeled ground LW absorbed with user OLR * eps * F * A
        tot["q_g"] = material.ir_emissivity * env.earth_longwave_w_m2 * (
            areas["bottom"] * views["bottom"].f_ground + areas["sides"] * views["sides"].f_ground
        )
    return tot


def solve_steady(
    cfg: SimulationConfig,
    material: Material,
    altitude_m: float | None = None,
    heat_w: float | None = None,
    wind: float | None = None,
    dt: datetime | None = None,
) -> NodeSolution:
    env = cfg.environment
    z = env.altitude_m if altitude_m is None else altitude_m
    q_elec = cfg.electronics.heat_load_w if heat_w is None else heat_w
    u = env.wind_speed_m_s if wind is None else wind
    if dt is None:
        dt = datetime(env.year, env.month, env.day, tzinfo=timezone.utc) + timedelta(hours=env.utc_hour)
    geom = Geometry(cfg.geometry.length_m, cfg.geometry.width_m, cfg.geometry.height_m, cfg.geometry.wall_thickness_m)
    atm = atmosphere_at(z, env.relative_humidity_surface, env.stratospheric_h2o_ppmv, env.precipitable_water_mm)
    t_g = _ground_temperature(cfg, atm)
    solar = _solar_on_faces(cfg, geom, atm, material, dt)
    q_solar = solar["top"] + solar["bottom"] + solar["sides"]
    k_wall = alloy_k(cfg.alloy)
    wall = WallConduction(k_wall, geom.area_total, geom.thickness)
    g_ws = wall.conductance()
    g_ew = cfg.electronics.mount_conductance_w_k
    use_air = cfg.environment.internal_air_model
    h_int = 8.0  # W/m2K ASSUMED internal natural convection if MODEL B
    area_int = geom.area_internal

    def residual(x: np.ndarray) -> np.ndarray:
        t_e, t_w, t_s = x[:3]
        rad = _radiative_loads(material, t_s, atm, geom, cfg, t_g)
        conv = combined_h(atm, t_s, u, geom.char_length)
        q_c = q_conv(conv.h_combined, geom.area_total, t_s, atm.temperature_k)
        q_cond_ws = g_ws * (t_w - t_s)
        if use_air:
            t_a = x[3]
            q_e_air = h_int * 0.1 * (t_e - t_a)  # 0.1 m2 electronics ASSUMED
            q_air_w = h_int * area_int * (t_a - t_w)
            q_e_w = g_ew * (t_e - t_w)
            r_e = q_elec - q_e_air - q_e_w
            r_a = q_e_air - q_air_w
            r_w = q_e_w + q_air_w - q_cond_ws
        else:
            q_e_w = g_ew * (t_e - t_w)
            r_e = q_elec - q_e_w
            r_w = q_e_w - q_cond_ws
            r_a = 0.0
        # Surface node (thin): conduction in + solar + atm + ground - emission - convection
        r_s = q_cond_ws + q_solar + rad["q_atm"] + rad["q_g"] - rad["q_out"] - q_c
        if use_air:
            return np.array([r_e, r_w, r_s, r_a])
        return np.array([r_e, r_w, r_s])

    t_guess = atm.temperature_k + 20.0
    x0 = np.array([t_guess + 10, t_guess + 2, t_guess] + ([t_guess] if use_air else []))
    sol = root(residual, x0, method="hybr")
    x = sol.x
    t_e, t_w, t_s = x[0], x[1], x[2]
    t_a = x[3] if use_air else t_e
    rad = _radiative_loads(material, t_s, atm, geom, cfg, t_g)
    conv = combined_h(atm, t_s, u, geom.char_length)
    q_c = q_conv(conv.h_combined, geom.area_total, t_s, atm.temperature_k)
    q_cond = g_ws * (t_w - t_s)
    # Energy: sources into the closed enclosure+surface system
    q_in = q_elec + q_solar + rad["q_atm"] + rad["q_g"]
    q_out = rad["q_out"] + q_c
    err = q_in - q_out
    rel = abs(err) / max(abs(q_in), 1e-12)
    # Net environmental heat rejected by the enclosure surface, split into its
    # three physical channels (all W, positive = leaves the surface to the
    # environment):
    #     q_env_rad_net = emitted LW - absorbed atm LW - absorbed ground LW
    #     q_env_conv    = convective exchange h A (T_s - T_air)
    #     q_env_solar   = -absorbed solar (a source, so it appears negative here)
    # Steady-state surface balance (residual r_s = 0) gives exactly
    #     q_env_conv + q_env_rad_net + q_env_solar = q_cond
    # and the wall/electronics balance gives q_cond = q_elec. Hence
    #     q_env_conv + q_env_rad_net - q_solar = Q_internal
    # i.e. p_cool_net_w equals the internal heat load at steady state BY
    # CONSTRUCTION (first law), not because it is echoed. It is a genuine
    # consistency check, and the split makes the rejection path visible.
    q_env_rad_net = rad["q_out"] - rad["q_atm"] - rad["q_g"]
    q_env_conv = q_c
    q_env_solar = -q_solar
    p_cool = q_env_rad_net + q_env_conv + q_env_solar

    t_sky_bb, _ = effective_sky_temperature(atm)
    t_sky = rad["t_sky"] if rad["t_sky"] > 0 else t_sky_bb
    ledger = ParameterLedger()
    atm.to_ledger(ledger)
    ledger.record("alpha_solar", material.solar_absorptance, "-", material.source, material.status, material.uncertainty_solar)
    ledger.record("epsilon_IR", material.ir_emissivity, "-", material.source, material.status, material.uncertainty_ir)
    ledger.record("wind_speed", u, "m/s", "USER INPUT scenario", Status.USER_INPUT)
    ledger.record("Q_electronics", q_elec, "W", "USER INPUT scenario", Status.USER_INPUT)
    ledger.record("T_sky_effective", t_sky, "K", "Derived from atmospheric RT or Dai-Fang 2014", Status.DERIVED)
    ledger.record("T_air", atm.temperature_k, "K", "US Standard Atmosphere 1976", Status.STANDARD_ATMOSPHERE)
    mass = mass_budget(geom, cfg.alloy, material)
    return NodeSolution(
        t_electronics_k=float(t_e),
        t_wall_k=float(t_w),
        t_surface_k=float(t_s),
        t_internal_air_k=float(t_a),
        t_air_k=atm.temperature_k,
        t_sky_effective_k=float(t_sky),
        q_electronics_w=q_elec,
        q_cond_w=float(q_cond),
        q_conv_w=float(q_c),
        q_solar_w=float(q_solar),
        q_atm_in_w=float(rad["q_atm"]),
        q_ground_lw_w=float(rad["q_g"]),
        q_rad_out_w=float(rad["q_out"]),
        q_window_out_w=float(rad["q_win"]),
        q_space_w=float(rad["q_space"]),
        q_atm_absorbed_emitted_w=float(rad["q_atm_abs"]),
        p_cool_net_w=float(q_env_rad_net + q_env_conv + q_env_solar),
        q_env_rad_net_w=float(q_env_rad_net),
        q_env_conv_w=float(q_env_conv),
        q_env_solar_w=float(q_env_solar),
        h_w_m2_k=conv.h_combined,
        energy_balance_error_w=float(err),
        relative_energy_error=float(rel),
        flagged=bool(rel > cfg.energy_balance_flag or not sol.success),
        solar_direct_w_m2=float(solar["g_direct_horizontal"]),
        solar_diffuse_w_m2=float(solar["g_diffuse"]),
        day_night=str(solar["day"]),
        mass=mass,
        material_id=material.material_id,
        altitude_m=z,
        ledger=ledger,
    )


def comparison_table(results: dict[str, NodeSolution]) -> list[dict]:
    rows = []
    for name, r in results.items():
        rows.append(
            {
                "material": name,
                "altitude_m": r.altitude_m,
                "T_surface_C": r.t_surface_c,
                "T_wall_C": r.t_wall_c,
                "T_electronics_C": r.t_electronics_c,
                "T_air_C": r.t_air_k - 273.15,
                "T_sky_C": r.t_sky_effective_k - 273.15,
                "Q_rad_out_W": r.q_rad_out_w,
                "Q_atm_in_W": r.q_atm_in_w,
                "Q_solar_W": r.q_solar_w,
                "Q_conv_W": r.q_conv_w,
                "Q_env_rejected_W": r.p_cool_net_w,
                "Q_env_rad_net_W": r.q_env_rad_net_w,
                "Q_env_conv_W": r.q_env_conv_w,
                "Q_env_solar_W": r.q_env_solar_w,
                "mass_total_kg": r.mass.total_enclosure_kg,
                "mass_aao_kg": r.mass.aao_kg,
                "power_consumption_W": 0.0,
                "energy_balance_error_W": r.energy_balance_error_w,
                "flagged": r.flagged,
            }
        )
    return rows
