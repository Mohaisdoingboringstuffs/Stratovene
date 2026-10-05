"""NASA/US Standard Atmosphere 1976 and dry-air thermophysical properties.

Source: NOAA/NASA/USAF, U.S. Standard Atmosphere, 1976, NASA-TM-X-74335.

T_air from this module is NOT an effective sky temperature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parameters import ParameterLedger, Status

# COESA 1976 defining constants
G0 = 9.80665  # m/s^2, STANDARD ATMOSPHERE
R_STAR = 8.31432  # J/(mol K), STANDARD ATMOSPHERE
M_AIR = 0.0289644  # kg/mol, STANDARD ATMOSPHERE
R_SPECIFIC = R_STAR / M_AIR  # J/(kg K)
R_EARTH = 6_356_766.0  # m, STANDARD ATMOSPHERE
P0 = 101_325.0  # Pa
T0 = 288.15  # K
GAMMA_AIR = 1.4  # ASSUMED dry air, calorically perfect for speed of sound

# Geopotential bases (m), temperature (K), lapse rate (K/m)
_LAYERS = (
    (0.0, 288.15, -0.0065),
    (11_000.0, 216.65, 0.0),
    (20_000.0, 216.65, 0.001),
    (32_000.0, 228.65, 0.0028),
    (47_000.0, 270.65, 0.0),
    (51_000.0, 270.65, -0.0028),
    (71_000.0, 214.65, -0.002),
    (84_852.0, 186.946, 0.0),
)

# Sutherland viscosity (dry air): White, Viscous Fluid Flow / ISO 2533 related
MU0 = 1.716e-5  # Pa s at T0_suth
T_SUTH_REF = 273.15  # K
S_SUTH = 110.4  # K
PR_AIR = 0.71  # LITERATURE typical dry air Prandtl number near 250 K
CP_DRY_AIR = 1004.0  # J/(kg K), LITERATURE nearly constant 200-300 K


def geometric_to_geopotential(z_m: float) -> float:
    """H = r0 Z / (r0 + Z), U.S. Standard Atmosphere 1976."""
    return R_EARTH * z_m / (R_EARTH + z_m)


def geopotential_to_geometric(h_m: float) -> float:
    return R_EARTH * h_m / (R_EARTH - h_m)


def _temperature_pressure(h_m: float) -> tuple[float, float]:
    if h_m < 0.0:
        raise ValueError("Altitude below sea level is not supported.")
    if h_m > 84_852.0:
        raise ValueError("U.S. Standard Atmosphere 1976 implementation limited to 84.852 km geopotential.")

    p = P0
    for i in range(len(_LAYERS) - 1):
        h_b, t_b, lapse = _LAYERS[i]
        h_next = _LAYERS[i + 1][0]
        h_top = min(h_m, h_next)
        dh = h_top - h_b
        if lapse == 0.0:
            t = t_b
            p = p * np.exp(-G0 * dh / (R_SPECIFIC * t_b))
        else:
            t = t_b + lapse * dh
            p = p * (t / t_b) ** (-G0 / (R_SPECIFIC * lapse))
        if h_m <= h_next:
            return float(t), float(p)
    return float(_LAYERS[-1][1]), float(p)


def sutherland_viscosity(t_k: float) -> float:
    """μ = μ0 (T/Tref)^(3/2) (Tref+S)/(T+S)."""
    return MU0 * (t_k / T_SUTH_REF) ** 1.5 * (T_SUTH_REF + S_SUTH) / (t_k + S_SUTH)


def air_thermal_conductivity(t_k: float, mu: float | None = None) -> float:
    """k = μ cp / Pr for dry air. Pr and cp are literature constants."""
    if mu is None:
        mu = sutherland_viscosity(t_k)
    return mu * CP_DRY_AIR / PR_AIR


@dataclass
class AtmosphereState:
    geometric_altitude_m: float
    geopotential_altitude_m: float
    temperature_k: float
    pressure_pa: float
    density_kg_m3: float
    dynamic_viscosity_pa_s: float
    thermal_conductivity_w_m_k: float
    cp_j_kg_k: float
    prandtl: float
    speed_of_sound_m_s: float
    relative_humidity: float | None
    water_vapor_mole_fraction: float
    precipitable_water_above_mm: float

    @property
    def temperature_c(self) -> float:
        return self.temperature_k - 273.15

    def to_ledger(self, ledger: ParameterLedger) -> None:
        ledger.record(
            "T_air",
            self.temperature_k,
            "K",
            "U.S. Standard Atmosphere 1976, NASA-TM-X-74335",
            Status.STANDARD_ATMOSPHERE,
        )
        ledger.record(
            "P_air",
            self.pressure_pa,
            "Pa",
            "U.S. Standard Atmosphere 1976",
            Status.STANDARD_ATMOSPHERE,
        )
        ledger.record(
            "rho_air",
            self.density_kg_m3,
            "kg/m^3",
            "U.S. Standard Atmosphere 1976, rho = P/(R T)",
            Status.STANDARD_ATMOSPHERE,
        )


def saturation_vapor_pressure_pa(t_k: float) -> float:
    """Wexler/Sonntag-type over water, approx. Murphy & Koop not required.

    Buck (1981) / WMO-style: e_s (hPa) = 6.1121 exp(17.502 T/(240.97+T)), T in C.
    LITERATURE: Buck, JAM 1981. Valid near tropospheric temperatures; stratosphere
    uses mole-fraction override instead of RH extrapolation.
    """
    tc = t_k - 273.15
    return 6.1121e2 * np.exp(17.502 * tc / (240.97 + tc))


def water_mole_fraction(
    z_m: float,
    t_k: float,
    p_pa: float,
    rh_surface: float,
    strat_h2o_ppmv: float,
) -> float:
    tropopause = 11_000.0
    e_s0 = saturation_vapor_pressure_pa(T0)
    x0 = rh_surface * e_s0 / P0
    h_v = 2500.0  # m, ASSUMED water-vapor density scale height
    if z_m <= tropopause:
        x = x0 * np.exp(-z_m / h_v)
        x = min(x, 0.95 * saturation_vapor_pressure_pa(t_k) / p_pa)
    else:
        x = strat_h2o_ppmv * 1e-6
    return float(x)


def water_profile(
    z_m: float,
    t_k: float,
    p_pa: float,
    rh_surface: float,
    strat_h2o_ppmv: float,
    compute_column: bool = True,
) -> tuple[float, float]:
    """Return (mole_fraction, precipitable_water_above_mm).

    Troposphere: exponential decay of water-vapor density with scale height 2.5 km
    (LITERATURE order-of-magnitude; actual PW is USER/climate dependent).
    Stratosphere: do NOT extrapolate surface RH. Use a dry mixing ratio
    (WMO/SPARC typical 3-6 ppmv; default 5 ppmv ASSUMED within that range).
    """
    x = water_mole_fraction(z_m, t_k, p_pa, rh_surface, strat_h2o_ppmv)
    if not compute_column:
        return x, 0.0
    tropopause = 11_000.0
    e_s0 = saturation_vapor_pressure_pa(T0)
    x0 = rh_surface * e_s0 / P0
    h_v = 2500.0
    m_v = 0.01801528
    z_grid = np.linspace(z_m, 80_000.0, 41)
    pw = 0.0
    for za, zb in zip(z_grid[:-1], z_grid[1:]):
        zm = 0.5 * (za + zb)
        hm = geometric_to_geopotential(zm)
        tm, pm = _temperature_pressure(hm)
        rhom = pm / (R_SPECIFIC * tm)
        if zm <= tropopause:
            xm = x0 * np.exp(-zm / h_v)
        else:
            xm = strat_h2o_ppmv * 1e-6
        rho_v = xm * (m_v / M_AIR) * rhom
        pw += rho_v * (zb - za)
    return float(x), float(pw)


def atmosphere_at(
    geometric_altitude_m: float,
    rh_surface: float = 0.40,
    strat_h2o_ppmv: float = 5.0,
    precipitable_water_mm: float | None = None,
    compute_pw: bool = True,
) -> AtmosphereState:
    h = geometric_to_geopotential(geometric_altitude_m)
    t, p = _temperature_pressure(h)
    rho = p / (R_SPECIFIC * t)
    mu = sutherland_viscosity(t)
    k = air_thermal_conductivity(t, mu)
    a = np.sqrt(GAMMA_AIR * R_SPECIFIC * t)
    x_h2o, pw = water_profile(
        geometric_altitude_m, t, p, rh_surface, strat_h2o_ppmv, compute_column=compute_pw
    )
    if precipitable_water_mm is not None:
        pw = precipitable_water_mm
    rh = None
    if geometric_altitude_m < 11_000.0:
        rh = float(np.clip(x_h2o * p / saturation_vapor_pressure_pa(t), 0.0, 1.0))
    return AtmosphereState(
        geometric_altitude_m=geometric_altitude_m,
        geopotential_altitude_m=h,
        temperature_k=t,
        pressure_pa=p,
        density_kg_m3=rho,
        dynamic_viscosity_pa_s=mu,
        thermal_conductivity_w_m_k=k,
        cp_j_kg_k=CP_DRY_AIR,
        prandtl=PR_AIR,
        speed_of_sound_m_s=a,
        relative_humidity=rh,
        water_vapor_mole_fraction=x_h2o,
        precipitable_water_above_mm=pw,
    )


def layer_table(z_max_m: float = 80_000.0, n: int = 40) -> list[AtmosphereState]:
    zs = np.linspace(0.0, z_max_m, n)
    return [atmosphere_at(float(z)) for z in zs]
