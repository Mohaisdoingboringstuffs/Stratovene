"""Atmospheric and surface radiative transfer.

MODE 1 — effective sky temperature / broadband sky emissivity
MODE 2 — spectral layered atmosphere

Dai et al. 2016 (Adv. Space Res. 58:2453–2459, doi:10.1016/j.asr.2016.08.014)
estimates *upward* IR transmittance vs altitude. It does NOT publish an
effective sky temperature T_sky(z) for downwelling radiation.

The closest defensible published broadband downwelling model from the same
group is Dai & Fang (2014), Adv. Space Res. 54:1044–1048,
doi:10.1016/j.asr.2014.05.028:

    ε_sky = (c1 + c2 P_w^{c3}) (P_a / 1013)^{c4}

with c1=0.48, c2=0.17, c3=0.22, c4=0.45, P_w and P_a in hPa
(coefficients as tabulated by Li et al. 2017 from Dai & Fang 2014).

T_sky_effective = T_air * ε_sky**(1/4)
This is NOT 3 K. 3 K is the cosmic microwave background, negligible here.

Sign convention
---------------
P_rad_out     : surface thermal emission, always >= 0
P_atm_in      : absorbed atmospheric + Earth longwave, always >= 0
P_solar_abs   : absorbed solar (direct+diffuse+albedo), always >= 0
P_conv        : hA(T_s - T_air); >0 if surface hotter than air
P_cool_net    : P_rad_out - P_atm_in - P_solar_abs - P_conv
                (at T_s; electronics heat is a separate node source)
Gross emission is never called "cooling power".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from functools import lru_cache

from atmosphere import AtmosphereState, atmosphere_at, saturation_vapor_pressure_pa
from materials import Material
from spectral import (
    SIGMA_SB,
    WINDOW_UM,
    band_integral,
    integrate_spectrum,
    planck_exitance_lambda,
    thermal_wavelengths_m,
)

# Dai & Fang 2014 coefficients (LITERATURE via Li et al. 2017 tabulation)
DAI_C1 = 0.48
DAI_C2 = 0.17
DAI_C3 = 0.22
DAI_C4 = 0.45

# Diffusivity factor for isotropic IR streams (Elsasser)
DIFFUSIVITY = 1.66


def dai_fang_2014_sky_emissivity(atm: AtmosphereState) -> float:
    """Broadband clear-sky emissivity. LITERATURE: Dai & Fang 2014.

    P_w is water-vapor pressure in hPa. At stratospheric altitudes the
    mixing ratio is the dry ppmv profile, not surface RH extrapolated.
    """
    p_hpa = atm.pressure_pa / 100.0
    p_w_hpa = atm.water_vapor_mole_fraction * p_hpa
    p_w_hpa = max(p_w_hpa, 1e-6)
    eps = (DAI_C1 + DAI_C2 * p_w_hpa**DAI_C3) * (p_hpa / 1013.0) ** DAI_C4
    return float(np.clip(eps, 0.0, 0.99))


def effective_sky_temperature(atm: AtmosphereState) -> tuple[float, float]:
    eps = dai_fang_2014_sky_emissivity(atm)
    t_sky = atm.temperature_k * eps**0.25
    return t_sky, eps


def cloud_longwave_boost(condition: str, cloud_fraction: float) -> float:
    """Increase broadband ε_sky toward 1 as clouds thicken. ASSUMED engineering."""
    c = condition.lower()
    if c == "clear":
        return 0.0
    if c == "partly_cloudy":
        f = 0.5 if cloud_fraction <= 0 else cloud_fraction
        return 0.4 * f
    if c == "cloudy":
        return 0.85
    return 0.0


def band_absorption_coefficient(wavelength_m: np.ndarray) -> dict[str, np.ndarray]:
    """Simplified spectral absorption weights [1 / (kg m^-2)] for H2O, CO2, O3.

    These are NOT HITRAN line-by-line. They are a coarse window-preserving
    band model tuned so that:
      * 8–13 μm is relatively transparent
      * 6.3 μm H2O and 15 μm CO2 are opaque
      * 9.6 μm O3 is weakly absorbing
    Status: ASSUMED engineering band model, calibrated to typical ground
    downwelling LW order of magnitude when combined with US76 water columns.
    """
    lam = wavelength_m * 1e6
    k_h2o = np.zeros_like(lam)
    k_co2 = np.zeros_like(lam)
    k_o3 = np.zeros_like(lam)
    # Water vapor: strong 5–8 μm and >17 μm, weak in 8–13
    k_h2o += 40.0 * np.exp(-((lam - 6.3) ** 2) / (2 * 0.8**2))
    k_h2o += 8.0 * np.exp(-((lam - 2.7) ** 2) / (2 * 0.4**2))
    k_h2o += np.where(lam > 17.0, 15.0 * ((lam - 17.0) / 10.0) ** 2, 0.0)
    k_h2o += 0.4  # continuum floor
    k_h2o = np.where((lam >= 8.0) & (lam <= 13.0), k_h2o * 0.08, k_h2o)
    # CO2 15 μm
    k_co2 += 25.0 * np.exp(-((lam - 15.0) ** 2) / (2 * 1.2**2))
    k_co2 += 4.0 * np.exp(-((lam - 4.3) ** 2) / (2 * 0.3**2))
    # O3 9.6 μm
    k_o3 += 3.0 * np.exp(-((lam - 9.6) ** 2) / (2 * 0.5**2))
    return {"h2o": k_h2o, "co2": k_co2, "o3": k_o3}


@lru_cache(maxsize=64)
def _layer_columns(z_m: float, n_layers: int = 24, z_top: float = 80_000.0):
    z_m = float(round(z_m, 0))
    z = np.linspace(z_m, z_top, n_layers + 1)
    states = [atmosphere_at(float(zm), compute_pw=False) for zm in 0.5 * (z[:-1] + z[1:])]
    dz = np.diff(z)
    u_h2o = np.array([s.water_vapor_mole_fraction * (18.01528 / 28.9644) * s.density_kg_m3 * d for s, d in zip(states, dz)])
    # CO2 415 ppmv well mixed. USER/LITERATURE ~2020s mean. Not 1976.
    x_co2 = 415e-6
    u_co2 = np.array([x_co2 * (44.01 / 28.9644) * s.density_kg_m3 * d for s, d in zip(states, dz)])
    # Crude ozone bump in 15–35 km
    u_o3 = []
    for s, d in zip(states, dz):
        zmid = s.geometric_altitude_m
        bump = np.exp(-((zmid - 25_000.0) / 8_000.0) ** 2)
        x_o3 = 8e-6 * bump  # ASSUMED peak mixing
        u_o3.append(x_o3 * (48.0 / 28.9644) * s.density_kg_m3 * d)
    return z, states, u_h2o, u_co2, np.array(u_o3)


def spectral_downwelling(
    atm: AtmosphereState,
    wavelength_m: np.ndarray,
    condition: str = "clear",
    cloud_fraction: float = 0.0,
) -> np.ndarray:
    """L_down(λ) as π-steradian exitance [W/m^2/m] incident on an upward face.

    Kirchhoff: layer emissivity = 1 - transmittance of that layer.
    """
    ks = band_absorption_coefficient(wavelength_m)
    _, states, u_h2o, u_co2, u_o3 = _layer_columns(atm.geometric_altitude_m)
    i_down = np.zeros_like(wavelength_m, dtype=float)
    for s, uh, uc, uo in zip(reversed(states), reversed(u_h2o), reversed(u_co2), reversed(u_o3)):
        tau = DIFFUSIVITY * (ks["h2o"] * uh + ks["co2"] * uc + ks["o3"] * uo)
        trans = np.exp(-tau)
        eps = 1.0 - trans
        b_exit = planck_exitance_lambda(wavelength_m, s.temperature_k)
        i_down = i_down * trans + eps * b_exit
    # Clouds: mix toward blackbody at local air T (ASSUMED)
    boost = cloud_longwave_boost(condition, cloud_fraction)
    if boost > 0:
        cloud_bb = planck_exitance_lambda(wavelength_m, atm.temperature_k)
        i_down = (1.0 - boost) * i_down + boost * cloud_bb
    return i_down


def spectral_upward_transmittance(atm: AtmosphereState, wavelength_m: np.ndarray) -> np.ndarray:
    """Band-model transmittance from altitude to space (Dai 2016 *motivation*).

    Dai et al. 2016 give an empirical upward-IR transmittance vs altitude;
    their fitted coefficients are not reproduced here (full tables not in
    the open abstract). This function uses the same layered optical-depth
    model as downwelling. Status: DERIVED from the ASSUMED band model.
    """
    ks = band_absorption_coefficient(wavelength_m)
    _, _, u_h2o, u_co2, u_o3 = _layer_columns(atm.geometric_altitude_m)
    tau = DIFFUSIVITY * (
        ks["h2o"] * float(np.sum(u_h2o))
        + ks["co2"] * float(np.sum(u_co2))
        + ks["o3"] * float(np.sum(u_o3))
    )
    return np.exp(-tau)


@dataclass
class FaceViews:
    f_sky: float
    f_ground: float
    f_aircraft: float = 0.0

    def __post_init__(self) -> None:
        s = self.f_sky + self.f_ground + self.f_aircraft
        if abs(s - 1.0) > 1e-9:
            raise ValueError(f"View factors must sum to 1, got {s}")


def box_face_views() -> dict[str, FaceViews]:
    """First-order views for a rectangular box above an infinite Earth.

    Top (zenith): F_sky=1
    Bottom (nadir): F_ground=1
    Sides: F_sky=F_ground=0.5 (vertical surface, infinite plane)
    Aircraft structure is 0 unless the user supplies blockage.
    """
    return {
        "top": FaceViews(1.0, 0.0, 0.0),
        "bottom": FaceViews(0.0, 1.0, 0.0),
        "sides": FaceViews(0.5, 0.5, 0.0),
    }


@dataclass
class SpectralBudget:
    wavelength_m: np.ndarray
    e_surface: np.ndarray
    e_down: np.ndarray
    e_ground: np.ndarray
    q_rad_out: float
    q_atm_in: float
    q_ground_in: float
    q_window_out: float
    q_absorbed_by_atmosphere: float
    q_escaping_to_space: float
    t_sky_effective: float


def face_spectral_budget(
    material: Material,
    t_surface: float,
    atm: AtmosphereState,
    area: float,
    views: FaceViews,
    t_ground: float,
    eps_ground: float,
    condition: str,
    cloud_fraction: float,
) -> SpectralBudget:
    lam = thermal_wavelengths_m()
    eps = material.epsilon_on(lam)
    e_s = eps * planck_exitance_lambda(lam, t_surface)
    e_down = spectral_downwelling(atm, lam, condition, cloud_fraction)
    e_g = eps_ground * planck_exitance_lambda(lam, t_ground)
    # Absorbed incoming uses Kirchhoff: alpha_IR = eps for opaque IR
    q_out = area * integrate_spectrum(lam, e_s)
    q_atm = area * views.f_sky * integrate_spectrum(lam, eps * e_down)
    q_g = area * views.f_ground * integrate_spectrum(lam, eps * e_g)
    q_win = area * band_integral(lam, e_s, WINDOW_UM[0] * 1e-6, WINDOW_UM[1] * 1e-6)
    trans = spectral_upward_transmittance(atm, lam)
    # Fraction of surface emission directed to sky that reaches space
    q_space = area * views.f_sky * integrate_spectrum(lam, e_s * trans)
    q_atm_abs = area * views.f_sky * integrate_spectrum(lam, e_s * (1.0 - trans))
    g_down = integrate_spectrum(lam, e_down)
    t_sky_eff = (g_down / SIGMA_SB) ** 0.25 if g_down > 0 else 0.0
    return SpectralBudget(
        lam, e_s, e_down, e_g, q_out, q_atm, q_g, q_win, q_atm_abs, q_space, t_sky_eff
    )


def broadband_face_budget(
    material: Material,
    t_surface: float,
    atm: AtmosphereState,
    area: float,
    views: FaceViews,
    t_ground: float,
    eps_ground: float,
    condition: str,
    cloud_fraction: float,
) -> SpectralBudget:
    t_sky, eps_sky = effective_sky_temperature(atm)
    eps_sky = float(np.clip(eps_sky + cloud_longwave_boost(condition, cloud_fraction), 0, 0.99))
    t_sky = atm.temperature_k * eps_sky**0.25
    eps = material.ir_emissivity
    q_out = area * eps * SIGMA_SB * t_surface**4
    q_atm = area * views.f_sky * eps * eps_sky * SIGMA_SB * atm.temperature_k**4
    q_g = area * views.f_ground * eps * eps_ground * SIGMA_SB * t_ground**4
    # Window fraction of a 300 K blackbody between 8–13 μm is ~0.26; scale
    from spectral import thermal_wavelengths_m as tw, planck_exitance_lambda as pel, integrate_spectrum as integ

    lam = tw()
    b = pel(lam, t_surface)
    frac = band_integral(lam, b, WINDOW_UM[0] * 1e-6, WINDOW_UM[1] * 1e-6) / max(integ(lam, b), 1e-18)
    q_win = frac * q_out
    # Dai 2016-motivated broadband upward transmittance: 1 - eps_sky
    tau_up = 1.0 - eps_sky
    q_space = views.f_sky * q_out * tau_up
    q_atm_abs = views.f_sky * q_out * (1.0 - tau_up)
    dummy = np.array([8e-6, 13e-6])
    return SpectralBudget(
        dummy,
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        q_out,
        q_atm,
        q_g,
        q_win,
        q_atm_abs,
        q_space,
        t_sky,
    )
