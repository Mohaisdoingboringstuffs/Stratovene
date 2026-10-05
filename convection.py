"""Forced and natural convection.

h is never a single constant for all altitudes. It is computed from
local density, viscosity, conductivity, wind speed, and length.

Q_conv = h A (T_surface - T_air)

Correlations: Incropera/DeWitt fundamentals (LITERATURE).
Wind speed is USER INPUT and is not an aircraft airspeed unless the user says so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atmosphere import AtmosphereState


@dataclass
class ConvectionResult:
    h_forced: float
    h_natural: float
    h_combined: float
    re: float
    ra: float
    nu: float
    regime: str


def _film_properties(atm: AtmosphereState, t_s: float) -> tuple[float, float, float, float, float]:
    t_f = 0.5 * (t_s + atm.temperature_k)
    # Scale viscosity ~ T^0.7 around the standard-atmosphere state
    mu = atm.dynamic_viscosity_pa_s * (t_f / atm.temperature_k) ** 0.7
    k = atm.thermal_conductivity_w_m_k * (t_f / atm.temperature_k) ** 0.8
    rho = atm.pressure_pa / (287.05287 * t_f)
    nu = mu / rho
    alpha = k / (rho * atm.cp_j_kg_k)
    pr = nu / alpha
    return rho, nu, k, pr, alpha


def forced_flat_plate(atm: AtmosphereState, t_s: float, u_m_s: float, l_m: float) -> tuple[float, float, float]:
    rho, nu, k, pr, _ = _film_properties(atm, t_s)
    re = u_m_s * l_m / max(nu, 1e-16)
    if re < 1.0:
        nu_nusselt = 0.0
    elif re < 5e5:
        nu_nusselt = 0.664 * re**0.5 * pr ** (1.0 / 3.0)
    else:
        nu_nusselt = (0.037 * re**0.8 - 871.0) * pr ** (1.0 / 3.0)
        nu_nusselt = max(nu_nusselt, 0.664 * re**0.5 * pr ** (1.0 / 3.0))
    h = nu_nusselt * k / l_m
    return float(h), float(re), float(nu_nusselt)


def natural_vertical(atm: AtmosphereState, t_s: float, l_m: float) -> tuple[float, float, float]:
    """Churchill and Chu (1975) vertical plate, all Ra."""
    _, nu, k, pr, alpha = _film_properties(atm, t_s)
    beta = 1.0 / (0.5 * (t_s + atm.temperature_k))  # ideal gas
    g = 9.80665
    ra = g * beta * abs(t_s - atm.temperature_k) * l_m**3 / (nu * alpha)
    ra = max(ra, 0.0)
    term = (0.492 / max(pr, 1e-12)) ** (9.0 / 16.0)
    nusselt = (0.825 + 0.387 * ra ** (1.0 / 6.0) / (1.0 + term) ** (8.0 / 27.0)) ** 2
    h = nusselt * k / l_m
    return float(h), float(ra), float(nusselt)


def combined_h(atm: AtmosphereState, t_s: float, u_m_s: float, l_m: float) -> ConvectionResult:
    h_f, re, nu_f = forced_flat_plate(atm, t_s, u_m_s, l_m)
    h_n, ra, nu_n = natural_vertical(atm, t_s, l_m)
    # Incropera mixed-convection combination
    h = (h_f**3 + h_n**3) ** (1.0 / 3.0)
    if h_f > 1.2 * h_n:
        regime = "forced"
        nu = nu_f
    elif h_n > 1.2 * h_f:
        regime = "natural"
        nu = nu_n
    else:
        regime = "mixed"
        nu = (nu_f**3 + nu_n**3) ** (1.0 / 3.0)
    return ConvectionResult(h_f, h_n, float(h), re, ra, float(nu), regime)


def q_conv(h: float, area_m2: float, t_s: float, t_air: float) -> float:
    """Positive when the surface loses heat to air."""
    return h * area_m2 * (t_s - t_air)
