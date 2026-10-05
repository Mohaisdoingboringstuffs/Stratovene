"""Solar geometry, broadband irradiance, and spectral solar models.

Do not assume 1000 W/m^2. Direct absorbed power is zero if cos(theta) <= 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import DATA_DIR
from spectral import integrate_spectrum, solar_wavelengths_m

# Kopp & Lean 2011 total solar irradiance (AM0, mean)
TSI_W_M2 = 1361.0  # LITERATURE
# ASTM E490 uses 1366.1 W/m^2; both are retained as alternatives.


@dataclass
class SolarPosition:
    zenith_deg: float
    elevation_deg: float
    azimuth_deg: float
    air_mass: float
    cos_zenith: float
    is_day: bool


def _day_of_year(dt: datetime) -> float:
    start = datetime(dt.year, 1, 1, tzinfo=timezone.utc)
    return (dt - start).total_seconds() / 86400.0 + 1.0


def solar_position(
    latitude_deg: float,
    longitude_deg: float,
    dt: datetime,
) -> SolarPosition:
    """NOAA solar-position approximation (Michalsky / Spencer-type).

    Accurate to ~0.01-0.5 deg; NOT the full NREL SPA. Good enough for
    thermal-balance geometry. USER INPUT lat/lon/time required for a claim
    about a specific flight.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    n = _day_of_year(dt)
    # Declination (Spencer 1971)
    gamma = 2.0 * np.pi / 365.0 * (n - 1.0)
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    # Equation of time (minutes)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    time_offset = eqtime + 4.0 * longitude_deg
    true_solar_min = dt.hour * 60 + dt.minute + dt.second / 60 + time_offset
    hour_angle = np.deg2rad((true_solar_min / 4.0) - 180.0)
    lat = np.deg2rad(latitude_deg)
    cos_zen = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    cos_zen = float(np.clip(cos_zen, -1.0, 1.0))
    zen = float(np.rad2deg(np.arccos(cos_zen)))
    elev = 90.0 - zen
    # Azimuth (degrees from north)
    sin_az = -np.cos(decl) * np.sin(hour_angle) / max(np.sin(np.deg2rad(zen)), 1e-12)
    cos_az = (
        np.sin(decl) * np.cos(lat) - np.cos(decl) * np.cos(hour_angle) * np.sin(lat)
    ) / max(np.sin(np.deg2rad(zen)), 1e-12)
    az = float(np.rad2deg(np.arctan2(sin_az, cos_az))) % 360.0
    # Kasten-Young 1989 air mass; infinite at night
    if elev <= 0.0:
        am = np.inf
        day = False
        cos_zen = 0.0
    else:
        am = 1.0 / (cos_zen + 0.50572 * (6.07995 + elev) ** -1.6364)
        day = True
    return SolarPosition(zen, elev, az, float(am), float(max(cos_zen, 0.0)), day)


def cloud_scale(condition: str, cloud_fraction: float) -> tuple[float, float]:
    """Return (direct_scale, diffuse_scale). USER/SCENARIO, not measured weather."""
    c = condition.lower()
    if c == "clear":
        return 1.0, 1.0
    if c == "partly_cloudy":
        f = 0.5 if cloud_fraction <= 0 else cloud_fraction
        return max(0.0, 1.0 - 0.75 * f), 1.0 + 0.3 * f
    if c == "cloudy":
        return 0.05, 1.4
    raise ValueError(f"Unknown sky condition {condition}")


def default_broadband(
    altitude_m: float,
    position: SolarPosition,
    user_direct: float | None,
    user_diffuse: float | None,
    condition: str,
    cloud_fraction: float,
) -> tuple[float, float]:
    """If the user supplies irradiance, use it.

    Otherwise a crude clear-sky estimate: AM0 TSI attenuated by a simple
    broadband optical depth that decreases with altitude. This is NOT a
    radiative-transfer solar code. Label: ASSUMED engineering estimate.
    """
    s_dir, s_dif = cloud_scale(condition, cloud_fraction)
    if not position.is_day:
        return 0.0, 0.0
    if user_direct is not None:
        diffuse = 0.0 if user_diffuse is None else user_diffuse
        return user_direct * s_dir, diffuse * s_dif
    # Simple Beer: tau_vert ≈ 0.25 at sea level (ASSUMED order of magnitude)
    tau0 = 0.25
    scale_h = 8500.0
    tau = tau0 * np.exp(-altitude_m / scale_h) * min(position.air_mass, 10.0)
    direct_beam = TSI_W_M2 * np.exp(-tau)  # on a surface normal to the beam
    # Horizontal direct = beam * cos(zen)
    direct_horiz = direct_beam * position.cos_zenith
    diffuse = 0.1 * TSI_W_M2 * position.cos_zenith * np.exp(-0.5 * tau)
    return float(direct_horiz * s_dir), float(diffuse * s_dif)


def load_solar_spectrum(path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Wavelength [m], spectral irradiance [W/m^2/m] normalized later.

    File columns: wavelength_um, irradiance_W_m2_um
    """
    path = path or (DATA_DIR / "solar" / "am0_reduced.csv")
    data = np.genfromtxt(path, delimiter=",", names=True)
    lam = np.asarray(data["wavelength_um"], dtype=float) * 1e-6
    irr_um = np.asarray(data["irradiance_W_m2_um"], dtype=float)
    irr_m = irr_um / 1e-6  # W/m^2/m
    return lam, irr_m


def scale_spectrum_to_g(lam_m: np.ndarray, irr: np.ndarray, g_w_m2: float) -> np.ndarray:
    current = integrate_spectrum(lam_m, irr)
    if current <= 0:
        return irr
    return irr * (g_w_m2 / current)


def absorbed_solar_spectral(
    alpha_lambda: np.ndarray,
    irr_lambda: np.ndarray,
    lam_m: np.ndarray,
    cos_theta: float,
    area_m2: float,
) -> float:
    if cos_theta <= 0.0:
        return 0.0
    return area_m2 * cos_theta * integrate_spectrum(lam_m, alpha_lambda * irr_lambda)


def absorbed_solar_broadband(alpha: float, g_w_m2: float, cos_theta: float, area_m2: float) -> float:
    if cos_theta <= 0.0:
        return 0.0
    return alpha * g_w_m2 * area_m2 * cos_theta


def day_night_flag(position: SolarPosition) -> str:
    if not position.is_day:
        return "night"
    if position.elevation_deg < 6.0:
        return "sunrise_or_sunset"
    return "day"
