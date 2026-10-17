"""Planck's law, wavelength grids, and spectral integration.

B_λ(T) = (2 h c^2 / λ^5) / (exp(h c / (λ k T)) - 1)   [W / m^2 / sr / m]
Hemispheric blackbody exitance M_λ = π B_λ for a Lambertian surface.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad, trapezoid

# CODATA 2018 / SI defining constants
H_PLANCK = 6.62607015e-34  # J s
C_LIGHT = 299792458.0  # m/s
K_BOLTZMANN = 1.380649e-23  # J/K
SIGMA_SB = 5.670374419e-8  # W/m^2/K^4, CODATA

WINDOW_UM = (8.0, 13.0)


def planck_B_lambda(wavelength_m: np.ndarray | float, temperature_k: float) -> np.ndarray | float:
    lam = np.asarray(wavelength_m, dtype=np.float64)
    t = float(temperature_k)
    if t <= 0.0:
        raise ValueError("Temperature must be positive.")
    x = H_PLANCK * C_LIGHT / (lam * K_BOLTZMANN * t)
    # Prevent overflow
    x = np.clip(x, None, 700.0)
    num = 2.0 * H_PLANCK * C_LIGHT**2 / lam**5
    den = np.expm1(x)
    out = num / den
    if np.isscalar(wavelength_m):
        return float(out)
    return out


def planck_exitance_lambda(wavelength_m: np.ndarray | float, temperature_k: float):
    return np.pi * planck_B_lambda(wavelength_m, temperature_k)


def stefan_boltzmann(temperature_k: float) -> float:
    return SIGMA_SB * temperature_k**4


def thermal_wavelengths_m(lam_min_um: float = 2.0, lam_max_um: float = 50.0, n: int = 400) -> np.ndarray:
    return np.linspace(lam_min_um, lam_max_um, n) * 1e-6


def solar_wavelengths_m(lam_min_um: float = 0.28, lam_max_um: float = 4.0, n: int = 400) -> np.ndarray:
    return np.linspace(lam_min_um, lam_max_um, n) * 1e-6


def integrate_spectrum(wavelength_m: np.ndarray, values: np.ndarray) -> float:
    return float(trapezoid(values, wavelength_m))


def band_integral(wavelength_m: np.ndarray, values: np.ndarray, lam_min_m: float, lam_max_m: float) -> float:
    mask = (wavelength_m >= lam_min_m) & (wavelength_m <= lam_max_m)
    if np.count_nonzero(mask) < 2:
        return 0.0
    return float(trapezoid(values[mask], wavelength_m[mask]))


def planck_sigma_check(temperature_k: float = 300.0) -> dict[str, float]:
    """∫ π B_λ dλ over 1 nm – 10 mm vs σ T^4.

    The originally used 1–200 μm window is physically complete for 300 K, but
    integrating the Planck shape over a finite band leaves a small truncation
    residue. Widening to 1 nm – 10 mm (and tightening the quadrature) makes the
    residual ~1e-9, confirming that the earlier ~6.5e-4 mismatch was pure
    integration-range truncation and NOT a wrong Stefan-Boltzmann constant.
    """
    lam_min_m = 1e-9
    lam_max_m = 1e-2

    def integrand(lam: float) -> float:
        return float(planck_exitance_lambda(lam, temperature_k))

    integral, err = quad(integrand, lam_min_m, lam_max_m, epsabs=1e-12, epsrel=1e-13, limit=2000)
    exact = stefan_boltzmann(temperature_k)
    return {
        "T_K": temperature_k,
        "lam_min_m": lam_min_m,
        "lam_max_m": lam_max_m,
        "integral_W_m2": integral,
        "sigma_T4_W_m2": exact,
        "rel_error": abs(integral - exact) / exact,
        "quad_err_est": err,
    }


def wien_peak_m(temperature_k: float) -> float:
    """Wien displacement for B_λ: λ_max T = 2.897771955e-3 m K (CODATA)."""
    return 2.897771955e-3 / temperature_k
