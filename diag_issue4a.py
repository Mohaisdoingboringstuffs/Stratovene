import numpy as np
from spectral import SIGMA_SB, thermal_wavelengths_m, planck_exitance_lambda, integrate_spectrum
from materials import load_materials

def test_planck_vs_sigma():
    t = 300.0
    # Standard sigma * T^4
    q_sigma = SIGMA_SB * t**4
    
    # Integration of Planck's law
    lam = thermal_wavelengths_m()
    b = planck_exitance_lambda(lam, t)
    q_planck = integrate_spectrum(lam, b)
    
    rel_error = abs(q_planck - q_sigma) / q_sigma
    print(f"T={t}K: q_sigma={q_sigma:.4f}, q_planck={q_planck:.4f}, rel_error={rel_error:.4e}")
    
    # Try finer grid or wider range
    lam_fine = np.linspace(1e-7, 1e-4, 100000)
    b_fine = planck_exitance_lambda(lam_fine, t)
    q_planck_fine = integrate_spectrum(lam_fine, b_fine)
    rel_error_fine = abs(q_planck_fine - q_sigma) / q_sigma
    print(f"Finer grid: q_planck_fine={q_planck_fine:.4f}, rel_error_fine={rel_error_fine:.4e}")

if __name__ == "__main__":
    test_planck_vs_sigma()
