"""Material optical database.

Solar and thermal-IR properties are independent.
High visible reflectance does not imply high IR emissivity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import DATA_DIR
from parameters import Status
from spectral import WINDOW_UM, band_integral, integrate_spectrum, planck_exitance_lambda


@dataclass
class Material:
    material_id: str
    name: str
    status: Status
    solar_reflectance: float
    solar_absorptance: float
    ir_emissivity: float
    source: str
    uncertainty_solar: float
    uncertainty_ir: float
    notes: str = ""
    aao_thickness_um: float | None = None
    porosity: float | None = None
    interpore_distance_nm: float | None = None
    electrolyte_family: str | None = None
    spectral_wavelength_um: np.ndarray | None = None
    spectral_reflectance: np.ndarray | None = None
    spectral_emissivity: np.ndarray | None = None
    finish: str | None = None

    def epsilon_on(self, wavelength_m: np.ndarray) -> np.ndarray:
        if self.spectral_wavelength_um is None or self.spectral_emissivity is None:
            return np.full_like(wavelength_m, self.ir_emissivity, dtype=float)
        lam_um = wavelength_m * 1e6
        return np.clip(np.interp(lam_um, self.spectral_wavelength_um, self.spectral_emissivity), 0.0, 1.0)

    def reflectance_on(self, wavelength_m: np.ndarray) -> np.ndarray:
        if self.spectral_wavelength_um is None or self.spectral_reflectance is None:
            # Split: solar-like below 4 μm, IR reflectance = 1 - eps
            r = np.where(
                wavelength_m < 4e-6,
                self.solar_reflectance,
                1.0 - self.ir_emissivity,
            )
            return np.clip(r, 0.0, 1.0)
        lam_um = wavelength_m * 1e6
        return np.clip(np.interp(lam_um, self.spectral_wavelength_um, self.spectral_reflectance), 0.0, 1.0)

    def alpha_on(self, wavelength_m: np.ndarray) -> np.ndarray:
        # Opaque Kirchhoff: alpha = epsilon = 1 - rho for opaque solids.
        return np.clip(1.0 - self.reflectance_on(wavelength_m), 0.0, 1.0)

    def band_ir_emissivity(self, t_k: float = 300.0) -> float:
        if self.spectral_wavelength_um is None:
            return self.ir_emissivity
        lam = np.linspace(2.0, 50.0, 400) * 1e-6
        e = self.epsilon_on(lam)
        b = planck_exitance_lambda(lam, t_k)
        tot = integrate_spectrum(lam, b)
        if tot <= 0:
            return self.ir_emissivity
        return integrate_spectrum(lam, e * b) / tot

    def window_emissivity(self, t_k: float = 300.0) -> float:
        lam = np.linspace(WINDOW_UM[0], WINDOW_UM[1], 200) * 1e-6
        e = self.epsilon_on(lam)
        b = planck_exitance_lambda(lam, t_k)
        den = integrate_spectrum(lam, b)
        if den <= 0:
            return self.ir_emissivity
        return integrate_spectrum(lam, e * b) / den


def _status(s: str) -> Status:
    for st in Status:
        if st.value == s or st.name == s:
            return st
    return Status.ASSUMED


def load_materials(path: Path | None = None) -> dict[str, Material]:
    path = path or (DATA_DIR / "materials" / "materials.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, Material] = {}
    for item in raw["materials"]:
        spec_w = spec_r = spec_e = None
        if item.get("spectral_file"):
            sp = np.genfromtxt(DATA_DIR / "materials" / item["spectral_file"], delimiter=",", names=True)
            spec_w = np.asarray(sp["wavelength_um"], dtype=float)
            spec_r = np.asarray(sp["reflectance"], dtype=float)
            spec_e = np.asarray(sp["emissivity"], dtype=float)
        mid = item["id"]
        out[mid] = Material(
            material_id=mid,
            name=item["name"],
            status=_status(item["status"]),
            solar_reflectance=item["solar_reflectance"],
            solar_absorptance=item["solar_absorptance"],
            ir_emissivity=item["IR_emissivity"],
            source=item["source"],
            uncertainty_solar=item.get("uncertainty_solar", 0.05),
            uncertainty_ir=item.get("uncertainty_ir", 0.05),
            notes=item.get("notes", ""),
            aao_thickness_um=item.get("aao_thickness_um"),
            porosity=item.get("porosity"),
            interpore_distance_nm=item.get("interpore_distance_nm"),
            electrolyte_family=item.get("electrolyte_family"),
            spectral_wavelength_um=spec_w,
            spectral_reflectance=spec_r,
            spectral_emissivity=spec_e,
            finish=item.get("finish"),
        )
    return out


def apply_user_optics(
    material: Material,
    solar_reflectance: float | None = None,
    ir_emissivity: float | None = None,
    aao_thickness_um: float | None = None,
    porosity: float | None = None,
) -> Material:
    """Optical parameters may be entered independently of morphology."""
    if solar_reflectance is not None:
        material.solar_reflectance = solar_reflectance
        material.solar_absorptance = 1.0 - solar_reflectance
        material.status = Status.USER_INPUT
    if ir_emissivity is not None:
        material.ir_emissivity = ir_emissivity
        material.status = Status.USER_INPUT
    if aao_thickness_um is not None:
        material.aao_thickness_um = aao_thickness_um
    if porosity is not None:
        material.porosity = porosity
    return material
