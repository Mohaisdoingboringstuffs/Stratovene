"""Literature benchmark reproduction and discrepancy analysis.

The Díaz-Lobo et al. case is a VALIDATION EXERCISE.
The model is not required to hit 8.0 C exactly.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone

from config import SimulationConfig
from enclosure import solve_steady
from materials import Material, load_materials


@dataclass
class BenchmarkResult:
    name: str
    reported: dict
    modeled: dict
    discrepancy: dict
    notes: str


DIAZ_LOBO = {
    "ambient_C": 34.6,
    "irradiance_W_m2": 736.0,
    "RH": 0.273,
    "deltaT_C": 8.0,
    "peak_cooling_W_m2": 175.0,
    "source": "Díaz-Lobo et al., ACS Appl. Opt. Mater. 2024, 2, 980-990, doi:10.1021/acsaom.3c00216",
}

FU_2023 = {
    "deltaT_C": 15.4,
    "material": "multilayer nanoparticle-polymer metamaterial (NOT STRATOVANE)",
    "source": "Fu et al., Adv. Space Res. 72 (2023) 541-551, doi:10.1016/j.asr.2023.03.004",
    "notes": "External stratospheric-airship benchmark only. Do not equate to AAO-Al.",
}

AAO_OPTIONAL_A = {
    "solar_reflectance": 0.94,
    "IR_emissivity": 0.90,
    "cooling_W_m2": 64.0,
    "deltaT_C": 2.6,
    "status": "LITERATURE comparison case only",
}

AAO_OPTIONAL_B = {
    "IR_emissivity": 0.97,
    "deltaT_K": 7.5,
    "irradiance_W_m2": 850.0,
    "RH": 0.70,
    "status": "LITERATURE comparison case only",
}


def diaz_lobo_reproduction(cfg: SimulationConfig | None = None) -> BenchmarkResult:
    """Ground-level horizontal sample vs ambient, not the aircraft box.

    We still use the enclosure solver but:
      * altitude 0
      * user irradiance 736 W/m2
      * T_air is forced by replacing US76 sea-level T with 34.6 C via a
        local override of the convection air temperature... US76 T0 is 15 C,
        so we cannot silently swap T_air. We set a USER ground-test mode:
        solve at 0 km then report using 34.6 C as T_air by constructing a
        custom config and monkeypatching atmosphere after solve is wrong.

    Instead: use sea-level pressure/density from US76, but pass ambient
    through an explicit ground-test hook in config by setting a high
    'earth_skin' and documenting that T_air for this case is USER INPUT
    34.6 C, not US76.
    """
    from atmosphere import atmosphere_at
    from enclosure import Geometry, NodeSolution
    import numpy as np
    from scipy.optimize import brentq
    from radiation import broadband_face_budget, FaceViews
    from convection import combined_h, q_conv
    from spectral import SIGMA_SB

    materials = load_materials()
    aao = materials["AAO_AL_DIAZ_LOBO"]
    bare = materials["BARE_ALUMINUM_OXIDIZED"]
    # Horizontal 0.05 m x 0.05 m coupon (ASSUMED; paper sample size not used as ours)
    area = 0.05 * 0.05
    t_air = 34.6 + 273.15  # USER/LITERATURE test condition
    atm = atmosphere_at(0.0, rh_surface=0.273)
    # Override T for convection/sky while keeping P, rho of sea level? Density should
    # follow the measured T. Rebuild a light copy:
    atm.temperature_k = t_air
    atm.density_kg_m3 = atm.pressure_pa / (287.05287 * t_air)
    g = DIAZ_LOBO["irradiance_W_m2"]
    views = FaceViews(1.0, 0.0, 0.0)
    u = 1.0  # m/s ASSUMED; paper wind not specified in the abstract numbers

    def t_eq(mat: Material) -> tuple[float, float, float]:
        def f(ts: float) -> float:
            b = broadband_face_budget(mat, ts, atm, area, views, t_air, 0.9, "clear", 0.0)
            h = combined_h(atm, ts, u, 0.05).h_combined
            qc = q_conv(h, area, ts, t_air)
            qsol = mat.solar_absorptance * g * area
            return b.q_rad_out + qc - b.q_atm_in - qsol

        ts = brentq(f, 250.0, 400.0)
        b = broadband_face_budget(mat, ts, atm, area, views, t_air, 0.9, "clear", 0.0)
        qsol = mat.solar_absorptance * g * area
        h = combined_h(atm, ts, u, 0.05).h_combined
        qc = q_conv(h, area, ts, t_air)
        # Peak cooling power at ambient temperature (usual PRC definition)
        b_amb = broadband_face_budget(mat, t_air, atm, area, views, t_air, 0.9, "clear", 0.0)
        p_cool_peak = (b_amb.q_rad_out - b_amb.q_atm_in - mat.solar_absorptance * g * area) / area
        return ts, p_cool_peak, h

    t_aao, p_aao, h = t_eq(aao)
    t_bare, p_bare, _ = t_eq(bare)
    dT_model = (t_bare - t_aao)
    dT_rep = DIAZ_LOBO["deltaT_C"]
    p_rep = DIAZ_LOBO["peak_cooling_W_m2"]
    notes = (
        "Reported ΔT is vs the paper's control, not necessarily this oxidized-aluminum "
        "placeholder. Wind speed was not taken from the paper (ASSUMED 1 m/s). "
        "US76 does not set T_air=34.6 C; that ambient is a LITERATURE test condition. "
        "Broadband Dai-Fang sky used. Spectral mismatch, convection, and control-surface "
        "difference are the expected discrepancy sources."
    )
    modeled = {
        "T_AAO_C": t_aao - 273.15,
        "T_bare_C": t_bare - 273.15,
        "deltaT_C": dT_model,
        "peak_cooling_W_m2": p_aao,
        "h_W_m2_K": h,
        "T_air_C": 34.6,
        "G_W_m2": g,
    }
    reported = {"deltaT_C": dT_rep, "peak_cooling_W_m2": p_rep, "T_air_C": 34.6, "G_W_m2": g, "RH": 0.273}
    disc = {
        "deltaT_model_minus_reported_K": dT_model - dT_rep,
        "Pcool_model_minus_reported_W_m2": p_aao - p_rep,
    }
    return BenchmarkResult("Diaz-Lobo AAO-Al 2023/24", reported, modeled, disc, notes)
