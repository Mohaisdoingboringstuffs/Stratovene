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

    DEFINITION MISMATCH (documented, not tuned away)
    ------------------------------------------------
    The paper's headline ``deltaT = 8.0 C`` is the temperature reduction of the
    AAO coupon relative to the paper's EMPTY-BOX reference surface, both sitting
    in the sun. It is NOT a material contrast and NOT the sample-vs-ambient
    offset. The paper's ``peak_cooling = 175 W/m2`` is a CALCULATED radiative
    peak using the recorded weather with an assumed convective coefficient
    ``h_CC = 12 W/m2K`` and evaluated at the hot daytime surface temperature, not
    at the ambient temperature.

    This harness reports several clearly-labelled quantities so the comparison
    is honest:
      * ``deltaT_contrast_C``       = T_bare - T_AAO  (what the old code called deltaT)
      * ``deltaT_subambient_AAO_C`` = T_air  - T_AAO  (reduction below ambient)
      * ``peak_cooling_at_ambient_W_m2``  (usual PRC definition, at T_air)
      * ``peak_cooling_at_surface_W_m2``  (radiative net at the solved T_s)

    Wind speed is not specified in the abstract numbers, so 1 m/s is ASSUMED and
    a wind sweep is reported. US76 does not set T_air = 34.6 C at sea level; that
    ambient is a LITERATURE ground-test condition. The sea-level value is
    replaced by a thermodynamically CONSISTENT state at 34.6 C (density,
    viscosity, conductivity, speed of sound and, importantly, water-vapor
    content all recomputed at that temperature).
    """
    from dataclasses import replace

    import numpy as np
    from scipy.optimize import brentq

    from atmosphere import (
        GAMMA_AIR,
        R_SPECIFIC,
        air_thermal_conductivity,
        atmosphere_at,
        saturation_vapor_pressure_pa,
        sutherland_viscosity,
    )
    from convection import combined_h, q_conv
    from radiation import FaceViews, broadband_face_budget, effective_sky_temperature

    materials = load_materials()
    aao = materials["AAO_AL_DIAZ_LOBO"]
    bare = materials["BARE_ALUMINUM_OXIDIZED"]
    # Horizontal 0.05 m x 0.05 m coupon (ASSUMED; paper sample size not used as ours)
    area = 0.05 * 0.05
    t_air = DIAZ_LOBO["ambient_C"] + 273.15  # USER/LITERATURE test condition
    rh = DIAZ_LOBO["RH"]
    atm0 = atmosphere_at(0.0, rh_surface=rh, compute_pw=False)
    # FIX: rebuild a consistent atmosphere state at the ground-test temperature.
    # Previously only T and density were overwritten, leaving the water-vapor
    # mole fraction (=> P_w => eps_sky) and transport properties at US76 15 C.
    atm = replace(
        atm0,
        temperature_k=t_air,
        density_kg_m3=atm0.pressure_pa / (R_SPECIFIC * t_air),
        dynamic_viscosity_pa_s=sutherland_viscosity(t_air),
        thermal_conductivity_w_m_k=air_thermal_conductivity(t_air),
        speed_of_sound_m_s=float(np.sqrt(GAMMA_AIR * R_SPECIFIC * t_air)),
        relative_humidity=rh,
        water_vapor_mole_fraction=rh * saturation_vapor_pressure_pa(t_air) / atm0.pressure_pa,
    )
    g = DIAZ_LOBO["irradiance_W_m2"]
    views = FaceViews(1.0, 0.0, 0.0)
    t_sky_eff, eps_sky = effective_sky_temperature(atm)

    def balance(mat: Material, ts: float, u: float) -> dict:
        b = broadband_face_budget(mat, ts, atm, area, views, t_air, 0.9, "clear", 0.0)
        h = combined_h(atm, ts, u, 0.05).h_combined
        qc = q_conv(h, area, ts, t_air)
        qsol = mat.solar_absorptance * g * area
        q_rad_net = b.q_rad_out - b.q_atm_in
        # Steady surface balance residual: emission + convection = absorbed atm + solar
        residual = q_rad_net + qc - qsol
        return {
            "q_rad_out_W": b.q_rad_out,
            "q_atm_in_W": b.q_atm_in,
            "q_rad_net_W": q_rad_net,
            "q_conv_W": qc,
            "q_solar_W": qsol,
            "residual_W": residual,
            "h_W_m2_K": h,
        }

    def solve(mat: Material, u: float = 1.0) -> dict:
        ts = brentq(lambda x: balance(mat, x, u)["residual_W"], 250.0, 400.0)
        d_s = balance(mat, ts, u)
        d_a = balance(mat, t_air, u)
        d_s["T_surface_C"] = ts - 273.15
        d_s["peak_cooling_at_surface_W_m2"] = d_s["q_rad_net_W"] / area
        d_a["peak_cooling_at_ambient_W_m2"] = d_a["q_rad_net_W"] / area
        return {"surface": d_s, "ambient": d_a}

    aao_sol = solve(aao)
    bare_sol = solve(bare)
    t_aao = aao_sol["surface"]["T_surface_C"] + 273.15
    t_bare = bare_sol["surface"]["T_surface_C"] + 273.15

    wind_sweep = {}
    for u in (0.5, 1.0, 2.0, 5.0):
        a = solve(aao, u)["surface"]["T_surface_C"]
        b = solve(bare, u)["surface"]["T_surface_C"]
        wind_sweep[f"{u:g}"] = {
            "T_AAO_C": a,
            "T_bare_C": b,
            "deltaT_contrast_C": b - a,
            "deltaT_subambient_AAO_C": DIAZ_LOBO["ambient_C"] - a,
        }

    dT_contrast = t_bare - t_aao
    dT_sub_aao = t_air - t_aao
    dT_sub_bare = t_air - t_bare
    p_cool_ambient = aao_sol["ambient"]["peak_cooling_at_ambient_W_m2"]
    p_cool_surface = aao_sol["surface"]["peak_cooling_at_surface_W_m2"]
    dT_rep = DIAZ_LOBO["deltaT_C"]
    p_rep = DIAZ_LOBO["peak_cooling_W_m2"]
    notes = (
        "DEFINITION MISMATCH: reported deltaT=8.0 C is the AAO coupon reduction vs the "
        "paper's EMPTY-BOX reference (not the bare-Al contrast this model used to print, "
        "and not the sample-vs-ambient offset). reported peak cooling=175 W/m2 is a "
        "CALCULATED radiative peak at the hot daytime surface with assumed h_CC=12 W/m2K, "
        "not an at-ambient number. Model wind is ASSUMED (paper abstract silent); see "
        "wind_sweep. T_air=34.6 C is a LITERATURE ground-test ambient; US76 sea-level 15 C "
        "is replaced by a thermodynamically consistent 34.6 C state. Broadband Dai-Fang sky, "
        "oxidized-aluminum placeholder for the control, spectral mismatch and convection "
        "are the expected residual discrepancy sources."
    )
    modeled = {
        "T_AAO_C": t_aao - 273.15,
        "T_bare_C": t_bare - 273.15,
        "deltaT_C": dT_contrast,
        "deltaT_contrast_C": dT_contrast,
        "deltaT_subambient_AAO_C": dT_sub_aao,
        "deltaT_subambient_bare_C": dT_sub_bare,
        "peak_cooling_W_m2": p_cool_ambient,
        "peak_cooling_at_ambient_W_m2": p_cool_ambient,
        "peak_cooling_at_surface_W_m2": p_cool_surface,
        "h_W_m2_K": aao_sol["surface"]["h_W_m2_K"],
        "T_air_C": DIAZ_LOBO["ambient_C"],
        "G_W_m2": g,
        "eps_sky": eps_sky,
        "T_sky_effective_K": t_sky_eff,
        "water_vapor_pressure_hPa": atm.water_vapor_mole_fraction * atm.pressure_pa / 100.0,
        "wind_sweep": wind_sweep,
        "AAO_balance_W": {k: v for k, v in aao_sol["surface"].items() if k.endswith("_W")},
        "bare_balance_W": {k: v for k, v in bare_sol["surface"].items() if k.endswith("_W")},
    }
    reported = {
        "deltaT_C": dT_rep,
        "peak_cooling_W_m2": p_rep,
        "T_air_C": DIAZ_LOBO["ambient_C"],
        "G_W_m2": g,
        "RH": DIAZ_LOBO["RH"],
        "deltaT_definition": "AAO coupon vs empty-box reference (sunlit)",
        "peak_cooling_definition": "calculated at hot surface, h_CC=12 W/m2K",
    }
    disc = {
        "deltaT_model_minus_reported_K": dT_contrast - dT_rep,
        "deltaT_contrast_minus_reported_K": dT_contrast - dT_rep,
        "deltaT_subambient_minus_reported_K": dT_sub_aao - dT_rep,
        "Pcool_model_minus_reported_W_m2": p_cool_ambient - p_rep,
        "Pcool_at_surface_model_minus_reported_W_m2": p_cool_surface - p_rep,
    }
    return BenchmarkResult("Diaz-Lobo AAO-Al 2023/24", reported, modeled, disc, notes)
