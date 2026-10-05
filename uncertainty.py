"""Monte Carlo uncertainty and one-at-a-time sensitivity.

Default 10_000 samples on a broadband 3-node residual (spectral RT is too
expensive for 10k full solves). Optical draws use the material uncertainties.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import SimulationConfig
from enclosure import NodeSolution, solve_steady
from materials import Material
from parameters import Status


@dataclass
class MonteCarloResult:
    t_electronics_c: np.ndarray
    mean: float
    median: float
    std: float
    p5: float
    p95: float
    n: int
    ranking_message: str


@dataclass
class SensitivityResult:
    names: list[str]
    dT_dparam: dict[str, float]
    tornado: list[tuple[str, float]]
    next_experiment: str


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def monte_carlo(
    cfg: SimulationConfig,
    material: Material,
    altitude_m: float,
    n: int | None = None,
) -> MonteCarloResult:
    n = n or cfg.uncertainty.n_samples
    rng = np.random.default_rng(cfg.uncertainty.seed)
    u = cfg.uncertainty
    saved_mode = cfg.environment.radiation_mode
    cfg.environment.radiation_mode = "effective_sky"
    temps = np.empty(n)
    base_alpha = material.solar_absorptance
    base_eps = material.ir_emissivity
    base_q = cfg.electronics.heat_load_w
    base_g = cfg.environment.irradiance_direct_w_m2
    for i in range(n):
        material.solar_absorptance = _clip(rng.normal(base_alpha, u.rel_alpha * max(base_alpha, 0.02)), 0.01, 0.99)
        material.solar_reflectance = 1.0 - material.solar_absorptance
        material.ir_emissivity = _clip(rng.normal(base_eps, u.rel_eps * max(base_eps, 0.05)), 0.02, 0.99)
        cfg.electronics.heat_load_w = max(0.0, rng.normal(base_q, u.rel_q_internal * max(base_q, 1.0)))
        if base_g is not None:
            cfg.environment.irradiance_direct_w_m2 = max(0.0, rng.normal(base_g, u.rel_g * max(base_g, 1.0)))
        sol = solve_steady(cfg, material, altitude_m=altitude_m)
        temps[i] = sol.t_electronics_c
    material.solar_absorptance = base_alpha
    material.solar_reflectance = 1.0 - base_alpha
    material.ir_emissivity = base_eps
    cfg.electronics.heat_load_w = base_q
    cfg.environment.irradiance_direct_w_m2 = base_g
    cfg.environment.radiation_mode = saved_mode
    mean = float(np.mean(temps))
    std = float(np.std(temps, ddof=1))
    p5, p95 = float(np.percentile(temps, 5)), float(np.percentile(temps, 95))
    msg = (
        f"Electronics temperature (broadband MC, n={n}): "
        f"mean {mean:.1f} C, median {float(np.median(temps)):.1f} C, "
        f"sd {std:.1f} K, 5th {p5:.1f} C, 95th {p95:.1f} C."
    )
    return MonteCarloResult(temps, mean, float(np.median(temps)), std, p5, p95, n, msg)


def sensitivity(cfg: SimulationConfig, material: Material, altitude_m: float) -> SensitivityResult:
    saved_mode = cfg.environment.radiation_mode
    cfg.environment.radiation_mode = "effective_sky"
    base = solve_steady(cfg, material, altitude_m=altitude_m)

    def dT(mutate, restore) -> float:
        mutate()
        s = solve_steady(cfg, material, altitude_m=altitude_m)
        restore()
        return s.t_electronics_c - base.t_electronics_c

    names = []
    effects = {}

    a0, e0 = material.solar_absorptance, material.ir_emissivity
    da = 0.02
    material.solar_absorptance = a0 + da
    material.solar_reflectance = 1.0 - material.solar_absorptance
    s1 = solve_steady(cfg, material, altitude_m=altitude_m)
    material.solar_absorptance = a0
    material.solar_reflectance = 1.0 - a0
    effects["alpha_solar"] = (s1.t_electronics_c - base.t_electronics_c) / da

    de = 0.02
    material.ir_emissivity = min(0.99, e0 + de)
    s2 = solve_steady(cfg, material, altitude_m=altitude_m)
    material.ir_emissivity = e0
    effects["epsilon_IR"] = (s2.t_electronics_c - base.t_electronics_c) / de

    g0 = cfg.environment.irradiance_direct_w_m2
    if g0 is None:
        g_use = base.solar_direct_w_m2
    else:
        g_use = g0
    cfg.environment.irradiance_direct_w_m2 = g_use + 50.0
    s3 = solve_steady(cfg, material, altitude_m=altitude_m)
    cfg.environment.irradiance_direct_w_m2 = g0
    effects["G_solar"] = (s3.t_electronics_c - base.t_electronics_c) / 50.0

    q0 = cfg.electronics.heat_load_w
    cfg.electronics.heat_load_w = q0 + 5.0
    s4 = solve_steady(cfg, material, altitude_m=altitude_m)
    cfg.electronics.heat_load_w = q0
    effects["Q_internal"] = (s4.t_electronics_c - base.t_electronics_c) / 5.0

    w0 = cfg.environment.wind_speed_m_s
    cfg.environment.wind_speed_m_s = w0 + 1.0
    s5 = solve_steady(cfg, material, altitude_m=altitude_m)
    cfg.environment.wind_speed_m_s = w0
    effects["wind_speed"] = (s5.t_electronics_c - base.t_electronics_c) / 1.0

    h_note = base.h_w_m2_k
    effects["h_implied_via_wind"] = effects["wind_speed"]

    # T_sky via cloud boost / radiation mode is implicit; perturb Earth LW as proxy for environment
    olr0 = cfg.environment.earth_longwave_w_m2
    cfg.environment.earth_longwave_w_m2 = 240.0 + 20.0
    s6 = solve_steady(cfg, material, altitude_m=altitude_m)
    cfg.environment.earth_longwave_w_m2 = 240.0
    s6b = solve_steady(cfg, material, altitude_m=altitude_m)
    cfg.environment.earth_longwave_w_m2 = olr0
    effects["earth_longwave"] = (s6.t_electronics_c - s6b.t_electronics_c) / 20.0

    cfg.environment.radiation_mode = saved_mode

    # Rank by |ΔT| for a representative uncertainty step
    u = cfg.uncertainty
    scaled = {
        "alpha_solar": abs(effects["alpha_solar"] * u.rel_alpha * max(a0, 0.02)),
        "epsilon_IR": abs(effects["epsilon_IR"] * u.rel_eps * max(e0, 0.05)),
        "G_solar": abs(effects["G_solar"] * u.rel_g * max(g_use, 50.0)),
        "Q_internal": abs(effects["Q_internal"] * u.rel_q_internal * max(q0, 1.0)),
        "wind_speed": abs(effects["wind_speed"] * 2.0),  # 2 m/s ASSUMED wind uncertainty
        "earth_longwave": abs(effects["earth_longwave"] * 20.0),
    }
    tornado = sorted(scaled.items(), key=lambda kv: kv[1], reverse=True)
    top = tornado[0][0]
    rec = {
        "alpha_solar": "Measure solar reflectance of the actual STRATOVANE sample (spectrophotometer, 0.3-2.5 μm).",
        "epsilon_IR": "Measure spectral IR emissivity of the actual sample (FTIR, at least 2-25 μm, ideally 2-50 μm).",
        "G_solar": "Record site/aircraft irradiance, solar geometry, and attitude rather than assuming a constant G.",
        "Q_internal": "Measure the actual avionics heat load versus time; scenario watts are not flight data.",
        "wind_speed": "Measure local airspeed/boundary-layer conditions on the enclosure.",
        "earth_longwave": "Improve atmospheric/Earth radiative boundary conditions (view factors, OLR, sky RT).",
        "atmospheric_transmission": "Improve the atmospheric radiative model (line-by-line or measured downwelling IR).",
    }
    next_exp = rec.get(top, "Measure the dominant uncertain parameter listed in the tornado plot.")
    return SensitivityResult(list(scaled), effects, tornado, next_exp)
