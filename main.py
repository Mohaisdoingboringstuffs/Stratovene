#!/usr/bin/env python3
"""STRATOVANE thermal model CLI.

python main.py
python main.py --mode 3
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DEFAULT_ALTITUDES_M,
    HEAT_LOAD_SCENARIOS_W,
    RH_SCENARIOS,
    WIND_SCENARIOS_M_S,
    SimulationConfig,
)
from enclosure import comparison_table, solve_steady
from experiments import load_experiment_csv, metrics as exp_metrics
from materials import load_materials
from plotting import (
    plot_altitude_temperatures,
    plot_mc,
    plot_residual,
    plot_spectra,
    plot_tornado,
    plot_transient,
    plot_vs_x,
)
from report import write_report
from spectral import planck_sigma_check
from transient import simulate_24h
from uncertainty import monte_carlo, sensitivity
from validation import diaz_lobo_reproduction, FU_2023, AAO_OPTIONAL_A, AAO_OPTIONAL_B


MENU = """
STRATOVANE thermal model — not a sales tool, not flight-proven
==============================================================
1. Literature benchmark (Díaz-Lobo reproduction)
2. Material comparison (bare / white / STRATOVANE literature AAO)
3. Altitude simulation (0-20 km, optional arbitrary 0-25 km)
4. Spectral simulation
5. Enclosure simulation (multi-node)
6. 24-hour simulation
7. Experimental data analysis
8. Monte Carlo uncertainty
9. Sensitivity analysis
10. Generate all plots + report
0. Exit
"""


def _trio(cfg: SimulationConfig):
    mats = load_materials()
    return {
        "bare_aluminum": mats[cfg.bare_material_id],
        "white_reference": mats[cfg.white_material_id],
        "stratovane_aao_literature": mats[cfg.stratovane_material_id],
    }


def run_benchmark(cfg: SimulationConfig) -> None:
    b = diaz_lobo_reproduction(cfg)
    print(json.dumps({"name": b.name, "reported": b.reported, "modeled": b.modeled, "discrepancy": b.discrepancy, "notes": b.notes}, indent=2))
    print("\nOptional AAO comparison cases (NOT merged into STRATOVANE):")
    print(AAO_OPTIONAL_A)
    print(AAO_OPTIONAL_B)
    print("\nFu et al. 2023 airship (different material):")
    print(FU_2023)


def run_comparison(cfg: SimulationConfig, altitudes=None) -> tuple[list[dict], dict]:
    altitudes = altitudes or [cfg.environment.altitude_m]
    rows = []
    last = {}
    for z in altitudes:
        sols = {}
        for name, mat in _trio(cfg).items():
            sols[name] = solve_steady(cfg, mat, altitude_m=z)
            print(
                f"{name:28s} z={z/1000:5.1f} km  "
                f"T_s={sols[name].t_surface_c:7.2f} C  "
                f"T_e={sols[name].t_electronics_c:7.2f} C  "
                f"Q_env={sols[name].p_cool_net_w:7.2f} W "
                f"(rad={sols[name].q_env_rad_net_w:7.2f}, conv={sols[name].q_env_conv_w:7.2f}, "
                f"sol={sols[name].q_env_solar_w:7.2f})  "
                f"bal={sols[name].relative_energy_error:.2e}"
            )
        last = sols
        rows.extend(comparison_table(sols))
    return rows, last


def run_altitude(cfg: SimulationConfig, z_user: float | None = None):
    zs = list(DEFAULT_ALTITUDES_M)
    if z_user is not None:
        zs.append(z_user)
    return run_comparison(cfg, zs)


def answers_20km(rows: list[dict]) -> str:
    r20 = [r for r in rows if abs(r["altitude_m"] - 20000) < 1]
    if len(r20) < 3:
        return "20 km comparison not present in this run."
    by = {r["material"]: r for r in r20}
    te = {k: v["T_electronics_C"] for k, v in by.items()}
    adv_bare = te.get("bare_aluminum", np.nan) - te.get("stratovane_aao_literature", np.nan)
    adv_white = te.get("white_reference", np.nan) - te.get("stratovane_aao_literature", np.nan)
    return (
        f"At 20 km (US76 air, scenario heat/wind/solar as configured, literature AAO optics):\n"
        f"- T_elec bare aluminum: {te.get('bare_aluminum'):.2f} C\n"
        f"- T_elec white reference: {te.get('white_reference'):.2f} C\n"
        f"- T_elec STRATOVANE literature AAO: {te.get('stratovane_aao_literature'):.2f} C\n"
        f"- Advantage vs bare (positive => cooler internals): {adv_bare:.2f} K\n"
        f"- Advantage vs white (positive => cooler internals): {adv_white:.2f} K\n"
        f"These are model predictions with literature optics, not measurements.\n"
        f"If the white reference is already cooler, the model must be believed on that point."
    )


def generate_all(cfg: SimulationConfig) -> None:
    out = cfg.ensure_output()
    plots = out / "plots"
    check = planck_sigma_check(300.0)
    print("Planck vs sigma T^4 relative error:", check["rel_error"])

    run_benchmark(cfg)
    print("\n--- Altitude sweep ---")
    rows, last = run_altitude(cfg)
    pd.DataFrame(rows).to_csv(out / "csv" / "altitude_comparison.csv", index=False)

    # Rebuild series
    mats = _trio(cfg)
    zs = list(DEFAULT_ALTITUDES_M)
    series_s, series_e, series_adv, series_p = {}, {}, {}, {}
    for name, mat in mats.items():
        ts, te, pc = [], [], []
        for z in zs:
            s = solve_steady(cfg, mat, altitude_m=z)
            ts.append(s.t_surface_c)
            te.append(s.t_electronics_c)
            pc.append(s.p_cool_net_w)
        series_s[name] = np.array(ts)
        series_e[name] = np.array(te)
        series_p[name] = np.array(pc)
    series_adv["vs_bare"] = series_e["bare_aluminum"] - series_e["stratovane_aao_literature"]
    series_adv["vs_white"] = series_e["white_reference"] - series_e["stratovane_aao_literature"]
    plot_altitude_temperatures(zs, series_s, "Surface temperature (C)", plots / "01_surface_temperature_vs_altitude.png")
    plot_altitude_temperatures(zs, series_e, "Electronics temperature (C)", plots / "02_electronics_temperature_vs_altitude.png")
    plot_altitude_temperatures(zs, series_adv, "Cooling advantage of literature AAO (K)", plots / "05_cooling_advantage_vs_altitude.png")
    plot_altitude_temperatures(zs, series_p, "Net environmental cooling Q_env (W)", plots / "06_cooling_power_vs_altitude.png")

    gs = np.array([0.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0])
    g_series = {}
    orig_g = cfg.environment.irradiance_direct_w_m2
    for name, mat in mats.items():
        vals = []
        for g in gs:
            cfg.environment.irradiance_direct_w_m2 = float(g)
            vals.append(solve_steady(cfg, mat, altitude_m=20000.0).t_electronics_c)
        g_series[name] = np.array(vals)
    cfg.environment.irradiance_direct_w_m2 = orig_g
    plot_vs_x(gs, g_series, "User direct irradiance on horizontal (W/m2)", "T_elec (C)", plots / "03_temperature_vs_solar.png")

    q_series = {}
    for name, mat in mats.items():
        q_series[name] = np.array(
            [solve_steady(cfg, mat, altitude_m=20000.0, heat_w=q).t_electronics_c for q in HEAT_LOAD_SCENARIOS_W]
        )
    plot_vs_x(HEAT_LOAD_SCENARIOS_W, q_series, "Electronics heat (W)", "T_elec (C)", plots / "04_temperature_vs_heat_load.png")

    plot_spectra(cfg, mats["stratovane_aao_literature"], plots)

    print("\n--- 24 h ---")
    trans_plots = {}
    for name, mat in mats.items():
        tr = simulate_24h(cfg, mat, altitude_m=20000.0)
        trans_plots[name] = tr
        print(f"{name}: min {tr.t_min_c:.1f} C  max {tr.t_max_c:.1f} C  swing {tr.swing_k:.1f} K")
    plot_transient(trans_plots["stratovane_aao_literature"], plots / "11_24h_enclosure_temperature.png")
    fig_swing_x = list(trans_plots)
    fig_swing_y = [trans_plots[k].swing_k for k in fig_swing_x]
    plot_vs_x(range(len(fig_swing_x)), {"swing_K": np.array(fig_swing_y)}, "material index", "day-night swing (K)", plots / "12_day_night_swing.png")

    rh_series = {}
    rh0 = cfg.environment.relative_humidity_surface
    for name, mat in mats.items():
        rh_series[name] = np.array(
            [
                solve_steady(
                    deepcopy(cfg) if False else cfg, mat, altitude_m=0.0
                ).t_electronics_c
                if False
                else _rh_point(cfg, mat, rh)
                for rh in RH_SCENARIOS
            ]
        )
    cfg.environment.relative_humidity_surface = rh0
    plot_vs_x(np.array(RH_SCENARIOS) * 100, rh_series, "Surface RH (%) at 0 km", "T_elec (C)", plots / "13_humidity_sensitivity.png")

    w_series = {}
    w0 = cfg.environment.wind_speed_m_s
    for name, mat in mats.items():
        w_series[name] = np.array(
            [solve_steady(cfg, mat, altitude_m=0.0, wind=w).t_electronics_c for w in WIND_SCENARIOS_M_S]
        )
    cfg.environment.wind_speed_m_s = w0
    plot_vs_x(WIND_SCENARIOS_M_S, w_series, "Wind (m/s) at 0 km", "T_elec (C)", plots / "14_wind_convection_sensitivity.png")

    print("\n--- Monte Carlo (10k, broadband) ---")
    mc = monte_carlo(cfg, mats["stratovane_aao_literature"], 20000.0)
    print(mc.ranking_message)
    plot_mc(mc, plots / "15_monte_carlo.png")
    pd.Series(mc.t_electronics_c, name="T_elec_C").to_csv(out / "csv" / "monte_carlo_T_elec.csv", index=False)

    print("\n--- Sensitivity ---")
    sens = sensitivity(cfg, mats["stratovane_aao_literature"], 20000.0)
    print("Tornado:", sens.tornado)
    print(sens.next_experiment)
    plot_tornado(sens, plots / "18_sensitivity_tornado.png")

    print("\n--- Experiment example ---")
    df = load_experiment_csv()
    met = exp_metrics(df, cfg=cfg)
    print(f"DATA SOURCE: {met.data_source}")
    print(met)
    # residual plot vs ambient-surface as placeholder if no coupled predictions
    plot_vs_x(
        np.arange(len(df)),
        {"surface": df["surface_temperature_C"].to_numpy(), "internal": df["internal_temperature_C"].to_numpy()},
        "sample index",
        "T (C)",
        plots / "16_model_vs_experiment_timeseries.png",
    )
    resid = df["surface_temperature_C"].to_numpy() - df["ambient_temperature_C"].to_numpy()
    plot_vs_x(np.arange(len(resid)), {"Tsurf-Tamb": resid}, "index", "K", plots / "17_residuals.png")

    ans = answers_20km(rows)
    print("\n" + ans)
    write_report(
        out / "reports" / "stratovane_report.md",
        cfg_text=json.dumps(
            {
                "geometry_m": [cfg.geometry.length_m, cfg.geometry.width_m, cfg.geometry.height_m],
                "heat_W": cfg.electronics.heat_load_w,
                "wind": cfg.environment.wind_speed_m_s,
                "lat": cfg.environment.latitude_deg,
                "date": f"{cfg.environment.year}-{cfg.environment.month:02d}-{cfg.environment.day:02d}",
                "radiation_mode": cfg.environment.radiation_mode,
                "alloy": cfg.alloy,
            },
            indent=2,
        ),
        solutions=last,
        comparison_rows=rows,
        mc_text=mc.ranking_message,
        sensitivity_text=str(sens.tornado),
        next_experiment=sens.next_experiment,
        validation_text=json.dumps(diaz_lobo_reproduction(cfg).discrepancy, indent=2),
        answers=ans,
    )
    print("Wrote", out)


def _rh_point(cfg, mat, rh):
    cfg.environment.relative_humidity_surface = rh
    return solve_steady(cfg, mat, altitude_m=0.0).t_electronics_c


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="STRATOVANE thermal model")
    p.add_argument("--mode", type=int, default=None)
    p.add_argument("--altitude-km", type=float, default=None)
    p.add_argument("--heat-w", type=float, default=None)
    p.add_argument("--wind", type=float, default=None)
    p.add_argument("--irradiance", type=float, default=None)
    p.add_argument("--radiation-mode", choices=["spectral", "effective_sky"], default="spectral")
    args = p.parse_args(argv)
    cfg = SimulationConfig()
    cfg.environment.radiation_mode = args.radiation_mode
    if args.heat_w is not None:
        cfg.electronics.heat_load_w = args.heat_w
    if args.wind is not None:
        cfg.environment.wind_speed_m_s = args.wind
    if args.irradiance is not None:
        cfg.environment.irradiance_direct_w_m2 = args.irradiance
    if args.altitude_km is not None:
        cfg.environment.altitude_m = args.altitude_km * 1000.0

    mode = args.mode
    if mode is None and sys.stdin.isatty():
        print(MENU)
        try:
            mode = int(input("Select mode: ").strip() or "10")
        except EOFError:
            mode = 10
    if mode is None:
        mode = 10

    if mode == 0:
        return 0
    if mode == 1:
        run_benchmark(cfg)
    elif mode == 2:
        run_comparison(cfg, [0.0, 10000.0, 15000.0, 20000.0])
    elif mode == 3:
        z = None if args.altitude_km is None else args.altitude_km * 1000.0
        run_altitude(cfg, z)
    elif mode == 4:
        from plotting import plot_spectra

        cfg.ensure_output()
        plot_spectra(cfg, load_materials()[cfg.stratovane_material_id], cfg.output_dir / "plots")
        print("Spectral plots written.")
    elif mode == 5:
        run_comparison(cfg, [cfg.environment.altitude_m])
    elif mode == 6:
        mats = _trio(cfg)
        for n, m in mats.items():
            tr = simulate_24h(cfg, m)
            print(n, tr.t_min_c, tr.t_max_c, tr.swing_k)
    elif mode == 7:
        df = load_experiment_csv()
        print(df.head())
        met = exp_metrics(df)
        print(f"DATA SOURCE: {met.data_source}")
        print(met)
    elif mode == 8:
        mc = monte_carlo(cfg, load_materials()[cfg.stratovane_material_id], cfg.environment.altitude_m)
        print(mc.ranking_message)
    elif mode == 9:
        s = sensitivity(cfg, load_materials()[cfg.stratovane_material_id], cfg.environment.altitude_m)
        print(s.tornado)
        print(s.next_experiment)
    elif mode == 10:
        generate_all(cfg)
    else:
        print("Unknown mode")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
