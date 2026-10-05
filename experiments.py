"""Experimental CSV import and model-experiment metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, SimulationConfig
from enclosure import solve_steady
from materials import load_materials


REQUIRED = [
    "timestamp",
    "sample",
    "surface_temperature_C",
    "internal_temperature_C",
    "ambient_temperature_C",
    "humidity_percent",
    "wind_speed_m_s",
    "irradiance_W_m2",
]


@dataclass
class ExperimentMetrics:
    n: int
    rmse: float
    mae: float
    bias: float
    ci95: tuple[float, float]
    heating_rate_c_s: float
    cooling_rate_c_s: float
    deltaT_mean: float
    model_rmse: float | None


def load_experiment_csv(path: Path | None = None) -> pd.DataFrame:
    path = path or (DATA_DIR / "experiments" / "example_ground_test.csv")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def metrics(df: pd.DataFrame, cfg: SimulationConfig | None = None) -> ExperimentMetrics:
    ts = pd.to_datetime(df["timestamp"])
    dt = ts.diff().dt.total_seconds().to_numpy()
    t_s = df["surface_temperature_C"].to_numpy()
    dTdt = np.diff(t_s) / np.clip(dt[1:], 1e-6, None)
    heat = float(np.nanmax(dTdt))
    cool = float(np.nanmin(dTdt))
    dT = (df["ambient_temperature_C"] - df["surface_temperature_C"]).to_numpy()
    mean = float(np.mean(dT))
    std = float(np.std(dT, ddof=1)) if len(dT) > 1 else 0.0
    se = std / max(np.sqrt(len(dT)), 1.0)
    ci = (mean - 1.96 * se, mean + 1.96 * se)
    model_rmse = None
    if cfg is not None:
        mats = load_materials()
        preds = []
        for _, row in df.iterrows():
            sample = str(row["sample"])
            key = {
                "bare": "BARE_ALUMINUM_OXIDIZED",
                "white": "WHITE_REFERENCE_PLACEHOLDER",
                "stratovane": "STRATOVANE_AAO_LITERATURE",
                "aao": "AAO_AL_DIAZ_LOBO",
            }.get(sample.lower(), "STRATOVANE_AAO_LITERATURE")
            cfg.environment.irradiance_direct_w_m2 = float(row["irradiance_W_m2"])
            cfg.environment.wind_speed_m_s = float(row["wind_speed_m_s"])
            cfg.environment.relative_humidity_surface = float(row["humidity_percent"]) / 100.0
            cfg.environment.altitude_m = 0.0
            sol = solve_steady(cfg, mats[key], altitude_m=0.0)
            preds.append(sol.t_surface_c)
        resid = np.asarray(preds) - t_s
        model_rmse = float(np.sqrt(np.mean(resid**2)))
    resid_ss = dT - mean
    rmse = float(np.sqrt(np.mean(resid_ss**2)))
    mae = float(np.mean(np.abs(resid_ss)))
    return ExperimentMetrics(len(df), rmse, mae, mean, ci, heat, cool, mean, model_rmse)
