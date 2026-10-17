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
    # Which quantity err_metrics are measured against. When a config is given,
    # rmse/mae/model_rmse all use the SAME reference (measured surface temp) and
    # the SAME sorted rows; otherwise rmse/mae describe the measured
    # ambient-surface difference.
    metric_reference: str = "surface_temperature_C"
    deltaT_ambient_surface_rmse: float = 0.0
    deltaT_ambient_surface_mae: float = 0.0
    data_source: str = "unspecified"


def load_experiment_csv(path: Path | None = None) -> pd.DataFrame:
    path = path or (DATA_DIR / "experiments" / "example_ground_test.csv")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if "example" in Path(path).name.lower():
        df.attrs["data_source"] = (
            "SYNTHETIC EXAMPLE placeholder (see data/experiments/README.md); "
            "NOT a STRATOVANE measurement campaign"
        )
    else:
        df.attrs["data_source"] = str(path)
    return df


def metrics(df: pd.DataFrame, cfg: SimulationConfig | None = None) -> ExperimentMetrics:
    df = df.sort_values(["sample", "timestamp"]).reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"])
    data_source = str(df.attrs.get("data_source", "unspecified"))

    # Calculate rates per sample to avoid cross-sample jumps
    heat_rates = []
    cool_rates = []
    for sample, group in df.groupby("sample"):
        s_ts = pd.to_datetime(group["timestamp"])
        dt = s_ts.diff().dt.total_seconds().to_numpy()
        t_s = group["surface_temperature_C"].to_numpy()

        if len(group) < 2:
            continue

        # Validate time is strictly increasing
        if np.any(dt[1:] <= 0):
            raise ValueError(f"Timestamps for sample {sample} must be strictly increasing.")

        dTdt = np.diff(t_s) / dt[1:]
        heat_rates.append(np.nanmax(dTdt))
        cool_rates.append(np.nanmin(dTdt))

    heat = float(np.nanmax(heat_rates)) if heat_rates else 0.0
    cool = float(np.nanmin(cool_rates)) if cool_rates else 0.0

    # Data descriptor: measured surface-vs-ambient difference, same rows.
    dT = (df["ambient_temperature_C"] - df["surface_temperature_C"]).to_numpy()
    mean = float(np.mean(dT))
    std = float(np.std(dT, ddof=1)) if len(dT) > 1 else 0.0
    se = std / max(np.sqrt(len(dT)), 1.0)
    ci = (mean - 1.96 * se, mean + 1.96 * se)
    deltaT_rmse = float(np.sqrt(np.mean(dT**2)))
    deltaT_mae = float(np.mean(np.abs(dT)))

    model_rmse = None
    model_mae = None
    preds = None
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
        resid = np.asarray(preds) - df["surface_temperature_C"].to_numpy()
        assert resid.shape[0] == len(df)
        model_rmse = float(np.sqrt(np.mean(resid**2)))
        model_mae = float(np.mean(np.abs(resid)))

    if cfg is not None:
        # Same reference (measured surface temp) and same rows as model_rmse.
        rmse = model_rmse
        mae = model_mae
        reference = "surface_temperature_C (model vs measured, same sorted rows)"
    else:
        # No model available; report the data descriptor explicitly.
        rmse = deltaT_rmse
        mae = deltaT_mae
        reference = "ambient - surface (data descriptor; no model config supplied)"

    return ExperimentMetrics(
        n=len(df),
        rmse=rmse,
        mae=mae,
        bias=mean,
        ci95=ci,
        heating_rate_c_s=heat,
        cooling_rate_c_s=cool,
        deltaT_mean=mean,
        model_rmse=model_rmse,
        metric_reference=reference,
        deltaT_ambient_surface_rmse=deltaT_rmse,
        deltaT_ambient_surface_mae=deltaT_mae,
        data_source=data_source,
    )
