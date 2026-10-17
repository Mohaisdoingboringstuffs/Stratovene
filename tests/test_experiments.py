import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from experiments import metrics, ExperimentMetrics, load_experiment_csv
from config import SimulationConfig

def test_heating_cooling_rates():
    # Create synthetic data: 3 samples, 3 time points each
    # Sample A: heating at 0.1 C/s
    # Sample B: cooling at -0.2 C/s
    # Sample C: constant temp
    data = {
        "timestamp": [
            "2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", "2026-01-01T00:00:02Z",
            "2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", "2026-01-01T00:00:02Z",
            "2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", "2026-01-01T00:00:02Z",
        ],
        "sample": ["A"]*3 + ["B"]*3 + ["C"]*3,
        "surface_temperature_C": [
            10.0, 10.1, 10.2,  # A: +0.1 C/s
            20.0, 19.8, 19.6,  # B: -0.2 C/s
            30.0, 30.0, 30.0,  # C: 0.0 C/s
        ],
        "internal_temperature_C": [0]*9,
        "ambient_temperature_C": [25]*9,
        "humidity_percent": [50]*9,
        "wind_speed_m_s": [1]*9,
        "irradiance_W_m2": [500]*9,
    }
    df = pd.DataFrame(data)
    cfg = SimulationConfig()
    
    res = metrics(df, cfg)
    
    assert np.isclose(res.heating_rate_c_s, 0.1)
    assert np.isclose(res.cooling_rate_c_s, -0.2)

def test_invalid_timestamps():
    data = {
        "timestamp": [
            "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", # Duplicate
            "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z",
        ],
        "sample": ["A"]*3 + ["B"]*3,
        "surface_temperature_C": [10, 11, 12, 20, 21, 22],
        "internal_temperature_C": [0]*6,
        "ambient_temperature_C": [25]*6,
        "humidity_percent": [50]*6,
        "wind_speed_m_s": [1]*6,
        "irradiance_W_m2": [500]*6,
    }
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="must be strictly increasing"):
        metrics(df)


def test_bundled_example_is_labelled_synthetic():
    df = load_experiment_csv()
    assert "SYNTHETIC" in df.attrs["data_source"]


def test_model_rmse_and_rmse_share_reference_and_rows():
    df = load_experiment_csv()
    cfg = SimulationConfig()
    res = metrics(df, cfg=cfg)
    assert res.model_rmse is not None
    # Same reference (measured surface temp) and same rows as model_rmse.
    assert res.rmse == res.model_rmse
    assert res.metric_reference.startswith("surface_temperature_C")
    # The ambient-surface data descriptor is reported separately.
    assert res.deltaT_ambient_surface_rmse > 0
    assert res.data_source and "SYNTHETIC" in res.data_source


def test_rmse_reference_without_model_is_explicit():
    df = load_experiment_csv()
    res = metrics(df)
    assert res.model_rmse is None
    assert res.metric_reference.startswith("ambient - surface")
    assert np.isclose(res.rmse, res.deltaT_ambient_surface_rmse)
