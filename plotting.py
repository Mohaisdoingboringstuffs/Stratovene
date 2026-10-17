"""Automatic figures. Saved under outputs/plots/."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import SimulationConfig
from materials import Material
from spectral import WINDOW_UM, planck_exitance_lambda, thermal_wavelengths_m, solar_wavelengths_m
from radiation import band_absorption_coefficient, spectral_downwelling, spectral_upward_transmittance
from atmosphere import atmosphere_at


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_altitude_temperatures(alt_m, series: dict[str, np.ndarray], ylabel: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, y in series.items():
        ax.plot(np.asarray(alt_m) / 1000.0, y, marker="o", label=name)
    ax.set_xlabel("Altitude (km)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save(fig, path)


def plot_vs_x(x, series: dict[str, np.ndarray], xlabel, ylabel, path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, y in series.items():
        ax.plot(x, y, marker="o", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save(fig, path)


def plot_spectra(cfg: SimulationConfig, material: Material, path_dir: Path) -> list[Path]:
    atm = atmosphere_at(cfg.environment.altitude_m, cfg.environment.relative_humidity_surface)
    lam = thermal_wavelengths_m()
    lam_um = lam * 1e6
    e_s = material.epsilon_on(lam)
    r_s = material.reflectance_on(lam)
    down = spectral_downwelling(atm, lam, cfg.environment.sky_condition, cfg.environment.cloud_fraction)
    trans = spectral_upward_transmittance(atm, lam)
    t_s = 300.0
    emit = e_s * planck_exitance_lambda(lam, t_s)
    ks = band_absorption_coefficient(lam)

    outs = []
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lam_um, emit / emit.max(), label="surface emission (norm.)")
    ax.axvspan(WINDOW_UM[0], WINDOW_UM[1], color="0.85", label="8–13 μm window")
    ax.set_xlabel("Wavelength (μm)")
    ax.set_ylabel("Normalized spectral flux")
    ax.set_title("Net radiative flux ingredients vs wavelength")
    ax.legend()
    outs.append(_save(fig, path_dir / "07_net_radiative_flux_wavelength.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lam_um, 1.0 - r_s)
    ax.set_xlim(0.3, 4)
    ax.set_xlabel("Wavelength (μm)")
    ax.set_ylabel("Absorptance")
    ax.set_title(f"Solar-range absorptance — {material.name}")
    outs.append(_save(fig, path_dir / "08_solar_absorptance_spectrum.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lam_um, e_s)
    ax.axvspan(WINDOW_UM[0], WINDOW_UM[1], color="0.85")
    ax.set_xlabel("Wavelength (μm)")
    ax.set_ylabel("Emissivity")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"IR emissivity spectrum — {material.name} ({material.status.value})")
    outs.append(_save(fig, path_dir / "09_ir_emissivity_spectrum.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lam_um, trans, label="upward transmittance")
    ax.plot(lam_um, down / max(down.max(), 1e-30), label="downwelling (norm.)")
    ax.axvspan(WINDOW_UM[0], WINDOW_UM[1], color="0.85")
    ax.set_xlabel("Wavelength (μm)")
    ax.set_ylabel("Transmittance / normalized radiance")
    ax.set_title(f"Atmospheric IR at {cfg.environment.altitude_m/1000:.0f} km (band model)")
    ax.legend()
    outs.append(_save(fig, path_dir / "10_atmospheric_transmission_spectrum.png"))
    return outs


def plot_transient(tr, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    hours = tr.time_s / 3600.0
    ax.plot(hours, tr.t_electronics_k - 273.15, label="electronics")
    ax.plot(hours, tr.t_wall_k - 273.15, label="wall")
    ax.plot(hours, tr.t_air_k - 273.15, label="US76 T_air")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Temperature (C)")
    ax.set_title(f"24 h transient  min={tr.t_min_c:.1f} C  max={tr.t_max_c:.1f} C  swing={tr.swing_k:.1f} K")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save(fig, path)


def plot_mc(mc, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(mc.t_electronics_c, bins=40, color="0.4")
    ax.axvline(mc.p5, color="k", ls="--", label="5th")
    ax.axvline(mc.p95, color="k", ls="--", label="95th")
    ax.axvline(mc.mean, color="C1", label="mean")
    ax.set_xlabel("Electronics temperature (C)")
    ax.set_ylabel("Count")
    ax.legend()
    return _save(fig, path)


def plot_tornado(sens, path: Path) -> Path:
    names = [n for n, _ in reversed(sens.tornado)]
    vals = [v for _, v in reversed(sens.tornado)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(names, vals)
    ax.set_xlabel("|ΔT| from representative uncertainty (K)")
    ax.set_title("Sensitivity ranking")
    return _save(fig, path)


def plot_residual(df, pred, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    meas = df["surface_temperature_C"].to_numpy()
    ax.scatter(meas, pred, c="0.2")
    lims = [min(meas.min(), pred.min()), max(meas.max(), pred.max())]
    ax.plot(lims, lims, "k--")
    ax.set_xlabel("Measured surface T (C)")
    ax.set_ylabel("Modeled surface T (C)")
    return _save(fig, path)
