"""User-configurable simulation settings.

Defaults are SCENARIOS or literature placeholders, not aircraft measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from parameters import Status


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


@dataclass
class GeometryConfig:
    length_m: float = 0.20
    width_m: float = 0.15
    height_m: float = 0.10
    wall_thickness_m: float = 0.002
    # Prototype box, NOT an aircraft enclosure.
    source: str = "USER INPUT default prototype 0.20 x 0.15 x 0.10 m"
    status: Status = Status.USER_INPUT


@dataclass
class ElectronicsConfig:
    heat_load_w: float = 20.0
    mass_kg: float = 0.50
    cp_j_kg_k: float = 850.0
    mount_conductance_w_k: float = 2.0
    time_varying: bool = False
    day_heat_load_w: float = 20.0
    night_heat_load_w: float = 20.0
    notes: str = "SCENARIO defaults (5/10/20/30/50 W), not measured avionics."


@dataclass
class EnvironmentConfig:
    altitude_m: float = 20000.0
    latitude_deg: float = 35.0
    longitude_deg: float = -106.0
    year: int = 2026
    month: int = 6
    day: int = 21
    utc_hour: float = 18.0  # noon-ish for ~105 W US
    wind_speed_m_s: float = 5.0
    relative_humidity_surface: float = 0.40
    precipitable_water_mm: float | None = None
    stratospheric_h2o_ppmv: float = 5.0
    earth_albedo: float = 0.30
    earth_skin_temperature_k: float | None = None
    earth_longwave_w_m2: float | None = None
    irradiance_direct_w_m2: float | None = None
    irradiance_diffuse_w_m2: float | None = 50.0
    sky_condition: str = "clear"  # clear, partly_cloudy, cloudy
    cloud_fraction: float = 0.0
    top_tilt_deg: float = 0.0  # 0 = zenith-facing lid
    radiating_mode: str = "all_faces"  # all_faces | top_only
    radiation_mode: str = "spectral"  # spectral | effective_sky
    internal_air_model: bool = False
    air_mass_override: float | None = None


@dataclass
class TransientConfig:
    duration_s: float = 24.0 * 3600.0
    dt_s: float = 30.0
    t_init_offset_k: float = 0.0


@dataclass
class UncertaintyConfig:
    n_samples: int = 10_000
    seed: int = 42
    rel_alpha: float = 0.20
    rel_eps: float = 0.05
    t_sky_k: float = 10.0
    rel_h: float = 0.30
    rel_g: float = 0.10
    rel_q_internal: float = 0.20
    rel_k_wall: float = 0.05
    rel_atm_trans: float = 0.20


@dataclass
class SimulationConfig:
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    electronics: ElectronicsConfig = field(default_factory=ElectronicsConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    transient: TransientConfig = field(default_factory=TransientConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    alloy: str = "6061-T6"
    aluminum_finish: str = "oxidized"  # polished | oxidized | rough
    stratovane_material_id: str = "STRATOVANE_AAO_LITERATURE"
    white_material_id: str = "WHITE_REFERENCE_PLACEHOLDER"
    bare_material_id: str = "BARE_ALUMINUM_OXIDIZED"
    energy_balance_flag: float = 1e-3
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)

    def ensure_output(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "csv").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        return self.output_dir


DEFAULT_ALTITUDES_M = (0.0, 5000.0, 10000.0, 15000.0, 20000.0)
HEAT_LOAD_SCENARIOS_W = (5.0, 10.0, 20.0, 30.0, 50.0)
WIND_SCENARIOS_M_S = (0.0, 1.0, 5.0, 10.0)
RH_SCENARIOS = (0.20, 0.40, 0.60, 0.80)
