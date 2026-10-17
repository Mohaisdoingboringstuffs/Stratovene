import numpy as np
import pandas as pd
from transient import simulate_24h
from materials import load_materials
from config import SimulationConfig, EnvironmentConfig, ElectronicsConfig, GeometryConfig, TransientConfig
from datetime import datetime

def diagnose_issue2():
    # Setup config
    cfg = SimulationConfig(
        environment=EnvironmentConfig(altitude_m=20000),
        electronics=ElectronicsConfig(heat_load_w=20.0),
        geometry=GeometryConfig(length_m=0.3, width_m=0.3, height_m=0.2, wall_thickness_m=0.005),
        transient=TransientConfig(duration_s=86400, dt_s=60)
    )
    
    mats = load_materials()
    test_mats = {
        "bare": mats["BARE_ALUMINUM_OXIDIZED"],
        "white": mats["WHITE_REFERENCE_PLACEHOLDER"],
        "aao": mats["STRATOVANE_AAO_LITERATURE"]
    }
    
    results = {}
    for name, mat in test_mats.items():
        res = simulate_24h(cfg, mat)
        results[name] = res.t_electronics_k - 273.15

    df = pd.DataFrame(results)
    print("--- Min Temperatures ---")
    print(df.min())
    
    print("\n--- Time of Min ---")
    for col in df.columns:
        print(f"{col}: index {df[col].idxmin()}")
    
    print("\n--- First 5 values ---")
    print(df.head())

if __name__ == "__main__":
    diagnose_issue2()
