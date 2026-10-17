import numpy as np

from config import SimulationConfig
from materials import load_materials
from transient import simulate_24h


def _mins():
    cfg = SimulationConfig()
    mats = load_materials()
    out = {}
    for mid in ("BARE_ALUMINUM_OXIDIZED", "WHITE_REFERENCE_PLACEHOLDER", "STRATOVANE_AAO_LITERATURE"):
        r = simulate_24h(cfg, mats[mid])
        out[mid] = (r, int(np.argmin(r.t_electronics_k)))
    return out


def test_minimum_is_not_an_initial_condition_artifact():
    # Regression: the old fixed IC T_air+15 made index 0 the coldest point and
    # gave the SAME reported minimum for every material.
    result = _mins()
    for mid, (r, idx) in result.items():
        assert idx > 0, f"{mid} minimum still pinned at t=0"
        assert idx == int(np.argmin(r.t_electronics_k))
        assert r.t_min_c == r.t_electronics_k[idx] - 273.15


def test_material_signal_survives_spinup():
    result = _mins()
    minima = [r.t_min_c for r, _ in result.values()]
    assert len(set(round(m, 3) for m in minima)) == len(minima)
    # AAO should not be identical to bare aluminum.
    assert abs(result["BARE_ALUMINUM_OXIDIZED"][0].t_min_c - result["STRATOVANE_AAO_LITERATURE"][0].t_min_c) > 0.1
