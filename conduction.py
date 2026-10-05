"""Wall conduction and aluminum alloy properties.

Q_cond = (k A / L) (T_hot - T_cold)
"""

from __future__ import annotations

from dataclasses import dataclass

# ASM Handbook / MatWeb typical room-temperature values. LITERATURE.
# Do not assume all alloys have identical k.
ALLOYS = {
    "6061-T6": {
        "k_w_m_k": 167.0,
        "rho_kg_m3": 2700.0,
        "cp_j_kg_k": 896.0,
        "source": "ASM Handbook typical 6061-T6 at ~25 C",
        "dk_dT": 0.0,  # no k(T) table bundled; constant unless user supplies
    },
    "6063": {
        "k_w_m_k": 200.0,
        "rho_kg_m3": 2700.0,
        "cp_j_kg_k": 900.0,
        "source": "ASM/MatWeb typical 6063-T5 ~25 C",
        "dk_dT": 0.0,
    },
    "1050": {
        "k_w_m_k": 222.0,
        "rho_kg_m3": 2705.0,
        "cp_j_kg_k": 900.0,
        "source": "ASM/MatWeb typical commercially pure 1050",
        "dk_dT": 0.0,
    },
    "user-defined": {
        "k_w_m_k": 167.0,
        "rho_kg_m3": 2700.0,
        "cp_j_kg_k": 896.0,
        "source": "USER INPUT placeholder (defaults to 6061-T6 until replaced)",
        "dk_dT": 0.0,
    },
}

# Al2O3 solid density. CRC Handbook. Porous effective density = rho (1-phi)
RHO_AL2O3_KG_M3 = 3950.0
CP_AL2O3_J_KG_K = 775.0  # LITERATURE order of magnitude, room T alumina


@dataclass
class WallConduction:
    k_w_m_k: float
    area_m2: float
    thickness_m: float

    def conductance(self) -> float:
        return self.k_w_m_k * self.area_m2 / self.thickness_m

    def q(self, t_hot: float, t_cold: float) -> float:
        return self.conductance() * (t_hot - t_cold)


def alloy_k(name: str, t_k: float | None = None) -> float:
    meta = ALLOYS[name]
    k = meta["k_w_m_k"]
    if t_k is not None and meta["dk_dT"]:
        k = k + meta["dk_dT"] * (t_k - 298.15)
    return k
