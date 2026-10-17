import numpy as np
from validation import diaz_lobo_reproduction
from radiation import broadband_face_budget, FaceViews
from convection import combined_h, q_conv
from materials import load_materials
from atmosphere import atmosphere_at
from scipy.optimize import brentq

def diagnose_diaz_lobo():
    materials = load_materials()
    aao = materials["AAO_AL_DIAZ_LOBO"]
    bare = materials["BARE_ALUMINUM_OXIDIZED"]
    
    area = 0.05 * 0.05
    t_air = 34.6 + 273.15
    atm = atmosphere_at(0.0, rh_surface=0.273)
    atm.temperature_k = t_air
    atm.density_kg_m3 = atm.pressure_pa / (287.05287 * t_air)
    g = 736.0
    views = FaceViews(1.0, 0.0, 0.0)
    u = 1.0
    
    def get_balance_terms(mat, ts):
        b = broadband_face_budget(mat, ts, atm, area, views, t_air, 0.9, "clear", 0.0)
        h_obj = combined_h(atm, ts, u, 0.05)
        qc = q_conv(h_obj.h_combined, area, ts, t_air)
        qsol = mat.solar_absorptance * g * area
        
        return {
            "T_s": ts - 273.15,
            "q_rad_out": b.q_rad_out / area,
            "q_atm_in": b.q_atm_in / area,
            "q_net_rad": (b.q_rad_out - b.q_atm_in) / area,
            "q_conv": qc / area,
            "q_sol": qsol / area,
            "q_total": (b.q_rad_out + qc - b.q_atm_in - qsol) / area,
            "h": h_obj.h_combined,
            "Re": h_obj.re,
            "Nu": h_obj.nu
        }

    def f(ts, mat):
        b = broadband_face_budget(mat, ts, atm, area, views, t_air, 0.9, "clear", 0.0)
        h = combined_h(atm, ts, u, 0.05).h_combined
        qc = q_conv(h, area, ts, t_air)
        qsol = mat.solar_absorptance * g * area
        return b.q_rad_out + qc - b.q_atm_in - qsol

    ts_aao = brentq(f, 250, 400, args=(aao,))
    ts_bare = brentq(f, 250, 400, args=(bare,))
    
    print("--- Surface Energy Balance at Steady State ---")
    for name, mat, ts in [("AAO-Al", aao, ts_aao), ("Bare Al", bare, ts_bare)]:
        terms = get_balance_terms(mat, ts)
        print(f"\n{name}:")
        print(f"  T_s: {terms['T_s']:.2f} C")
        print(f"  Absorbed Solar:  {terms['q_sol']:>10.2f} W/m2")
        print(f"  Emitted LW:      {terms['q_rad_out']:>10.2f} W/m2")
        print(f"  Atmospheric LW:  {-terms['q_atm_in']:>10.2f} W/m2")
        print(f"  Net Radiative:   {terms['q_net_rad']:>10.2f} W/m2")
        print(f"  Convection:     {terms['q_conv']:>10.2f} W/m2")
        print(f"  Total Balance:   {terms['q_total']:>10.2e} W/m2")
        print(f"  h: {terms['h']:.2f} W/m2K, Re: {terms['Re']:.2f}, Nu: {terms['Nu']:.2f}")

    # Peak Cooling check
    def peak_cooling(mat):
        b_amb = broadband_face_budget(mat, t_air, atm, area, views, t_air, 0.9, "clear", 0.0)
        return (b_amb.q_rad_out - b_amb.q_atm_in - mat.solar_absorptance * g * area) / area

    print("\n--- Peak Cooling Analysis ---")
    print(f"AAO-Al Peak Cooling: {peak_cooling(aao):.2f} W/m2")
    print(f"Bare Al Peak Cooling: {peak_cooling(bare):.2f} W/m2")

    print("\n--- Wind Sensitivity Sweep ---")
    for wind in [0.5, 1.0, 2.0, 5.0]:
        def f_wind(ts, mat):
            b = broadband_face_budget(mat, ts, atm, area, views, t_air, 0.9, "clear", 0.0)
            h = combined_h(atm, ts, wind, 0.05).h_combined
            qc = q_conv(h, area, ts, t_air)
            qsol = mat.solar_absorptance * g * area
            return b.q_rad_out + qc - b.q_atm_in - qsol
        
        ts_aao_w = brentq(f_wind, 250, 400, args=(aao,))
        ts_bare_w = brentq(f_wind, 250, 400, args=(bare,))
        print(f"Wind {wind} m/s: deltaT = {ts_bare_w - ts_aao_w:.2f} K")

if __name__ == "__main__":
    diagnose_diaz_lobo()
