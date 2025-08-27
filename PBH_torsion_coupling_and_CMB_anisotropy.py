import numpy as np
import pandas as pd
from astropy import units as u
from astropy.constants import G, c, M_sun

# Constants in parsec-based units
G_pcMsun = G.to(u.pc**3 / (u.M_sun * u.s**2))
c_pc = c.to(u.pc / u.s)
M_sun_kg = M_sun.to(u.kg)

def rho_D_torsion(r, M_pbh, r_c=1e-3 * u.pc):
    """Compute defect density using SFIT torsion model"""
    M = (M_pbh.to(u.kg) / M_sun_kg) * u.M_sun
    r_s = (2 * G_pcMsun * M / c_pc**2).to(u.pc)

    r_pc = r.to(u.pc)
    r_clipped = max(r_pc.value, 1e-6 * r_s.value) * u.pc

    term = 1 + (r_s / r_clipped).value
    gaussian = np.exp(-(r_pc.value / r_c.to(u.pc).value) ** 2)
    return 0.15 * term * gaussian

def generate_dataset(n_masses=100, n_radii=20):
    """Sweep over PBH masses and radial distances"""
    mass_range = np.geomspace(1e15, 1e20, n_masses) * u.g  # PBH masses (g)
    radii = np.linspace(1e-6, 1e-2, n_radii) * u.pc         # Radii from micro-parsec to centi-parsec

    rows = []
    for M in mass_range:
        M_kg = M.to(u.kg)
        M_solar = M_kg.value / M_sun_kg.value
        r_s = (2 * G_pcMsun * M_kg / c_pc**2).to(u.pc).value

        for r in radii:
            rho_D = rho_D_torsion(r, M)
            row = {
                "M_pbh_g": M.value,
                "M_pbh_kg": M_kg.value,
                "M_pbh_Msun": M_solar,
                "Schwarzschild_radius_pc": r_s,
                "r_eval_pc": r.to(u.pc).value,
                "rho_D": rho_D
            }
            rows.append(row)

    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("PBH_torsion_dataset.csv", index=False)
    print("✅ Dataset saved to PBH_torsion_dataset.csv")
    print(f"{len(df)} rows generated across mass and radius sweeps.")
