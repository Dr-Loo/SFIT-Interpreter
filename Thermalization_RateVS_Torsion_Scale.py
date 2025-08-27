import numpy as np
import matplotlib.pyplot as plt

Lambda_values = np.linspace(0.1, 10.0, 100)  # GeV^-1
dNeff_dt_values = [dNeff_dt(Lambda_torsion=Lambda) for Lambda in Lambda_values]

plt.figure(figsize=(10, 6))
plt.plot(Lambda_values, dNeff_dt_values, 'b-', lw=2)
plt.axhline(0.25, color='r', linestyle='--', label="Planck ΔN_eff < 0.25")
plt.xlabel("Torsion Scale Λ_torsion (GeV⁻¹)")
plt.ylabel("dN_eff/dt")
plt.title("Thermalization Rate vs. Torsion Scale")
plt.legend()
plt.grid(True)
plt.show()