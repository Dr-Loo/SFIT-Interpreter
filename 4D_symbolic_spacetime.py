import sympy as sp
from sympy.diffgeom import Manifold, Patch, CoordSystem
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 1. Setup a 4D manifold with coordinates (t, x, y, z)
M = Manifold('M', 4)
patch = Patch('P', M)
coords = CoordSystem('X', patch, ['t', 'x', 'y', 'z'])
t, x, y, z = coords.coord_functions()

# 2. Define a fully symbolic metric tensor g_{µν}(t,x,y,z)
g = sp.Matrix([
    [sp.Function('g00')(t,x,y,z), sp.Function('g01')(t,x,y,z), sp.Function('g02')(t,x,y,z), sp.Function('g03')(t,x,y,z)],
    [sp.Function('g10')(t,x,y,z), sp.Function('g11')(t,x,y,z), sp.Function('g12')(t,x,y,z), sp.Function('g13')(t,x,y,z)],
    [sp.Function('g20')(t,x,y,z), sp.Function('g21')(t,x,y,z), sp.Function('g22')(t,x,y,z), sp.Function('g23')(t,x,y,z)],
    [sp.Function('g30')(t,x,y,z), sp.Function('g31')(t,x,y,z), sp.Function('g32')(t,x,y,z), sp.Function('g33')(t,x,y,z)]
])

# 3. Introduce symbolic Ricci tensor R_{µν} and scalar curvature R
Ricci = sp.Matrix([
    [sp.Function(f'R{i}{j}')(t,x,y,z) for j in range(4)]
    for i in range(4)
])
R = sp.Function('R')(t,x,y,z)

# 4. Form the Einstein tensor: G_{µν} = R_{µν} - 1/2 R g_{µν}
Einstein = Ricci - sp.Rational(1,2) * R * g

# 5. Define torsion and curvature flux functions and their combined dynamics
T = sp.Function('T')(t,x,y,z)
C = sp.Function('C')(t,x,y,z)
flux_dynamics = T + C

# 6. Quick visualization of mock torsion-curvature flux over time
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(8, 5))
t_vals = np.linspace(0, 10, 200)
torsion_vals = np.sin(t_vals)        # mock torsion flux
curvature_vals = np.cos(t_vals)      # mock curvature flux

ax.plot(t_vals, torsion_vals, label='Torsion Flux', color='C0')
ax.plot(t_vals, curvature_vals, label='Curvature Flux', color='C1')
ax.set_title('Mock Torsion–Curvature Flux Dynamics')
ax.set_xlabel('Time')
ax.set_ylabel('Flux Intensity')
ax.legend()

# Save the figure
output_path = 'torsion_curvature_flux_dynamics.png'
fig.savefig(output_path)
plt.close()

# 7. Output the symbolic Einstein tensor and file location
sp.pprint(Einstein)
print("Saved flux dynamics plot to:", output_path)
