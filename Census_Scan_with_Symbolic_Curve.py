import snappy
import numpy as np

# Target volume range (tune as needed)
min_vol = 2.0
max_vol = 3.0

# Initialize storage
results = []

# Loop through orientable cusped census
for name in snappy.OrientableCuspedCensus:
    M = snappy.Manifold(name)
    vol = M.volume()
    
    if min_vol <= vol <= max_vol:
        # Example symbolic proxy: lambda_c candidate as volume/torsion
        try:
            torsion = abs(M.alexander_polynomial().coefficients()[0])  # symbolic stand-in
            if torsion != 0:
                lambda_c = vol / torsion
                results.append((name, vol, torsion, lambda_c))
        except:
            continue

# Sort and print results
sorted_results = sorted(results, key=lambda x: x[3])  # Sort by lambda_c
for name, vol, torsion, lc in sorted_results:
    print(f"{name}: Volume={vol:.4f}, Torsion={torsion}, λ_c={lc:.4f}")
