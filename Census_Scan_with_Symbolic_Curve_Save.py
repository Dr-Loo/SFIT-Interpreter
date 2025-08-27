import snappy
import csv
import numpy as np

# Settings: adjust volume window as needed
min_vol = 1.0
max_vol = 4.0
output_file = 'lambda_c_scan_results.csv'

# Initialize results container
results = []

# Open CSV file for writing results
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Manifold', 'Volume', 'Homology_Rank', 'Torsion_Order_Proxy', 'Lambda_c'])

    # Loop through manifolds in the OrientableCuspedCensus
    for name in snappy.OrientableCuspedCensus:
        M = snappy.Manifold(name)
        vol = M.volume()

        if min_vol <= vol <= max_vol:
            try:
                # Homology group extraction
                homology = M.homology()
                homology_rank = len(homology)

                # Torsion order proxy: product of finite group orders
                torsion_order = np.prod([
                    group.order() for group in homology
                    if hasattr(group, 'order') and group.order() != 0
                ])

                # Avoid divide by zero
                if torsion_order > 0:
                    lambda_c = vol / torsion_order
                    results.append((name, vol, homology_rank, torsion_order, lambda_c))
                    print(f"{name}: Vol={float(vol):.4f}, Torsion={torsion_order}, λ_c={float(lambda_c):.4f}")
                    writer.writerow([name, float(vol), homology_rank, torsion_order, float(lambda_c)])
            except Exception as e:
                print(f"Failed on {name}: {e}")
