import snappy
import csv

min_vol = 1.0
max_vol = 4.0
results = []

with open('lambda_c_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Manifold', 'Volume', 'Torsion', 'Lambda_c'])

    for name in snappy.OrientableCuspedCensus:
        M = snappy.Manifold(name)
        vol = M.volume()

        if min_vol <= vol <= max_vol:
            try:
                torsion = abs(M.alexander_polynomial().coefficients()[0])
                if torsion != 0:
                    lambda_c = vol / torsion
                    results.append((name, vol, torsion, lambda_c))
                    print(f"{name}: Volume={vol:.4f}, Torsion={torsion}, λ_c={lambda_c:.4f}")
                    writer.writerow([name, vol, torsion, lambda_c])
            except Exception as e:
                print(f"Failed on {name}: {e}")
