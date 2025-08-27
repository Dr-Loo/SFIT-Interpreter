import numpy as np
from sfit_sgf.core import davis_kahan_synthetic

def main():
    # small symmetric perturbation; expect drift ≲ ||E||/gap
    drift, E_norm, gap = davis_kahan_synthetic(n=40, k=3, eps=1e-3, seed=0)
    print(f"||P - P'|| = {drift:.6e}")
    print(f"||E||/gap  = {(E_norm/gap):.6e}")
    print("DK bound satisfied:", drift <= E_norm/gap + 5e-3)

if __name__ == "__main__":
    main()
