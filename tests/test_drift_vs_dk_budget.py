from ._helpers import davis_kahan_synthetic

def test_davis_kahan_budget_kernel_block():
    drift, Enorm, gap = davis_kahan_synthetic(n=40, k=3, eps=1e-3)
    # Davis–Kahan-style bound: ||P - P'|| <= ||E|| / gap (up to tiny numerical slack)
    assert drift <= Enorm / gap + 5e-4
