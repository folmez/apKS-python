import numpy as np
import src
import samples

def test_penKS_works_ok_continous_zero_penalty_case():
    assert_penKS_works('EPL1', samples.bounds_EPL1, samples.n_EPL1)
    assert_penKS_works('EPL2', samples.bounds_EPL2, samples.n_EPL2)
    assert_penKS_works('EPL3', samples.bounds_EPL3, samples.n_EPL3)

def assert_penKS_works(sample_rule, bounds_pl, n):
    # Generate an array of exponents
    alpha_pl_vec = np.arange(1.5, 3.00, 0.1)

    plot_sample = False
    relative_tolerance = 0.10
    for a in alpha_pl_vec:
        X = src.gsdf(sample_rule, a, bounds_pl, n, plot_sample)
        a_hat, _, _, _, _ = src.penKS(X, 'REAL')
        np.testing.assert_allclose(a_hat, a, rtol = relative_tolerance)
