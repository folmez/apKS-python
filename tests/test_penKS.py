import numpy as np
import src

def test_penKS_works_ok_continous_zero_penalty_case():
    # Generate an array of exponents from 1.10 to 3.00 with stepsize 0.10
    alpha_pl_vec = np.arange(1.1, 2.00, 0.1)

    # Generate an EPL1 random sample
    n_pl = 2000
    xmin_pl, xmax_pl = 1, 100
    plot_sample = False

    for a in alpha_pl_vec:
        X = src.gsdf('EPL1', a, [xmin_pl, xmax_pl], n_pl, plot_sample)
        alpha_hat, _, _, _, _ = src.penKS(X, 'REAL')
        np.testing.assert_almost_equal(a, alpha_hat, decimal=1)
