import numpy as np
import src

def test_power_law_estimation_for_continuous_EPL1_samples():
    # Generate an array of exponents from 1.10 to 3.00 with stepsize 0.10
    alpha_pl_vec = np.arange(1.1, 3.00, 0.1)

    # Define random sample generation parameters
    n_pl = 1000
    xmin_pl, xmax_pl = 1, 100
    plot_sample = False

    for a in alpha_pl_vec:
        # Generate an EPL1 random sample
        X = src.gsdf('EPL1', a, [xmin_pl, xmax_pl], n_pl, plot_sample)
        # Estimate a continous power-law exponent
        a_hat = src.estexp(X, xmin_pl, xmax_pl, 'REAL')
        # Check whether estimation is close enough or not
        np.testing.assert_almost_equal(a, a_hat, decimal=1)

def test_power_law_estimation_for_discrete_EPL1_samples():
    # Generate an array of exponents from 1.10 to 3.00 with stepsize 0.10
    alpha_pl_vec = np.arange(1.1, 2.00, 0.1)

    # Define random sample generation parameters
    n_pl = 1000
    xmin_pl, xmax_pl = 10, 1000
    plot_sample = False

    for a in alpha_pl_vec:
        # Generate an EPL1 random sample
        X = src.gsdf('EPL1', a, [xmin_pl, xmax_pl], n_pl, plot_sample)
        # Estimate a discrete power-law exponent
        a_hat = src.estexp(X, xmin_pl, xmax_pl, 'INTS')
        # Check whether estimation is close enough or not
        np.testing.assert_almost_equal(a, a_hat, decimal=1)