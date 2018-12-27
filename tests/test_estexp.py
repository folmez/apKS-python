import numpy as np
import src
import samples

def test_power_law_estimation_for_continuous_samples():
    assert_exponent_estimation_works('EPL1', samples.bounds_EPL1, \
                        samples.n_EPL1, samples.xmin_EPL1, samples.xmax_EPL1)
    assert_exponent_estimation_works('EPL2', samples.bounds_EPL2, \
                        samples.n_EPL2, samples.xmin_EPL2, samples.xmax_EPL2)
    assert_exponent_estimation_works('EPL3', samples.bounds_EPL3, \
                        samples.n_EPL3, samples.xmin_EPL3, samples.xmax_EPL3)

def assert_exponent_estimation_works(sample_rule, bounds_pl, n, xmin_pl, xmax_pl):
    plot_sample = False

    # Generate an array of exponents
    alpha_pl_vec = np.arange(1.1, 3.00, 0.1) # about 20 choices

    relative_tolerance = 0.10
    for a in alpha_pl_vec:
        # Generate a random sample
        X = src.gsdf(sample_rule, a, bounds_pl, n, plot_sample)
        # Estimate a continous power-law exponent
        a_hat = src.estexp(X, xmin_pl, xmax_pl, 'REAL')
        # Check whether estimation is close enough or not
        np.testing.assert_allclose(a_hat, a, rtol = relative_tolerance)


def test_power_law_estimation_for_discrete_EPL1_samples():
    # Generate an array of exponents from 1.10 to 3.00 with stepsize 0.10
    alpha_pl_vec = np.arange(1.1, 3.00, 0.1)

    # Define random sample generation parameters
    n_pl = 1000
    xmin_pl, xmax_pl = 10, 1000
    plot_sample = False

    relative_tolerance = 0.10
    for a in alpha_pl_vec:
        # Generate an EPL1 random sample
        X = src.gsdf('EPL1', a, [xmin_pl, xmax_pl], n_pl, plot_sample)
        # Estimate a discrete power-law exponent
        a_hat = src.estexp(X, xmin_pl, xmax_pl, 'INTS')
        # Check whether estimation is close enough or not
        np.testing.assert_allclose(a_hat, a, rtol = relative_tolerance)
