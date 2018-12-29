import numpy as np
import pytest
import src
import samples

def test_a_ridiculously_large_penalty_coefficient_for_EPL1():
    # EPL1 is a pure bounded power-law, a ridiculously large penalty coefficient
    # should return the whole sample as a valid power-law fit
    a_hat, xmin_hat, xmax_hat, _ =  src.penKS(samples.X_EPL1, 'REAL', pen_slope = 1e5)
    assert [xmin_hat, xmax_hat] == [np.amin(samples.X_EPL1), np.amax(samples.X_EPL1)]

def test_interval_has_data():
    assert_interval_has_data_works(samples.X_EPL1)
    assert_interval_has_data_works(samples.X_EPL2)
    assert_interval_has_data_works(samples.X_EPL3)

def assert_interval_has_data_works(X):
    assert src.interval_has_data(X, np.amin(X), np.amax(X)) == True
    assert src.interval_has_data(X, 0.1 * np.amin(X), np.amin(X)) == False
    assert src.interval_has_data(X, np.amax(X), 10 * np.amax(X)) == False

@pytest.mark.slow
def test_penKS_works_ok_continous_zero_penalty_case():
    assert_penKS_works('EPL1', samples.bounds_EPL1, samples.n_EPL1)
    assert_penKS_works('EPL2', samples.bounds_EPL2, samples.n_EPL2)
    assert_penKS_works('EPL3', samples.bounds_EPL3, samples.n_EPL3)

def assert_penKS_works(sample_rule, bounds_pl, n):
    # Generate an array of exponents
    alpha_pl_vec = np.arange(1.5, 3.00, 0.5)

    plot_sample = False
    relative_tolerance = 0.10
    for a in alpha_pl_vec:
        # Generate a random sample
        X = src.gsdf(sample_rule, a, bounds_pl, n, plot_sample)

        # Test KS method for bounded power-law fitting
        a_hat, _, _, _ = src.penKS(X, 'REAL')
        np.testing.assert_allclose(a_hat, a, rtol = relative_tolerance)

        # Test penalized KS method for bounded power-law fitting with a tiny penalty
        a_hat, _, _, _ = src.penKS(X, 'REAL', pen_slope = 1e-4)
        np.testing.assert_allclose(a_hat, a, rtol = relative_tolerance)

def test_penalty_matrix_computation():
    """
    Test whether the penalty term computation works as expected
    """
    pen = 0.1

    KS_mat = np.zeros([3, 4])
    xmin_vec = np.asarray([1.0, 2.0, 3.0])
    xmax_vec = np.asarray([4.0, 5.0, 6.0, 7.0])
    penalty_term = np.asarray([[(-1)*pen*np.log(xmax_vec[0]/xmin_vec[0]), \
                                (-1)*pen*np.log(xmax_vec[1]/xmin_vec[0]), \
                                (-1)*pen*np.log(xmax_vec[2]/xmin_vec[0]), \
                                (-1)*pen*np.log(xmax_vec[3]/xmin_vec[0]), \
                                ], \
                                [(-1)*pen*np.log(xmax_vec[0]/xmin_vec[1]), \
                                (-1)*pen*np.log(xmax_vec[1]/xmin_vec[1]), \
                                (-1)*pen*np.log(xmax_vec[2]/xmin_vec[1]), \
                                (-1)*pen*np.log(xmax_vec[3]/xmin_vec[1]), \
                                ], \
                                [(-1)*pen*np.log(xmax_vec[0]/xmin_vec[2]), \
                                (-1)*pen*np.log(xmax_vec[1]/xmin_vec[2]), \
                                (-1)*pen*np.log(xmax_vec[2]/xmin_vec[2]), \
                                (-1)*pen*np.log(xmax_vec[3]/xmin_vec[2]), \
                                ]])
    np.testing.assert_array_almost_equal(penalty_term, \
        src.calc_penalized_KS_metric(KS_mat, pen, xmin_vec, xmax_vec), \
        decimal=16)

    KS_mat = np.zeros([1, 4])
    xmin_vec = np.asarray([3.0])
    xmax_vec = np.asarray([4.0, 5.0, 6.0, 7.0])
    penalty_term = np.asarray([[(-1)*pen*np.log(xmax_vec[0]/xmin_vec[0]), \
                                (-1)*pen*np.log(xmax_vec[1]/xmin_vec[0]), \
                                (-1)*pen*np.log(xmax_vec[2]/xmin_vec[0]), \
                                (-1)*pen*np.log(xmax_vec[3]/xmin_vec[0]), \
                                ]])
    np.testing.assert_array_almost_equal(penalty_term, \
        src.calc_penalized_KS_metric(KS_mat, pen, xmin_vec, xmax_vec), \
        decimal=16)

    KS_mat = np.zeros([3, 1])
    xmin_vec = np.asarray([1.0, 2.0, 3.0])
    xmax_vec = np.asarray([4.0])
    penalty_term = np.asarray([[(-1)*pen*np.log(xmax_vec[0]/xmin_vec[0])], \
                                [(-1)*pen*np.log(xmax_vec[0]/xmin_vec[1])], \
                                [(-1)*pen*np.log(xmax_vec[0]/xmin_vec[2])]])
    np.testing.assert_array_almost_equal(penalty_term, \
        src.calc_penalized_KS_metric(KS_mat, pen, xmin_vec, xmax_vec), \
        decimal=16)
