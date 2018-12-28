import numpy as np
import src
import samples

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
        a_hat, _, _, _ = src.penKS(X, 'REAL', pen_slope = 0.000001)
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
