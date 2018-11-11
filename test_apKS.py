import numpy as np

from gsdf import gsdf
from estexp import estexp
from elspd import elspd
from estKS import estKS

def test_estKS1():
    alpha_pl = 2
    n_pl = 1000
    xmin_pl = 1
    xmax_pl = 100
    plot_sample = False
    X = gsdf('EPL1', alpha_pl, [xmin_pl, xmax_pl], n_pl, plot_sample)
    result_bool = True
    for d_alpha in np.arange(20,100)/100:
        result_bool = result_bool and \
                    estKS(X, xmin_pl, xmax_pl, alpha_pl, 'REAL') < \
                    estKS(X, xmin_pl, xmax_pl, alpha_pl-d_alpha, 'REAL')\
                    and \
                    estKS(X, xmin_pl, xmax_pl, alpha_pl, 'REAL') < \
                    estKS(X, xmin_pl, xmax_pl, alpha_pl+d_alpha, 'REAL')
    assert result_bool

def test_elspd_1():
    # Generate numbers from 1 to 100
    X = np.arange(100) + 1
    # The number of logarithmically spaced points in each decade
    # m = 4 should return [1,2,4,7,14,27,52,100]
    m = 4
    np.testing.assert_array_equal(elspd(X, m), \
                                [1.0,2.0,4.0,7.0,14.0,27.0,52.0,100.0])

def test_elspd_2():
    # Generate numbers from 1 to 100
    X = np.arange(100) + 1
    # The number of logarithmically spaced points in each decade
    # m = 4 should return [1,2,4,7,14,27,52,100]
    m = 100
    np.testing.assert_array_equal(elspd(X, m), X)

def test_estexp_EPL1_REAL():
    alpha_pl_vec = np.arange(1.1, 3.00, 0.1)
    n_pl = 1000
    xmin_pl = 1
    xmax_pl = 100
    plot_sample = False
    alpha_hat_pl_vec = [\
    estexp(gsdf('EPL1', a, [xmin_pl, xmax_pl], n_pl, plot_sample), \
                                            xmin_pl, xmax_pl, 'REAL') \
                                            for a in alpha_pl_vec]
    np.testing.assert_almost_equal(alpha_hat_pl_vec, alpha_pl_vec, decimal=1)

def test_estexp_EPL1_INTS():
    alpha_pl_vec = np.arange(1.1, 2.00, 0.1)
    n_pl = 1000
    xmin_pl = 10
    xmax_pl = 1000
    plot_sample = False
    alpha_hat_pl_vec = [\
    estexp(gsdf('EPL1', a, [xmin_pl, xmax_pl], n_pl, plot_sample), \
                                            xmin_pl, xmax_pl, 'INTS') \
                                            for a in alpha_pl_vec]
    np.testing.assert_almost_equal(alpha_hat_pl_vec, alpha_pl_vec, decimal=1)

## toy testing
#def func(x):
#    return x + 1
#
#def test_answer_1():
#    assert func(4) == 5
#
#def test_answer_2():
#    assert func(5) == 6
#
#def test_answer_3():
#    assert func(7) == 6
#
#def test_answer_4():
#    assert func(10) == 11
