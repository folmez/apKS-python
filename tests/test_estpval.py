import numpy as np
import src

def test_estpval_EPL1():
    # Generate an EPL1 random sample
    n_pl = 1000
    xmin_pl, xmax_pl = 1, 100
    alpha_pl = 2
    plot_sample = False
    X = src.gsdf('EPL1', alpha_pl, [xmin_pl, xmax_pl], n_pl, plot_sample)

    # Estimate a power-law fit using KS method bounded power-law fit
    alpha_hat, xmin_hat, xmax_hat, KS_val, _ = src.penKS(X, 'REAL')

    # Assert that p-value is greater than the threshold
    assert src.P_VAL_THRESHOLD < \
                src.estpval(X, 'REAL', alpha_hat, xmin_hat, xmax_hat, KS_val)
