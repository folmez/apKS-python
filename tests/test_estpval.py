import numpy as np
import src

"""
To test p-value estimation implementation code, many random samples containing a
bounded power-law will be tested. Not each of these random samples will contain
a valid power-law. Therefore, the assertion will be that the fraction of valid
power-laws is greater than a fraction
"""
NR_TRIALS = 10
VALID_POWER_LAW_FRACTION_MIN = 0.5

def test_estpval_EPL1():
    # Define EPL1 random sample parameters
    n_pl = 1000
    xmin_pl, xmax_pl = 1, 100
    alpha_pl = 2
    plot_sample = False

    # Generate data, estimate power-law fit, validate fit and count
    valid_power_law_count = 0
    for i in range(NR_TRIALS):
        # Generate an EPL1 sample
        X = src.gsdf('EPL1', alpha_pl, [xmin_pl, xmax_pl], n_pl, plot_sample)

        # Estimate a power-law fit using KS method bounded power-law fit
        alpha_hat, xmin_hat, xmax_hat, KS_val, _ = src.penKS(X, 'REAL')

        # Increment count if p-value is greater than the p-value threshold
        if src.P_VAL_THRESHOLD < \
                src.estpval(X, 'REAL', alpha_hat, xmin_hat, xmax_hat, KS_val):
            valid_power_law_count = valid_power_law_count + 1

    assert valid_power_law_count/NR_TRIALS > VALID_POWER_LAW_FRACTION_MIN
