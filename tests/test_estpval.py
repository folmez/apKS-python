import numpy as np
import pytest
import src
import samples

"""
To test p-value estimation implementation code, many random samples containing a
bounded power-law will be tested. Not each of these random samples will contain
a valid power-law. Therefore, the assertion will be that the fraction of valid
power-laws is greater than a fraction
"""
NR_TRIALS = 10
VALID_POWER_LAW_FRACTION_MIN = 0.5

@pytest.mark.slow
def test_estpval():
    assert_pval_estimation_works('EPL1', samples.alpha_EPL1, \
                                    samples.bounds_EPL1, samples.n_EPL1)
    assert_pval_estimation_works('EPL2', samples.alpha_EPL2, \
                                    samples.bounds_EPL2, samples.n_EPL2)
    assert_pval_estimation_works('EPL3', samples.alpha_EPL3, \
                                    samples.bounds_EPL3, samples.n_EPL3)

def assert_pval_estimation_works(sample_rule, alpha_pl, bounds_pl, n):
    # Generate data, estimate power-law fit, validate fit and count
    plot_sample = False
    valid_power_law_count = 0
    for i in range(NR_TRIALS):
        # Generate data
        X = src.gsdf(sample_rule, alpha_pl, bounds_pl, n, plot_sample)

        # Estimate a power-law fit using KS method bounded power-law fit
        alpha_hat, xmin_hat, xmax_hat, KS_val = src.penKS(X, 'REAL')

        # Increment count if p-value is greater than the p-value threshold
        if src.P_VAL_THRESHOLD < \
                src.estpval(X, 'REAL', alpha_hat, xmin_hat, xmax_hat, KS_val):
            valid_power_law_count = valid_power_law_count + 1

    assert valid_power_law_count/NR_TRIALS > VALID_POWER_LAW_FRACTION_MIN
