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
NR_TRIALS = 2
VALID_POWER_LAW_FRACTION_MIN = 0.49

def test_automatic_pval_estimation_fail_due_to_very_large_KS_value():
    print()
    assert src.estpval(samples.X_EPL3, 'REAL', 1.21, 0.02, 91.63, 0.3053) == -1.0

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
        X, _ = src.gsdf(sample_rule, alpha_pl, bounds_pl, n, plot_sample)

        # Estimate a power-law fit using KS method bounded power-law fit
        alpha_hat, xmin_hat, xmax_hat, KS_val = src.penKS(X, 'REAL')

        # Increment count if p-value is greater than the p-value threshold
        if src.P_VAL_THRESHOLD < \
                src.estpval(X, 'REAL', alpha_hat, xmin_hat, xmax_hat, KS_val):
            valid_power_law_count = valid_power_law_count + 1

        # Quit early if goal is achieved
        if goal_is_achieved(valid_power_law_count, NR_TRIALS, VALID_POWER_LAW_FRACTION_MIN):
            break

    assert goal_is_achieved(valid_power_law_count, NR_TRIALS, VALID_POWER_LAW_FRACTION_MIN)

def goal_is_achieved(valid_power_law_count, NR_TRIALS, VALID_POWER_LAW_FRACTION_MIN):
    return valid_power_law_count/NR_TRIALS > VALID_POWER_LAW_FRACTION_MIN
