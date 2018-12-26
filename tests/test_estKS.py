import numpy as np
import src
import samples

def test_KS_distance_is_worse_when_power_law_exponent_is_wrong_continuous():
    assert_KS_estimation_works(samples.X_EPL1, samples.xmin_EPL1, \
                                        samples.xmax_EPL1, samples.alpha_EPL1)
    assert_KS_estimation_works(samples.X_EPL2, samples.xmin_EPL2, \
                                        samples.xmax_EPL2, samples.alpha_EPL2)

def assert_KS_estimation_works(X, xmin_pl, xmax_pl, alpha_pl):
    # Compute KS distance between X and true power-law
    KS_dist = src.estKS(X, xmin_pl, xmax_pl, alpha_pl, 'REAL')

    # Check that the KS distance is worse (larger!) when alpha is wrong
    for d_alpha in np.linspace(0.20, 1.00, num=100):
        assert KS_dist < src.estKS(X, xmin_pl, xmax_pl, alpha_pl-d_alpha, 'REAL')
        assert KS_dist < src.estKS(X, xmin_pl, xmax_pl, alpha_pl+d_alpha, 'REAL')
