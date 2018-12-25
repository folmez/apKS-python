import numpy as np
import src

def test_KS_distance_is_worse_when_power_law_exponent_is_wrong_continuous():
    # Generate an EPL1 random sample
    n_pl = 1000
    xmin_pl, xmax_pl = 1, 100
    alpha_pl = 2
    plot_sample = False
    X = src.gsdf('EPL1', alpha_pl, [xmin_pl, xmax_pl], n_pl, plot_sample)

    # Compute KS distance between X and true power-law
    KS_dist = src.estKS(X, xmin_pl, xmax_pl, alpha_pl, 'REAL')

    # Check that the KS distance is worse (larger!) when alpha is wrong
    for d_alpha in np.linspace(0.20, 1.00, num=100):
        assert KS_dist < src.estKS(X, xmin_pl, xmax_pl, alpha_pl-d_alpha, 'REAL')
        assert KS_dist < src.estKS(X, xmin_pl, xmax_pl, alpha_pl+d_alpha, 'REAL')
