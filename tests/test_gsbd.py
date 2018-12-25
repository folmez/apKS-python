import numpy as np
import src

def test_semiparametric_bootstrap_sample_generation():
    # Generate an EPL1 random sample
    n_pl = 2000
    xmin_pl, xmax_pl = 1, 100
    alpha_pl = 2
    plot_sample = False
    X = src.gsdf('EPL1', alpha_pl, [xmin_pl, xmax_pl], n_pl, plot_sample)

    # Semiparametric bootstrap sample parameters
    s_xmin, s_xmax = xmin_pl * 2, xmax_pl / 2

    # Check whether new power-law is in the sample
    for s_alpha in [1.0, 1.5, 2.5]:
        for dat_type in ['REAL']:#['REAL', 'INTS']:
            sX = src.gsbd(X, s_alpha, s_xmin, s_xmax, dat_type)
            s_alpha_hat = src.estexp(sX, s_xmin, s_xmax, dat_type)
            np.testing.assert_almost_equal(s_alpha, s_alpha_hat, decimal=1)
