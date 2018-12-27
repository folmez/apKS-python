import numpy as np
import src
import samples

def test_semiparametric_bootstrap_sample_generation():
    assert_sb_works(samples.X_EPL1, samples.xmin_EPL1, \
                                        samples.xmax_EPL1, samples.alpha_EPL1)
    assert_sb_works(samples.X_EPL2, samples.xmin_EPL2, \
                                        samples.xmax_EPL2, samples.alpha_EPL2)
    assert_sb_works(samples.X_EPL3, samples.xmin_EPL3, \
                                        samples.xmax_EPL3, samples.alpha_EPL3)

def assert_sb_works(X, xmin_pl, xmax_pl, alpha_pl):
    # Semiparametric bootstrap sample parameters
    s_xmin, s_xmax = xmin_pl * 2, xmax_pl / 2

    # Check whether new power-law is in the sample
    relative_tolerance = 0.10
    for s_alpha in [1.0, 1.5, 2.5]:
        for dat_type in ['REAL', 'INTS']:
            sX = src.gsbd(X, s_alpha, s_xmin, s_xmax, dat_type)
            s_alpha_hat = src.estexp(sX, s_xmin, s_xmax, dat_type)
            np.testing.assert_allclose(s_alpha_hat, s_alpha, \
                                                    rtol = relative_tolerance)
