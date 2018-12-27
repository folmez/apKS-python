import numpy as np
import src

def test_lognormal_helpers():
    mu    = [0, 0, 0, 0, 0]
    sigma = [1, 1, 1, 1, 1]
    x_vec = [0, 1, 2, 3, 4]
    pdf_x = [0.0, 0.3989422804, 0.1568740193, 0.07272825614, 0.03815345651]
    cdf_x = [0.0, 0.5, 0.7558914042, 0.8640313923, 0.917171481]

    for m, s, x, p, c in zip(mu, sigma, x_vec, pdf_x, cdf_x):
        np.testing.assert_almost_equal(src.my_lognorm_pdf(x,m,s), p, decimal=8)
        np.testing.assert_almost_equal(src.my_lognorm_cdf(x,m,s), c, decimal=8)
        np.testing.assert_almost_equal(src.my_lognorm_inv_cdf(c,m,s), x, decimal=8)
